//! Verifies architectural DAG conformance between `src/architecture.rs` and source modules.
//!
//! Checks:
//! 1. Every `ArchitectureComponent` variant has a matching root file declaring
//!    `architecture_component!(<Variant>);`.
//! 2. Top-level `src/*.rs` library modules match the catalog one-to-one.
//! 3. Production code (after lexing away comments, literals, and `#[cfg(...test...)]` items)
//!    only imports internal components declared as dependencies in `define_architecture!`.
//! 4. Crates listed under `owns_external` in `define_architecture!` are referenced in production
//!    code only by their owning component.
//! 5. Declared component submodule DAGs (`modules: { ... }`) match on-disk files and allowed
//!    sibling dependencies, and `crates/adfontes-py/src/lib.rs` delegates strictly to `adfontes::cli`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use adfontes::architecture::ArchitectureComponent;

#[test]
fn every_catalog_component_has_matching_root_declaration() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let missing = collect_missing_declarations(&src_dir);
    assert_eq!(missing, Vec::<String>::new());
}

#[test]
fn top_level_src_modules_match_architecture_catalog() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let actual = collect_actual_library_stems(&src_dir).expect("src directory must be readable");
    assert_eq!(actual, collect_expected_library_stems());
}

#[test]
fn production_imports_respect_declared_component_dag_and_external_ownership() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let violations = collect_dag_and_ownership_violations(manifest_dir)
        .expect("architecture files and component sources must be readable");
    assert_eq!(violations, Vec::<String>::new());
}

#[test]
fn declared_submodules_match_files_and_respect_submodule_dag() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let violations =
        collect_submodule_dag_violations(&src_dir).expect("declared submodules must be readable");
    assert_eq!(violations, Vec::<String>::new());
}

#[test]
fn non_production_stripper_preserves_code_after_inline_test_item_and_catches_grouped_imports() {
    let synthetic = indoc::indoc! {r#"
        #[cfg(test)]
        impl Foo {
            fn test_only() { let _ = crate::transport::HttpClient::default(); }
        }
        // comment with crate::resolve::resolve_target
        /* block comment /* nested */ crate::stage::stage_document */
        pub fn prod() {
            let escaped = ['\'', '"', '…'];
            let _ = escaped;
            let url = "https://example.org/crate::pdf::x";
            use crate::{model, transport};
        }
    "#};
    let stripped = strip_non_production_rust(synthetic);
    let references = extract_module_references(&stripped);
    assert!(references.crate_targets.contains("model"));
    assert!(references.crate_targets.contains("transport"));
    assert!(!references.crate_targets.contains("resolve"));
    assert!(!references.crate_targets.contains("pdf"));
}

fn collect_dag_and_ownership_violations(manifest_dir: &Path) -> io::Result<Vec<String>> {
    let src_dir = manifest_dir.join("src");
    let architecture_source = fs::read_to_string(src_dir.join("architecture.rs"))?;
    let spec = parse_architecture_spec(&architecture_source)?;
    let known_modules: BTreeSet<String> = spec.components.keys().cloned().collect();

    let mut violations = Vec::new();
    for (module_name, rule) in &spec.components {
        for file_path in collect_component_source_files(&src_dir, module_name)? {
            let source = fs::read_to_string(&file_path)?;
            let production_code = strip_non_production_rust(&source);
            let references = extract_module_references(&production_code);
            check_component_edges(
                &file_path,
                module_name,
                rule,
                &known_modules,
                &references,
                &mut violations,
            );
            check_external_ownership(
                &file_path,
                module_name,
                &spec.external_owners,
                &references,
                &mut violations,
            );
        }
    }

    let py_bridge = manifest_dir
        .join("crates")
        .join("adfontes-py")
        .join("src")
        .join("lib.rs");
    if py_bridge.is_file() {
        let source = fs::read_to_string(&py_bridge)?;
        let production_code = strip_non_production_rust(&source);
        let references = extract_module_references(&production_code);
        let allowed_cli = ComponentRule {
            allowed_deps: BTreeSet::from(["cli".to_owned()]),
        };
        check_component_edges(
            &py_bridge,
            "bin",
            &allowed_cli,
            &known_modules,
            &references,
            &mut violations,
        );
        check_external_ownership(
            &py_bridge,
            "bin",
            &spec.external_owners,
            &references,
            &mut violations,
        );
    }
    Ok(violations)
}

fn collect_submodule_dag_violations(src_dir: &Path) -> io::Result<Vec<String>> {
    let architecture_source = fs::read_to_string(src_dir.join("architecture.rs"))?;
    let spec = parse_architecture_spec(&architecture_source)?;
    let mut violations = Vec::new();

    for (component_name, submodules) in &spec.submodules {
        let component_dir = src_dir.join(component_name);
        let declared_names: BTreeSet<String> = submodules.keys().cloned().collect();
        let actual_names = collect_rust_file_stems(&component_dir)?;
        if actual_names != declared_names {
            violations.push(format!(
                "{}: submodule files {:?} do not match declared {:?}",
                component_dir.display(),
                actual_names,
                declared_names
            ));
        }

        for (submodule, allowed_deps) in submodules {
            let file_path = component_dir.join(format!("{submodule}.rs"));
            let source = fs::read_to_string(&file_path)?;
            let production_code = strip_non_production_rust(&source);
            let references = extract_module_references(&production_code);
            check_submodule_edges(
                &file_path,
                component_name,
                submodule,
                allowed_deps,
                &declared_names,
                &references,
                &mut violations,
            );
        }
    }

    Ok(violations)
}

struct ArchitectureSpec {
    components: BTreeMap<String, ComponentRule>,
    external_owners: BTreeMap<String, String>,
    submodules: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
}

struct ComponentRule {
    allowed_deps: BTreeSet<String>,
}

fn parse_architecture_spec(source: &str) -> io::Result<ArchitectureSpec> {
    let mut components = BTreeMap::new();
    let mut external_owners = BTreeMap::new();
    let mut submodules: BTreeMap<String, BTreeMap<String, BTreeSet<String>>> = BTreeMap::new();
    let Some(start) = source.find("define_architecture! {") else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "define_architecture! macro invocation not found",
        ));
    };
    let body = &source[start..];
    let mut active_submodule_owner: Option<String> = None;

    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(ref owner) = active_submodule_owner {
            if trimmed.starts_with("})") {
                active_submodule_owner = None;
                continue;
            }
            if let Some((sub_lhs, sub_rhs)) = trimmed.split_once("=>") {
                let sub_name = sub_lhs.trim().to_owned();
                let deps = parse_bracket_list(sub_rhs).into_iter().collect();
                submodules
                    .entry(owner.clone())
                    .or_default()
                    .insert(sub_name, deps);
            }
            continue;
        }

        let Some((lhs, rhs)) = trimmed.split_once("=>") else {
            continue;
        };
        let variant = lhs.trim();
        if !variant
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_uppercase())
        {
            continue;
        }
        let module_name = pascal_to_snake(variant);
        let deps = parse_bracket_list(rhs)
            .into_iter()
            .map(|dep| pascal_to_snake(&dep))
            .collect();
        if let Some((_, external_tail)) = rhs.split_once("owns_external:") {
            for external_crate in parse_bracket_list(external_tail) {
                let cleaned = external_crate.trim_matches('"').to_owned();
                if !cleaned.is_empty() {
                    external_owners.insert(cleaned, module_name.clone());
                }
            }
        }
        if rhs.contains("modules: {") {
            active_submodule_owner = Some(module_name.clone());
        }
        components.insert(module_name, ComponentRule { allowed_deps: deps });
    }

    Ok(ArchitectureSpec {
        components,
        external_owners,
        submodules,
    })
}

fn parse_bracket_list(text: &str) -> Vec<String> {
    let Some(open) = text.find('[') else {
        return Vec::new();
    };
    let rest = &text[open + 1..];
    let Some(close) = rest.find(']') else {
        return Vec::new();
    };
    rest[..close]
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn collect_component_source_files(src_dir: &Path, module_name: &str) -> io::Result<Vec<PathBuf>> {
    if module_name == "bin" {
        return Ok(vec![src_dir.join("bin").join("adfontes.rs")]);
    }
    let mut files = vec![src_dir.join(format!("{module_name}.rs"))];
    let subdir = src_dir.join(module_name);
    if subdir.is_dir() {
        for entry in fs::read_dir(&subdir)? {
            let path = entry?.path();
            if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }
    Ok(files)
}

fn collect_rust_file_stems(dir: &Path) -> io::Result<BTreeSet<String>> {
    let mut stems = BTreeSet::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("rs")
            && let Some(stem) = path.file_stem().and_then(|value| value.to_str())
        {
            stems.insert(stem.to_owned());
        }
    }
    Ok(stems)
}

/// Strips comments, string/char literals, and `#[cfg(...test...)]`-attributed items cleanly while
/// preserving all surrounding production Rust code.
fn strip_non_production_rust(source: &str) -> String {
    let stripped_tokens = strip_comments_and_literals(source);
    strip_test_configuration_items(&stripped_tokens)
}

fn strip_comments_and_literals(source: &str) -> String {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut idx = 0;

    while idx < len {
        let byte = bytes[idx];
        if byte == b'/' && idx + 1 < len && bytes[idx + 1] == b'/' {
            idx += 2;
            while idx < len && bytes[idx] != b'\n' {
                idx += 1;
            }
            continue;
        }
        if byte == b'/' && idx + 1 < len && bytes[idx + 1] == b'*' {
            idx = skip_nested_block_comment(bytes, idx + 2);
            out.push(' ');
            continue;
        }
        if let Some(after_raw) = try_skip_raw_string(bytes, idx) {
            idx = after_raw;
            out.push(' ');
            continue;
        }
        if (byte == b'b' || byte == b'c') && idx + 1 < len && bytes[idx + 1] == b'"' {
            idx = skip_quoted_literal(bytes, idx + 2, b'"');
            out.push(' ');
            continue;
        }
        if byte == b'"' {
            idx = skip_quoted_literal(bytes, idx + 1, b'"');
            out.push(' ');
            continue;
        }
        if byte == b'b' && idx + 1 < len && bytes[idx + 1] == b'\'' {
            idx = skip_quoted_literal(bytes, idx + 2, b'\'');
            out.push(' ');
            continue;
        }
        if byte == b'\''
            && let Some(after_char) = try_skip_char_literal(source, idx)
        {
            idx = after_char;
            out.push(' ');
            continue;
        }
        let character = source[idx..].chars().next().unwrap_or(' ');
        out.push(character);
        idx += character.len_utf8();
    }

    out
}

const fn skip_nested_block_comment(bytes: &[u8], mut idx: usize) -> usize {
    let len = bytes.len();
    let mut depth = 1usize;
    while idx + 1 < len && depth > 0 {
        if bytes[idx] == b'/' && bytes[idx + 1] == b'*' {
            depth += 1;
            idx += 2;
        } else if bytes[idx] == b'*' && bytes[idx + 1] == b'/' {
            depth -= 1;
            idx += 2;
        } else {
            idx += 1;
        }
    }
    if depth > 0 { len } else { idx }
}

const fn try_skip_raw_string(bytes: &[u8], start: usize) -> Option<usize> {
    let len = bytes.len();
    let prev_ident = start > 0 && is_ident_byte(bytes[start - 1]);
    if prev_ident {
        return None;
    }
    let mut idx = start;
    if matches!(bytes[idx], b'b' | b'c') {
        idx += 1;
    }
    if idx >= len || bytes[idx] != b'r' {
        return None;
    }
    idx += 1;
    let mut hashes = 0usize;
    while idx < len && bytes[idx] == b'#' {
        hashes += 1;
        idx += 1;
    }
    if idx >= len || bytes[idx] != b'"' {
        return None;
    }
    idx += 1;
    while idx < len {
        if bytes[idx] == b'"' {
            let mut matched = 0usize;
            while matched < hashes && idx + 1 + matched < len && bytes[idx + 1 + matched] == b'#' {
                matched += 1;
            }
            if matched == hashes {
                return Some(idx + 1 + hashes);
            }
        }
        idx += 1;
    }
    Some(len)
}

fn skip_quoted_literal(bytes: &[u8], mut idx: usize, delimiter: u8) -> usize {
    let len = bytes.len();
    while idx < len {
        if bytes[idx] == b'\\' {
            idx = (idx + 2).min(len);
        } else if bytes[idx] == delimiter {
            return idx + 1;
        } else {
            idx += 1;
        }
    }
    len
}

fn try_skip_char_literal(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    if start + 2 >= len {
        return None;
    }
    if bytes[start + 1] == b'\\' {
        let mut idx = start + 3;
        while idx < len && idx < start + 12 {
            if bytes[idx] == b'\'' {
                return Some(idx + 1);
            }
            idx += 1;
        }
        return None;
    }
    let character = source.get(start + 1..)?.chars().next()?;
    let close_index = start + 1 + character.len_utf8();
    if bytes.get(close_index) == Some(&b'\'') {
        return Some(close_index + 1);
    }
    None
}

fn strip_test_configuration_items(code: &str) -> String {
    let bytes = code.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut idx = 0;

    while idx < len {
        if bytes[idx] == b'#'
            && idx + 1 < len
            && bytes[idx + 1] == b'['
            && let Some(attr_end) = find_balanced_delimiter(bytes, idx + 1, b'[', b']')
        {
            let attr_inner = &code[idx + 2..attr_end - 1];
            if is_test_configuration_attribute(attr_inner) {
                idx = skip_attributed_item(bytes, attr_end);
                out.push('\n');
                continue;
            }
        }
        let character = code[idx..].chars().next().unwrap_or(' ');
        out.push(character);
        idx += character.len_utf8();
    }

    out
}

fn is_test_configuration_attribute(inner: &str) -> bool {
    let compact: String = inner
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect();
    if !compact.starts_with("cfg(") || !compact.ends_with(')') {
        return false;
    }
    let body = &compact[4..compact.len() - 1];
    if body.contains("not(test)") {
        return false;
    }
    body == "test" || body.split([',', '(', ')']).any(|part| part == "test")
}

fn skip_attributed_item(bytes: &[u8], mut idx: usize) -> usize {
    let len = bytes.len();
    loop {
        while idx < len && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx + 1 < len
            && bytes[idx] == b'#'
            && bytes[idx + 1] == b'['
            && let Some(end) = find_balanced_delimiter(bytes, idx + 1, b'[', b']')
        {
            idx = end;
            continue;
        }
        break;
    }

    while idx < len {
        if bytes[idx] == b';' {
            return idx + 1;
        }
        if bytes[idx] == b'{' {
            return find_balanced_delimiter(bytes, idx, b'{', b'}').unwrap_or(len);
        }
        idx += 1;
    }
    len
}

const fn find_balanced_delimiter(
    bytes: &[u8],
    open_index: usize,
    open: u8,
    close: u8,
) -> Option<usize> {
    let mut depth = 0usize;
    let mut idx = open_index;
    while idx < bytes.len() {
        if bytes[idx] == open {
            depth += 1;
        } else if bytes[idx] == close {
            depth -= 1;
            if depth == 0 {
                return Some(idx + 1);
            }
        }
        idx += 1;
    }
    None
}

struct ModuleReferences {
    crate_targets: BTreeSet<String>,
    adfontes_targets: BTreeSet<String>,
    super_targets: BTreeSet<String>,
    extract_targets: BTreeSet<String>,
    external_roots: BTreeSet<String>,
}

fn extract_module_references(code: &str) -> ModuleReferences {
    let mut references = ModuleReferences {
        crate_targets: BTreeSet::new(),
        adfontes_targets: BTreeSet::new(),
        super_targets: BTreeSet::new(),
        extract_targets: BTreeSet::new(),
        external_roots: BTreeSet::new(),
    };

    for statement in code.split(';') {
        if let Some(use_tail) = extract_use_tree(statement) {
            let mut paths = Vec::new();
            expand_use_tree(use_tail, &[], &mut paths);
            for path in paths {
                record_path_segments(&path, &mut references);
            }
        }
        if let Some(extern_crate) = extract_extern_crate_name(statement) {
            references.external_roots.insert(extern_crate);
        }
    }

    scan_qualified_paths(code, &mut references);
    references
}

fn extract_use_tree(statement: &str) -> Option<&str> {
    let bytes = statement.as_bytes();
    let mut idx = 0;
    while idx + 3 < bytes.len() {
        if &bytes[idx..idx + 3] == b"use"
            && (idx == 0 || !is_ident_byte(bytes[idx - 1]))
            && bytes[idx + 3].is_ascii_whitespace()
        {
            return Some(statement[idx + 3..].trim());
        }
        idx += 1;
    }
    None
}

fn extract_extern_crate_name(statement: &str) -> Option<String> {
    let tokens: Vec<&str> = statement.split_whitespace().collect();
    for window in tokens.windows(3) {
        if window[0] == "extern" && window[1] == "crate" {
            let name = window[2].trim_start_matches("::");
            if !name.is_empty() {
                return Some(name.to_owned());
            }
        }
    }
    None
}

fn expand_use_tree(tree: &str, prefix: &[String], out: &mut Vec<Vec<String>>) {
    let trimmed = tree.trim().trim_start_matches("::");
    if trimmed.is_empty() {
        return;
    }
    if let Some(brace_open) = trimmed.find('{')
        && let Some(brace_close) = trimmed.rfind('}')
    {
        let head = trimmed[..brace_open].trim().trim_end_matches("::");
        let mut next_prefix = prefix.to_vec();
        for segment in head.split("::").filter(|segment| !segment.is_empty()) {
            next_prefix.push(segment.trim().to_owned());
        }
        let inner = &trimmed[brace_open + 1..brace_close];
        for item in split_top_level_commas(inner) {
            expand_use_tree(item, &next_prefix, out);
        }
        return;
    }

    let without_alias = trimmed.split(" as ").next().unwrap_or(trimmed).trim();
    let mut full = prefix.to_vec();
    for segment in without_alias
        .split("::")
        .map(str::trim)
        .filter(|segment| !segment.is_empty() && *segment != "self")
    {
        full.push(segment.to_owned());
    }
    if !full.is_empty() {
        out.push(full);
    }
}

fn split_top_level_commas(input: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (idx, byte) in input.bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                parts.push(&input[start..idx]);
                start = idx + 1;
            }
            _ => {}
        }
    }
    parts.push(&input[start..]);
    parts
}

fn scan_qualified_paths(code: &str, references: &mut ModuleReferences) {
    let bytes = code.as_bytes();
    let len = bytes.len();
    let mut idx = 0;

    while idx < len {
        if !is_ident_start_byte(bytes[idx]) {
            idx += 1;
            continue;
        }
        let mut segments = Vec::new();
        let mut cursor = idx;
        loop {
            let seg_start = cursor;
            while cursor < len && is_ident_byte(bytes[cursor]) {
                cursor += 1;
            }
            segments.push(code[seg_start..cursor].to_owned());
            if cursor + 2 < len
                && bytes[cursor] == b':'
                && bytes[cursor + 1] == b':'
                && is_ident_start_byte(bytes[cursor + 2])
            {
                cursor += 2;
            } else {
                break;
            }
        }
        if segments.len() >= 2 {
            record_path_segments(&segments, references);
        }
        idx = cursor;
    }
}

fn record_path_segments(segments: &[String], references: &mut ModuleReferences) {
    let Some(first) = segments.first().map(String::as_str) else {
        return;
    };
    let second = segments.get(1).map(String::as_str);
    let third = segments.get(2).map(String::as_str);

    match first {
        "crate" => {
            if let Some(module) = second {
                references.crate_targets.insert(module.to_owned());
                if module == "extract"
                    && let Some(sub) = third
                {
                    references.extract_targets.insert(sub.to_owned());
                }
            }
        }
        "adfontes" => {
            if let Some(module) = second {
                references.adfontes_targets.insert(module.to_owned());
            }
        }
        "super" => {
            if let Some(module) = second {
                references.super_targets.insert(module.to_owned());
                if module == "super"
                    && let Some(parent_sibling) = third
                {
                    references.crate_targets.insert(parent_sibling.to_owned());
                }
            }
        }
        "self" | "std" | "core" | "alloc" => {}
        other => {
            references.external_roots.insert(other.to_owned());
        }
    }
}

fn check_component_edges(
    file_path: &Path,
    module_name: &str,
    rule: &ComponentRule,
    known_modules: &BTreeSet<String>,
    references: &ModuleReferences,
    violations: &mut Vec<String>,
) {
    let targets = if module_name == "bin" {
        &references.adfontes_targets
    } else {
        &references.crate_targets
    };
    for target in targets {
        if !known_modules.contains(target)
            || target == module_name
            || target == "bin"
            || rule.allowed_deps.contains(target)
        {
            continue;
        }
        violations.push(format!(
            "{}: undeclared dependency `{module_name}` -> `{target}`",
            file_path.display()
        ));
    }
}

fn check_external_ownership(
    file_path: &Path,
    module_name: &str,
    external_owners: &BTreeMap<String, String>,
    references: &ModuleReferences,
    violations: &mut Vec<String>,
) {
    for (external_crate, owner) in external_owners {
        if owner == module_name {
            continue;
        }
        if references.external_roots.contains(external_crate) {
            violations.push(format!(
                "{}: `{module_name}` references `{external_crate}` owned exclusively by `{owner}`",
                file_path.display()
            ));
        }
    }
}

fn check_submodule_edges(
    file_path: &Path,
    component_name: &str,
    submodule: &str,
    allowed_deps: &BTreeSet<String>,
    declared_submodules: &BTreeSet<String>,
    references: &ModuleReferences,
    violations: &mut Vec<String>,
) {
    for candidate in declared_submodules {
        if candidate == submodule || allowed_deps.contains(candidate) {
            continue;
        }
        let via_super = references.super_targets.contains(candidate);
        let via_crate =
            component_name == "extract" && references.extract_targets.contains(candidate);
        if via_super || via_crate {
            violations.push(format!(
                "{}: forbidden `{component_name}` submodule dependency `{submodule}` -> `{candidate}`",
                file_path.display()
            ));
        }
    }
}

const fn is_ident_start_byte(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

const fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn pascal_to_snake(name: &str) -> String {
    let mut snake = String::with_capacity(name.len() + 4);
    for (index, character) in name.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index > 0 {
                snake.push('_');
            }
            snake.push(character.to_ascii_lowercase());
        } else {
            snake.push(character);
        }
    }
    snake
}

fn component_root_path(src_dir: &Path, variant_name: &str) -> PathBuf {
    if variant_name == "Bin" {
        src_dir.join("bin").join("adfontes.rs")
    } else {
        src_dir.join(format!("{}.rs", pascal_to_snake(variant_name)))
    }
}

fn collect_missing_declarations(src_dir: &Path) -> Vec<String> {
    ArchitectureComponent::all()
        .iter()
        .filter_map(|component| {
            let variant_name = component.as_ref();
            let root_path = component_root_path(src_dir, variant_name);
            let content = match fs::read_to_string(&root_path) {
                Ok(text) => text,
                Err(io_error) => {
                    return Some(format!(
                        "missing root file {} for component {variant_name}: {io_error}",
                        root_path.display()
                    ));
                }
            };
            let direct_decl = format!("architecture_component!({variant_name});");
            let qualified_decl = format!("adfontes::architecture_component!({variant_name});");
            let declared = content.lines().any(|line| {
                let trimmed = line.trim();
                trimmed == direct_decl || trimmed == qualified_decl
            });
            if declared {
                None
            } else {
                Some(format!(
                    "{} must declare {direct_decl}",
                    root_path.display()
                ))
            }
        })
        .collect()
}

fn collect_expected_library_stems() -> BTreeSet<String> {
    ArchitectureComponent::all()
        .iter()
        .map(AsRef::as_ref)
        .filter(|variant_name| *variant_name != "Bin")
        .map(pascal_to_snake)
        .collect()
}

fn collect_actual_library_stems(src_dir: &Path) -> io::Result<BTreeSet<String>> {
    let mut stems = BTreeSet::new();
    for entry in fs::read_dir(src_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|value| value.to_str())
            && stem != "lib"
        {
            stems.insert(stem.to_owned());
        }
    }
    Ok(stems)
}
