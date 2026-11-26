//! Architecture component catalog completeness, visibility, and Python bridge boundary tests.

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
fn crate_public_surface_exposes_only_cli_and_architecture() {
    let lib_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("lib.rs");
    let lib_source = fs::read_to_string(&lib_path).expect("src/lib.rs must be readable");
    let public_decls = collect_public_lib_decls(&lib_source);
    let macro_exports = collect_macro_exports(&lib_source);
    assert_eq!(
        public_decls,
        BTreeSet::from([
            "pub mod architecture;".to_owned(),
            "pub mod cli;".to_owned()
        ])
    );
    assert_eq!(macro_exports, vec!["architecture_component".to_owned()]);
}

#[test]
fn adfontes_py_manifest_depends_only_on_adfontes_and_pyo3() {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("adfontes-py")
        .join("Cargo.toml");
    let manifest =
        fs::read_to_string(&manifest_path).expect("crates/adfontes-py/Cargo.toml must be readable");
    let dependency_headers = collect_toml_dependency_headers(&manifest);
    let lib_entries = collect_toml_section_entries(&manifest, "[lib]");
    let dependency_keys: BTreeSet<String> =
        collect_toml_section_entries(&manifest, "[dependencies]")
            .into_keys()
            .collect();
    assert_eq!(
        dependency_headers,
        BTreeSet::from(["[dependencies]".to_owned()])
    );
    assert_eq!(lib_entries.get("test").map(String::as_str), Some("false"));
    assert_eq!(
        dependency_keys,
        BTreeSet::from(["adfontes".to_owned(), "pyo3".to_owned()])
    );
}

#[test]
fn adfontes_py_bridge_calls_cli_only() {
    let py_src_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("adfontes-py")
        .join("src");
    let rust_files =
        collect_rust_files(&py_src_dir).expect("crates/adfontes-py/src must be readable");
    let (saw_cli_call, violations) =
        scan_adfontes_paths(&rust_files).expect("Python bridge Rust file must be readable");
    assert_ne!(rust_files, Vec::<PathBuf>::new());
    assert!(saw_cli_call);
    assert_eq!(violations, Vec::<String>::new());
}

fn collect_public_lib_decls(lib_source: &str) -> BTreeSet<String> {
    lib_source
        .lines()
        .map(strip_line_comment)
        .filter(|line| line.starts_with("pub ") || line.contains(" pub "))
        .map(str::to_owned)
        .collect()
}

fn collect_macro_exports(lib_source: &str) -> Vec<String> {
    let mut exports = Vec::new();
    let mut waiting_for_macro = false;
    for raw_line in lib_source.lines() {
        let line = strip_line_comment(raw_line);
        if line.is_empty() {
            continue;
        }
        if waiting_for_macro {
            if line.starts_with("#[") {
                continue;
            }
            waiting_for_macro = false;
            let macro_name = line
                .strip_prefix("macro_rules!")
                .and_then(|rest| rest.trim().split([' ', '{', '(']).next())
                .filter(|name| !name.is_empty())
                .unwrap_or("<non-macro_rules>");
            exports.push(macro_name.to_owned());
            continue;
        }
        if line.contains("macro_export") {
            waiting_for_macro = true;
        }
    }
    if waiting_for_macro {
        exports.push("<dangling-macro_export>".to_owned());
    }
    exports
}

fn collect_toml_dependency_headers(toml_source: &str) -> BTreeSet<String> {
    toml_source
        .lines()
        .filter_map(|raw_line| {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || !trimmed.starts_with('[') {
                return None;
            }
            let header = trimmed.split('#').next().unwrap_or("").trim();
            header
                .to_ascii_lowercase()
                .contains("dependencies")
                .then(|| header.to_owned())
        })
        .collect()
}

fn collect_toml_section_entries(
    toml_source: &str,
    section_header: &str,
) -> BTreeMap<String, String> {
    let mut in_section = false;
    let mut entries = BTreeMap::new();
    for trimmed in toml_source.lines().map(str::trim) {
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with('[') {
            let header = trimmed.split('#').next().unwrap_or("").trim();
            in_section = header == section_header;
            continue;
        }
        if in_section && let Some((key, value)) = trimmed.split_once('=') {
            let clean_value = value.split('#').next().unwrap_or("").trim();
            entries.insert(key.trim().to_owned(), clean_value.to_owned());
        }
    }
    entries
}

fn collect_rust_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            files.extend(collect_rust_files(&path)?);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn scan_adfontes_paths(rust_files: &[PathBuf]) -> io::Result<(bool, Vec<String>)> {
    const PREFIX: &str = "adfontes::";
    let mut saw_cli_call = false;
    let mut violations = Vec::new();
    for path in rust_files {
        let source = fs::read_to_string(path)?;
        for (index, raw_line) in source.lines().enumerate() {
            let line = strip_line_comment(raw_line);
            let mut rest = line;
            while let Some(pos) = rest.find(PREFIX) {
                let after = &rest[pos + PREFIX.len()..];
                if after.starts_with("cli::") {
                    saw_cli_call = true;
                } else {
                    violations.push(format!("{}:{}: {line}", path.display(), index + 1));
                }
                rest = after;
            }
        }
    }
    Ok((saw_cli_call, violations))
}

fn strip_line_comment(line: &str) -> &str {
    line.split_once("//").map_or(line, |(head, _)| head).trim()
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
            (!declared).then(|| format!("{} must declare {direct_decl}", root_path.display()))
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
        if path.extension().and_then(|extension| extension.to_str()) == Some("rs")
            && let Some(stem) = path.file_stem().and_then(|value| value.to_str())
            && stem != "lib"
        {
            stems.insert(stem.to_owned());
        }
    }
    Ok(stems)
}
