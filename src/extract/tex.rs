//! Zero-regex `arXiv` LaTeX archive extractor and bounded `\input` flattener.
//!
//! Safely unpacks `.tar.gz`, uncompressed `.tar`, single-file `.tex.gz`, and plain `.tex`
//! submissions via `Archive`, flattens `\input{}` / `\include{}` trees under cycle, depth,
//! inclusion-count (`2,000`), and cumulative expanded-byte (`10 MiB`) caps, and converts
//! structural LaTeX commands to Markdown iteratively while preserving math spans verbatim.

use std::collections::{BTreeMap, BTreeSet};

use super::title::{
    ExtractedText, normalize_markdown_spacing, push_markdown_heading, sanitize_inline_text,
};
use crate::archive::{
    DEFAULT_MAX_ARCHIVE_BYTES, DEFAULT_MAX_TAR_FILES, is_gzip_magic, rejected_archive_format,
    safe_extract_tar_bytes, safe_gunzip_bytes,
};
use crate::model::AdfontesError;

const TEX_FORMAT_LABEL: &str = "TeX";
const TEX_EXTENSION: &str = ".tex";
const CMD_INPUT: &str = "input";
const DEFAULT_MAIN_TEX: &str = "main.tex";
const BEGIN_DOCUMENT_MARKER: &str = "\\begin{document}";
const END_DOCUMENT_MARKER: &str = "\\end{document}";
const DOCUMENT_CLASS_MARKER: &str = "\\documentclass";
const ITEM_COMMAND: &str = "\\item";
const MAX_TEX_INCLUDE_DEPTH: usize = 12;
const MAX_TEX_INCLUSIONS: usize = 2_000;
const MAX_EXPANDED_TEX_BYTES: usize = 10 * 1024 * 1024;
const MAX_FAILED_LOOKAHEAD_BYTES: usize = 1024 * 1024;
const MAX_INLINE_TEX_WRAP_DEPTH: usize = 32;
const FLAG_NO_CLOSE_BRACKET_MATH: u8 = 1 << 0;
const FLAG_NO_CLOSE_PAREN_MATH: u8 = 1 << 1;
const FLAG_NO_CLOSE_DOUBLE_DOLLAR: u8 = 1 << 2;
const FLAG_NO_CLOSE_SINGLE_DOLLAR: u8 = 1 << 3;
const PREFERRED_ROOT_NAMES: &[&str] = &[DEFAULT_MAIN_TEX, "paper.tex", "ms.tex", "manuscript.tex"];

/// Unpacks an `arXiv` `/src/<id>` archive (`.tar.gz`, `.tar`, single `.tex.gz`, or plain `.tex`)
/// and converts the flattened LaTeX source into math-preserving Markdown.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if decompression, archive limits, or `\input` expansion
/// caps are exceeded, or [`AdfontesError::Extraction`] if no valid LaTeX entrypoint exists.
pub fn extract_arxiv_source_archive(raw_bytes: &[u8]) -> Result<ExtractedText, AdfontesError> {
    if let Some(format_name) = rejected_archive_format(raw_bytes) {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("unsupported TeX archive format `{format_name}` is blocked"),
        });
    }

    let decompressed_buf: Vec<u8>;
    let payload: &[u8] = if is_gzip_magic(raw_bytes) {
        decompressed_buf = safe_gunzip_bytes(raw_bytes, DEFAULT_MAX_ARCHIVE_BYTES)?;
        &decompressed_buf
    } else {
        if raw_bytes.len() > DEFAULT_MAX_ARCHIVE_BYTES {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "TeX payload exceeded {DEFAULT_MAX_ARCHIVE_BYTES}-byte safety ceiling"
                ),
            });
        }
        raw_bytes
    };

    if let Some(format_name) = rejected_archive_format(payload) {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("unsupported inner TeX archive format `{format_name}` is blocked"),
        });
    }

    match safe_extract_tar_bytes(payload, DEFAULT_MAX_ARCHIVE_BYTES, DEFAULT_MAX_TAR_FILES) {
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Ok(entries) if !entries.is_empty() => {
            let mut tex_files = BTreeMap::new();
            for entry in entries {
                if entry.path.to_ascii_lowercase().ends_with(TEX_EXTENSION) {
                    tex_files.insert(
                        entry.path,
                        String::from_utf8_lossy(&entry.bytes).into_owned(),
                    );
                }
            }
            let Some(entrypoint) = select_primary_tex_file(&tex_files) else {
                return Err(AdfontesError::Extraction {
                    format: TEX_FORMAT_LABEL,
                    detail: "tar archive contained no readable `.tex` files".to_owned(),
                });
            };
            return flatten_and_convert_tex(&tex_files, &entrypoint);
        }
        Ok(_) | Err(AdfontesError::Extraction { .. }) => {}
        Err(other) => return Err(other),
    }

    let single_tex = String::from_utf8_lossy(payload).into_owned();
    if single_tex.contains(DOCUMENT_CLASS_MARKER) || single_tex.contains(BEGIN_DOCUMENT_MARKER) {
        let mut files = BTreeMap::new();
        files.insert(DEFAULT_MAIN_TEX.to_owned(), single_tex);
        return flatten_and_convert_tex(&files, DEFAULT_MAIN_TEX);
    }

    Err(AdfontesError::Extraction {
        format: TEX_FORMAT_LABEL,
        detail: "archive did not contain a readable LaTeX `.tex` document".to_owned(),
    })
}

/// Flattens `\input{}` / `\include{}` directives starting from `entry_path` and renders Markdown.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if expanded TeX output exceeds `10 MiB` or `2,000`
/// inclusions, or [`AdfontesError::Extraction`] if `entry_path` is missing from `files`.
pub fn flatten_and_convert_tex(
    files: &BTreeMap<String, String>,
    entry_path: &str,
) -> Result<ExtractedText, AdfontesError> {
    if !files.contains_key(entry_path) {
        return Err(AdfontesError::Extraction {
            format: TEX_FORMAT_LABEL,
            detail: format!("entry TeX file `{entry_path}` not found"),
        });
    }

    let mut active_stack = BTreeSet::new();
    let mut state = ExpansionBudget {
        expanded_bytes: 0,
        inclusions: 0,
        scan_budget: TexScanBudget::new(),
    };
    let mut flattened = String::new();
    expand_tex_file(
        files,
        entry_path,
        0,
        &mut active_stack,
        &mut state,
        &mut flattened,
    )?;

    let title = extract_command_argument(&flattened, "title")
        .map(|raw| sanitize_inline_text(&clean_inline_tex(raw)))
        .filter(|value| !value.is_empty());

    let body = slice_document_body(&flattened);
    let converted = convert_tex_body_to_markdown(body, title.as_deref());
    Ok(ExtractedText {
        title,
        markdown: normalize_markdown_spacing(&converted),
    })
}

fn select_primary_tex_file(files: &BTreeMap<String, String>) -> Option<String> {
    for &preferred in PREFERRED_ROOT_NAMES {
        if let Some(content) = files.get(preferred)
            && content.contains(BEGIN_DOCUMENT_MARKER)
        {
            return Some(preferred.to_owned());
        }
    }
    let mut candidates: Vec<TexCandidate<'_>> = files
        .iter()
        .filter(|(_, content)| {
            content.contains(DOCUMENT_CLASS_MARKER) || content.contains(BEGIN_DOCUMENT_MARKER)
        })
        .map(|(path, content)| TexCandidate {
            path,
            length: content.len(),
        })
        .collect();
    candidates.sort_by(|left, right| {
        let left_depth = left.path.matches('/').count();
        let right_depth = right.path.matches('/').count();
        left_depth
            .cmp(&right_depth)
            .then_with(|| right.length.cmp(&left.length))
            .then_with(|| left.path.cmp(right.path))
    });
    candidates
        .first()
        .map(|candidate| candidate.path.clone())
        .or_else(|| files.keys().next().cloned())
}

struct TexCandidate<'a> {
    path: &'a String,
    length: usize,
}

struct ExpansionBudget {
    expanded_bytes: usize,
    inclusions: usize,
    scan_budget: TexScanBudget,
}

fn expand_tex_file(
    files: &BTreeMap<String, String>,
    current_path: &str,
    depth: usize,
    active_stack: &mut BTreeSet<String>,
    budget: &mut ExpansionBudget,
    out: &mut String,
) -> Result<(), AdfontesError> {
    if depth > MAX_TEX_INCLUDE_DEPTH || !active_stack.insert(current_path.to_owned()) {
        return Ok(());
    }

    let source = files.get(current_path).map_or("", String::as_str);
    let current_dir = current_path.rsplit_once('/').map_or("", |(dir, _)| dir);

    for raw_line in source.lines() {
        let line = strip_tex_comment(raw_line);
        let mut cursor: usize = 0;
        while let Some(slice) = line.get(cursor..) {
            let Some(directive) = find_next_include_directive(slice, &mut budget.scan_budget)
            else {
                append_bounded_tex(out, slice, &mut budget.expanded_bytes)?;
                break;
            };
            if let Some(before) = slice.get(..directive.start) {
                append_bounded_tex(out, before, &mut budget.expanded_bytes)?;
            }
            if let Some(resolved_key) =
                resolve_tex_include_key(files, current_dir, directive.target)
            {
                budget.inclusions = budget.inclusions.saturating_add(1);
                if budget.inclusions > MAX_TEX_INCLUSIONS {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "TeX \\input inclusion count exceeded {MAX_TEX_INCLUSIONS}-file ceiling"
                        ),
                    });
                }
                append_bounded_tex(out, "\n", &mut budget.expanded_bytes)?;
                expand_tex_file(files, &resolved_key, depth + 1, active_stack, budget, out)?;
                append_bounded_tex(out, "\n", &mut budget.expanded_bytes)?;
            }
            cursor += directive.end;
        }
        append_bounded_tex(out, "\n", &mut budget.expanded_bytes)?;
    }

    active_stack.remove(current_path);
    Ok(())
}

fn append_bounded_tex(
    out: &mut String,
    chunk: &str,
    expanded_bytes: &mut usize,
) -> Result<(), AdfontesError> {
    *expanded_bytes = expanded_bytes.saturating_add(chunk.len());
    if *expanded_bytes > MAX_EXPANDED_TEX_BYTES {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("TeX \\input expansion exceeded {MAX_EXPANDED_TEX_BYTES}-byte ceiling"),
        });
    }
    out.push_str(chunk);
    Ok(())
}

/// Strips TeX `%` comments using exact preceding-backslash parity (`\%` kept, `\\%` stripped).
fn strip_tex_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut index: usize = 0;
    while let Some(&byte) = bytes.get(index) {
        if byte == b'%' {
            let mut backslashes: usize = 0;
            let mut back = index;
            while back > 0 && bytes.get(back - 1) == Some(&b'\\') {
                backslashes += 1;
                back -= 1;
            }
            if backslashes.is_multiple_of(2) {
                return line.get(..index).unwrap_or(line);
            }
        }
        index += 1;
    }
    line
}

fn resolve_tex_include_key(
    files: &BTreeMap<String, String>,
    current_dir: &str,
    raw_target: &str,
) -> Option<String> {
    let cleaned = raw_target.trim().trim_start_matches("./");
    if cleaned.is_empty()
        || cleaned.contains("..")
        || cleaned.starts_with('/')
        || cleaned.contains('\\')
        || cleaned.contains('\0')
    {
        return None;
    }
    let with_extension = if cleaned.to_ascii_lowercase().ends_with(TEX_EXTENSION) {
        cleaned.to_owned()
    } else {
        format!("{cleaned}{TEX_EXTENSION}")
    };
    let mut candidates = Vec::new();
    if !current_dir.is_empty() {
        candidates.push(format!("{current_dir}/{with_extension}"));
        candidates.push(format!("{current_dir}/{cleaned}"));
    }
    candidates.push(with_extension);
    candidates.push(cleaned.to_owned());

    candidates.into_iter().find(|key| files.contains_key(key))
}

fn extract_command_argument<'a>(source: &'a str, command_name: &str) -> Option<&'a str> {
    let needle = format!("\\{command_name}");
    let mut scan_budget = TexScanBudget::new();
    let mut search_from: usize = 0;
    while let Some(rel) = source.get(search_from..)?.find(&needle) {
        let cmd_start = search_from + rel;
        let cmd_end = cmd_start + needle.len();
        if count_preceding_backslashes(source.as_bytes(), cmd_start) % 2 == 1 {
            search_from = cmd_end;
            continue;
        }
        let after = source.get(cmd_end..)?;
        if after
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic())
        {
            search_from = cmd_end;
            continue;
        }
        if let Some(arg) = read_command_braced_argument_bounded(source, cmd_end, &mut scan_budget) {
            return Some(arg.content);
        }
        search_from = cmd_end;
    }
    None
}

fn slice_document_body(source: &str) -> &str {
    let after_begin = source
        .find(BEGIN_DOCUMENT_MARKER)
        .and_then(|pos| source.get(pos + BEGIN_DOCUMENT_MARKER.len()..))
        .unwrap_or(source);
    after_begin
        .find(END_DOCUMENT_MARKER)
        .and_then(|pos| after_begin.get(..pos))
        .unwrap_or(after_begin)
}

fn convert_tex_body_to_markdown(body: &str, title: Option<&str>) -> String {
    let mut out = String::with_capacity(body.len());
    if let Some(heading) = title {
        push_markdown_heading(&mut out, 1, heading);
    }

    let mut scan_budget = TexScanBudget::new();
    let mut body_missing_flags: u8 = 0;
    let mut cursor: usize = 0;
    let bytes = body.as_bytes();
    while cursor < bytes.len() {
        let rest = body.get(cursor..).unwrap_or_default();

        // Atomic escaped control symbols (`\\` line break and `\$` literal dollar).
        if rest.starts_with("\\\\") {
            out.push_str("\\\\");
            cursor += 2;
            continue;
        }
        if rest.starts_with("\\$") {
            out.push_str("\\$");
            cursor += 2;
            continue;
        }

        // Preserve display and inline math spans verbatim before any command rewriting.
        if let Some(math_len) =
            consume_delimited_math(rest, &mut out, &mut scan_budget, &mut body_missing_flags)
        {
            cursor += math_len;
            continue;
        }

        if let Some(env_match) = try_consume_math_environment(rest, &mut scan_budget) {
            out.push_str("\n\n$$\n");
            if env_match.keep_wrapper {
                out.push_str(env_match.full.trim());
            } else {
                out.push_str(env_match.inner.trim());
            }
            out.push_str("\n$$\n\n");
            cursor += env_match.consumed;
            continue;
        }

        if let Some(env_match) = try_consume_named_environment(rest, "abstract", &mut scan_budget) {
            let clean = clean_inline_tex_with_budget(env_match.inner, &mut scan_budget);
            out.push_str("\n\n## Abstract\n\n");
            out.push_str(&clean);
            out.push_str("\n\n");
            cursor += env_match.consumed;
            continue;
        }

        if let Some(heading) = try_consume_heading_command(body, cursor, &mut scan_budget) {
            out.push_str("\n\n");
            let text = sanitize_inline_text(&clean_inline_tex_with_budget(
                heading.text,
                &mut scan_budget,
            ));
            if !text.is_empty() {
                push_markdown_heading(&mut out, heading.level, &text);
            }
            cursor = heading.next_index;
            continue;
        }

        if let Some(stripped) = rest.strip_prefix(ITEM_COMMAND)
            && !stripped
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic())
        {
            out.push_str("\n- ");
            cursor += ITEM_COMMAND.len();
            continue;
        }

        if let Some(step) =
            try_consume_body_inline_command(body, cursor, &mut out, &mut scan_budget)
        {
            cursor = step;
            continue;
        }

        let character = rest.chars().next().unwrap_or(' ');
        out.push(character);
        cursor += character.len_utf8();
    }

    out
}

fn clean_inline_tex(input: &str) -> String {
    let mut scan_budget = TexScanBudget::new();
    clean_inline_tex_with_budget(input, &mut scan_budget)
}

/// Memoizes unclosed environments and bounds cumulative failed lookahead scans to prevent
/// $O(N^2)$ CPU exhaustion on adversarial TeX files with thousands of unclosed openers.
struct TexScanBudget {
    unclosed_envs: BTreeSet<&'static str>,
    failed_lookahead_remaining: usize,
}

impl TexScanBudget {
    const fn new() -> Self {
        Self {
            unclosed_envs: BTreeSet::new(),
            failed_lookahead_remaining: MAX_FAILED_LOOKAHEAD_BYTES,
        }
    }

    const fn has_lookahead_budget(&self) -> bool {
        self.failed_lookahead_remaining > 0
    }

    const fn charge_failed_scan(&mut self, scanned_bytes: usize) {
        let cost = if scanned_bytes == 0 { 1 } else { scanned_bytes };
        self.failed_lookahead_remaining = self.failed_lookahead_remaining.saturating_sub(cost);
    }
}

struct IncludeDirective<'a> {
    start: usize,
    end: usize,
    target: &'a str,
}

fn find_next_include_directive<'a>(
    line: &'a str,
    scan_budget: &mut TexScanBudget,
) -> Option<IncludeDirective<'a>> {
    let bytes = line.as_bytes();
    let mut index: usize = 0;
    while let Some(&byte) = bytes.get(index) {
        if byte == b'\\' {
            if bytes.get(index + 1) == Some(&b'\\') {
                index += 2;
                continue;
            }
            let after = line.get(index + 1..).unwrap_or_default();
            for cmd in [CMD_INPUT, "include"] {
                if let Some(rest) = after.strip_prefix(cmd)
                    && !rest
                        .bytes()
                        .next()
                        .is_some_and(|next| next.is_ascii_alphabetic())
                {
                    let trimmed_offset = rest.len() - rest.trim_start().len();
                    let arg_start = index + 1 + cmd.len() + trimmed_offset;
                    if let Some(braced) = read_braced_argument_bounded(line, arg_start, scan_budget)
                    {
                        return Some(IncludeDirective {
                            start: index,
                            end: braced.next_index,
                            target: braced.content.trim(),
                        });
                    }
                    if cmd == CMD_INPUT
                        && trimmed_offset > 0
                        && let Some(token_slice) = line.get(arg_start..)
                    {
                        let tok_len = token_slice
                            .bytes()
                            .take_while(|&byte| {
                                !byte.is_ascii_whitespace() && !matches!(byte, b'{' | b'}')
                            })
                            .count();
                        if tok_len > 0
                            && let Some(target) = token_slice.get(..tok_len)
                        {
                            return Some(IncludeDirective {
                                start: index,
                                end: arg_start + tok_len,
                                target,
                            });
                        }
                    }
                }
            }
        }
        index += 1;
    }
    None
}

struct MathEnvSpan<'a> {
    full: &'a str,
    inner: &'a str,
    consumed: usize,
    keep_wrapper: bool,
}

fn try_consume_math_environment<'a>(
    rest: &'a str,
    scan_budget: &mut TexScanBudget,
) -> Option<MathEnvSpan<'a>> {
    for &(env, keep_wrapper) in &[
        ("equation", false),
        ("equation*", false),
        ("align", true),
        ("align*", true),
        ("gather", true),
        ("gather*", true),
        ("multline", true),
        ("multline*", true),
    ] {
        if let Some(span) = try_consume_named_environment(rest, env, scan_budget) {
            let full = rest.get(..span.consumed)?;
            return Some(MathEnvSpan {
                full,
                inner: span.inner,
                consumed: span.consumed,
                keep_wrapper,
            });
        }
    }
    None
}

struct EnvSpan<'a> {
    inner: &'a str,
    consumed: usize,
}

fn try_consume_named_environment<'a>(
    rest: &'a str,
    env: &'static str,
    scan_budget: &mut TexScanBudget,
) -> Option<EnvSpan<'a>> {
    if scan_budget.unclosed_envs.contains(env) {
        return None;
    }
    let open = format!("\\begin{{{env}}}");
    let after_open = rest.strip_prefix(&open)?;
    let close = format!("\\end{{{env}}}");
    let Some(close_pos) = after_open.find(&close) else {
        scan_budget.unclosed_envs.insert(env);
        return None;
    };
    let inner = after_open.get(..close_pos)?;
    Some(EnvSpan {
        inner,
        consumed: open.len() + close_pos + close.len(),
    })
}

struct TexHeading<'a> {
    level: usize,
    text: &'a str,
    next_index: usize,
}

fn try_consume_heading_command<'a>(
    source: &'a str,
    cursor: usize,
    scan_budget: &mut TexScanBudget,
) -> Option<TexHeading<'a>> {
    let rest = source.get(cursor..)?;
    let specs: &[(&str, usize)] = &[
        ("\\chapter", 2),
        ("\\section", 2),
        ("\\subsection", 3),
        ("\\subsubsection", 4),
        ("\\paragraph", 5),
    ];
    for &(prefix, level) in specs {
        if let Some(after) = rest.strip_prefix(prefix)
            && !after
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic())
        {
            let arg =
                read_command_braced_argument_bounded(source, cursor + prefix.len(), scan_budget)?;
            return Some(TexHeading {
                level,
                text: arg.content,
                next_index: arg.next_index,
            });
        }
    }
    None
}

fn try_consume_body_inline_command(
    source: &str,
    cursor: usize,
    out: &mut String,
    scan_budget: &mut TexScanBudget,
) -> Option<usize> {
    let slice = source.get(cursor..)?;
    match match_inline_command(slice, scan_budget)? {
        InlineMatch::Wrap {
            wrapper,
            inner,
            consumed,
        } => {
            out.push_str(wrapper);
            clean_inline_tex_into(inner, out, scan_budget);
            out.push_str(wrapper);
            Some(cursor + consumed)
        }
        InlineMatch::Drop { consumed } => Some(cursor + consumed),
    }
}

fn clean_inline_tex_with_budget(input: &str, scan_budget: &mut TexScanBudget) -> String {
    let mut out = String::with_capacity(input.len());
    clean_inline_tex_into(input, &mut out, scan_budget);
    out
}

fn clean_inline_tex_into(root_input: &str, out: &mut String, scan_budget: &mut TexScanBudget) {
    let mut stack: Vec<TexInlineFrame<'_>> = vec![TexInlineFrame::Slice {
        text: root_input,
        depth: 0,
    }];

    while let Some(frame) = stack.pop() {
        match frame {
            TexInlineFrame::Emit(text) => out.push_str(text),
            TexInlineFrame::Slice { text: slice, depth } => {
                let mut cursor: usize = 0;
                let bytes = slice.as_bytes();
                let mut slice_missing_flags: u8 = 0;
                while cursor < bytes.len() {
                    let rest = slice.get(cursor..).unwrap_or_default();
                    if rest.starts_with("\\\\") {
                        out.push_str("\\\\");
                        cursor += 2;
                        continue;
                    }
                    if rest.starts_with("\\$") {
                        out.push_str("\\$");
                        cursor += 2;
                        continue;
                    }
                    if let Some(math_len) =
                        consume_delimited_math(rest, out, scan_budget, &mut slice_missing_flags)
                    {
                        cursor += math_len;
                        continue;
                    }
                    if depth < MAX_INLINE_TEX_WRAP_DEPTH
                        && let Some(matched) = match_inline_command(rest, scan_budget)
                    {
                        match matched {
                            InlineMatch::Wrap {
                                wrapper,
                                inner,
                                consumed,
                            } => {
                                if let Some(tail) = rest.get(consumed..)
                                    && !tail.is_empty()
                                {
                                    stack.push(TexInlineFrame::Slice { text: tail, depth });
                                }
                                if !wrapper.is_empty() {
                                    stack.push(TexInlineFrame::Emit(wrapper));
                                }
                                stack.push(TexInlineFrame::Slice {
                                    text: inner,
                                    depth: depth + 1,
                                });
                                out.push_str(wrapper);
                                break;
                            }
                            InlineMatch::Drop { consumed } => {
                                cursor += consumed;
                                continue;
                            }
                        }
                    }
                    let character = rest.chars().next().unwrap_or(' ');
                    out.push(character);
                    cursor += character.len_utf8();
                }
            }
        }
    }
}

enum TexInlineFrame<'a> {
    Slice { text: &'a str, depth: usize },
    Emit(&'static str),
}

fn consume_delimited_math(
    rest: &str,
    out: &mut String,
    scan_budget: &mut TexScanBudget,
    missing_flags: &mut u8,
) -> Option<usize> {
    if rest.starts_with("\\[") {
        if (*missing_flags & FLAG_NO_CLOSE_BRACKET_MATH) != 0 || !scan_budget.has_lookahead_budget()
        {
            return None;
        }
        let after = rest.get(2..)?;
        if let Some(end_rel) = find_unescaped_sequence(after, "\\]") {
            let math = rest.get(2..2 + end_rel)?.trim();
            out.push_str("\n\n$$\n");
            out.push_str(math);
            out.push_str("\n$$\n\n");
            return Some(2 + end_rel + 2);
        }
        *missing_flags |= FLAG_NO_CLOSE_BRACKET_MATH;
        scan_budget.charge_failed_scan(after.len());
        return None;
    }

    if rest.starts_with("\\(") {
        if (*missing_flags & FLAG_NO_CLOSE_PAREN_MATH) != 0 || !scan_budget.has_lookahead_budget() {
            return None;
        }
        let after = rest.get(2..)?;
        if let Some(end_rel) = find_unescaped_sequence(after, "\\)") {
            let math = rest.get(2..2 + end_rel)?.trim();
            out.push('$');
            out.push_str(math);
            out.push('$');
            return Some(2 + end_rel + 2);
        }
        *missing_flags |= FLAG_NO_CLOSE_PAREN_MATH;
        scan_budget.charge_failed_scan(after.len());
        return None;
    }

    if rest.starts_with("$$") {
        if (*missing_flags & FLAG_NO_CLOSE_DOUBLE_DOLLAR) != 0
            || !scan_budget.has_lookahead_budget()
        {
            return None;
        }
        let after = rest.get(2..)?;
        if let Some(end_rel) = find_unescaped_sequence(after, "$$") {
            let math = rest.get(2..2 + end_rel)?.trim();
            out.push_str("\n\n$$\n");
            out.push_str(math);
            out.push_str("\n$$\n\n");
            return Some(2 + end_rel + 2);
        }
        *missing_flags |= FLAG_NO_CLOSE_DOUBLE_DOLLAR;
        scan_budget.charge_failed_scan(after.len());
        return None;
    }

    if rest.starts_with('$') {
        if (*missing_flags & FLAG_NO_CLOSE_SINGLE_DOLLAR) != 0
            || !scan_budget.has_lookahead_budget()
        {
            return None;
        }
        let after = rest.get(1..)?;
        if let Some(end_rel) = find_unescaped_single_dollar(after) {
            let full = rest.get(..1 + end_rel + 1)?;
            out.push_str(full);
            return Some(1 + end_rel + 1);
        }
        *missing_flags |= FLAG_NO_CLOSE_SINGLE_DOLLAR;
        scan_budget.charge_failed_scan(after.len());
        return None;
    }

    None
}

fn find_unescaped_sequence(slice: &str, needle: &str) -> Option<usize> {
    let mut from: usize = 0;
    while let Some(rel) = slice.get(from..)?.find(needle) {
        let pos = from + rel;
        if count_preceding_backslashes(slice.as_bytes(), pos).is_multiple_of(2) {
            return Some(pos);
        }
        from = pos + needle.len();
    }
    None
}

fn count_preceding_backslashes(bytes: &[u8], pos: usize) -> usize {
    let mut count: usize = 0;
    let mut back = pos;
    while back > 0 && bytes.get(back - 1) == Some(&b'\\') {
        count += 1;
        back -= 1;
    }
    count
}

fn find_unescaped_single_dollar(slice: &str) -> Option<usize> {
    let bytes = slice.as_bytes();
    let mut index: usize = 0;
    while let Some(&byte) = bytes.get(index) {
        if byte == b'\\' {
            let step = slice
                .get(index + 1..)
                .and_then(|rest_slice| rest_slice.chars().next())
                .map_or(0, char::len_utf8);
            index += 1 + step;
            continue;
        }
        if byte == b'$' {
            return Some(index);
        }
        index += 1;
    }
    None
}

enum InlineMatch<'a> {
    Wrap {
        wrapper: &'static str,
        inner: &'a str,
        consumed: usize,
    },
    Drop {
        consumed: usize,
    },
}

fn match_inline_command<'a>(
    rest: &'a str,
    scan_budget: &mut TexScanBudget,
) -> Option<InlineMatch<'a>> {
    if !rest.starts_with('\\') {
        return None;
    }

    for &(cmd, wrapper) in &[
        ("\\textbf", "**"),
        ("\\emph", "*"),
        ("\\textit", "*"),
        ("\\texttt", "`"),
        ("\\text", ""),
    ] {
        if let Some(after) = rest.strip_prefix(cmd)
            && !after
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic())
            && let Some(arg) = read_command_braced_argument_bounded(rest, cmd.len(), scan_budget)
        {
            return Some(InlineMatch::Wrap {
                wrapper,
                inner: arg.content,
                consumed: arg.next_index,
            });
        }
    }

    for dropped_word in [
        "\\maketitle",
        "\\tableofcontents",
        "\\centering",
        "\\noindent",
        "\\clearpage",
        "\\newpage",
    ] {
        if let Some(after) = rest.strip_prefix(dropped_word)
            && !after
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic())
        {
            return Some(InlineMatch::Drop {
                consumed: dropped_word.len(),
            });
        }
    }

    for dropped_env in [
        "\\begin{itemize}",
        "\\end{itemize}",
        "\\begin{enumerate}",
        "\\end{enumerate}",
    ] {
        if rest.starts_with(dropped_env) {
            return Some(InlineMatch::Drop {
                consumed: dropped_env.len(),
            });
        }
    }

    for dropped_with_arg in [
        "\\label",
        "\\title",
        "\\vspace",
        "\\hspace",
        "\\bibliographystyle",
        "\\bibliography",
    ] {
        if let Some(after) = rest.strip_prefix(dropped_with_arg)
            && !after
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic())
            && let Some(arg) =
                read_command_braced_argument_bounded(rest, dropped_with_arg.len(), scan_budget)
        {
            return Some(InlineMatch::Drop {
                consumed: arg.next_index,
            });
        }
    }

    None
}

struct BracedArg<'a> {
    content: &'a str,
    next_index: usize,
}

fn read_command_braced_argument_bounded<'a>(
    source: &'a str,
    after_cmd: usize,
    scan_budget: &mut TexScanBudget,
) -> Option<BracedArg<'a>> {
    let skipped = skip_optional_star_and_brackets_bounded(source, after_cmd, scan_budget);
    if let Some(arg) = read_braced_argument_bounded(source, skipped.next_cursor, scan_budget) {
        return Some(arg);
    }
    if skipped.scanned_bytes > 0 {
        scan_budget.charge_failed_scan(skipped.scanned_bytes);
    }
    None
}

struct SkippedBrackets {
    next_cursor: usize,
    scanned_bytes: usize,
}

fn read_braced_argument_bounded<'a>(
    text: &'a str,
    open_brace_index: usize,
    scan_budget: &mut TexScanBudget,
) -> Option<BracedArg<'a>> {
    if text.as_bytes().get(open_brace_index) != Some(&b'{') {
        return None;
    }
    if !scan_budget.has_lookahead_budget() {
        return None;
    }
    let result = read_braced_argument(text, open_brace_index);
    if result.is_none() {
        scan_budget.charge_failed_scan(text.len().saturating_sub(open_brace_index));
    }
    result
}

fn read_braced_argument(text: &str, open_brace_index: usize) -> Option<BracedArg<'_>> {
    let bytes = text.as_bytes();
    if bytes.get(open_brace_index) != Some(&b'{') {
        return None;
    }
    let mut depth: usize = 0;
    let mut cursor = open_brace_index;
    while let Some(&byte) = bytes.get(cursor) {
        if byte == b'\\' {
            let escaped_len = text
                .get(cursor + 1..)
                .and_then(|slice| slice.chars().next())
                .map_or(0, char::len_utf8);
            cursor += 1 + escaped_len;
            continue;
        }
        if byte == b'{' {
            depth += 1;
        } else if byte == b'}' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                let inner = text.get(open_brace_index + 1..cursor)?;
                return Some(BracedArg {
                    content: inner,
                    next_index: cursor + 1,
                });
            }
        }
        cursor += 1;
    }
    None
}

fn skip_optional_star_and_brackets_bounded(
    source: &str,
    mut cursor: usize,
    scan_budget: &mut TexScanBudget,
) -> SkippedBrackets {
    let bytes = source.as_bytes();
    while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    }
    if bytes.get(cursor) == Some(&b'*') {
        cursor += 1;
    }
    while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    }
    let mut scanned_bytes: usize = 0;
    if bytes.get(cursor) == Some(&b'[') && scan_budget.has_lookahead_budget() {
        let bracket_start = cursor;
        let mut depth: usize = 0;
        let mut closed = false;
        while let Some(&byte) = bytes.get(cursor) {
            if byte == b'\\' {
                let escaped_len = source
                    .get(cursor + 1..)
                    .and_then(|slice| slice.chars().next())
                    .map_or(0, char::len_utf8);
                cursor += 1 + escaped_len;
                continue;
            }
            if byte == b'[' {
                depth += 1;
            } else if byte == b']' {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    cursor += 1;
                    closed = true;
                    break;
                }
            }
            cursor += 1;
        }
        if closed {
            scanned_bytes = cursor.saturating_sub(bracket_start);
        } else {
            scan_budget.charge_failed_scan(source.len().saturating_sub(bracket_start));
            cursor = bracket_start;
        }
    }
    while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    }
    SkippedBrackets {
        next_cursor: cursor,
        scanned_bytes,
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use indoc::indoc;

    use super::*;
    use crate::archive::test_gzip_bytes;

    fn sample_multifile_tex() -> BTreeMap<String, String> {
        let mut files = BTreeMap::new();
        files.insert(
            "main.tex".to_owned(),
            indoc! {r"
                \documentclass{article}
                \title{Fractional \textbf{Sizing} with {Nested {Braces}} \\}
                \begin{document}
                \begin{abstract}
                Shows 50\% growth \\% trailing comment after line break
                \end{abstract}
                \input{sections/intro}
                \end{document}
            "}
            .to_owned(),
        );
        files.insert(
            "sections/intro.tex".to_owned(),
            indoc! {r"
                \section*{Introduction}
                Inline math $\textbf{W}\textbf{x} + P(\text{event})$ stays intact.
                Line 1\\[6pt] Line 2 before \[ x = 1 \] and \\input{ignored}.
                Custom macro \centeringbox stays intact.
                \begin{align}
                a &= b \\
                c &= d
                \end{align}
                \begin{equation}
                g(f) = \mu f - \frac{1}{2} \sigma^2 f^2
                \end{equation}
            "}
            .to_owned(),
        );
        files
    }

    fn build_tar_bytes(entries: &[(&str, &[u8], u8)]) -> Vec<u8> {
        let mut tar_buf = Vec::new();
        for &(name, content, typeflag) in entries {
            let mut header = [0_u8; 512];
            header[..name.len()].copy_from_slice(name.as_bytes());
            header[100..108].copy_from_slice(b"0000644\0");
            header[108..116].copy_from_slice(b"0000000\0");
            header[116..124].copy_from_slice(b"0000000\0");
            let size_octal = format!("{:011o}\0", content.len());
            header[124..136].copy_from_slice(size_octal.as_bytes());
            header[136..148].copy_from_slice(b"00000000000\0");
            header[148..156].copy_from_slice(b"        ");
            header[156] = typeflag;
            header[257..263].copy_from_slice(b"ustar\0");
            header[263..265].copy_from_slice(b"00");
            let sum: u32 = header.iter().map(|&byte| u32::from(byte)).sum();
            let cksum = format!("{sum:06o}\0 ");
            header[148..156].copy_from_slice(cksum.as_bytes());
            tar_buf.extend_from_slice(&header);
            tar_buf.extend_from_slice(content);
            let pad = (512 - (content.len() % 512)) % 512;
            tar_buf.resize(tar_buf.len() + pad, 0);
        }
        tar_buf.resize(tar_buf.len() + 1024, 0);
        tar_buf
    }

    #[test]
    fn flattens_nested_inputs_and_preserves_title_and_inline_math() {
        let files = sample_multifile_tex();
        let extracted = flatten_and_convert_tex(&files, "main.tex").unwrap();
        assert_eq!(
            extracted.title.as_deref(),
            Some(r"Fractional **Sizing** with {Nested {Braces}} \\")
        );
        assert!(extracted.markdown.contains("## Introduction"));
        assert!(
            extracted
                .markdown
                .contains(r"$\textbf{W}\textbf{x} + P(\text{event})$")
        );
    }

    #[test]
    fn preserves_line_breaks_custom_macros_and_math_environments() {
        let files = sample_multifile_tex();
        let extracted = flatten_and_convert_tex(&files, "main.tex").unwrap();
        assert!(extracted.markdown.contains("Line 2 before"));
        assert!(extracted.markdown.contains(r"\centeringbox"));
        assert!(extracted.markdown.contains(r"\begin{align}"));
        assert!(
            extracted
                .markdown
                .contains(r"g(f) = \mu f - \frac{1}{2} \sigma^2 f^2")
        );
    }

    #[test]
    fn strips_comments_using_exact_backslash_parity() {
        let files = sample_multifile_tex();
        let extracted = flatten_and_convert_tex(&files, "main.tex").unwrap();
        assert!(extracted.markdown.contains("50\\% growth"));
        assert!(!extracted.markdown.contains("trailing comment"));
    }

    #[test]
    fn unpacks_tar_gz_and_single_tex_gz_archives() {
        let tex_body =
            br"\documentclass{article}\begin{document}\section{Hello}World\end{document}";
        let good_tar = build_tar_bytes(&[("main.tex", tex_body, b'0')]);
        let good_targz = test_gzip_bytes(&good_tar);
        let single_gz = test_gzip_bytes(tex_body);

        let parsed_tar = extract_arxiv_source_archive(&good_targz).unwrap();
        let parsed_single = extract_arxiv_source_archive(&single_gz).unwrap();
        assert!(parsed_tar.markdown.contains("## Hello"));
        assert!(parsed_single.markdown.contains("## Hello"));
    }

    #[test]
    fn propagates_safety_block_for_symlink_tar_containing_valid_tex() {
        let tex_body =
            br"\documentclass{article}\begin{document}\section{Hello}World\end{document}";
        let evil_tar = build_tar_bytes(&[("main.tex", tex_body, b'0'), ("escape.tex", b"", b'2')]);
        let evil_targz = test_gzip_bytes(&evil_tar);
        assert!(matches!(
            extract_arxiv_source_archive(&evil_targz).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn handles_deeply_nested_inline_commands_without_stack_overflow() {
        let open = r"\text{".repeat(4_000);
        let close = "}".repeat(4_000);
        let tex = format!(
            r"\documentclass{{article}}\begin{{document}}{open}DeepMathLeaf{close}\end{{document}}"
        );

        let extracted = extract_arxiv_source_archive(tex.as_bytes()).unwrap();
        assert!(extracted.markdown.contains("DeepMathLeaf"));
    }

    fn build_repeat_inputs(target: &str) -> String {
        let mut text = String::new();
        for _ in 0..16 {
            let _ = writeln!(text, "\\input{{{target}}}");
        }
        text
    }

    #[test]
    fn blocks_diamond_dag_byte_expansion_bomb() {
        let mut byte_bomb = BTreeMap::new();
        byte_bomb.insert("a.tex".to_owned(), build_repeat_inputs("b"));
        byte_bomb.insert("b.tex".to_owned(), build_repeat_inputs("c"));
        byte_bomb.insert("c.tex".to_owned(), build_repeat_inputs("d"));
        byte_bomb.insert("d.tex".to_owned(), build_repeat_inputs("e"));
        byte_bomb.insert("e.tex".to_owned(), "A".repeat(4096));
        assert!(matches!(
            flatten_and_convert_tex(&byte_bomb, "a.tex").unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn blocks_empty_leaf_diamond_dag_inclusion_count_bomb() {
        let mut count_bomb = BTreeMap::new();
        count_bomb.insert("a.tex".to_owned(), build_repeat_inputs("b"));
        count_bomb.insert("b.tex".to_owned(), build_repeat_inputs("c"));
        count_bomb.insert("c.tex".to_owned(), build_repeat_inputs("d"));
        count_bomb.insert("d.tex".to_owned(), String::new());
        assert!(matches!(
            flatten_and_convert_tex(&count_bomb, "a.tex").unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn bounds_unclosed_tex_openers_in_linear_time() {
        let openers = r"\title{ \input{ \section[ \begin{equation} \[ \textbf{ ".repeat(15_000);
        let tex = format!(
            r"\documentclass{{article}}\begin{{document}}{openers}TailToken\end{{document}}"
        );
        let extracted = extract_arxiv_source_archive(tex.as_bytes()).unwrap();
        assert!(extracted.markdown.contains("TailToken"));
    }

    #[test]
    fn subslice_unclosed_dollar_and_non_braced_command_do_not_corrupt_subsequent_tex() {
        let padding = "x".repeat(MAX_FAILED_LOOKAHEAD_BYTES + 1024);
        let tex = format!(
            r"\documentclass{{article}}\begin{{document}}\section{{Cost is $5}}\vspace 1em {padding} \section{{Methods}} Formula $\textbf{{W}}\textbf{{x}}$.\end{{document}}"
        );
        let extracted = extract_arxiv_source_archive(tex.as_bytes()).unwrap();
        assert!(extracted.markdown.contains("## Methods"));
        assert!(extracted.markdown.contains(r"$\textbf{W}\textbf{x}$"));
    }

    #[test]
    fn bounds_repeated_unclosed_inputs_and_unbraced_nested_brackets_in_linear_time() {
        let repeated_inputs = r"\input{\input a ".repeat(30_000);
        let nested_brackets = format!("{}{}", r"\section[".repeat(30_000), "]".repeat(30_000));
        let nested_text = format!("{}DeepLeaf{}", r"\text{".repeat(30_000), "}".repeat(30_000));
        let tex = format!(
            r"\documentclass{{article}}\begin{{document}}{repeated_inputs} {nested_brackets} {nested_text}\end{{document}}"
        );
        let extracted = extract_arxiv_source_archive(tex.as_bytes()).unwrap();
        assert!(extracted.markdown.contains("DeepLeaf"));
    }
}
