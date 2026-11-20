//! Shared text normalization, Markdown formatting, and title extraction helpers.

use std::fmt::Write as _;
use std::num::NonZeroUsize;

use crate::model::TocEntry;

const MAX_HEADER_SCAN_LINES: usize = 40;
const MIN_SETEXT_UNDERLINE_LEN: usize = 3;
const MIN_FENCE_LEN: usize = 3;
const MAX_COMMONMARK_INDENT: usize = 3;
const MAX_HEADING_LEVEL: usize = 6;
const MAX_CODE_LANGUAGE_LEN: usize = 32;
const TITLE_HEADER_PREFIX: &str = "title:";
const YAML_FRONTMATTER_DELIMITER: &str = "---";
const YAML_FRONTMATTER_END_DOTS: &str = "...";

/// Normalized full-text Markdown alongside an optional extracted document title.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedText {
    /// Extracted title when detected in document metadata or headings.
    pub title: Option<String>,
    /// Clean UTF-8 Markdown or plain text body.
    pub markdown: String,
}

/// Active `CommonMark` §4.5 code-fence delimiter state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CodeFenceState {
    marker: u8,
    length: usize,
}

/// Collapses ASCII whitespace and control characters into single spaces and trims ends.
#[must_use]
pub fn sanitize_inline_text(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut pending_space = false;
    for character in input.chars() {
        if character.is_whitespace() || character.is_control() {
            if !output.is_empty() {
                pending_space = true;
            }
        } else {
            if pending_space {
                output.push(' ');
                pending_space = false;
            }
            output.push(character);
        }
    }
    output
}

/// Normalizes line endings, strips trailing horizontal whitespace, and collapses excessive blank
/// lines outside `CommonMark` fenced code blocks.
#[must_use]
pub fn normalize_markdown_spacing(raw: &str) -> String {
    let unified = raw.replace("\r\n", "\n").replace('\r', "\n");
    let mut result = String::with_capacity(unified.len());
    let mut consecutive_blanks: usize = 0;
    let mut active_fence: Option<CodeFenceState> = None;

    for line in unified.split('\n') {
        let trimmed_end = line.trim_end_matches([' ', '\t']);
        update_code_fence_state(trimmed_end, &mut active_fence);
        if active_fence.is_none() && trimmed_end.is_empty() {
            if result.is_empty() {
                continue;
            }
            consecutive_blanks += 1;
            if consecutive_blanks <= 1 {
                result.push('\n');
            }
            continue;
        }
        consecutive_blanks = 0;
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(trimmed_end);
    }

    while result.ends_with('\n') {
        result.pop();
    }
    result
}

/// Extracts a document title from YAML frontmatter, PEP/RFC-822 headers, Markdown H1, or RST titles.
#[must_use]
pub fn extract_text_title(text: &str) -> Option<String> {
    let header_lines: Vec<&str> = text.lines().take(MAX_HEADER_SCAN_LINES + 2).collect();
    extract_yaml_frontmatter_title(&header_lines)
        .or_else(|| extract_pep_header_title(&header_lines))
        .or_else(|| extract_markdown_h1_title(text))
        .or_else(|| extract_rst_setext_title(&header_lines))
}

const MAX_TOC_ENTRIES: usize = 2_048;

/// Extracts a fence-aware, 1-indexed line-ranged Markdown Table of Contents.
///
/// Respects `CommonMark` §4.2–§4.5 indentation rules (ignoring 4+-space or tab-indented code
/// blocks), strips closing ATX `#` markers only when space-separated, skips opening YAML
/// frontmatter, recognizes both ATX (`#`) and Setext (`===` / `---`) headings, and streams in a
/// single bounded pass (`<= 2,048` entries) to prevent memory amplification on adversarial inputs.
#[must_use]
pub fn extract_markdown_toc(markdown: &str) -> Vec<TocEntry> {
    let start_idx = yaml_frontmatter_end_index(markdown.lines()).unwrap_or(0);
    let mut entries: Vec<TocEntry> = Vec::new();
    let mut active_fence: Option<CodeFenceState> = None;
    let mut last_entry_closed = false;
    let mut total_lines: usize = 0;

    let mut lines = markdown.lines().enumerate().peekable();
    while let Some((idx, line)) = lines.next() {
        total_lines = idx + 1;
        if idx < start_idx {
            continue;
        }
        let was_in_fence = active_fence.is_some();
        update_code_fence_state(line, &mut active_fence);
        if was_in_fence || active_fence.is_some() {
            continue;
        }

        let heading_found = parse_atx_heading(line).or_else(|| {
            let &(_, next_line) = lines.peek()?;
            let parsed = parse_setext_heading(line, next_line)?;
            let _ = lines.next();
            total_lines = idx + 2;
            Some(parsed)
        });

        if let Some(parsed) = heading_found {
            let start_line = idx + 1;
            if !last_entry_closed && let Some(previous) = entries.last_mut() {
                let end_line = start_line.saturating_sub(1).max(previous.start_line.get());
                if let Some(end_nz) = NonZeroUsize::new(end_line) {
                    previous.end_line = end_nz;
                }
            }
            if entries.len() < MAX_TOC_ENTRIES {
                if let Some(start_nz) = NonZeroUsize::new(start_line) {
                    entries.push(TocEntry {
                        level: parsed.level,
                        heading: parsed.heading,
                        start_line: start_nz,
                        end_line: start_nz,
                    });
                    last_entry_closed = false;
                }
            } else {
                last_entry_closed = true;
            }
        }
    }

    if !last_entry_closed && let Some(last) = entries.last_mut() {
        let end_line = total_lines.max(last.start_line.get());
        if let Some(end_nz) = NonZeroUsize::new(end_line) {
            last.end_line = end_nz;
        }
    }
    entries
}

/// Sanitizes a fenced code block language identifier to safe ASCII token characters.
#[must_use]
pub(super) fn sanitize_code_language(raw: &str) -> String {
    raw.trim()
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '_')
        })
        .take(MAX_CODE_LANGUAGE_LEN)
        .collect()
}

/// Appends a collision-safe fenced code block using a backtick fence longer than any backtick run
/// appearing inside `body`.
pub(super) fn push_fenced_code_block(out: &mut String, lang: &str, body: &str) {
    let trimmed = body.trim_matches('\n');
    if trimmed.trim().is_empty() {
        return;
    }
    let fence_len = max_backtick_run(trimmed)
        .saturating_add(1)
        .max(MIN_FENCE_LEN);
    let clean_lang = sanitize_code_language(lang);
    for _ in 0..fence_len {
        out.push('`');
    }
    out.push_str(&clean_lang);
    out.push('\n');
    out.push_str(trimmed);
    out.push('\n');
    for _ in 0..fence_len {
        out.push('`');
    }
    out.push_str("\n\n");
}

/// Appends a collision-safe `CommonMark` inline code span for `text`.
pub(super) fn push_inline_code(out: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    let delim_len = max_backtick_run(text).saturating_add(1);
    let needs_pad = text.starts_with('`')
        || text.ends_with('`')
        || (text.starts_with(' ') && text.ends_with(' ') && text.bytes().any(|byte| byte != b' '));
    for _ in 0..delim_len {
        out.push('`');
    }
    if needs_pad {
        out.push(' ');
    }
    out.push_str(text);
    if needs_pad {
        out.push(' ');
    }
    for _ in 0..delim_len {
        out.push('`');
    }
}

/// Renders a non-empty grid of pre-escaped cells as a GitHub-Flavored Markdown pipe table.
#[must_use]
pub(super) fn render_gfm_table(rows: &[Vec<String>]) -> String {
    let Some(header) = rows.first() else {
        return String::new();
    };
    let width = rows.iter().map(Vec::len).max().unwrap_or(header.len());
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    push_gfm_row(&mut out, header, width);
    out.push('|');
    for _ in 0..width {
        out.push_str(" --- |");
    }
    out.push('\n');
    for row in rows.iter().skip(1) {
        push_gfm_row(&mut out, row, width);
    }
    out
}

/// Formats a sanitized Markdown ATX heading into `out`.
pub(super) fn push_markdown_heading(out: &mut String, level: usize, text: &str) {
    let clamped = level.clamp(1, MAX_HEADING_LEVEL);
    for _ in 0..clamped {
        out.push('#');
    }
    let _ = writeln!(out, " {text}\n");
}

/// Updates `CommonMark` §4.5 fenced code block tracking for a single line.
pub(super) fn update_code_fence_state(line: &str, active_fence: &mut Option<CodeFenceState>) {
    let Some(trimmed) = strip_commonmark_indent(line) else {
        return;
    };
    let Some(parsed) = parse_fence_prefix(trimmed) else {
        return;
    };
    match *active_fence {
        None => {
            if parsed.marker != b'`' || !parsed.info.contains('`') {
                *active_fence = Some(CodeFenceState {
                    marker: parsed.marker,
                    length: parsed.length,
                });
            }
        }
        Some(open) => {
            if parsed.marker == open.marker
                && parsed.length >= open.length
                && parsed.info.trim().is_empty()
            {
                *active_fence = None;
            }
        }
    }
}

fn push_gfm_row(out: &mut String, row: &[String], width: usize) {
    out.push('|');
    for col in 0..width {
        out.push(' ');
        if let Some(cell) = row.get(col) {
            out.push_str(cell);
        }
        out.push_str(" |");
    }
    out.push('\n');
}

fn max_backtick_run(text: &str) -> usize {
    let mut longest: usize = 0;
    let mut current: usize = 0;
    for byte in text.bytes() {
        if byte == b'`' {
            current += 1;
            if current > longest {
                longest = current;
            }
        } else {
            current = 0;
        }
    }
    longest
}

fn extract_yaml_frontmatter_title(lines: &[&str]) -> Option<String> {
    if lines.first().map(|line| line.trim()) != Some(YAML_FRONTMATTER_DELIMITER) {
        return None;
    }
    for line in lines.iter().skip(1).take(MAX_HEADER_SCAN_LINES) {
        let trimmed = line.trim();
        if trimmed == YAML_FRONTMATTER_DELIMITER || trimmed == YAML_FRONTMATTER_END_DOTS {
            break;
        }
        if let Some(rest) = strip_key_ignore_case(trimmed, TITLE_HEADER_PREFIX) {
            let unquoted = strip_matching_yaml_quotes(rest.trim());
            let cleaned = sanitize_inline_text(unquoted);
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }
    None
}

fn strip_matching_yaml_quotes(value: &str) -> &str {
    let bytes = value.as_bytes();
    if bytes.len() >= 2
        && let (Some(&first), Some(&last)) = (bytes.first(), bytes.last())
        && first == last
        && matches!(first, b'"' | b'\'')
    {
        return value.get(1..value.len() - 1).unwrap_or(value);
    }
    value
}

fn extract_pep_header_title(lines: &[&str]) -> Option<String> {
    let mut saw_metadata_header = false;
    for line in lines.iter().take(MAX_HEADER_SCAN_LINES) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if saw_metadata_header {
                break;
            }
            continue;
        }
        if let Some(rest) = strip_key_ignore_case(trimmed, TITLE_HEADER_PREFIX) {
            let cleaned = sanitize_inline_text(rest);
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
        if let Some((key, _)) = trimmed.split_once(':')
            && !key.is_empty()
            && key
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b' '))
        {
            saw_metadata_header = true;
            continue;
        }
        if !saw_metadata_header {
            break;
        }
    }
    None
}

fn extract_markdown_h1_title(markdown: &str) -> Option<String> {
    let start_idx = yaml_frontmatter_end_index(markdown.lines()).unwrap_or(0);
    let mut active_fence: Option<CodeFenceState> = None;
    for line in markdown.lines().skip(start_idx) {
        let was_in_fence = active_fence.is_some();
        update_code_fence_state(line, &mut active_fence);
        if was_in_fence || active_fence.is_some() {
            continue;
        }
        if let Some(parsed) = parse_atx_heading(line)
            && parsed.level == 1
        {
            return Some(parsed.heading);
        }
    }
    None
}

fn extract_rst_setext_title(lines: &[&str]) -> Option<String> {
    for (index, line) in lines.iter().enumerate().take(MAX_HEADER_SCAN_LINES) {
        let candidate = line.trim();
        if candidate.is_empty() || is_rst_adornment(candidate) {
            continue;
        }
        let Some(next_line) = lines.get(index + 1).map(|next| next.trim()) else {
            continue;
        };
        let min_required = candidate.chars().count().max(MIN_SETEXT_UNDERLINE_LEN);
        if is_rst_adornment(next_line) && next_line.len() >= min_required {
            let cleaned = sanitize_inline_text(candidate);
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }
    None
}

fn strip_key_ignore_case<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let head = line.get(..key.len())?;
    head.eq_ignore_ascii_case(key)
        .then(|| line.get(key.len()..))
        .flatten()
}

struct ParsedHeading {
    level: u8,
    heading: String,
}

fn yaml_frontmatter_end_index<'a>(mut lines: impl Iterator<Item = &'a str>) -> Option<usize> {
    if lines.next()?.trim() != YAML_FRONTMATTER_DELIMITER {
        return None;
    }
    for (offset, line) in lines.enumerate().take(MAX_HEADER_SCAN_LINES) {
        let trimmed = line.trim();
        if trimmed == YAML_FRONTMATTER_DELIMITER || trimmed == YAML_FRONTMATTER_END_DOTS {
            return Some(offset + 2);
        }
    }
    None
}

fn parse_setext_heading(line: &str, next_line: &str) -> Option<ParsedHeading> {
    let candidate = strip_commonmark_indent(line)?;
    let trimmed_candidate = candidate.trim_end_matches([' ', '\t']);
    if trimmed_candidate.is_empty()
        || is_setext_excluded_prefix(trimmed_candidate)
        || parse_fence_prefix(trimmed_candidate).is_some()
        || parse_atx_heading(line).is_some()
    {
        return None;
    }

    let underline = strip_commonmark_indent(next_line)?.trim_end_matches([' ', '\t']);
    let bytes = underline.as_bytes();
    let &first = bytes.first()?;
    let level = match first {
        b'=' if bytes.iter().all(|&byte| byte == b'=') => 1,
        b'-' if bytes.iter().all(|&byte| byte == b'-') => 2,
        _ => return None,
    };
    let cleaned = sanitize_inline_text(trimmed_candidate);
    if cleaned.is_empty() {
        return None;
    }
    Some(ParsedHeading {
        level,
        heading: cleaned,
    })
}

fn is_setext_excluded_prefix(trimmed: &str) -> bool {
    if trimmed.starts_with('>')
        || trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
        || is_rst_adornment(trimmed)
    {
        return true;
    }
    let digits = trimmed.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0
        && let Some(rest) = trimmed.get(digits..)
        && (rest.starts_with(". ") || rest.starts_with(") "))
    {
        return true;
    }
    false
}

fn parse_atx_heading(line: &str) -> Option<ParsedHeading> {
    let trimmed = strip_commonmark_indent(line)?;
    let hashes = trimmed.bytes().take_while(|&byte| byte == b'#').count();
    if !(1..=MAX_HEADING_LEVEL).contains(&hashes) {
        return None;
    }
    let after = trimmed.get(hashes..)?;
    if !after.is_empty() && !after.starts_with(' ') && !after.starts_with('\t') {
        return None;
    }
    let stripped = strip_commonmark_closing_hashes(after.trim_start_matches([' ', '\t']));
    let cleaned = sanitize_inline_text(stripped);
    if cleaned.is_empty() {
        return None;
    }
    let level = u8::try_from(hashes).ok()?;
    Some(ParsedHeading {
        level,
        heading: cleaned,
    })
}

fn is_rst_adornment(line: &str) -> bool {
    let bytes = line.as_bytes();
    if bytes.len() < MIN_SETEXT_UNDERLINE_LEN {
        return false;
    }
    let Some(&first) = bytes.first() else {
        return false;
    };
    matches!(first, b'=' | b'-' | b'~' | b'#' | b'*') && bytes.iter().all(|&byte| byte == first)
}

struct ParsedFenceLine<'a> {
    marker: u8,
    length: usize,
    info: &'a str,
}

fn parse_fence_prefix(trimmed: &str) -> Option<ParsedFenceLine<'_>> {
    let &marker = trimmed.as_bytes().first()?;
    if !matches!(marker, b'`' | b'~') {
        return None;
    }
    let length = trimmed.bytes().take_while(|&byte| byte == marker).count();
    if length < MIN_FENCE_LEN {
        return None;
    }
    let info = trimmed.get(length..).unwrap_or_default();
    Some(ParsedFenceLine {
        marker,
        length,
        info,
    })
}

fn strip_commonmark_indent(line: &str) -> Option<&str> {
    let leading_spaces = line.bytes().take_while(|&byte| byte == b' ').count();
    if leading_spaces > MAX_COMMONMARK_INDENT {
        return None;
    }
    let rest = line.get(leading_spaces..)?;
    if rest.starts_with('\t') {
        return None;
    }
    Some(rest)
}

fn strip_commonmark_closing_hashes(content: &str) -> &str {
    let trimmed = content.trim_end_matches([' ', '\t']);
    let without_hashes = trimmed.trim_end_matches('#');
    if without_hashes.len() == trimmed.len() {
        return trimmed;
    }
    if without_hashes.is_empty() || without_hashes.ends_with(' ') || without_hashes.ends_with('\t')
    {
        without_hashes.trim_end_matches([' ', '\t'])
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use super::*;

    #[test]
    fn extracts_pep_rst_header_title() {
        let pep = indoc! {r"
            PEP: 695
            Title: Type Parameter Syntax
            Author: Eric Traut
            Status: Final

            Abstract
            ========
            Adds cleaner syntax for generics.
        "};
        assert_eq!(
            extract_text_title(pep).as_deref(),
            Some("Type Parameter Syntax")
        );
    }

    #[test]
    fn extracts_yaml_frontmatter_and_preserves_internal_unquoted_quotes() {
        let quoted = indoc! {r#"
            ---
            title: "Non-Lexical Lifetimes"
            ---
            Body text.
        "#};
        let unquoted_with_quotes = indoc! {r#"
            ---
            title: It's a "Test"
            ---
            Body text.
        "#};
        assert_eq!(
            extract_text_title(quoted).as_deref(),
            Some("Non-Lexical Lifetimes")
        );
        assert_eq!(
            extract_text_title(unquoted_with_quotes).as_deref(),
            Some(r#"It's a "Test""#)
        );
    }

    #[test]
    fn requires_rst_underline_at_least_as_long_as_title() {
        let prose_then_rule = indoc! {r"
            This is a regular prose paragraph that ends before a thematic break.
            ---
            # Actual Markdown Title
        "};
        let rst = indoc! {r"
            ================
            Overlined Header
            ================

            Paragraph.
        "};
        assert_eq!(
            extract_text_title(prose_then_rule).as_deref(),
            Some("Actual Markdown Title")
        );
        assert_eq!(extract_text_title(rst).as_deref(), Some("Overlined Header"));
    }

    #[test]
    fn ignores_headings_inside_nested_four_backtick_fences() {
        let doc = indoc! {r"
            ````markdown
            ```rust
            # Not A Heading
            ```
            ~~~
            # Still Inside Four-Backtick Fence
            ````

            # Real Heading
        "};
        assert_eq!(extract_text_title(doc).as_deref(), Some("Real Heading"));
    }

    #[test]
    fn preserves_blank_lines_inside_four_backtick_fences() {
        let raw = indoc! {r"
            ````


            ```


            ````


            After
        "};
        let preserved = normalize_markdown_spacing(raw);
        let expected_inner = indoc! {r"
            ````


            ```


            ````
        "};
        assert!(preserved.starts_with(expected_inner.trim_end()));
        assert!(preserved.ends_with("\n\nAfter"));
    }

    #[test]
    fn chooses_collision_free_code_fence_and_sanitizes_language() {
        let mut fenced = String::new();
        push_fenced_code_block(&mut fenced, "rust\n# injected", "let s = \"```\";");
        assert!(fenced.starts_with("````rustinjected\n"));
        assert!(fenced.ends_with("\n````\n\n"));
    }

    #[test]
    fn pads_inline_code_containing_backticks() {
        let mut inline = String::new();
        push_inline_code(&mut inline, "`foo`");
        assert_eq!(inline, "`` `foo` ``");
    }

    #[test]
    fn extracts_commonmark_toc_with_spans_fences_indents_trailing_hashes_and_setext() {
        let doc = concat!(
            "---\n",
            "# YAML comment must not appear in TOC\n",
            "title: Frontmatter Title\n",
            "---\n",
            "# C#\n",
            "Body line.\n",
            "    # four-space indented python comment\n",
            "\t# tab-indented comment\n",
            "    ```\n",
            "## Migrating to F#\n",
            "```rust\n",
            "# Inside backtick fence\n",
            "```\n",
            "~~~python\n",
            "## Inside tilde fence\n",
            "~~~\n",
            "### Title ###   \n",
            "## ###\n",
            "Setext Section\n",
            "==============\n",
            "Paragraph.\n",
            "\n",
            "---\n",
            "Subheading\n",
            "----------\n",
            "###### Deepest\n",
        );
        let toc = extract_markdown_toc(doc);
        let summary: Vec<(u8, &str, usize, usize)> = toc
            .iter()
            .map(|entry| {
                (
                    entry.level,
                    entry.heading.as_str(),
                    entry.start_line.get(),
                    entry.end_line.get(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                (1, "C#", 5, 9),
                (2, "Migrating to F#", 10, 16),
                (3, "Title", 17, 18),
                (1, "Setext Section", 19, 23),
                (2, "Subheading", 24, 25),
                (6, "Deepest", 26, 26),
            ]
        );
        assert_eq!(extract_markdown_h1_title("# C#").as_deref(), Some("C#"));
    }

    #[test]
    fn caps_markdown_toc_entries_under_streaming_budget() {
        let massive = "# Heading\n".repeat(MAX_TOC_ENTRIES + 500);
        let toc = extract_markdown_toc(&massive);
        assert_eq!(toc.len(), MAX_TOC_ENTRIES);
    }
}
