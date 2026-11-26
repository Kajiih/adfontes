//! Shared text normalization, Markdown formatting, `CommonMark` TOC extraction (`pulldown_cmark`), and title helpers.

use std::fmt::Write as _;
use std::num::NonZeroUsize;
use std::ops::{ControlFlow, Range};

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

use crate::model::TocEntry;

const MAX_HEADER_SCAN_LINES: usize = 40;
const MIN_SETEXT_UNDERLINE_LEN: usize = 3;
const MIN_FENCE_LEN: usize = 3;
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
    let fence_ranges = collect_fenced_code_ranges(&unified);
    let mut result = String::with_capacity(unified.len());
    let mut consecutive_blanks: usize = 0;
    let mut line_start: usize = 0;
    let mut fence_idx: usize = 0;

    for line in unified.split('\n') {
        while fence_ranges
            .get(fence_idx)
            .is_some_and(|range| line_start >= range.end)
        {
            fence_idx += 1;
        }
        let in_fence = fence_ranges
            .get(fence_idx)
            .is_some_and(|range| line_start >= range.start && line_start < range.end);
        let trimmed_end = line.trim_end_matches([' ', '\t']);
        line_start = line_start.saturating_add(line.len()).saturating_add(1);

        if !in_fence && trimmed_end.is_empty() {
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

/// Extracts a `CommonMark` 1-indexed line-ranged Markdown Table of Contents via `pulldown_cmark`.
///
/// Skips opening YAML frontmatter, ignores headings inside code fences or 4-space/tab indented
/// code blocks, skips empty headings (for example `## ###`), and bounds output at `<= 2,048`
/// entries so adversarial inputs cannot amplify memory.
#[must_use]
pub fn extract_markdown_toc(markdown: &str) -> Vec<TocEntry> {
    let mut entries: Vec<TocEntry> = Vec::new();
    let outcome = visit_markdown_headings(markdown, |parsed| {
        if let Some(previous) = entries.last_mut() {
            let end_line = parsed
                .start_line
                .saturating_sub(1)
                .max(previous.start_line.get());
            if let Some(end_nz) = NonZeroUsize::new(end_line) {
                previous.end_line = end_nz;
            }
        }
        if entries.len() < MAX_TOC_ENTRIES {
            if let Some(start_nz) = NonZeroUsize::new(parsed.start_line) {
                entries.push(TocEntry {
                    level: parsed.level,
                    heading: parsed.heading,
                    start_line: start_nz,
                    end_line: start_nz,
                });
            }
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    });

    if outcome.is_continue()
        && let Some(last) = entries.last_mut()
    {
        let total_lines = markdown.lines().count();
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

/// Normalizes and prefixes non-empty Markdown blockquote lines with `> ` into `out`.
pub(super) fn push_markdown_blockquote(out: &mut String, inner: &str) {
    let normalized = normalize_markdown_spacing(inner);
    if normalized.is_empty() {
        return;
    }
    for line in normalized.lines() {
        if line.is_empty() {
            out.push_str(">\n");
        } else {
            let _ = writeln!(out, "> {line}");
        }
    }
    out.push('\n');
}

fn collect_fenced_code_ranges(unified: &str) -> Vec<Range<usize>> {
    if !unified.contains("```") && !unified.contains("~~~") {
        return Vec::new();
    }
    Parser::new_ext(unified, Options::empty())
        .into_offset_iter()
        .filter_map(|(event, range)| match event {
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(_))) => Some(range),
            _ => None,
        })
        .collect()
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
    let scanned = scan_yaml_frontmatter(lines.iter().copied())?;
    for line in scanned.lines {
        if let Some(rest) = strip_key_ignore_case(line.trim(), TITLE_HEADER_PREFIX) {
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
    let mut title = None;
    let _ = visit_markdown_headings(markdown, |parsed| {
        // Restrict fallback H1 title extraction to ATX `#` headings so PEP/RST `===` documents
        // still pass through to `extract_rst_setext_title` (which enforces underline length).
        if parsed.level == 1 && parsed.is_atx {
            title = Some(parsed.heading);
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    title
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

struct ParsedMarkdownHeading {
    level: u8,
    heading: String,
    start_line: usize,
    is_atx: bool,
}

fn visit_markdown_headings<F>(markdown: &str, mut on_heading: F) -> ControlFlow<()>
where
    F: FnMut(ParsedMarkdownHeading) -> ControlFlow<()>,
{
    let post = split_post_frontmatter(markdown);
    if post.body.is_empty() {
        return ControlFlow::Continue(());
    }

    let mut cursor = 0usize;
    let mut current_line = post.start_line_base + 1;
    let mut container_depth = 0usize;
    let mut active: Option<ActiveHeading> = None;

    for (event, range) in Parser::new_ext(post.body, Options::empty()).into_offset_iter() {
        match event {
            Event::Start(Tag::BlockQuote(_) | Tag::List(_)) => {
                container_depth += 1;
            }
            Event::End(TagEnd::BlockQuote(_) | TagEnd::List(_)) => {
                container_depth = container_depth.saturating_sub(1);
            }
            Event::Start(Tag::Heading { level, .. }) if container_depth == 0 => {
                if range.start > cursor
                    && let Some(gap) = post.body.get(cursor..range.start)
                {
                    current_line += gap.bytes().filter(|&byte| byte == b'\n').count();
                    cursor = range.start;
                }
                let is_atx = post.body.get(range).is_some_and(is_atx_heading_raw);
                active = Some(ActiveHeading {
                    level: level as u8,
                    start_line: current_line,
                    is_atx,
                    raw_text: String::new(),
                });
            }
            Event::Text(text) | Event::Code(text) => {
                if let Some(ref mut heading) = active {
                    heading.raw_text.push_str(&text);
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some(ref mut heading) = active {
                    heading.raw_text.push(' ');
                }
            }
            Event::InlineHtml(html) => {
                if let Some(ref mut heading) = active
                    && is_inline_br_html(&html)
                {
                    heading.raw_text.push(' ');
                }
            }
            Event::End(TagEnd::Heading(_)) if container_depth == 0 => {
                if let Some(finished) = active.take() {
                    let heading = sanitize_inline_text(&finished.raw_text);
                    if !heading.is_empty()
                        && on_heading(ParsedMarkdownHeading {
                            level: finished.level,
                            heading,
                            start_line: finished.start_line,
                            is_atx: finished.is_atx,
                        })
                        .is_break()
                    {
                        return ControlFlow::Break(());
                    }
                }
            }
            _ => {}
        }
    }
    ControlFlow::Continue(())
}

fn is_inline_br_html(raw: &str) -> bool {
    const BR_OPEN_PREFIX: &[u8] = b"<br";
    let bytes = raw.trim_start().as_bytes();
    bytes
        .get(..BR_OPEN_PREFIX.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(BR_OPEN_PREFIX))
        && bytes
            .get(BR_OPEN_PREFIX.len())
            .is_none_or(|&next| matches!(next, b'>' | b'/' | b' ' | b'\t' | b'\r' | b'\n'))
}

fn is_atx_heading_raw(raw: &str) -> bool {
    let trimmed = raw.trim_start_matches(' ');
    let hashes = trimmed.bytes().take_while(|&byte| byte == b'#').count();
    (1..=MAX_HEADING_LEVEL).contains(&hashes)
        && trimmed
            .as_bytes()
            .get(hashes)
            .is_none_or(|&byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
}

struct ActiveHeading {
    level: u8,
    start_line: usize,
    is_atx: bool,
    raw_text: String,
}

struct PostFrontmatter<'a> {
    start_line_base: usize,
    body: &'a str,
}

fn split_post_frontmatter(markdown: &str) -> PostFrontmatter<'_> {
    let start_idx = scan_yaml_frontmatter(markdown.lines())
        .and_then(|scanned| scanned.closed_line_count)
        .unwrap_or(0);
    if start_idx == 0 {
        return PostFrontmatter {
            start_line_base: 0,
            body: markdown,
        };
    }
    let byte_offset: usize = markdown
        .split_inclusive('\n')
        .take(start_idx)
        .map(str::len)
        .sum();
    PostFrontmatter {
        start_line_base: start_idx,
        body: markdown.get(byte_offset..).unwrap_or_default(),
    }
}

struct YamlFrontmatter<'a> {
    lines: Vec<&'a str>,
    closed_line_count: Option<usize>,
}

fn scan_yaml_frontmatter<'a>(
    mut iter: impl Iterator<Item = &'a str>,
) -> Option<YamlFrontmatter<'a>> {
    if iter.next()?.trim() != YAML_FRONTMATTER_DELIMITER {
        return None;
    }
    let mut lines = Vec::new();
    for (offset, line) in iter.enumerate().take(MAX_HEADER_SCAN_LINES) {
        let trimmed = line.trim();
        if trimmed == YAML_FRONTMATTER_DELIMITER || trimmed == YAML_FRONTMATTER_END_DOTS {
            return Some(YamlFrontmatter {
                lines,
                closed_line_count: Some(offset + 2),
            });
        }
        lines.push(line);
    }
    Some(YamlFrontmatter {
        lines,
        closed_line_count: None,
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

#[cfg(test)]
mod tests {
    use indoc::{formatdoc, indoc};
    use rstest::rstest;

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
        insta::assert_snapshot!(preserved, @"
        ````


        ```


        ````

        After
        ");
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

    #[test]
    fn decodes_inline_markdown_in_headings_across_crlf_frontmatter() {
        let crlf_doc = concat!(
            "---\r\n",
            "title: Frontmatter\r\n",
            "---\r\n",
            "## *Emphasis*, `code`, &amp; [Link](https://example.org)\r\n",
            "Body.\r\n",
        );
        let toc = extract_markdown_toc(crlf_doc);
        assert_eq!(
            toc,
            vec![TocEntry {
                level: 2,
                heading: "Emphasis, code, & Link".to_owned(),
                start_line: NonZeroUsize::new(4).unwrap(),
                end_line: NonZeroUsize::new(5).unwrap(),
            }]
        );
    }

    #[rstest]
    #[case("Long Prose Before Short Equals")]
    #[case("#define FOO")]
    #[case("####### Long Prose")]
    fn ignores_short_setext_equals_in_title_extraction(#[case] headline: &str) {
        let doc = formatdoc! {"
            {headline}
            ===
            Paragraph.
        "};
        assert_eq!(extract_text_title(&doc), None);
    }

    #[test]
    fn ignores_headings_nested_inside_blockquotes_and_lists() {
        let doc = indoc! {r"
            > # Quoted Title
            - # List Heading
            # Top Title
        "};
        let toc = extract_markdown_toc(doc);
        assert_eq!(extract_text_title(doc).as_deref(), Some("Top Title"));
        assert_eq!(toc.len(), 1);
        assert_eq!(toc[0].heading, "Top Title");
    }

    #[test]
    fn separates_heading_words_across_inline_br_tag() {
        assert_eq!(
            extract_markdown_h1_title("# Alpha<br>Beta").as_deref(),
            Some("Alpha Beta")
        );
    }
}
