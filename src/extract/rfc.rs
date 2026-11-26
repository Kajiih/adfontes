//! Plain-text IETF RFC pagination cleaner (`\x0c` form-feed headers and footers).

use super::title::{ExtractedText, normalize_markdown_spacing, sanitize_inline_text};

const FORM_FEED: char = '\x0c';
const MIN_HEADER_SCAN_LINES: usize = 45;
const RFC_PREFIX_LEN: usize = 4;
const YEAR_DIGIT_COUNT: usize = 4;
const CENTERED_TITLE_MIN_INDENT: usize = 4;

/// Strips paginated `\x0c` running headers and `[Page N]` footers from plain-text IETF RFCs
/// while preserving ASCII diagrams, BNF grammars, and paragraph structure.
#[must_use]
pub fn clean_rfc_text(raw: &str) -> ExtractedText {
    let unified = raw.replace("\r\n", "\n").replace('\r', "\n");
    let lines = split_with_form_feed_markers(&unified);
    let mut skip = vec![false; lines.len()];

    for (index, entry) in lines.iter().enumerate() {
        if !entry.is_page_break {
            continue;
        }
        if let Some(flag) = skip.get_mut(index) {
            *flag = true;
        }
        mark_preceding_footer(&lines, &mut skip, index);
        mark_following_header(&lines, &mut skip, index);
    }

    let mut cleaned_lines = Vec::with_capacity(lines.len());
    for (index, entry) in lines.iter().enumerate() {
        if !skip.get(index).copied().unwrap_or(false) {
            cleaned_lines.push(entry.text);
        }
    }

    let title = detect_rfc_text_title(&cleaned_lines);
    let markdown = normalize_markdown_spacing(&cleaned_lines.join("\n"));
    ExtractedText { title, markdown }
}

struct RfcLine<'a> {
    text: &'a str,
    is_page_break: bool,
}

fn split_with_form_feed_markers(unified: &str) -> Vec<RfcLine<'_>> {
    let mut entries = Vec::new();
    for raw_line in unified.split('\n') {
        if !raw_line.contains(FORM_FEED) {
            entries.push(RfcLine {
                text: raw_line,
                is_page_break: false,
            });
            continue;
        }
        let mut segments = raw_line.split(FORM_FEED);
        if let Some(first) = segments.next()
            && !first.trim().is_empty()
        {
            entries.push(RfcLine {
                text: first,
                is_page_break: false,
            });
        }
        for segment in segments {
            entries.push(RfcLine {
                text: "",
                is_page_break: true,
            });
            if !segment.trim().is_empty() {
                entries.push(RfcLine {
                    text: segment,
                    is_page_break: false,
                });
            }
        }
    }
    entries
}

fn mark_preceding_footer(lines: &[RfcLine<'_>], skip: &mut [bool], form_feed_index: usize) {
    let mut cursor = form_feed_index;
    while cursor > 0 {
        cursor -= 1;
        let Some(entry) = lines.get(cursor) else {
            break;
        };
        if entry.is_page_break {
            break;
        }
        let trimmed = entry.text.trim();
        if trimmed.is_empty() {
            if let Some(flag) = skip.get_mut(cursor) {
                *flag = true;
            }
            continue;
        }
        if looks_like_rfc_page_footer(trimmed)
            && let Some(flag) = skip.get_mut(cursor)
        {
            *flag = true;
        }
        break;
    }
}

fn mark_following_header(lines: &[RfcLine<'_>], skip: &mut [bool], form_feed_index: usize) {
    let mut cursor = form_feed_index + 1;
    let mut stripped_header = false;
    while let Some(entry) = lines.get(cursor) {
        if entry.is_page_break {
            break;
        }
        let trimmed = entry.text.trim();
        if trimmed.is_empty() {
            if let Some(flag) = skip.get_mut(cursor) {
                *flag = true;
            }
            cursor += 1;
            continue;
        }
        if !stripped_header && looks_like_rfc_running_header(trimmed) {
            if let Some(flag) = skip.get_mut(cursor) {
                *flag = true;
            }
            stripped_header = true;
            cursor += 1;
            continue;
        }
        break;
    }
}

fn looks_like_rfc_page_footer(trimmed: &str) -> bool {
    let Some(start) = trimmed.rfind("[Page ") else {
        return false;
    };
    let rest = trimmed.get(start + 6..).unwrap_or_default();
    let Some(digits) = rest.strip_suffix(']') else {
        return false;
    };
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn looks_like_rfc_running_header(trimmed: &str) -> bool {
    let starts_rfc = trimmed
        .get(..RFC_PREFIX_LEN)
        .is_some_and(|head| head.eq_ignore_ascii_case("rfc "));
    let has_year = trimmed.split_whitespace().last().is_some_and(|token| {
        token.len() == YEAR_DIGIT_COUNT && token.bytes().all(|byte| byte.is_ascii_digit())
    });
    starts_rfc && has_year
}

fn detect_rfc_text_title(lines: &[&str]) -> Option<String> {
    let mut saw_meta_line = false;
    let mut saw_blank_after_meta = false;
    let mut title_parts = Vec::new();
    for line in lines.iter().take(MIN_HEADER_SCAN_LINES) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !title_parts.is_empty() {
                break;
            }
            if saw_meta_line {
                saw_blank_after_meta = true;
            }
            continue;
        }
        if !saw_blank_after_meta {
            saw_meta_line = true;
            continue;
        }
        if trimmed.eq_ignore_ascii_case("Abstract")
            || trimmed.eq_ignore_ascii_case("Status of This Memo")
        {
            break;
        }
        let leading_spaces = line.len().saturating_sub(line.trim_start().len());
        if leading_spaces >= CENTERED_TITLE_MIN_INDENT {
            title_parts.push(sanitize_inline_text(trimmed));
        } else if !title_parts.is_empty() {
            break;
        }
    }
    if title_parts.is_empty() {
        None
    } else {
        Some(title_parts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_form_feed_headers_and_footers_while_preserving_grammar() {
        let sample = concat!(
            "\n\n",
            "Internet Engineering Task Force (IETF)                        R. Fielding\n",
            "Request for Comments: 9110                                      June 2022\n",
            "\n",
            "                            HTTP Semantics\n",
            "\n",
            "Abstract\n",
            "\n",
            "   field-line = field-name \":\" OWS\n",
            "Fielding, et al.             Standards Track                    [Page 1]\n",
            "\x0c\n",
            "RFC 9110                     HTTP Semantics                    June 2022\n",
            "\n",
            "                field-value OWS\n",
        );
        let extracted = clean_rfc_text(sample);
        assert_eq!(extracted.title.as_deref(), Some("HTTP Semantics"));
        insta::assert_snapshot!(extracted.markdown, @r#"
        Internet Engineering Task Force (IETF)                        R. Fielding
        Request for Comments: 9110                                      June 2022

                                    HTTP Semantics

        Abstract

           field-line = field-name ":" OWS
                        field-value OWS
        "#);
    }

    #[test]
    fn preserves_inline_form_feed_prose_and_subsequent_rfc_citations() {
        let sample = concat!(
            "Network Working Group                                           J. Author\n",
            "Request for Comments: 9999                                   January 2026\n",
            "\n",
            "                            Test Protocol\n",
            "\n",
            "Author                       Standards Track                    [Page 1]\n",
            "\x0cRFC 9999                     Test Protocol                  January 2026\n",
            "RFC 7230 was superseded in June 2014\n",
            "\x0c   continued prose after bare form feed\n",
        );
        let extracted = clean_rfc_text(sample);
        assert_eq!(extracted.title.as_deref(), Some("Test Protocol"));
        insta::assert_snapshot!(extracted.markdown, @"
        Network Working Group                                           J. Author
        Request for Comments: 9999                                   January 2026

                                    Test Protocol

        RFC 7230 was superseded in June 2014
           continued prose after bare form feed
        ");
    }
}
