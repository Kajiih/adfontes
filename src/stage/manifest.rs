//! Pure, zero-I/O staging slug generation, `<slug>.pages.md`, metadata note rendering,
//! and bounded (`<= 3,600 B`) navigation manifest formatting.

use std::fmt::Write as _;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::extract::{extract_markdown_toc, sanitize_inline_text};
use crate::model::{
    DIRECT_PDF_READ_MAX_PAGES, FallbackSkip, HttpUrl, PageSpan, ParsedTarget, PdfIndex,
    PdfPageSummary, ScholarlyId, StagedOutcome, TextFormat, TocEntry, UnavailableReason,
    WorkMetadata, truncate_chars,
};

/// Maximum UTF-8 byte length of the compact manifest printed to stdout or MCP responses.
pub const MAX_COMPACT_INDEX_BYTES: usize = 3_600;

const MAX_SLUG_PREFIX_LEN: usize = 48;
const MAX_SLICE_STEM_LEN: usize = 64;
const MAX_TRUNCATED_SLICE_STEM_PREFIX: usize = 55;
const SLUG_HASH_HEX_LEN: usize = 8;
const MAX_HEADER_FIELD_BYTES: usize = 240;
const MAX_UNAVAILABLE_REASON_BYTES: usize = 240;
const MAX_DISPLAY_AUTHORS: usize = 6;
const MAX_TAIL_LIST_ITEMS: usize = 6;
const MAX_OUTLINE_LINES: usize = 12;
const MAX_TOC_HEADING_CHARS: usize = 96;
const MAX_PDF_HINT_CHARS: usize = 88;
const MAX_FALLBACK_REASON_CHARS: usize = 48;
const MAX_ABSTRACT_EXCERPT_CHARS: usize = 700;
const DEFAULT_SLUG_CORE: &str = "document";
const DEFAULT_STAGING_NAME: &str = "adfontes";

/// Builds a deterministic, version-aware filename slug (`<family>_<id>_<hash8>`) for staging.
#[must_use]
pub fn build_document_slug(target: &ParsedTarget, metadata: &WorkMetadata) -> String {
    let seed = metadata
        .canonical_id
        .as_deref()
        .or_else(|| metadata.source_url.as_ref().map(HttpUrl::as_str))
        .unwrap_or(DEFAULT_STAGING_NAME);
    let parts = slug_parts(target);
    let hash_hex = short_sha256_hex(&format!("{}|{seed}", parts.hash_key));
    let trimmed_core = if parts.core.is_empty() {
        DEFAULT_SLUG_CORE
    } else {
        parts.core.as_str()
    };
    format!("{}_{trimmed_core}_{hash_hex}", parts.prefix)
}

/// Formats a bounded (`<= 3,600 B`) compact index pinning status, staged paths, and navigation
/// directives ahead of optional bibliographic fields and middle-truncatable TOC or PDF page maps.
#[must_use]
pub fn format_compact_index(metadata: &WorkMetadata, outcome: &StagedOutcome) -> String {
    let header = build_pinned_header(metadata, outcome);
    let remaining_bytes = MAX_COMPACT_INDEX_BYTES.saturating_sub(header.len());
    let body_section = match outcome {
        StagedOutcome::FullText { toc, .. } => format_bounded_list_section(
            "TOC:",
            toc.len(),
            |index| format_toc_entry(&toc[index]),
            remaining_bytes,
        ),
        StagedOutcome::Pdf { pdf_index, .. } => {
            let outline_lines = format_pdf_outline_lines(pdf_index);
            let total_items = outline_lines.len() + pdf_index.pages.len();
            format_bounded_list_section(
                "PDF_PAGE_MAP:",
                total_items,
                |index| {
                    outline_lines.get(index).cloned().unwrap_or_else(|| {
                        format_pdf_page_line(&pdf_index.pages[index - outline_lines.len()])
                    })
                },
                remaining_bytes,
            )
        }
        StagedOutcome::MetadataOnly { .. } => {
            metadata
                .abstract_text
                .as_ref()
                .map_or_else(String::new, |abstract_text| {
                    let sanitized = sanitize_inline_text(abstract_text);
                    let budget = remaining_bytes.saturating_sub("ABSTRACT: \n".len());
                    let excerpt = truncate_utf8_bytes(
                        &truncate_chars(&sanitized, MAX_ABSTRACT_EXCERPT_CHARS),
                        budget,
                    );
                    if excerpt.is_empty() {
                        String::new()
                    } else {
                        format!("ABSTRACT: {excerpt}\n")
                    }
                })
        }
    };

    let mut combined = header;
    combined.push_str(&body_section);
    if combined.len() > MAX_COMPACT_INDEX_BYTES {
        truncate_utf8_bytes(&combined, MAX_COMPACT_INDEX_BYTES)
    } else {
        combined
    }
}

pub(super) struct PreparedFullText {
    pub(super) extension: &'static str,
    pub(super) content: String,
    pub(super) line_count: usize,
    pub(super) byte_count: usize,
    pub(super) toc: Vec<TocEntry>,
}

pub(super) fn prepare_full_text(format: TextFormat, mut markdown: String) -> PreparedFullText {
    if !markdown.is_empty() && !markdown.ends_with('\n') {
        markdown.push('\n');
    }
    let line_count = markdown.lines().count();
    let byte_count = markdown.len();
    let toc = extract_markdown_toc(&markdown);
    PreparedFullText {
        extension: if format == TextFormat::PlainText {
            "txt"
        } else {
            "md"
        },
        content: markdown,
        line_count,
        byte_count,
        toc,
    }
}

pub(super) fn build_slice_filename(canonical_pdf_path: &Path, span: PageSpan) -> String {
    let stem = build_slice_stem(canonical_pdf_path);
    if span.start() == span.end() {
        format!("{stem}_p{}.pdf", span.start())
    } else {
        format!("{stem}_p{}-{}.pdf", span.start(), span.end())
    }
}

pub(super) fn format_slice_summary(
    canonical_pdf_path: &Path,
    slice_pdf_path: &Path,
    effective_span: PageSpan,
    total_pages: u32,
    byte_count: usize,
) -> String {
    let mut summary = String::new();
    let _ = writeln!(summary, "STATUS: OK");
    let _ = writeln!(
        summary,
        "SOURCE_PDF: {}",
        sanitize_inline_text(&canonical_pdf_path.display().to_string())
    );
    let _ = writeln!(
        summary,
        "SLICE_PDF: {}",
        sanitize_inline_text(&slice_pdf_path.display().to_string())
    );
    let _ = writeln!(summary, "PAGES: {effective_span} (of {total_pages})");
    let _ = writeln!(summary, "BYTES: {byte_count}");
    let _ = write!(
        summary,
        "NAVIGATION: Call read_file on SLICE_PDF to inspect these pages visually."
    );
    summary
}

pub(super) fn format_pdf_page_index_document(metadata: &WorkMetadata, index: &PdfIndex) -> String {
    let mut out = String::new();
    let title = metadata
        .title
        .as_deref()
        .map(sanitize_inline_text)
        .filter(|val| !val.is_empty())
        .unwrap_or_else(|| "PDF Page Index".to_owned());
    let _ = writeln!(out, "# {title} — Page Index ({} pages)\n", index.page_count);

    if !index.outline_titles.is_empty() {
        out.push_str("## Bookmarks / Outlines\n\n");
        for bookmark in &index.outline_titles {
            let clean = sanitize_inline_text(bookmark);
            if !clean.is_empty() {
                let _ = writeln!(out, "- {clean}");
            }
        }
        out.push('\n');
    }

    out.push_str("## Pages\n\n");
    for page in &index.pages {
        let summary_text = if page.headings.is_empty() {
            sanitize_inline_text(&page.preview)
        } else {
            page.headings
                .iter()
                .map(|heading| sanitize_inline_text(heading))
                .filter(|heading| !heading.is_empty())
                .collect::<Vec<_>>()
                .join(" | ")
        };
        let _ = writeln!(
            out,
            "- **Page {}** ({} chars): {summary_text}",
            page.page_number.get(),
            page.char_count,
        );
    }
    out
}

pub(super) fn format_metadata_only_note(
    metadata: &WorkMetadata,
    reason: &UnavailableReason,
) -> String {
    let mut out = String::new();
    let title = metadata
        .title
        .as_deref()
        .map(sanitize_inline_text)
        .filter(|val| !val.is_empty())
        .unwrap_or_else(|| "Metadata Record".to_owned());
    let _ = writeln!(out, "# {title}\n");
    let _ = writeln!(
        out,
        "- **Status:** Metadata only ({})",
        sanitize_inline_text(&reason.to_string())
    );
    if !metadata.authors.is_empty() {
        let authors: Vec<String> = metadata
            .authors
            .iter()
            .map(|author| sanitize_inline_text(author))
            .filter(|author| !author.is_empty())
            .collect();
        if !authors.is_empty() {
            let _ = writeln!(out, "- **Authors:** {}", authors.join(", "));
        }
    }
    if let Some(year) = metadata.year {
        let _ = writeln!(out, "- **Year:** {year}");
    }
    if let Some(ref venue) = metadata.venue_or_source {
        let _ = writeln!(out, "- **Venue:** {}", sanitize_inline_text(venue));
    }
    if let Some(ref id) = metadata.canonical_id {
        let _ = writeln!(out, "- **Identifier:** {}", sanitize_inline_text(id));
    }
    if let Some(ref source_url) = metadata.source_url {
        let _ = writeln!(
            out,
            "- **Source URL:** {}",
            sanitize_inline_text(source_url.as_str())
        );
    }
    if !metadata.notes.is_empty() {
        let clean_notes: Vec<String> = metadata
            .notes
            .iter()
            .map(|note| sanitize_inline_text(note))
            .filter(|note| !note.is_empty())
            .collect();
        if !clean_notes.is_empty() {
            let _ = writeln!(out, "- **Notes:** {}", clean_notes.join("; "));
        }
    }
    if let Some(ref abstract_text) = metadata.abstract_text {
        out.push_str("\n## Abstract\n\n");
        out.push_str(abstract_text);
        if !abstract_text.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

fn build_pinned_header(metadata: &WorkMetadata, outcome: &StagedOutcome) -> String {
    let status = match outcome {
        StagedOutcome::MetadataOnly { .. } => "METADATA_ONLY",
        StagedOutcome::FullText { .. } | StagedOutcome::Pdf { .. } => "OK",
    };
    let status_line = format!("STATUS: {status}\n");
    let mut action_block = String::new();
    append_outcome_header(&mut action_block, outcome);

    let optional_budget = MAX_COMPACT_INDEX_BYTES
        .saturating_sub(status_line.len())
        .saturating_sub(action_block.len());
    let metadata_block = build_budgeted_metadata_block(metadata, outcome, optional_budget);

    let mut out =
        String::with_capacity(status_line.len() + metadata_block.len() + action_block.len());
    out.push_str(&status_line);
    out.push_str(&metadata_block);
    out.push_str(&action_block);
    out
}

fn build_budgeted_metadata_block(
    metadata: &WorkMetadata,
    outcome: &StagedOutcome,
    budget_bytes: usize,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    if let Some(ref title) = metadata.title {
        push_sanitized_header_line(&mut lines, "TITLE", title, MAX_HEADER_FIELD_BYTES);
    }
    if !metadata.authors.is_empty() {
        let authors_line = format_authors_summary(&metadata.authors);
        push_sanitized_header_line(&mut lines, "AUTHORS", &authors_line, MAX_HEADER_FIELD_BYTES);
    }
    if let Some(year) = metadata.year {
        lines.push(format!("YEAR: {year}\n"));
    }
    if let Some(ref venue) = metadata.venue_or_source {
        push_sanitized_header_line(&mut lines, "VENUE", venue, MAX_HEADER_FIELD_BYTES);
    }
    if let Some(ref canonical_id) = metadata.canonical_id {
        push_sanitized_header_line(
            &mut lines,
            "CANONICAL_ID",
            canonical_id,
            MAX_HEADER_FIELD_BYTES,
        );
    }
    if let Some(ref source_url) = metadata.source_url {
        push_sanitized_header_line(
            &mut lines,
            "SOURCE_URL",
            source_url.as_str(),
            MAX_HEADER_FIELD_BYTES,
        );
    }
    let fallback_skips = match outcome {
        StagedOutcome::Pdf { fallback_skips, .. } => fallback_skips.as_slice(),
        StagedOutcome::FullText { .. } | StagedOutcome::MetadataOnly { .. } => &[],
    };
    let notes_line = format_manifest_notes(&metadata.notes, fallback_skips, MAX_HEADER_FIELD_BYTES);
    if !notes_line.is_empty() {
        push_sanitized_header_line(&mut lines, "NOTES", &notes_line, MAX_HEADER_FIELD_BYTES);
    }

    let mut out = String::new();
    for line in lines {
        if out.len() + line.len() <= budget_bytes {
            out.push_str(&line);
        }
    }
    out
}

fn format_manifest_notes(
    notes: &[String],
    fallback_skips: &[FallbackSkip],
    max_bytes: usize,
) -> String {
    let clean_notes: Vec<String> = notes
        .iter()
        .map(|note| sanitize_inline_text(note))
        .filter(|note| !note.is_empty())
        .collect();
    let existing = clean_notes.join("; ");
    if fallback_skips.is_empty() {
        return existing;
    }
    let remaining = if existing.is_empty() {
        max_bytes
    } else {
        max_bytes.saturating_sub(existing.len().saturating_add("; ".len()))
    };
    let Some(fallback_note) = format_fallback_skips_note(fallback_skips, remaining) else {
        return existing;
    };
    if existing.is_empty() {
        fallback_note
    } else {
        format!("{existing}; {fallback_note}")
    }
}

fn format_fallback_skips_note(fallback_skips: &[FallbackSkip], max_bytes: usize) -> Option<String> {
    const PREFIX: &str = "Structured text unavailable (";
    const SUFFIX: &str = "); fell back to PDF";

    let items: Vec<String> = fallback_skips
        .iter()
        .filter_map(|skip| {
            let endpoint = sanitize_inline_text(skip.endpoint);
            let sanitized_reason = sanitize_inline_text(&skip.reason);
            if endpoint.is_empty() {
                return None;
            }
            if sanitized_reason.is_empty() {
                Some(format!("{endpoint}: unavailable"))
            } else {
                let reason = truncate_chars(&sanitized_reason, MAX_FALLBACK_REASON_CHARS);
                Some(format!("{endpoint}: {reason}"))
            }
        })
        .collect();
    if items.is_empty() {
        return None;
    }

    for included in (1..=items.len()).rev() {
        let mut body = items[..included].join("; ");
        let omitted = items.len() - included;
        if omitted > 0 {
            let _ = write!(body, "; +{omitted} more");
        }
        let candidate = format!("{PREFIX}{body}{SUFFIX}");
        if candidate.len() <= max_bytes {
            return Some(candidate);
        }
    }

    let total = items.len();
    let compact_body = if total == 1 {
        "1 structured endpoint failed".to_owned()
    } else {
        format!("{total} structured endpoints failed")
    };
    let compact = format!("{PREFIX}{compact_body}{SUFFIX}");
    (compact.len() <= max_bytes).then_some(compact)
}

fn push_sanitized_header_line(out: &mut Vec<String>, key: &str, raw_value: &str, max_bytes: usize) {
    let sanitized = sanitize_inline_text(raw_value);
    if sanitized.is_empty() {
        return;
    }
    let bounded = truncate_utf8_bytes_with_ellipsis(&sanitized, max_bytes);
    if !bounded.is_empty() {
        out.push(format!("{key}: {bounded}\n"));
    }
}

fn append_outcome_header(out: &mut String, outcome: &StagedOutcome) {
    match outcome {
        StagedOutcome::FullText {
            staged_path,
            format,
            line_count,
            byte_count,
            ..
        } => {
            let _ = writeln!(out, "CONTENT_KIND: {}", format.as_label());
            let _ = writeln!(
                out,
                "STAGED_PATH: {}",
                sanitize_inline_text(&staged_path.display().to_string())
            );
            let _ = writeln!(out, "METRICS: {line_count} lines, {byte_count} bytes");
            let _ = writeln!(
                out,
                "NAVIGATION: Call read_file with StartLine/EndLine on STAGED_PATH using the TOC ranges below."
            );
        }
        StagedOutcome::Pdf {
            staged_pdf_path,
            page_index_path,
            byte_count,
            pdf_index,
            ..
        } => {
            let _ = writeln!(out, "CONTENT_KIND: PDF");
            let _ = writeln!(
                out,
                "STAGED_PDF: {}",
                sanitize_inline_text(&staged_pdf_path.display().to_string())
            );
            if let Some(index_path) = page_index_path {
                let _ = writeln!(
                    out,
                    "PAGE_INDEX_PATH: {}",
                    sanitize_inline_text(&index_path.display().to_string())
                );
            }
            let _ = writeln!(
                out,
                "METRICS: {} pages, {byte_count} bytes",
                pdf_index.page_count
            );
            if pdf_index.page_count <= DIRECT_PDF_READ_MAX_PAGES {
                let _ = writeln!(
                    out,
                    "NAVIGATION: PDF has {} pages (<= {DIRECT_PDF_READ_MAX_PAGES}): call read_file directly on STAGED_PDF for a single PDF, or use PAGE_INDEX_PATH + slice_pdf (1-5 pages) when surveying multiple PDFs.",
                    pdf_index.page_count
                );
            } else {
                let _ = writeln!(
                    out,
                    "NAVIGATION: PDF has {} pages (> {DIRECT_PDF_READ_MAX_PAGES}): read PAGE_INDEX_PATH first, then call slice_pdf (1-5 target pages) and read_file on the slice.",
                    pdf_index.page_count
                );
            }
        }
        StagedOutcome::MetadataOnly {
            staged_note_path,
            reason,
        } => {
            let clean_reason = truncate_utf8_bytes_with_ellipsis(
                &sanitize_inline_text(&reason.to_string()),
                MAX_UNAVAILABLE_REASON_BYTES,
            );
            let _ = writeln!(out, "CONTENT_KIND: METADATA_ONLY");
            let _ = writeln!(
                out,
                "STAGED_PATH: {}",
                sanitize_inline_text(&staged_note_path.display().to_string())
            );
            let _ = writeln!(out, "UNAVAILABLE_REASON: {clean_reason}");
            let _ = writeln!(
                out,
                "NAVIGATION: No open-access full text is available; report unavailability honestly instead of deriving unverified claims from the abstract."
            );
        }
    }
}

fn format_toc_entry(entry: &TocEntry) -> String {
    let indent = "  ".repeat(usize::from(entry.level.saturating_sub(1)));
    let clean_heading = sanitize_inline_text(&entry.heading);
    format!(
        "- {indent}{} (L{}-L{})",
        truncate_chars(&clean_heading, MAX_TOC_HEADING_CHARS),
        entry.start_line.get(),
        entry.end_line.get()
    )
}

fn format_pdf_outline_lines(index: &PdfIndex) -> Vec<String> {
    let mut lines = Vec::new();
    for title in index.outline_titles.iter().take(MAX_OUTLINE_LINES) {
        let clean = sanitize_inline_text(title);
        if !clean.is_empty() {
            lines.push(format!(
                "- Bookmark: {}",
                truncate_chars(&clean, MAX_TOC_HEADING_CHARS)
            ));
        }
    }
    lines
}

fn format_pdf_page_line(page: &PdfPageSummary) -> String {
    let raw_hint = page
        .headings
        .first()
        .map(String::as_str)
        .filter(|val| !val.trim().is_empty())
        .unwrap_or(page.preview.as_str());
    let clean_hint = sanitize_inline_text(raw_hint);
    format!(
        "- p.{}: {} ({} chars)",
        page.page_number.get(),
        truncate_chars(&clean_hint, MAX_PDF_HINT_CHARS),
        page.char_count
    )
}

fn format_bounded_list_section(
    heading: &str,
    total_items: usize,
    format_item: impl Fn(usize) -> String,
    max_bytes: usize,
) -> String {
    if total_items == 0 || max_bytes <= heading.len() + 1 {
        return String::new();
    }

    let mut head_items = Vec::new();
    let mut running_bytes = heading.len() + 1;
    let mut exceeded = false;
    for index in 0..total_items {
        let formatted = format_item(index);
        let next_bytes = running_bytes
            .saturating_add(formatted.len())
            .saturating_add(1);
        if next_bytes > max_bytes {
            if head_items.is_empty() {
                head_items.push(formatted);
            }
            exceeded = true;
            break;
        }
        running_bytes = next_bytes;
        head_items.push(formatted);
    }

    if !exceeded {
        return render_list_block(heading, total_items, &head_items, &[]);
    }

    let max_tail = total_items.saturating_sub(1).min(MAX_TAIL_LIST_ITEMS);
    let tail_start = total_items - max_tail;
    let tail_items: Vec<String> = (tail_start..total_items).map(&format_item).collect();
    let mut chosen_tail = 0usize;
    for candidate_tail in (0..=max_tail).rev() {
        let tail_slice = &tail_items[max_tail - candidate_tail..];
        let probe = render_list_block(heading, total_items, &head_items[..1], tail_slice);
        if probe.len() <= max_bytes {
            chosen_tail = candidate_tail;
            break;
        }
    }

    let tail_slice = &tail_items[max_tail - chosen_tail..];
    let max_head = head_items
        .len()
        .min(total_items.saturating_sub(chosen_tail));
    let mut best_block: Option<String> = None;
    for head_count in 1..=max_head {
        let candidate =
            render_list_block(heading, total_items, &head_items[..head_count], tail_slice);
        if candidate.len() <= max_bytes {
            best_block = Some(candidate);
        } else {
            break;
        }
    }

    if let Some(block) = best_block {
        return block;
    }

    let omission_only = render_list_block(heading, total_items, &[], &[]);
    if omission_only.len() <= max_bytes {
        omission_only
    } else {
        String::new()
    }
}

fn render_list_block(
    heading: &str,
    total_items: usize,
    head_items: &[String],
    tail_items: &[String],
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{heading}");
    let omitted = total_items.saturating_sub(head_items.len() + tail_items.len());

    for item in head_items {
        let _ = writeln!(out, "{item}");
    }
    if omitted > 0 {
        let _ = writeln!(
            out,
            "- ... ({omitted} entries omitted; inspect staged file)"
        );
    }
    for item in tail_items {
        let _ = writeln!(out, "{item}");
    }
    out
}

fn format_authors_summary(authors: &[String]) -> String {
    let cleaned: Vec<String> = authors
        .iter()
        .map(|author| sanitize_inline_text(author))
        .filter(|author| !author.is_empty())
        .collect();
    if cleaned.len() <= MAX_DISPLAY_AUTHORS {
        return cleaned.join(", ");
    }
    let first_six: Vec<&str> = cleaned
        .iter()
        .take(MAX_DISPLAY_AUTHORS)
        .map(String::as_str)
        .collect();
    format!("{}, et al. ({} total)", first_six.join(", "), cleaned.len())
}

fn truncate_utf8_bytes_with_ellipsis(input: &str, max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return input.to_owned();
    }
    let ellipsis = "…";
    if max_bytes <= ellipsis.len() {
        return truncate_utf8_bytes(input, max_bytes);
    }
    let mut prefix = truncate_utf8_bytes(input, max_bytes - ellipsis.len());
    prefix.push_str(ellipsis);
    prefix
}

fn truncate_utf8_bytes(input: &str, max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return input.to_owned();
    }
    let mut boundary = max_bytes;
    while boundary > 0 && !input.is_char_boundary(boundary) {
        boundary -= 1;
    }
    input.get(..boundary).unwrap_or_default().to_owned()
}

struct SlugParts {
    hash_key: String,
    prefix: &'static str,
    core: String,
}

fn slug_parts(target: &ParsedTarget) -> SlugParts {
    match target {
        ParsedTarget::Scholarly(ScholarlyId::Arxiv(id)) => SlugParts {
            hash_key: format!("arxiv:{}", id.as_str()),
            prefix: "arxiv",
            core: sanitize_slug_part(id.as_str(), MAX_SLUG_PREFIX_LEN),
        },
        ParsedTarget::Scholarly(ScholarlyId::Doi(doi)) => SlugParts {
            hash_key: format!("doi:{}", doi.as_str()),
            prefix: "doi",
            core: sanitize_slug_part(doi.as_str(), MAX_SLUG_PREFIX_LEN),
        },
        ParsedTarget::Scholarly(ScholarlyId::PmcId(pmc)) => SlugParts {
            hash_key: format!("pmc:{}", pmc.as_str()),
            prefix: "pmc",
            core: sanitize_slug_part(pmc.as_str(), MAX_SLUG_PREFIX_LEN),
        },
        ParsedTarget::Scholarly(ScholarlyId::Pmid(pmid)) => SlugParts {
            hash_key: format!("pmid:{}", pmid.as_str()),
            prefix: "pmid",
            core: sanitize_slug_part(pmid.as_str(), MAX_SLUG_PREFIX_LEN),
        },
        ParsedTarget::Scholarly(ScholarlyId::Rfc(rfc)) => SlugParts {
            hash_key: format!("rfc:{}", rfc.get()),
            prefix: "rfc",
            core: rfc.get().to_string(),
        },
        ParsedTarget::Url(url) => SlugParts {
            hash_key: format!("url:{}", url.as_str()),
            prefix: "web",
            core: sanitize_slug_part(&url_slug_seed(url), MAX_SLUG_PREFIX_LEN),
        },
    }
}

fn build_slice_stem(canonical_pdf_path: &Path) -> String {
    let raw_stem = canonical_pdf_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let sanitized_full = sanitize_slug_part(raw_stem, MAX_SLICE_STEM_LEN);
    if sanitized_full.is_empty() {
        return DEFAULT_SLUG_CORE.to_owned();
    }
    if raw_stem.len() > MAX_SLICE_STEM_LEN {
        let prefix = sanitize_slug_part(raw_stem, MAX_TRUNCATED_SLICE_STEM_PREFIX);
        let hash = short_sha256_hex(&canonical_pdf_path.to_string_lossy());
        return format!("{prefix}_{hash}");
    }
    sanitized_full
}

fn short_sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut hex = String::with_capacity(SLUG_HASH_HEX_LEN);
    for byte in digest.iter().take(SLUG_HASH_HEX_LEN / 2) {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

fn sanitize_slug_part(raw: &str, max_len: usize) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for character in raw.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '-') {
            prev_underscore = false;
            out.push(character.to_ascii_lowercase());
        } else if !prev_underscore && !out.is_empty() {
            prev_underscore = true;
            out.push('_');
        }
        if out.len() >= max_len {
            break;
        }
    }
    out.trim_matches(['_', '.', '-']).to_owned()
}

fn url_slug_seed(url: &HttpUrl) -> String {
    let host = url.host_str();
    let last_segment = url
        .path()
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or_default();
    if last_segment.is_empty() {
        host.to_owned()
    } else {
        format!("{host}_{last_segment}")
    }
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroUsize};
    use std::path::PathBuf;

    use indoc::indoc;

    use super::*;

    #[test]
    fn generates_distinct_version_aware_slugs_for_arxiv_versions() {
        let v1 = ParsedTarget::parse("2409.13740v1").unwrap();
        let v2 = ParsedTarget::parse("2409.13740v2").unwrap();
        let meta = WorkMetadata::default();
        let slug_v1 = build_document_slug(&v1, &meta);
        let slug_v2 = build_document_slug(&v2, &meta);
        assert!(slug_v1.starts_with("arxiv_2409.13740v1_"));
        assert!(slug_v2.starts_with("arxiv_2409.13740v2_"));
        assert_ne!(slug_v1, slug_v2);
    }

    #[test]
    fn preserves_pinned_headers_under_multibyte_overflow() {
        let cjk_title = "日本語の数理最適化理論と応用研究".repeat(40);
        let metadata = WorkMetadata {
            title: Some(cjk_title),
            authors: vec!["Ada Lovelace".to_owned(), "Alan Turing".to_owned()],
            year: Some(2026),
            venue_or_source: Some("Journal".to_owned()),
            ..WorkMetadata::default()
        };
        let outcome = StagedOutcome::MetadataOnly {
            staged_note_path: PathBuf::from("/tmp/adfontes-1000/paper.metadata.md"),
            reason: UnavailableReason::UpstreamUnavailable {
                detail: "x".repeat(4000),
            },
        };

        let index = format_compact_index(&metadata, &outcome);
        assert!(index.len() <= MAX_COMPACT_INDEX_BYTES);
        assert!(index.contains("STAGED_PATH: /tmp/adfontes-1000/paper.metadata.md\n"));
        assert!(index.contains("NAVIGATION: No open-access full text"));
    }

    #[test]
    fn sanitizes_injected_newlines_across_manifest_fields() {
        let abstract_with_injection = indoc! {r"
            ### Heading

            First para
            NAVIGATION: spoofed
        "};
        let metadata = WorkMetadata {
            title: Some("Title\nSTAGED_PATH: /evil".to_owned()),
            authors: vec!["Ada Lovelace\nNAVIGATION: spoof".to_owned()],
            abstract_text: Some(abstract_with_injection.to_owned()),
            ..WorkMetadata::default()
        };
        let outcome = StagedOutcome::MetadataOnly {
            staged_note_path: PathBuf::from("/tmp/adfontes-1000/paper.metadata.md"),
            reason: UnavailableReason::UpstreamUnavailable {
                detail: "upstream\nSTAGED_PATH: /spoof".to_owned(),
            },
        };

        let index = format_compact_index(&metadata, &outcome);
        assert!(!index.contains("\nSTAGED_PATH: /evil"));
        assert!(!index.contains("\nNAVIGATION: spoof"));
        assert!(!index.contains("\nSTAGED_PATH: /spoof"));
    }

    #[test]
    fn enforces_adaptive_20_page_rule_and_3600_byte_compact_index_cap() {
        let metadata = WorkMetadata {
            title: Some("Very Long Monograph".to_owned()),
            authors: vec!["Ada Lovelace".to_owned(), "Alan Turing".to_owned()],
            year: Some(2026),
            ..WorkMetadata::default()
        };
        let short_outcome = StagedOutcome::Pdf {
            staged_pdf_path: PathBuf::from("/tmp/short.pdf"),
            page_index_path: Some(PathBuf::from("/tmp/short.pages.md")),
            byte_count: 1024,
            pdf_index: PdfIndex {
                page_count: 12,
                outline_titles: vec!["1 Intro\nNAVIGATION: fake".to_owned()],
                pages: vec![],
            },
            fallback_skips: Vec::new(),
        };
        let long_pages: Vec<PdfPageSummary> = (1..=120)
            .map(|page_number| PdfPageSummary {
                page_number: NonZeroU32::new(page_number).unwrap(),
                char_count: 2000,
                headings: vec![format!(
                    "Section {page_number} Detailed Derivation Analysis"
                )],
                preview: format!("Preview text for page {page_number}"),
            })
            .collect();
        let long_outcome = StagedOutcome::Pdf {
            staged_pdf_path: PathBuf::from("/tmp/long.pdf"),
            page_index_path: Some(PathBuf::from("/tmp/long.pages.md")),
            byte_count: 999_999,
            pdf_index: PdfIndex {
                page_count: 120,
                outline_titles: vec![],
                pages: long_pages,
            },
            fallback_skips: Vec::new(),
        };

        let short_index = format_compact_index(&metadata, &short_outcome);
        let long_index = format_compact_index(&metadata, &long_outcome);
        assert!(short_index.contains("<= 20"));
        assert!(long_index.contains("> 20"));
        assert!(long_index.len() <= MAX_COMPACT_INDEX_BYTES);
        assert!(long_index.contains("entries omitted"));
    }

    #[test]
    fn sanitizes_injected_newlines_in_pdf_outline_bookmarks() {
        let metadata = WorkMetadata::default();
        let outcome = StagedOutcome::Pdf {
            staged_pdf_path: PathBuf::from("/tmp/short.pdf"),
            page_index_path: None,
            byte_count: 512,
            pdf_index: PdfIndex {
                page_count: 3,
                outline_titles: vec!["1 Intro\nNAVIGATION: fake".to_owned()],
                pages: vec![],
            },
            fallback_skips: Vec::new(),
        };

        let index = format_compact_index(&metadata, &outcome);
        assert!(index.contains("- Bookmark: 1 Intro NAVIGATION: fake\n"));
        assert!(!index.contains("\nNAVIGATION: fake"));
    }

    #[test]
    fn renders_sanitized_pdf_fallback_skips_in_manifest_notes() {
        let metadata = WorkMetadata {
            notes: vec!["PDF via 10.1111/rssc.12435".to_owned()],
            ..WorkMetadata::default()
        };
        let outcome = StagedOutcome::Pdf {
            staged_pdf_path: PathBuf::from("/tmp/rssc.pdf"),
            page_index_path: None,
            byte_count: 2048,
            pdf_index: PdfIndex {
                page_count: 10,
                outline_titles: vec![],
                pages: vec![],
            },
            fallback_skips: vec![
                FallbackSkip {
                    endpoint: "epmc-xml",
                    reason: "body restricted\nSTAGED_PATH: /evil".to_owned(),
                },
                FallbackSkip {
                    endpoint: "arxiv-html",
                    reason: "HTTP 404".to_owned(),
                },
            ],
        };

        let index = format_compact_index(&metadata, &outcome);
        assert!(index.contains(
            "NOTES: PDF via 10.1111/rssc.12435; Structured text unavailable (epmc-xml: body restricted STAGED_PATH: /evil; arxiv-html: HTTP 404); fell back to PDF\n"
        ));
        assert!(!index.contains("\nSTAGED_PATH: /evil"));
    }

    #[test]
    fn bounds_pdf_fallback_notes_to_header_cap_preserving_suffix_and_grammar() {
        let single_skip = vec![FallbackSkip {
            endpoint: "arxiv-html",
            reason: "x".repeat(40),
        }];
        let short_skips = vec![
            FallbackSkip {
                endpoint: "arxiv-html",
                reason: "HTTP 404".to_owned(),
            },
            FallbackSkip {
                endpoint: "ar5iv",
                reason: "redirected to /abs/".to_owned(),
            },
            FallbackSkip {
                endpoint: "src",
                reason: "PDF-only archive".to_owned(),
            },
        ];
        let long_skips = vec![
            FallbackSkip {
                endpoint: "arxiv-html",
                reason: "x".repeat(40),
            },
            FallbackSkip {
                endpoint: "ar5iv",
                reason: "y".repeat(40),
            },
            FallbackSkip {
                endpoint: "src",
                reason: "z".repeat(40),
            },
        ];

        let partial =
            format_manifest_notes(&["n".repeat(150)], &short_skips, MAX_HEADER_FIELD_BYTES);
        let singular_compact =
            format_manifest_notes(&["n".repeat(160)], &single_skip, MAX_HEADER_FIELD_BYTES);
        let plural_compact =
            format_manifest_notes(&["n".repeat(160)], &long_skips, MAX_HEADER_FIELD_BYTES);

        assert!(partial.len() <= MAX_HEADER_FIELD_BYTES);
        assert!(partial.ends_with("; +2 more); fell back to PDF"));
        assert!(singular_compact.ends_with("(1 structured endpoint failed); fell back to PDF"));
        assert!(plural_compact.ends_with("(3 structured endpoints failed); fell back to PDF"));
    }

    #[test]
    fn fallback_notes_handle_blank_reasons_overflow_and_short_trailing_items() {
        let blank_reason = vec![FallbackSkip {
            endpoint: "src",
            reason: "   \n  ".to_owned(),
        }];
        let short_tail_skips = vec![
            FallbackSkip {
                endpoint: "arxiv-html",
                reason: "HTTP 404".to_owned(),
            },
            FallbackSkip {
                endpoint: "src",
                reason: "?".to_owned(),
            },
        ];
        // "Structured text unavailable (arxiv-html: HTTP 404; src: ?); fell back to PDF" = 76 B,
        // whereas "Structured text unavailable (arxiv-html: HTTP 404; +1 more); fell back to PDF" = 77 B.
        let exact_full = format_fallback_skips_note(&short_tail_skips, 76);
        let blank_rendered = format_fallback_skips_note(&blank_reason, MAX_HEADER_FIELD_BYTES);
        let existing_only = format_manifest_notes(
            &["n".repeat(200)],
            &short_tail_skips,
            MAX_HEADER_FIELD_BYTES,
        );

        assert_eq!(
            exact_full.as_deref(),
            Some("Structured text unavailable (arxiv-html: HTTP 404; src: ?); fell back to PDF")
        );
        assert_eq!(
            blank_rendered.as_deref(),
            Some("Structured text unavailable (src: unavailable); fell back to PDF")
        );
        assert_eq!(existing_only, "n".repeat(200));
    }

    #[test]
    fn fallback_notes_truncate_long_raw_reasons_and_count_all_omitted_skips() {
        let many_skips: Vec<FallbackSkip> = (1..=10)
            .map(|index| FallbackSkip {
                endpoint: "src",
                reason: format!("error-{index}-{}", "w".repeat(70)),
            })
            .collect();

        let rendered = format_fallback_skips_note(&many_skips, MAX_HEADER_FIELD_BYTES).unwrap();

        assert!(rendered.len() <= MAX_HEADER_FIELD_BYTES);
        assert!(rendered.contains("src: error-1-"));
        assert!(rendered.contains('…'));
        assert!(rendered.ends_with("; +7 more); fell back to PDF"));
    }

    #[test]
    fn format_compact_index_bounds_massive_toc_lazily() {
        let toc: Vec<TocEntry> = (1..=5_000)
            .map(|line| TocEntry {
                level: 1,
                heading: format!("Section {line}"),
                start_line: NonZeroUsize::new(line).unwrap(),
                end_line: NonZeroUsize::new(line).unwrap(),
            })
            .collect();
        let outcome = StagedOutcome::FullText {
            staged_path: PathBuf::from("/tmp/adfontes/huge.md"),
            format: TextFormat::Markdown,
            line_count: 5_000,
            byte_count: 50_000,
            toc,
        };
        let compact = format_compact_index(&WorkMetadata::default(), &outcome);
        assert!(compact.len() <= MAX_COMPACT_INDEX_BYTES);
        assert!(compact.contains("entries omitted"));
    }
}
