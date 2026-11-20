//! Confined filesystem staging, version-aware slugs, `<slug>.pages.md`, PDF slicing, and compact
//! (`<= 3,600 B`) navigation manifests.
//!
//! Enforces strict filesystem confinement (blocking configured forbidden prefixes, VCS/credential
//! directories, and agent configuration roots case-insensitively while permitting canonicalized
//! temp, `ADFONTES_EXTRA_ALLOWED_ROOTS`, and `ADFONTES_STAGING_DIR` roots), step-by-step
//! `openat(O_DIRECTORY | O_NOFOLLOW)` directory verification with `fstat` UID ownership and
//! `fchmod(0700)` enforcement on Unix, atomic `0600` file writes via `tempfile`, and the adaptive
//! `20`-page PDF consumption policy.

architecture_component!(Stage);

use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
#[cfg(unix)]
use std::fs::File;
#[cfg(not(unix))]
use std::fs::{DirBuilder, OpenOptions};
use std::io::{self, Read, Write as _};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
#[cfg(windows)]
use std::os::windows::fs::MetadataExt as WindowsMetadataExt;
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use rustix::fs::{CWD, Mode, OFlags, fchmod, mkdirat, openat};
#[cfg(unix)]
use rustix::io::Errno;
use sha2::{Digest, Sha256};
use tempfile::{Builder as TempFileBuilder, NamedTempFile};

use crate::extract::{extract_markdown_toc, sanitize_inline_text};
use crate::model::{
    AdfontesError, DIRECT_PDF_READ_MAX_PAGES, ExtractedDocument, ExtractedPayload, HttpUrl,
    PageSpan, ParsedTarget, PdfIndex, StagedOutcome, StagedResult, TextFormat, TocEntry,
    UnavailableReason, WorkMetadata,
};
use crate::pdf::{index_pdf_bytes, slice_pdf_bytes};

/// Maximum UTF-8 byte length of the compact manifest printed to stdout or MCP responses.
pub const MAX_COMPACT_INDEX_BYTES: usize = 3_600;

const MAX_LOCAL_PDF_BYTES: u64 = 100 * 1024 * 1024;
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
const MAX_ABSTRACT_EXCERPT_CHARS: usize = 700;
const DEFAULT_SLUG_CORE: &str = "document";
const DEFAULT_STAGING_NAME: &str = "adfontes";
#[cfg(unix)]
const FILE_MODE_OWNER_ONLY: u32 = 0o600;
#[cfg(unix)]
const MODE_GROUP_OTHER_MASK: u32 = 0o077;
#[cfg(windows)]
const WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;

const FORBIDDEN_PATH_SEGMENTS: &[&str] = &[".git", ".jj", ".hg", ".ssh", ".gnupg", ".aws", ".kube"];
const FORBIDDEN_HOME_ROOTS: &[&[&str]] = &[
    &[".gemini", "config"],
    &[".agents"],
    &[".config"],
    &[".local"],
];

/// Boundary-parsed filesystem confinement and PDF slicing policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagingPolicy {
    /// Process temp directory (`std::env::temp_dir()`).
    pub temp_dir: PathBuf,
    /// Conventional Unix `/tmp` root (resolves to `/private/tmp` on macOS).
    pub system_tmp_dir: PathBuf,
    /// User home directory (`HOME` / `USERPROFILE`), when available.
    pub home_dir: Option<PathBuf>,
    /// Optional operator-configured default staging root (`ADFONTES_STAGING_DIR`).
    pub configured_staging_dir: Option<PathBuf>,
    /// Additional operator-configured allowed staging roots (`ADFONTES_EXTRA_ALLOWED_ROOTS`).
    pub extra_allowed_roots: Vec<PathBuf>,
    /// Additional operator-configured absolute forbidden path prefixes
    /// (`ADFONTES_FORBIDDEN_PATH_PREFIXES`).
    pub forbidden_path_prefixes: Vec<PathBuf>,
    /// Disables local PDF slicing when `ADFONTES_DISABLE_PDF_SLICE` is truthy.
    pub disable_pdf_slice: bool,
    /// Effective Unix UID verified against existing staging directories.
    #[cfg(unix)]
    pub effective_uid: u32,
}

impl StagingPolicy {
    /// Constructs a [`StagingPolicy`] by reading environment variables once at startup.
    #[must_use]
    pub fn from_env() -> Self {
        let home_dir = std::env::var_os("HOME")
            .filter(|val| !val.is_empty())
            .or_else(|| std::env::var_os("USERPROFILE").filter(|val| !val.is_empty()))
            .map(PathBuf::from);
        let configured_staging_dir = std::env::var_os("ADFONTES_STAGING_DIR")
            .filter(|val| !val.is_empty())
            .map(PathBuf::from);
        let extra_allowed_roots =
            std::env::var_os("ADFONTES_EXTRA_ALLOWED_ROOTS").map_or_default(|raw| {
                std::env::split_paths(&raw)
                    .filter(|path| !path.as_os_str().is_empty())
                    .collect()
            });
        let forbidden_path_prefixes = std::env::var_os("ADFONTES_FORBIDDEN_PATH_PREFIXES")
            .map_or_default(|raw| {
                std::env::split_paths(&raw)
                    .filter(|path| !path.as_os_str().is_empty())
                    .collect()
            });
        let disable_pdf_slice =
            std::env::var("ADFONTES_DISABLE_PDF_SLICE").is_ok_and(|val| is_truthy_env(&val));
        Self {
            temp_dir: std::env::temp_dir(),
            system_tmp_dir: PathBuf::from("/tmp"),
            home_dir,
            configured_staging_dir,
            extra_allowed_roots,
            forbidden_path_prefixes,
            disable_pdf_slice,
            #[cfg(unix)]
            effective_uid: rustix::process::geteuid().as_raw(),
        }
    }
}

/// Output summary produced by [`slice_staged_pdf`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedPdfSlice {
    /// Original source PDF path.
    pub source_pdf_path: PathBuf,
    /// Newly written standalone vector PDF slice path (`0600`).
    pub slice_pdf_path: PathBuf,
    /// Total page count of the source PDF.
    pub total_pages: u32,
    /// Effective clamped 1-indexed page span contained in [`Self::slice_pdf_path`].
    pub effective_span: PageSpan,
    /// Byte size of the written PDF slice.
    pub byte_count: usize,
    /// Compact plain-text summary suitable for CLI stdout or MCP tool responses.
    pub summary: String,
}

/// Stages an already-resolved [`ExtractedDocument`] atomically (`0600`) inside a confined staging
/// directory and builds its bounded (`<= 3,600 B`) navigation manifest.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the target staging directory fails confinement or
/// ownership checks, [`AdfontesError::Extraction`] if PDF indexing fails, or [`AdfontesError::Io`] on
/// local filesystem write failures.
pub fn stage_document(
    target: &ParsedTarget,
    doc: ExtractedDocument,
    output_dir: Option<&Path>,
    include_pdf_index: bool,
    policy: &StagingPolicy,
) -> Result<StagedResult, AdfontesError> {
    let staging_dir = resolve_confined_staging_dir(output_dir, policy)?;
    let slug = build_document_slug(target, &doc.metadata);

    let outcome = match doc.payload {
        ExtractedPayload::FullText { format, markdown } => {
            let extension = if format == TextFormat::PlainText {
                "txt"
            } else {
                "md"
            };
            let staged_path = staging_dir.join(format!("{slug}.{extension}"));
            let mut content = markdown;
            if !content.is_empty() && !content.ends_with('\n') {
                content.push('\n');
            }
            write_file_atomic(&staged_path, content.as_bytes())?;
            let line_count = content.lines().count();
            let byte_count = content.len();
            let toc = extract_markdown_toc(&content);
            StagedOutcome::FullText {
                staged_path,
                format,
                line_count,
                byte_count,
                toc,
            }
        }
        ExtractedPayload::Pdf { bytes } => {
            let staged_pdf_path = staging_dir.join(format!("{slug}.pdf"));
            // Write into an owner-only temp file first so corrupt, encrypted, or bomb PDFs
            // rejected by `index_pdf_bytes` never persist on disk (NamedTempFile cleans up on drop).
            let temp_pdf = create_staged_temp_file(&staging_dir, &bytes)?;
            let pdf_index = index_pdf_bytes(&bytes, Some(temp_pdf.path()))?;
            persist_staged_temp_file(temp_pdf, &staged_pdf_path)?;
            let page_index_path = if include_pdf_index {
                let index_path = staging_dir.join(format!("{slug}.pages.md"));
                let page_doc = format_pdf_page_index_document(&doc.metadata, &pdf_index);
                write_file_atomic(&index_path, page_doc.as_bytes())?;
                Some(index_path)
            } else {
                None
            };
            StagedOutcome::Pdf {
                staged_pdf_path,
                page_index_path,
                byte_count: bytes.len(),
                pdf_index,
            }
        }
        ExtractedPayload::MetadataOnly { reason } => {
            let staged_note_path = staging_dir.join(format!("{slug}.metadata.md"));
            let note_text = format_metadata_only_note(&doc.metadata, &reason);
            write_file_atomic(&staged_note_path, note_text.as_bytes())?;
            StagedOutcome::MetadataOnly {
                staged_note_path,
                reason,
            }
        }
    };

    let compact_index = format_compact_index(&doc.metadata, &outcome);
    Ok(StagedResult {
        metadata: doc.metadata,
        outcome,
        compact_index,
    })
}

/// Validates, slices, and atomically writes (`0600`) a `1..=8`-page vector PDF slice from a local
/// confined staged PDF file.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if slicing is disabled by `ADFONTES_DISABLE_PDF_SLICE`, if
/// `pdf_path` escapes allowed staging roots, traverses a symlink, is a FIFO/non-regular file, or
/// exceeds `100 MiB`, or if `output_dir` fails confinement; returns [`AdfontesError::InvalidTarget`]
/// or [`AdfontesError::Extraction`] on invalid page bounds or PDF errors.
pub fn slice_staged_pdf(
    pdf_path: &Path,
    span: PageSpan,
    output_dir: Option<&Path>,
    policy: &StagingPolicy,
) -> Result<StagedPdfSlice, AdfontesError> {
    if policy.disable_pdf_slice {
        return Err(AdfontesError::SafetyBlock {
            reason: "PDF slicing is disabled by ADFONTES_DISABLE_PDF_SLICE".to_owned(),
        });
    }
    if let Some(explicit) = output_dir {
        validate_staging_dir_policy(Some(explicit), policy)?;
    }

    let confined_pdf = read_bounded_confined_pdf(pdf_path, policy)?;
    let canonical_pdf_path = confined_pdf.canonical_path;
    let sliced = slice_pdf_bytes(&confined_pdf.bytes, span)?;

    let target_dir = if let Some(explicit) = output_dir {
        resolve_confined_staging_dir(Some(explicit), policy)?
    } else {
        let parent = canonical_pdf_path
            .parent()
            .ok_or_else(|| AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF path `{}` has no parent directory",
                    canonical_pdf_path.display()
                ),
            })?;
        resolve_confined_staging_dir(Some(parent), policy)?
    };

    let stem = build_slice_stem(&canonical_pdf_path);
    let slice_name = if sliced.effective_span.start() == sliced.effective_span.end() {
        format!("{stem}_p{}.pdf", sliced.effective_span.start())
    } else {
        format!(
            "{stem}_p{}-{}.pdf",
            sliced.effective_span.start(),
            sliced.effective_span.end()
        )
    };
    let slice_pdf_path = target_dir.join(slice_name);
    write_file_atomic(&slice_pdf_path, &sliced.bytes)?;

    let byte_count = sliced.bytes.len();
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
    let _ = writeln!(
        summary,
        "PAGES: {} (of {})",
        sliced.effective_span, sliced.total_pages
    );
    let _ = writeln!(summary, "BYTES: {byte_count}");
    let _ = write!(
        summary,
        "NAVIGATION: Call read_file on SLICE_PDF to inspect these pages visually."
    );

    Ok(StagedPdfSlice {
        source_pdf_path: canonical_pdf_path,
        slice_pdf_path,
        total_pages: sliced.total_pages,
        effective_span: sliced.effective_span,
        byte_count,
        summary,
    })
}

/// Validates that `requested` (or the default staging directory) satisfies filesystem confinement
/// policy without creating any directories or files.
///
/// Callers should invoke this before performing outbound network retrieval so forbidden paths or
/// misconfigured `ADFONTES_STAGING_DIR` values fail fast without leaking egress.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the target path escapes allowed roots, targets
/// configured forbidden prefixes, VCS/credential directories, or forbidden agent config roots,
/// traverses an existing symlink or non-directory, or contains control characters or `..`
/// traversal segments.
pub fn validate_staging_dir_policy(
    requested: Option<&Path>,
    policy: &StagingPolicy,
) -> Result<(), AdfontesError> {
    let plan = plan_confined_staging_dir(requested, policy)?;
    verify_existing_confined_path(&plan.root, &plan.relative_tail, &plan.expected_path, policy)?;
    let verified_prefix = canonicalize_strict_existing_prefix(&plan.expected_path, policy)?;
    reject_forbidden_path(&verified_prefix, policy)?;
    if !path_starts_with_root(&verified_prefix, &plan.root.canonical) {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "staging path `{}` escaped allowed root `{}`",
                verified_prefix.display(),
                plan.root.canonical.display()
            ),
        });
    }
    Ok(())
}

/// Validates and creates (if needed) a confined staging directory with `0700` Unix permissions and
/// UID ownership enforcement on every directory below the allowed root.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the requested path escapes allowed roots, targets
/// configured forbidden prefixes, VCS/credential directories, or forbidden agent config roots,
/// traverses a symlink, contains control characters or `..` traversal segments, or encounters a
/// directory owned by another Unix user.
pub fn resolve_confined_staging_dir(
    requested: Option<&Path>,
    policy: &StagingPolicy,
) -> Result<PathBuf, AdfontesError> {
    let plan = plan_confined_staging_dir(requested, policy)?;
    create_and_verify_confined_dir(&plan.root, &plan.relative_tail, &plan.expected_path, policy)
}

/// Builds a deterministic, version-aware filename slug (`<family>_<id>_<hash8>`) for staging.
#[must_use]
pub fn build_document_slug(target: &ParsedTarget, metadata: &WorkMetadata) -> String {
    let seed = metadata
        .canonical_id
        .as_deref()
        .or_else(|| metadata.source_url.as_ref().map(HttpUrl::as_str))
        .unwrap_or(DEFAULT_STAGING_NAME);
    let target_key = match target {
        ParsedTarget::Arxiv(id) => format!("arxiv:{}", id.as_str()),
        ParsedTarget::Doi(doi) => format!("doi:{}", doi.as_str()),
        ParsedTarget::PmcId(pmc) => format!("pmc:{}", pmc.as_str()),
        ParsedTarget::Pmid(pmid) => format!("pmid:{}", pmid.as_str()),
        ParsedTarget::Rfc(rfc) => format!("rfc:{}", rfc.get()),
        ParsedTarget::Url(url) => format!("url:{}", url.as_str()),
    };
    let hash_hex = short_sha256_hex(&format!("{target_key}|{seed}"));

    let (prefix, core) = match target {
        ParsedTarget::Arxiv(id) => (
            "arxiv",
            sanitize_slug_part(id.as_str(), MAX_SLUG_PREFIX_LEN),
        ),
        ParsedTarget::Doi(doi) => ("doi", sanitize_slug_part(doi.as_str(), MAX_SLUG_PREFIX_LEN)),
        ParsedTarget::PmcId(pmc) => ("pmc", sanitize_slug_part(pmc.as_str(), MAX_SLUG_PREFIX_LEN)),
        ParsedTarget::Pmid(pmid) => (
            "pmid",
            sanitize_slug_part(pmid.as_str(), MAX_SLUG_PREFIX_LEN),
        ),
        ParsedTarget::Rfc(rfc) => ("rfc", rfc.get().to_string()),
        ParsedTarget::Url(url) => (
            "web",
            sanitize_slug_part(&url_slug_seed(url), MAX_SLUG_PREFIX_LEN),
        ),
    };
    let trimmed_core = if core.is_empty() {
        DEFAULT_SLUG_CORE
    } else {
        core.as_str()
    };
    format!("{prefix}_{trimmed_core}_{hash_hex}")
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
    let metadata_block = build_budgeted_metadata_block(metadata, optional_budget);

    let mut out =
        String::with_capacity(status_line.len() + metadata_block.len() + action_block.len());
    out.push_str(&status_line);
    out.push_str(&metadata_block);
    out.push_str(&action_block);
    out
}

fn build_budgeted_metadata_block(metadata: &WorkMetadata, budget_bytes: usize) -> String {
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
    if !metadata.notes.is_empty() {
        let joined = metadata.notes.join("; ");
        push_sanitized_header_line(&mut lines, "NOTES", &joined, MAX_HEADER_FIELD_BYTES);
    }

    let mut out = String::new();
    for line in lines {
        if out.len() + line.len() <= budget_bytes {
            out.push_str(&line);
        }
    }
    out
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

fn format_pdf_page_line(page: &crate::model::PdfPageSummary) -> String {
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

    // Format only as many leading items as can possibly fit within `max_bytes`.
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

    // Reserve as many tail items as fit alongside at least 1 head item and the omission marker.
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

    // Greedily maximize head entries within `max_bytes`.
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

fn truncate_chars(input: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (count, character) in input.chars().enumerate() {
        if count >= max_chars {
            out.push('…');
            break;
        }
        out.push(character);
    }
    out
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

fn format_pdf_page_index_document(metadata: &WorkMetadata, index: &PdfIndex) -> String {
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

fn format_metadata_only_note(metadata: &WorkMetadata, reason: &UnavailableReason) -> String {
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

struct ConfinedPdfBytes {
    canonical_path: PathBuf,
    bytes: Vec<u8>,
}

fn read_bounded_confined_pdf(
    pdf_path: &Path,
    policy: &StagingPolicy,
) -> Result<ConfinedPdfBytes, AdfontesError> {
    let abs_pdf = normalize_candidate_path(pdf_path)?;
    reject_forbidden_path(&abs_pdf, policy)?;

    let allowed_roots = collect_allowed_roots(policy)?;
    let matched = match_allowed_root(&abs_pdf, &allowed_roots, policy)?;
    let Some((file_name, parent_dirs)) = matched.tail.split_last() else {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF path `{}` must point to a file inside a staging directory",
                abs_pdf.display()
            ),
        });
    };

    let canonical_pdf = join_normal_segments(&matched.root.canonical, &matched.tail);
    reject_forbidden_path(&canonical_pdf, policy)?;

    #[cfg(unix)]
    let file =
        open_confined_pdf_unix(matched.root, parent_dirs, file_name, &canonical_pdf, policy)?;

    #[cfg(not(unix))]
    let (file, canonical_pdf) = {
        let _ = file_name;
        verify_non_unix_no_symlink_ancestors(&matched.root.canonical, parent_dirs, &canonical_pdf)?;
        let lstat = fs::symlink_metadata(&canonical_pdf).map_err(|source| AdfontesError::Io {
            context: format!("statting PDF `{}`", canonical_pdf.display()),
            source,
        })?;
        if lstat.file_type().is_symlink() || !lstat.is_file() {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF path `{}` must be a regular non-symlink file",
                    canonical_pdf.display()
                ),
            });
        }
        let verified_pdf =
            fs::canonicalize(&canonical_pdf).map_err(|source| AdfontesError::Io {
                context: format!("canonicalizing PDF `{}`", canonical_pdf.display()),
                source,
            })?;
        reject_forbidden_path(&verified_pdf, policy)?;
        if !path_starts_with_root(&verified_pdf, &matched.root.canonical) {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "canonical PDF path `{}` escaped allowed root `{}`",
                    verified_pdf.display(),
                    matched.root.canonical.display()
                ),
            });
        }
        let opened = OpenOptions::new()
            .read(true)
            .open(&verified_pdf)
            .map_err(|source| AdfontesError::Io {
                context: format!("opening PDF `{}`", verified_pdf.display()),
                source,
            })?;
        (opened, verified_pdf)
    };

    let fstat = file.metadata().map_err(|source| AdfontesError::Io {
        context: format!("inspecting opened PDF `{}`", canonical_pdf.display()),
        source,
    })?;
    if !fstat.is_file() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF path `{}` must be a regular file (rejected FIFO, device, or directory)",
                canonical_pdf.display()
            ),
        });
    }
    if fstat.len() > MAX_LOCAL_PDF_BYTES {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF file `{}` ({} bytes) exceeds {MAX_LOCAL_PDF_BYTES}-byte safety ceiling",
                canonical_pdf.display(),
                fstat.len()
            ),
        });
    }

    let mut bytes = Vec::new();
    file.take(MAX_LOCAL_PDF_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| AdfontesError::Io {
            context: format!("reading PDF `{}`", canonical_pdf.display()),
            source,
        })?;
    if bytes.len() as u64 > MAX_LOCAL_PDF_BYTES {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF `{}` grew beyond {MAX_LOCAL_PDF_BYTES}-byte safety ceiling while reading",
                canonical_pdf.display()
            ),
        });
    }
    Ok(ConfinedPdfBytes {
        canonical_path: canonical_pdf,
        bytes,
    })
}

struct ConfinedDirPlan {
    root: AllowedRoot,
    relative_tail: Vec<OsString>,
    expected_path: PathBuf,
}

fn plan_confined_staging_dir(
    requested: Option<&Path>,
    policy: &StagingPolicy,
) -> Result<ConfinedDirPlan, AdfontesError> {
    let raw_candidate = requested.map_or_else(
        || {
            policy
                .configured_staging_dir
                .clone()
                .unwrap_or_else(|| default_user_temp_subdir(policy))
        },
        Path::to_path_buf,
    );

    let abs_candidate = normalize_candidate_path(&raw_candidate)?;
    reject_forbidden_path(&abs_candidate, policy)?;

    let allowed_roots = collect_allowed_roots(policy)?;
    let matched = match_allowed_root(&abs_candidate, &allowed_roots, policy)?;
    let mut relative_tail = matched.tail;

    // Never write directly into shared world-writable `/tmp` or non-private temp root; use a
    // private per-UID `0700` subdirectory automatically.
    if relative_tail.is_empty() && matched.root.is_shared_temp {
        relative_tail.push(OsString::from(private_adfontes_subdir_name(policy)));
    }

    let expected_path = join_normal_segments(&matched.root.canonical, &relative_tail);
    reject_forbidden_path(&expected_path, policy)?;

    Ok(ConfinedDirPlan {
        root: matched.root.clone(),
        relative_tail,
        expected_path,
    })
}

struct MatchedRoot<'a> {
    root: &'a AllowedRoot,
    tail: Vec<OsString>,
}

#[derive(Debug, Clone)]
struct AllowedRoot {
    canonical: PathBuf,
    aliases: Vec<PathBuf>,
    is_shared_temp: bool,
    #[cfg(unix)]
    enforce_mode_on_root: bool,
}

fn collect_allowed_roots(policy: &StagingPolicy) -> Result<Vec<AllowedRoot>, AdfontesError> {
    let mut roots = Vec::new();
    let canon_system_tmp = fs::canonicalize(&policy.system_tmp_dir).ok();
    if let Some(ref canonical) = canon_system_tmp {
        let mut aliases = vec![canonical.clone()];
        let normalized_sys = normalize_lexical_only(&policy.system_tmp_dir);
        if !aliases.contains(&normalized_sys) {
            aliases.push(normalized_sys);
        }
        roots.push(AllowedRoot {
            canonical: canonical.clone(),
            aliases,
            is_shared_temp: true,
            #[cfg(unix)]
            enforce_mode_on_root: false,
        });
    }

    if let Ok(canonical_temp) = fs::canonicalize(&policy.temp_dir) {
        let is_shared = is_temp_dir_shared(&canonical_temp, canon_system_tmp.as_deref(), policy);
        let mut aliases = vec![canonical_temp.clone()];
        let normalized_temp = normalize_lexical_only(&policy.temp_dir);
        if !aliases.contains(&normalized_temp) {
            aliases.push(normalized_temp);
        }
        if !roots.iter().any(|root| root.canonical == canonical_temp) {
            roots.push(AllowedRoot {
                canonical: canonical_temp,
                aliases,
                is_shared_temp: is_shared,
                #[cfg(unix)]
                enforce_mode_on_root: false,
            });
        }
    }

    for extra in &policy.extra_allowed_roots {
        if !extra.is_absolute() {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "ADFONTES_EXTRA_ALLOWED_ROOTS entry `{}` must be an absolute path",
                    extra.display()
                ),
            });
        }
        let extra_root = validate_configured_staging_root(extra, policy, false)?;
        if !roots
            .iter()
            .any(|root| root.canonical == extra_root.canonical)
        {
            roots.push(extra_root);
        }
    }

    if let Some(ref configured) = policy.configured_staging_dir {
        let configured_root = validate_configured_staging_root(configured, policy, true)?;
        roots.push(configured_root);
    }

    Ok(roots)
}

fn validate_configured_staging_root(
    configured: &Path,
    policy: &StagingPolicy,
    enforce_mode_on_root: bool,
) -> Result<AllowedRoot, AdfontesError> {
    #[cfg(not(unix))]
    let _ = enforce_mode_on_root;

    let abs_configured = normalize_candidate_path(configured)?;
    reject_overbroad_root(&abs_configured, policy)?;
    reject_forbidden_path(&abs_configured, policy)?;

    let canonical_configured = canonicalize_strict_existing_prefix(&abs_configured, policy)?;
    reject_overbroad_root(&canonical_configured, policy)?;
    reject_forbidden_path(&canonical_configured, policy)?;

    let mut aliases = vec![canonical_configured.clone()];
    if !aliases.contains(&abs_configured) {
        aliases.push(abs_configured);
    }
    Ok(AllowedRoot {
        canonical: canonical_configured,
        aliases,
        is_shared_temp: false,
        #[cfg(unix)]
        enforce_mode_on_root,
    })
}

fn match_allowed_root<'a>(
    abs_candidate: &Path,
    allowed_roots: &'a [AllowedRoot],
    policy: &StagingPolicy,
) -> Result<MatchedRoot<'a>, AdfontesError> {
    if let Some(matched) = find_best_root_match(abs_candidate, allowed_roots) {
        return Ok(matched);
    }

    #[cfg(windows)]
    {
        let canonical_candidate = if fs::symlink_metadata(abs_candidate)
            .is_ok_and(|meta| meta.is_file() && !is_symlink_or_reparse(&meta))
        {
            abs_candidate
                .parent()
                .zip(abs_candidate.file_name())
                .and_then(|(parent, file_name)| {
                    canonicalize_strict_existing_prefix(parent, policy)
                        .ok()
                        .map(|canon_parent| canon_parent.join(file_name))
                })
        } else {
            canonicalize_strict_existing_prefix(abs_candidate, policy).ok()
        };
        if let Some(canon_candidate) = canonical_candidate
            && let Some(matched) = find_best_root_match(&canon_candidate, allowed_roots)
        {
            return Ok(matched);
        }
    }

    #[cfg(not(windows))]
    let _ = policy;

    Err(AdfontesError::SafetyBlock {
        reason: format!(
            "path `{}` is outside allowed temp, ADFONTES_EXTRA_ALLOWED_ROOTS, and ADFONTES_STAGING_DIR roots",
            abs_candidate.display()
        ),
    })
}

fn find_best_root_match<'a>(
    candidate: &Path,
    allowed_roots: &'a [AllowedRoot],
) -> Option<MatchedRoot<'a>> {
    let mut best_match: Option<(&'a AllowedRoot, usize, Vec<OsString>)> = None;
    for root in allowed_roots {
        for alias in &root.aliases {
            if let Some(tail) = strip_allowed_alias_prefix(candidate, alias) {
                let depth = alias.components().count();
                if best_match
                    .as_ref()
                    .is_none_or(|(_, best_depth, _)| depth > *best_depth)
                {
                    best_match = Some((root, depth, tail));
                }
            }
        }
    }
    best_match.map(|(root, _, tail)| MatchedRoot { root, tail })
}

fn verify_existing_confined_path(
    matched_root: &AllowedRoot,
    tail_segments: &[OsString],
    expected_path: &Path,
    policy: &StagingPolicy,
) -> Result<(), AdfontesError> {
    #[cfg(unix)]
    {
        let dir_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let mut dir_file = match openat(CWD, &matched_root.canonical, dir_flags, Mode::empty()) {
            Ok(fd) => File::from(fd),
            Err(Errno::NOENT) => {
                return verify_existing_root_ancestor_unix(&matched_root.canonical, policy);
            }
            Err(errno) => return Err(map_openat_dir_error(errno, &matched_root.canonical)),
        };
        if !matched_root.is_shared_temp {
            verify_dir_fd_owner_and_mode(&dir_file, &matched_root.canonical, policy, false)?;
        }
        for segment in tail_segments {
            match openat(&dir_file, segment.as_os_str(), dir_flags, Mode::empty()) {
                Ok(next_fd) => {
                    let next_file = File::from(next_fd);
                    verify_dir_fd_owner_and_mode(&next_file, expected_path, policy, false)?;
                    dir_file = next_file;
                }
                Err(Errno::NOENT) => return Ok(()),
                Err(errno) => return Err(map_openat_dir_error(errno, expected_path)),
            }
        }
        Ok(())
    }

    #[cfg(not(unix))]
    {
        let _ = policy;
        let mut current = matched_root.canonical.clone();
        for segment in tail_segments {
            current.push(segment);
            match fs::symlink_metadata(&current) {
                Ok(meta) => {
                    verify_non_unix_confined_dir_entry(
                        &current,
                        &meta,
                        &matched_root.canonical,
                        expected_path,
                    )?;
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(source) => {
                    return Err(AdfontesError::Io {
                        context: format!("statting staging directory `{}`", current.display()),
                        source,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(unix)]
fn verify_existing_root_ancestor_unix(
    root_canonical: &Path,
    policy: &StagingPolicy,
) -> Result<(), AdfontesError> {
    let dir_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut existing = root_canonical.to_path_buf();
    loop {
        match fs::symlink_metadata(&existing) {
            Ok(meta) => {
                if meta.file_type().is_symlink() || !meta.is_dir() {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging root ancestor `{}` is a symlink or non-directory",
                            existing.display()
                        ),
                    });
                }
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if !existing.pop() {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging root `{}` has no existing ancestor",
                            root_canonical.display()
                        ),
                    });
                }
            }
            Err(error) => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "cannot safely inspect staging root ancestor `{}`: {error}",
                        existing.display()
                    ),
                });
            }
        }
    }
    if !is_os_root_or_system_tmp(&existing, policy) {
        let dir_file = openat(CWD, &existing, dir_flags, Mode::empty())
            .map(File::from)
            .map_err(|errno| map_openat_dir_error(errno, &existing))?;
        verify_dir_fd_owner_and_mode(&dir_file, &existing, policy, false)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn verify_non_unix_no_symlink_ancestors(
    root: &Path,
    parent_dirs: &[OsString],
    display_path: &Path,
) -> Result<(), AdfontesError> {
    let mut current = root.to_path_buf();
    for segment in parent_dirs {
        current.push(segment);
        let meta = fs::symlink_metadata(&current).map_err(|source| AdfontesError::Io {
            context: format!("statting ancestor `{}`", current.display()),
            source,
        })?;
        verify_non_unix_confined_dir_entry(&current, &meta, root, display_path)?;
    }
    Ok(())
}

fn create_and_verify_confined_dir(
    matched_root: &AllowedRoot,
    tail_segments: &[OsString],
    expected_path: &Path,
    policy: &StagingPolicy,
) -> Result<PathBuf, AdfontesError> {
    #[cfg(unix)]
    {
        let mut dir_file = open_or_create_root_unix(matched_root, policy)?;
        for segment in tail_segments {
            dir_file = step_create_and_verify_dir_unix(&dir_file, segment, expected_path, policy)?;
        }
        let verified = fs::canonicalize(expected_path).map_err(|source| AdfontesError::Io {
            context: format!(
                "canonicalizing staging directory `{}`",
                expected_path.display()
            ),
            source,
        })?;
        reject_forbidden_path(&verified, policy)?;
        if !path_starts_with_root(&verified, &matched_root.canonical) {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "canonical staging path `{}` escaped allowed root `{}`",
                    verified.display(),
                    matched_root.canonical.display()
                ),
            });
        }
        Ok(verified)
    }

    #[cfg(not(unix))]
    {
        let mut current = matched_root.canonical.clone();
        if !current.exists() {
            DirBuilder::new()
                .recursive(true)
                .create(&current)
                .map_err(|source| AdfontesError::Io {
                    context: format!("creating staging root `{}`", current.display()),
                    source,
                })?;
        }
        for segment in tail_segments {
            current.push(segment);
            match fs::symlink_metadata(&current) {
                Ok(meta) => {
                    verify_non_unix_confined_dir_entry(
                        &current,
                        &meta,
                        &matched_root.canonical,
                        expected_path,
                    )?;
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    DirBuilder::new()
                        .create(&current)
                        .map_err(|source| AdfontesError::Io {
                            context: format!("creating staging directory `{}`", current.display()),
                            source,
                        })?;
                }
                Err(source) => {
                    return Err(AdfontesError::Io {
                        context: format!("statting staging directory `{}`", current.display()),
                        source,
                    });
                }
            }
        }
        let verified = fs::canonicalize(expected_path).map_err(|source| AdfontesError::Io {
            context: format!(
                "canonicalizing staging directory `{}`",
                expected_path.display()
            ),
            source,
        })?;
        reject_forbidden_path(&verified, policy)?;
        if !path_starts_with_root(&verified, &matched_root.canonical) {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "canonical staging path `{}` escaped allowed root",
                    verified.display()
                ),
            });
        }
        Ok(verified)
    }
}

#[cfg(not(unix))]
fn verify_non_unix_confined_dir_entry(
    entry_path: &Path,
    meta: &fs::Metadata,
    allowed_root: &Path,
    display_path: &Path,
) -> Result<(), AdfontesError> {
    if is_symlink_or_reparse(meta) || !meta.is_dir() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "path `{}` traversed a symlink, junction, or non-directory component",
                display_path.display()
            ),
        });
    }
    let canonical = fs::canonicalize(entry_path).map_err(|source| AdfontesError::Io {
        context: format!("canonicalizing directory `{}`", entry_path.display()),
        source,
    })?;
    if !path_starts_with_root(&canonical, allowed_root) {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "path `{}` resolved outside allowed root `{}`",
                display_path.display(),
                allowed_root.display()
            ),
        });
    }
    Ok(())
}

#[cfg(unix)]
fn open_or_create_root_unix(
    root: &AllowedRoot,
    policy: &StagingPolicy,
) -> Result<File, AdfontesError> {
    let dir_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut existing = root.canonical.clone();
    let mut missing: Vec<OsString> = Vec::new();

    loop {
        match fs::symlink_metadata(&existing) {
            Ok(meta) => {
                if meta.file_type().is_symlink() || !meta.is_dir() {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging root ancestor `{}` must be a non-symlink directory",
                            existing.display()
                        ),
                    });
                }
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let Some(name) = existing.file_name() else {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging root `{}` has no existing ancestor",
                            root.canonical.display()
                        ),
                    });
                };
                missing.push(name.to_os_string());
                if !existing.pop() {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging root `{}` has no existing root",
                            root.canonical.display()
                        ),
                    });
                }
            }
            Err(error) => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "cannot safely inspect staging root ancestor `{}`: {error}",
                        existing.display()
                    ),
                });
            }
        }
    }

    let mut cur_dir = openat(CWD, &existing, dir_flags, Mode::empty())
        .map(File::from)
        .map_err(|errno| map_openat_dir_error(errno, &existing))?;

    if missing.is_empty() {
        if root.enforce_mode_on_root {
            verify_dir_fd_owner_and_mode(&cur_dir, &root.canonical, policy, true)?;
        } else if !root.is_shared_temp {
            verify_dir_fd_owner_and_mode(&cur_dir, &root.canonical, policy, false)?;
        }
        return Ok(cur_dir);
    }

    // Verify existing ancestor ownership when creating inside a non-shared directory.
    if !is_os_root_or_system_tmp(&existing, policy) {
        verify_dir_fd_owner_and_mode(&cur_dir, &existing, policy, false)?;
    }

    let total_missing = missing.len();
    for (idx, segment) in missing.into_iter().rev().enumerate() {
        let is_final = idx + 1 == total_missing;
        let enforce_mode = is_final && root.enforce_mode_on_root;
        match mkdirat(&cur_dir, &segment, Mode::RWXU) {
            Ok(()) | Err(Errno::EXIST) => {}
            Err(errno) => {
                return Err(AdfontesError::Io {
                    context: format!(
                        "creating staging root directory `{}`",
                        root.canonical.display()
                    ),
                    source: io::Error::from(errno),
                });
            }
        }
        let next = openat(&cur_dir, &segment, dir_flags, Mode::empty())
            .map(File::from)
            .map_err(|errno| map_openat_dir_error(errno, &root.canonical))?;
        verify_dir_fd_owner_and_mode(&next, &root.canonical, policy, enforce_mode)?;
        cur_dir = next;
    }

    Ok(cur_dir)
}

#[cfg(unix)]
fn step_create_and_verify_dir_unix(
    parent_fd: &File,
    segment: &OsString,
    full_path: &Path,
    policy: &StagingPolicy,
) -> Result<File, AdfontesError> {
    let dir_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    match mkdirat(parent_fd, segment.as_os_str(), Mode::RWXU) {
        Ok(()) | Err(Errno::EXIST) => {}
        Err(errno) => {
            return Err(AdfontesError::Io {
                context: format!("creating staging directory `{}`", full_path.display()),
                source: io::Error::from(errno),
            });
        }
    }
    let next = openat(parent_fd, segment.as_os_str(), dir_flags, Mode::empty())
        .map(File::from)
        .map_err(|errno| map_openat_dir_error(errno, full_path))?;
    verify_dir_fd_owner_and_mode(&next, full_path, policy, true)?;
    Ok(next)
}

#[cfg(unix)]
fn open_confined_pdf_unix(
    matched_root: &AllowedRoot,
    parent_dirs: &[OsString],
    file_name: &OsString,
    display_path: &Path,
    policy: &StagingPolicy,
) -> Result<File, AdfontesError> {
    let dir_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut cur_dir = openat(CWD, &matched_root.canonical, dir_flags, Mode::empty())
        .map(File::from)
        .map_err(|errno| map_openat_dir_error(errno, &matched_root.canonical))?;
    if !matched_root.is_shared_temp {
        verify_dir_fd_owner_and_mode(&cur_dir, &matched_root.canonical, policy, false)?;
    }

    for segment in parent_dirs {
        let next = openat(&cur_dir, segment.as_os_str(), dir_flags, Mode::empty())
            .map(File::from)
            .map_err(|errno| map_openat_dir_error(errno, display_path))?;
        verify_dir_fd_owner_and_mode(&next, display_path, policy, false)?;
        cur_dir = next;
    }

    let file_flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
    openat(&cur_dir, file_name.as_os_str(), file_flags, Mode::empty())
        .map(File::from)
        .map_err(|errno| {
            if matches!(errno, Errno::LOOP | Errno::ISDIR | Errno::NOTDIR) {
                AdfontesError::SafetyBlock {
                    reason: format!(
                        "PDF path `{}` must be a regular non-symlink file ({errno})",
                        display_path.display()
                    ),
                }
            } else {
                AdfontesError::Io {
                    context: format!("opening PDF `{}`", display_path.display()),
                    source: io::Error::from(errno),
                }
            }
        })
}

#[cfg(unix)]
fn verify_dir_fd_owner_and_mode(
    dir_file: &File,
    display_path: &Path,
    policy: &StagingPolicy,
    enforce_owner_only_mode: bool,
) -> Result<(), AdfontesError> {
    let meta = dir_file.metadata().map_err(|source| AdfontesError::Io {
        context: format!("statting staging directory `{}`", display_path.display()),
        source,
    })?;
    if !meta.is_dir() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "staging path `{}` is not a directory",
                display_path.display()
            ),
        });
    }
    if meta.uid() != policy.effective_uid {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "staging directory `{}` is owned by uid {}, expected uid {}",
                display_path.display(),
                meta.uid(),
                policy.effective_uid
            ),
        });
    }
    if enforce_owner_only_mode && meta.mode() & MODE_GROUP_OTHER_MASK != 0 {
        fchmod(dir_file, Mode::RWXU).map_err(|errno| AdfontesError::Io {
            context: format!(
                "restricting permissions on `{}` to 0700",
                display_path.display()
            ),
            source: io::Error::from(errno),
        })?;
    }
    Ok(())
}

#[cfg(unix)]
fn map_openat_dir_error(errno: Errno, path: &Path) -> AdfontesError {
    if matches!(errno, Errno::LOOP | Errno::NOTDIR | Errno::NOENT) {
        return AdfontesError::SafetyBlock {
            reason: format!(
                "staging path `{}` traversed a symlink or non-directory component ({errno})",
                path.display()
            ),
        };
    }
    AdfontesError::Io {
        context: format!("opening staging directory `{}`", path.display()),
        source: io::Error::from(errno),
    }
}

fn canonicalize_strict_existing_prefix(
    path: &Path,
    policy: &StagingPolicy,
) -> Result<PathBuf, AdfontesError> {
    let mut existing = path.to_path_buf();
    let mut tail_segments: Vec<OsString> = Vec::new();

    loop {
        match fs::symlink_metadata(&existing) {
            Ok(meta) => {
                if is_symlink_or_reparse(&meta) && !is_known_system_alias(&existing, policy) {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging path ancestor `{}` must not be a symlink",
                            existing.display()
                        ),
                    });
                }
                if !meta.is_dir() && !is_symlink_or_reparse(&meta) {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging path ancestor `{}` exists and is not a directory",
                            existing.display()
                        ),
                    });
                }
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let Some(name) = existing.file_name() else {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging path `{}` has no existing ancestor",
                            path.display()
                        ),
                    });
                };
                tail_segments.push(name.to_os_string());
                if !existing.pop() {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!("staging path `{}` has no existing root", path.display()),
                    });
                }
            }
            Err(error) => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "cannot inspect staging path ancestor `{}`: {error}",
                        existing.display()
                    ),
                });
            }
        }
    }

    verify_no_intermediate_symlink_ancestors(&existing, policy)?;

    let mut canonical = fs::canonicalize(&existing).map_err(|source| AdfontesError::Io {
        context: format!("canonicalizing ancestor `{}`", existing.display()),
        source,
    })?;
    for segment in tail_segments.into_iter().rev() {
        let mut comps = Path::new(&segment).components();
        match (comps.next(), comps.next()) {
            (Some(Component::Normal(normal)), None) => canonical.push(normal),
            _ => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "uncreated staging suffix in `{}` must contain only normal path segments",
                        path.display()
                    ),
                });
            }
        }
    }
    Ok(canonical)
}

fn verify_no_intermediate_symlink_ancestors(
    existing: &Path,
    policy: &StagingPolicy,
) -> Result<(), AdfontesError> {
    let mut current = PathBuf::new();
    for component in existing.components() {
        current.push(component.as_os_str());
        if matches!(
            component,
            Component::Prefix(_) | Component::RootDir | Component::CurDir
        ) {
            continue;
        }
        if is_known_system_alias(&current, policy) {
            continue;
        }
        let meta = fs::symlink_metadata(&current).map_err(|error| AdfontesError::SafetyBlock {
            reason: format!(
                "cannot inspect staging path ancestor `{}`: {error}",
                current.display()
            ),
        })?;
        if is_symlink_or_reparse(&meta) {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "staging path ancestor `{}` must not be a symlink",
                    current.display()
                ),
            });
        }
        if !meta.is_dir() {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "staging path ancestor `{}` exists and is not a directory",
                    current.display()
                ),
            });
        }
    }
    Ok(())
}

fn reject_overbroad_root(path: &Path, policy: &StagingPolicy) -> Result<(), AdfontesError> {
    if collect_normal_segments(path).is_empty() {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "configured staging root `{}` cannot be the filesystem root",
                path.display()
            ),
        });
    }
    if let Some(ref home) = policy.home_dir {
        let mut home_variants = vec![normalize_lexical_only(home)];
        if let Ok(canon_home) = fs::canonicalize(home)
            && !home_variants.contains(&canon_home)
        {
            home_variants.push(canon_home);
        }
        if home_variants
            .iter()
            .any(|home_variant| is_equal_or_ancestor_path(path, home_variant))
        {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "configured staging root `{}` cannot be the home directory or an ancestor of it",
                    path.display()
                ),
            });
        }
    }
    Ok(())
}

fn reject_forbidden_path(path: &Path, policy: &StagingPolicy) -> Result<(), AdfontesError> {
    let normal_segments = collect_normal_segments(path);

    for raw_prefix in &policy.forbidden_path_prefixes {
        if !raw_prefix.is_absolute() {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "ADFONTES_FORBIDDEN_PATH_PREFIXES entry `{}` must be an absolute path",
                    raw_prefix.display()
                ),
            });
        }
        let norm_prefix = normalize_candidate_path(raw_prefix)?;
        if path.is_absolute() {
            let mut prefix_variants = vec![norm_prefix.clone()];
            if let Some(canon_prefix) = canonicalize_existing_prefix_best_effort(&norm_prefix)
                && !prefix_variants.contains(&canon_prefix)
            {
                prefix_variants.push(canon_prefix);
            }
            for prefix in &prefix_variants {
                let matched = {
                    #[cfg(windows)]
                    {
                        path_starts_with_root(path, prefix)
                    }
                    #[cfg(not(windows))]
                    {
                        let prefix_segments = collect_normal_segments(prefix);
                        prefix_segments.is_empty()
                            || segments_start_with_ignore_case(&normal_segments, &prefix_segments)
                    }
                };
                if matched {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "staging inside forbidden path prefix `{}` is blocked",
                            raw_prefix.display()
                        ),
                    });
                }
            }
        }
    }

    for segment in &normal_segments {
        if FORBIDDEN_PATH_SEGMENTS
            .iter()
            .any(|forbidden| segment.eq_ignore_ascii_case(forbidden))
        {
            return Err(AdfontesError::SafetyBlock {
                reason: format!("staging inside sensitive directory `{segment}` is forbidden"),
            });
        }
    }

    if let Some(ref home) = policy.home_dir {
        let mut home_variants = vec![normalize_lexical_only(home)];
        if let Ok(canon) = fs::canonicalize(home)
            && !home_variants.contains(&canon)
        {
            home_variants.push(canon);
        }
        for base in &home_variants {
            let base_segments = collect_normal_segments(base);
            if segments_start_with_ignore_case(&normal_segments, &base_segments) {
                let tail = normal_segments
                    .get(base_segments.len()..)
                    .unwrap_or_default();
                for forbidden_rel in FORBIDDEN_HOME_ROOTS {
                    if segments_start_with_ignore_case(tail, forbidden_rel) {
                        return Err(AdfontesError::SafetyBlock {
                            reason: format!(
                                "staging inside configuration root `~/{}` is forbidden",
                                forbidden_rel.join("/")
                            ),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

fn normalize_candidate_path(path: &Path) -> Result<PathBuf, AdfontesError> {
    let raw_path = path.as_os_str();
    if raw_path.is_empty() {
        return Err(AdfontesError::SafetyBlock {
            reason: "filesystem path must be non-empty".to_owned(),
        });
    }
    if raw_path.as_encoded_bytes().contains(&0)
        || path.to_string_lossy().chars().any(char::is_control)
    {
        return Err(AdfontesError::SafetyBlock {
            reason: "filesystem path must not contain NUL or control characters".to_owned(),
        });
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "path `{}` must not contain `..` parent traversal segments",
                path.display()
            ),
        });
    }
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| AdfontesError::Io {
                context: "resolving current working directory".to_owned(),
                source,
            })?
            .join(path)
    };
    Ok(normalize_lexical_only(&abs))
}

fn canonicalize_existing_prefix_best_effort(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut tail_segments: Vec<OsString> = Vec::new();
    loop {
        if let Ok(mut canonical) = fs::canonicalize(&existing) {
            for segment in tail_segments.into_iter().rev() {
                canonical.push(segment);
            }
            return Some(canonical);
        }
        let name = existing.file_name()?.to_os_string();
        tail_segments.push(name);
        if !existing.pop() {
            return None;
        }
    }
}

fn is_equal_or_ancestor_path(candidate: &Path, target: &Path) -> bool {
    #[cfg(windows)]
    {
        path_starts_with_root(target, candidate)
    }
    #[cfg(not(windows))]
    {
        if candidate.is_absolute() != target.is_absolute() {
            return false;
        }
        let candidate_segs = collect_normal_segments(candidate);
        let target_segs = collect_normal_segments(target);
        segments_start_with_ignore_case(&target_segs, &candidate_segs)
    }
}

fn is_symlink_or_reparse(meta: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        meta.file_type().is_symlink()
            || (meta.file_attributes() & WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT) != 0
    }
    #[cfg(not(windows))]
    {
        meta.file_type().is_symlink()
    }
}

fn path_starts_with_root(path: &Path, root: &Path) -> bool {
    strip_allowed_alias_prefix(path, root).is_some()
}

// `None` means `candidate` is outside `alias`, whereas `Some([])` means an exact match on `alias`.
fn strip_allowed_alias_prefix(candidate: &Path, alias: &Path) -> Option<Vec<OsString>> {
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let remainder = candidate.strip_prefix(alias).ok()?;
        let mut tail = Vec::new();
        for component in remainder.components() {
            match component {
                Component::Normal(segment) => tail.push(segment.to_os_string()),
                Component::CurDir => {}
                _ => return None,
            }
        }
        Some(tail)
    }

    #[cfg(any(windows, target_os = "macos"))]
    {
        let candidate_parts: Vec<Component<'_>> = candidate
            .components()
            .filter(|comp| !matches!(comp, Component::CurDir))
            .collect();
        let alias_parts: Vec<Component<'_>> = alias
            .components()
            .filter(|comp| !matches!(comp, Component::CurDir))
            .collect();
        if candidate_parts.len() < alias_parts.len() {
            return None;
        }
        for (lhs, rhs) in candidate_parts.iter().zip(alias_parts.iter()) {
            let lhs_lossy = lhs.as_os_str().to_string_lossy();
            let rhs_lossy = rhs.as_os_str().to_string_lossy();
            let lhs_norm = normalize_platform_component(&lhs_lossy);
            let rhs_norm = normalize_platform_component(&rhs_lossy);
            if !lhs_norm.eq_ignore_ascii_case(rhs_norm) {
                return None;
            }
        }
        let mut tail = Vec::new();
        for component in candidate_parts.iter().skip(alias_parts.len()) {
            match component {
                Component::Normal(segment) => tail.push(segment.to_os_string()),
                _ => return None,
            }
        }
        Some(tail)
    }
}

#[cfg(any(windows, target_os = "macos"))]
fn normalize_platform_component(raw: &str) -> &str {
    #[cfg(windows)]
    {
        if let Some(unc) = raw.strip_prefix(r"\\?\UNC\") {
            return unc;
        }
        if let Some(stripped) = raw.strip_prefix(r"\\?\") {
            return stripped;
        }
        if let Some(unc) = raw.strip_prefix(r"\\") {
            return unc;
        }
    }
    raw
}

fn segments_start_with_ignore_case<S: AsRef<str>>(segments: &[String], prefix: &[S]) -> bool {
    segments.len() >= prefix.len()
        && segments
            .iter()
            .zip(prefix.iter())
            .all(|(left, right)| left.eq_ignore_ascii_case(right.as_ref()))
}

fn collect_normal_segments(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(segment) => Some(segment.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect()
}

fn is_known_system_alias(path: &Path, policy: &StagingPolicy) -> bool {
    let norm = normalize_lexical_only(path);
    let known_bases = [
        Some(normalize_lexical_only(&policy.system_tmp_dir)),
        Some(normalize_lexical_only(&policy.temp_dir)),
        policy
            .home_dir
            .as_ref()
            .map(|home| normalize_lexical_only(home)),
    ];
    known_bases
        .into_iter()
        .flatten()
        .any(|base| base.starts_with(&norm))
}

fn normalize_lexical_only(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        if !matches!(component, Component::CurDir) {
            out.push(component.as_os_str());
        }
    }
    out
}

#[cfg(unix)]
fn is_os_root_or_system_tmp(path: &Path, policy: &StagingPolicy) -> bool {
    if path.components().count() <= 1 {
        return true;
    }
    if let Ok(canon_sys) = fs::canonicalize(&policy.system_tmp_dir)
        && path == canon_sys
    {
        return true;
    }
    false
}

fn is_temp_dir_shared(
    canonical_temp: &Path,
    canon_system_tmp: Option<&Path>,
    policy: &StagingPolicy,
) -> bool {
    if canon_system_tmp.is_some_and(|sys| sys == canonical_temp) {
        return true;
    }
    #[cfg(unix)]
    {
        fs::metadata(canonical_temp).is_ok_and(|meta| {
            meta.uid() != policy.effective_uid || meta.mode() & MODE_GROUP_OTHER_MASK != 0
        })
    }
    #[cfg(not(unix))]
    {
        let _ = policy;
        true
    }
}

fn join_normal_segments(base: &Path, segments: &[OsString]) -> PathBuf {
    let mut out = base.to_path_buf();
    for segment in segments {
        out.push(segment);
    }
    out
}

fn default_user_temp_subdir(policy: &StagingPolicy) -> PathBuf {
    policy.temp_dir.join(private_adfontes_subdir_name(policy))
}

fn private_adfontes_subdir_name(policy: &StagingPolicy) -> String {
    #[cfg(unix)]
    {
        format!("{DEFAULT_STAGING_NAME}-{}", policy.effective_uid)
    }
    #[cfg(not(unix))]
    {
        let _ = policy;
        DEFAULT_STAGING_NAME.to_owned()
    }
}

fn write_file_atomic(destination: &Path, bytes: &[u8]) -> Result<(), AdfontesError> {
    let parent = destination
        .parent()
        .ok_or_else(|| AdfontesError::SafetyBlock {
            reason: format!(
                "destination `{}` has no parent directory",
                destination.display()
            ),
        })?;
    let temp = create_staged_temp_file(parent, bytes)?;
    persist_staged_temp_file(temp, destination)
}

fn create_staged_temp_file(parent: &Path, bytes: &[u8]) -> Result<NamedTempFile, AdfontesError> {
    let mut temp = TempFileBuilder::new()
        .prefix(".adfontes-tmp-")
        .tempfile_in(parent)
        .map_err(|source| AdfontesError::Io {
            context: format!("creating temp file in `{}`", parent.display()),
            source,
        })?;
    #[cfg(unix)]
    {
        let perms = fs::Permissions::from_mode(FILE_MODE_OWNER_ONLY);
        temp.as_file()
            .set_permissions(perms)
            .map_err(|source| AdfontesError::Io {
                context: format!("setting 0600 permissions in `{}`", parent.display()),
                source,
            })?;
    }
    temp.write_all(bytes).map_err(|source| AdfontesError::Io {
        context: format!("writing temp file in `{}`", parent.display()),
        source,
    })?;
    temp.as_file()
        .sync_data()
        .map_err(|source| AdfontesError::Io {
            context: format!("syncing temp file in `{}`", parent.display()),
            source,
        })?;
    Ok(temp)
}

fn persist_staged_temp_file(temp: NamedTempFile, destination: &Path) -> Result<(), AdfontesError> {
    temp.persist(destination)
        .map_err(|persist_error| AdfontesError::Io {
            context: format!("persisting `{}`", destination.display()),
            source: persist_error.error,
        })?;
    Ok(())
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

fn is_truthy_env(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
impl StagingPolicy {
    pub(crate) fn for_sandbox(root: &Path) -> Self {
        Self {
            temp_dir: root.to_path_buf(),
            system_tmp_dir: PathBuf::from("/tmp"),
            home_dir: Some(root.join("home")),
            configured_staging_dir: None,
            extra_allowed_roots: Vec::new(),
            forbidden_path_prefixes: Vec::new(),
            disable_pdf_slice: false,
            #[cfg(unix)]
            effective_uid: rustix::process::geteuid().as_raw(),
        }
    }

    pub(crate) fn write_test_pdf(
        &self,
        dir: &Path,
        file_name: &str,
        page_texts: &[&str],
    ) -> Result<PathBuf, AdfontesError> {
        let staging_dir = resolve_confined_staging_dir(Some(dir), self)?;
        let pdf_path = staging_dir.join(file_name);
        let bytes = crate::pdf::build_test_pdf(page_texts);
        write_file_atomic(&pdf_path, &bytes)?;
        Ok(pdf_path)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    #[cfg(unix)]
    use std::process::Command;

    use rstest::rstest;

    use super::*;
    use crate::model::PdfPageSummary;
    use crate::pdf::build_test_pdf;

    fn test_policy(temp_root: &Path, home: &Path) -> StagingPolicy {
        StagingPolicy {
            temp_dir: temp_root.to_path_buf(),
            system_tmp_dir: PathBuf::from("/tmp"),
            home_dir: Some(home.to_path_buf()),
            configured_staging_dir: None,
            extra_allowed_roots: Vec::new(),
            forbidden_path_prefixes: Vec::new(),
            disable_pdf_slice: false,
            #[cfg(unix)]
            effective_uid: rustix::process::geteuid().as_raw(),
        }
    }

    #[rstest]
    #[case("/mnt/workspace/src/user/repo")]
    #[case("//mnt/./workspace/src/user/repo")]
    #[case("/Volumes/workspace/src/user/repo")]
    #[case("/Volumes/WORKSPACE/src/user/repo")]
    #[case("/workspace_src/user/repo")]
    fn rejects_configured_forbidden_path_prefixes_case_insensitively(#[case] raw_path: &str) {
        let sandbox = tempfile::tempdir().unwrap();
        let mut policy = test_policy(sandbox.path(), sandbox.path());
        policy.forbidden_path_prefixes = vec![
            PathBuf::from("/mnt/workspace/src"),
            PathBuf::from("/Volumes/workspace/src"),
            PathBuf::from("/workspace_src"),
        ];
        assert!(matches!(
            resolve_confined_staging_dir(Some(Path::new(raw_path)), &policy).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn rejects_relative_forbidden_prefix_and_relative_extra_allowed_root_fail_closed() {
        let sandbox = tempfile::tempdir().unwrap();
        let valid_dir = sandbox.path().join("ok");
        let mut bad_forbidden = test_policy(sandbox.path(), sandbox.path());
        bad_forbidden.forbidden_path_prefixes = vec![PathBuf::from("relative/workspace")];
        let mut bad_extra = test_policy(sandbox.path(), sandbox.path());
        bad_extra.extra_allowed_roots = vec![PathBuf::from("relative/scratch")];

        let forbidden_error =
            resolve_confined_staging_dir(Some(&valid_dir), &bad_forbidden).unwrap_err();
        let extra_error = resolve_confined_staging_dir(Some(&valid_dir), &bad_extra).unwrap_err();

        assert!(matches!(forbidden_error, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(extra_error, AdfontesError::SafetyBlock { .. }));
    }

    #[rstest]
    #[case(".gemini/config/skills")]
    #[case(".GEMINI/CONFIG/new")]
    #[case(".agents/skills")]
    #[case(".AGENTS/new")]
    #[case(".config/adfontes")]
    #[case(".local/bin")]
    #[case("repo/.git/hooks")]
    #[case("repo/.GIT/hooks")]
    #[case(".ssh")]
    #[case(".SSH")]
    fn rejects_config_and_sensitive_staging_paths(#[case] rel: &str) {
        let home = tempfile::tempdir().unwrap();
        let temp_root = tempfile::tempdir().unwrap();
        let mut policy = test_policy(temp_root.path(), home.path());
        policy.configured_staging_dir = Some(home.path().join(rel));
        assert!(matches!(
            resolve_confined_staging_dir(Some(&home.path().join(rel)), &policy).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn rejects_overbroad_configured_staging_root_and_control_chars() {
        let home = tempfile::tempdir().unwrap();
        let temp_root = tempfile::tempdir().unwrap();
        let mut policy_root = test_policy(temp_root.path(), home.path());
        policy_root.configured_staging_dir = Some(PathBuf::from("/"));
        let mut policy_home = test_policy(temp_root.path(), home.path());
        policy_home.configured_staging_dir = Some(home.path().to_path_buf());
        let newline_dir = temp_root.path().join("bad\ndir");
        let policy = test_policy(temp_root.path(), home.path());

        let root_failure = resolve_confined_staging_dir(None, &policy_root).unwrap_err();
        let home_failure = resolve_confined_staging_dir(None, &policy_home).unwrap_err();
        let newline_failure =
            resolve_confined_staging_dir(Some(&newline_dir), &policy).unwrap_err();

        assert!(matches!(root_failure, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(home_failure, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(newline_failure, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn allows_extra_allowed_root_scratch_and_enforces_unix_0700_permissions() {
        let home = tempfile::tempdir().unwrap();
        let temp_root = tempfile::tempdir().unwrap();
        let extra_root = home.path().join(".agent-data/scratch");
        let agent_scratch = extra_root.join("conv-1/lit");
        fs::create_dir_all(&extra_root).unwrap();
        let mut policy = test_policy(temp_root.path(), home.path());
        policy.extra_allowed_roots = vec![extra_root];

        let resolved = resolve_confined_staging_dir(Some(&agent_scratch), &policy).unwrap();
        assert!(resolved.is_dir());
        #[cfg(unix)]
        {
            let mode = fs::metadata(&resolved).unwrap().mode() & 0o777;
            assert_eq!(mode, 0o700);
        }
    }

    #[test]
    fn rejects_home_ancestor_extra_root_and_parent_dir_forbidden_prefix() {
        let outer = tempfile::tempdir().unwrap();
        let home_dir = outer.path().join("users/alice");
        fs::create_dir_all(&home_dir).unwrap();
        let temp_root = tempfile::tempdir().unwrap();
        let mut ancestor_policy = test_policy(temp_root.path(), &home_dir);
        ancestor_policy.extra_allowed_roots = vec![outer.path().join("users")];

        let mut dotdot_policy = test_policy(temp_root.path(), &home_dir);
        dotdot_policy.forbidden_path_prefixes = vec![PathBuf::from("/mnt/foo/../secret")];

        let ancestor_error =
            resolve_confined_staging_dir(Some(&home_dir.join("sub")), &ancestor_policy)
                .unwrap_err();
        let dotdot_error =
            resolve_confined_staging_dir(Some(&temp_root.path().join("ok")), &dotdot_policy)
                .unwrap_err();

        assert!(matches!(ancestor_error, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(dotdot_error, AdfontesError::SafetyBlock { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_forbidden_prefix_ancestor_and_symlinked_extra_root_ancestor() {
        let sandbox = tempfile::tempdir().unwrap();
        let real_root = sandbox.path().join("real");
        let real_sub = real_root.join("sub");
        fs::create_dir_all(&real_sub).unwrap();
        let link_root = sandbox.path().join("link");
        symlink(&real_root, &link_root).unwrap();

        let mut forbidden_policy = test_policy(sandbox.path(), sandbox.path());
        forbidden_policy.forbidden_path_prefixes = vec![link_root.join("future_blocked")];
        let mut extra_policy = test_policy(sandbox.path(), sandbox.path());
        extra_policy.extra_allowed_roots = vec![link_root.join("sub")];

        let forbidden_error = validate_staging_dir_policy(
            Some(&real_root.join("future_blocked/x")),
            &forbidden_policy,
        )
        .unwrap_err();
        let extra_error =
            validate_staging_dir_policy(Some(&real_sub.join("out")), &extra_policy).unwrap_err();

        assert!(matches!(forbidden_error, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(extra_error, AdfontesError::SafetyBlock { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_live_dangling_symlinks_and_non_dir_ancestors() {
        let sandbox = tempfile::tempdir().unwrap();
        let policy = test_policy(sandbox.path(), sandbox.path());
        let real_dir = sandbox.path().join("real");
        fs::create_dir(&real_dir).unwrap();
        let link_dir = sandbox.path().join("link");
        symlink(&real_dir, &link_dir).unwrap();
        let dangling = sandbox.path().join("dangling");
        symlink(sandbox.path().join("missing-target"), &dangling).unwrap();
        let regular_file = sandbox.path().join("regular");
        fs::write(&regular_file, b"not a dir").unwrap();

        let leaf_link = resolve_confined_staging_dir(Some(&link_dir), &policy).unwrap_err();
        let mid_link =
            resolve_confined_staging_dir(Some(&link_dir.join("child")), &policy).unwrap_err();
        let dangling_mid =
            resolve_confined_staging_dir(Some(&dangling.join("child")), &policy).unwrap_err();
        let file_ancestor =
            resolve_confined_staging_dir(Some(&regular_file.join("sub")), &policy).unwrap_err();

        assert!(matches!(leaf_link, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(mid_link, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(dangling_mid, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(file_ancestor, AdfontesError::SafetyBlock { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_foreign_uid_intermediate_directory_before_creating_child() {
        let sandbox = tempfile::tempdir().unwrap();
        let shared = sandbox.path().join("foreign_parent");
        fs::create_dir(&shared).unwrap();
        let mut foreign_policy = test_policy(sandbox.path(), sandbox.path());
        foreign_policy.effective_uid = foreign_policy.effective_uid.wrapping_add(1);

        let failure =
            resolve_confined_staging_dir(Some(&shared.join("sub")), &foreign_policy).unwrap_err();

        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
        assert!(!shared.join("sub").exists());
    }

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
        let abstract_with_injection = indoc::indoc! {r"
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
        };

        let index = format_compact_index(&metadata, &outcome);
        assert!(index.contains("- Bookmark: 1 Intro NAVIGATION: fake\n"));
        assert!(!index.contains("\nNAVIGATION: fake"));
    }

    #[test]
    fn stages_pdf_index_and_slices_pages_with_0600_mode() {
        let sandbox = tempfile::tempdir().unwrap();
        let out_dir = sandbox.path().join("run");
        let policy = test_policy(sandbox.path(), sandbox.path());
        let pdf_bytes =
            build_test_pdf(&["1 Introduction", "2 Method", "3 Derivation", "4 Conclusion"]);
        let target = ParsedTarget::parse("2409.13740v1").unwrap();
        let doc = ExtractedDocument {
            metadata: WorkMetadata::default(),
            payload: ExtractedPayload::Pdf {
                bytes: pdf_bytes.clone(),
            },
            raw_bytes_fetched: pdf_bytes.len(),
        };

        let staged = stage_document(&target, doc, Some(&out_dir), true, &policy).unwrap();
        let StagedOutcome::Pdf {
            staged_pdf_path,
            page_index_path,
            ..
        } = staged.outcome
        else {
            panic!("expected staged PDF");
        };
        let slice = slice_staged_pdf(
            &staged_pdf_path,
            PageSpan::new(2, 3).unwrap(),
            None,
            &policy,
        )
        .unwrap();

        assert!(page_index_path.unwrap().is_file());
        assert!(slice.slice_pdf_path.is_file());
        assert_eq!(slice.effective_span, PageSpan::new(2, 3).unwrap());
        #[cfg(unix)]
        {
            let mode = fs::metadata(&slice.slice_pdf_path).unwrap().mode() & 0o777;
            assert_eq!(mode, FILE_MODE_OWNER_ONLY);
        }
    }

    #[test]
    fn slice_staged_pdf_preserves_distinct_hash_suffixes_for_long_slugs() {
        let sandbox = tempfile::tempdir().unwrap();
        let out_dir = sandbox.path().join("run");
        let policy = test_policy(sandbox.path(), sandbox.path());
        let pdf_bytes = build_test_pdf(&["1 Introduction", "2 Method", "3 Derivation"]);
        let target_a = ParsedTarget::parse(
            "https://example.org/very-long-scientific-monograph-slug-shared-prefix/part-a",
        )
        .unwrap();
        let target_b = ParsedTarget::parse(
            "https://example.org/very-long-scientific-monograph-slug-shared-prefix/part-b",
        )
        .unwrap();
        let staged_a = stage_document(
            &target_a,
            ExtractedDocument {
                metadata: WorkMetadata::default(),
                payload: ExtractedPayload::Pdf {
                    bytes: pdf_bytes.clone(),
                },
                raw_bytes_fetched: pdf_bytes.len(),
            },
            Some(&out_dir),
            false,
            &policy,
        )
        .unwrap();
        let staged_b = stage_document(
            &target_b,
            ExtractedDocument {
                metadata: WorkMetadata::default(),
                payload: ExtractedPayload::Pdf {
                    bytes: pdf_bytes.clone(),
                },
                raw_bytes_fetched: pdf_bytes.len(),
            },
            Some(&out_dir),
            false,
            &policy,
        )
        .unwrap();
        let StagedOutcome::Pdf {
            staged_pdf_path: pdf_a,
            ..
        } = staged_a.outcome
        else {
            panic!("expected staged PDF A");
        };
        let StagedOutcome::Pdf {
            staged_pdf_path: pdf_b,
            ..
        } = staged_b.outcome
        else {
            panic!("expected staged PDF B");
        };

        let slice_a =
            slice_staged_pdf(&pdf_a, PageSpan::new(1, 2).unwrap(), None, &policy).unwrap();
        let slice_b =
            slice_staged_pdf(&pdf_b, PageSpan::new(1, 2).unwrap(), None, &policy).unwrap();

        assert_ne!(slice_a.slice_pdf_path, slice_b.slice_pdf_path);
        assert!(slice_a.slice_pdf_path.is_file());
        assert!(slice_b.slice_pdf_path.is_file());
    }

    #[cfg(unix)]
    #[test]
    fn slice_staged_pdf_blocks_unconfined_and_symlink_inputs() {
        let allowed_sandbox = tempfile::tempdir().unwrap();
        let outside_dir = tempfile::tempdir().unwrap();
        let mut policy = test_policy(allowed_sandbox.path(), allowed_sandbox.path());
        policy.system_tmp_dir = allowed_sandbox.path().join("fake-sys-tmp");

        let outside_pdf = outside_dir.path().join("secret.pdf");
        fs::write(&outside_pdf, build_test_pdf(&["Secret"])).unwrap();
        let real_pdf = allowed_sandbox.path().join("paper.pdf");
        fs::write(&real_pdf, build_test_pdf(&["Page 1"])).unwrap();
        let link_pdf = allowed_sandbox.path().join("link.pdf");
        symlink(&real_pdf, &link_pdf).unwrap();

        let outside_failure =
            slice_staged_pdf(&outside_pdf, PageSpan::new(1, 1).unwrap(), None, &policy)
                .unwrap_err();
        let link_failure =
            slice_staged_pdf(&link_pdf, PageSpan::new(1, 1).unwrap(), None, &policy).unwrap_err();

        assert!(matches!(outside_failure, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(link_failure, AdfontesError::SafetyBlock { .. }));
    }

    #[cfg(unix)]
    #[test]
    fn slice_staged_pdf_rejects_fifo_without_blocking() {
        let allowed_sandbox = tempfile::tempdir().unwrap();
        let policy = test_policy(allowed_sandbox.path(), allowed_sandbox.path());
        let fifo_path = allowed_sandbox.path().join("trap.pdf");
        let status = Command::new("mkfifo").arg(&fifo_path).status().unwrap();

        let failure =
            slice_staged_pdf(&fifo_path, PageSpan::new(1, 1).unwrap(), None, &policy).unwrap_err();

        assert!(status.success());
        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn honors_adfontes_disable_pdf_slice_kill_switch() {
        let sandbox = tempfile::tempdir().unwrap();
        let mut policy = test_policy(sandbox.path(), sandbox.path());
        policy.disable_pdf_slice = true;
        let pdf_path = sandbox.path().join("paper.pdf");
        fs::write(&pdf_path, build_test_pdf(&["P1", "P2"])).unwrap();

        assert!(matches!(
            slice_staged_pdf(&pdf_path, PageSpan::new(1, 1).unwrap(), None, &policy).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn corrupt_pdf_leaves_no_persisted_file_in_staging_dir() {
        let sandbox = tempfile::tempdir().unwrap();
        let policy = test_policy(sandbox.path(), sandbox.path());
        let out_dir = sandbox.path().join("corrupt-pdf");
        let doc = ExtractedDocument {
            metadata: WorkMetadata::default(),
            payload: ExtractedPayload::Pdf {
                bytes: b"%PDF-1.5\nnot actually a valid pdf".to_vec(),
            },
            raw_bytes_fetched: 33,
        };
        let target = ParsedTarget::parse("https://example.org/broken.pdf").unwrap();

        let failure = stage_document(&target, doc, Some(&out_dir), true, &policy).unwrap_err();
        let remaining_entries: Vec<_> = fs::read_dir(&out_dir).unwrap().collect();
        assert!(matches!(failure, AdfontesError::Extraction { .. }));
        assert!(remaining_entries.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn validate_staging_dir_policy_rejects_existing_intermediate_symlink() {
        let sandbox = tempfile::tempdir().unwrap();
        let policy = test_policy(sandbox.path(), sandbox.path());
        let target_dir = sandbox.path().join("real");
        let sub_dir = target_dir.join("nested");
        fs::create_dir_all(&sub_dir).unwrap();
        let link_dir = sandbox.path().join("link");
        symlink(&target_dir, &link_dir).unwrap();

        let failure =
            validate_staging_dir_policy(Some(&link_dir.join("nested")), &policy).unwrap_err();
        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn slice_staged_pdf_rejects_forbidden_output_dir_before_reading_input() {
        let sandbox = tempfile::tempdir().unwrap();
        let policy = test_policy(sandbox.path(), sandbox.path());
        let missing_pdf = sandbox.path().join("does-not-exist.pdf");
        let forbidden_out = sandbox.path().join(".ssh/blocked");
        let failure = slice_staged_pdf(
            &missing_pdf,
            PageSpan::new(1, 1).unwrap(),
            Some(&forbidden_out),
            &policy,
        )
        .unwrap_err();
        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn format_compact_index_bounds_massive_toc_lazily() {
        let toc: Vec<TocEntry> = (1..=5_000)
            .map(|line| TocEntry {
                level: 1,
                heading: format!("Section {line}"),
                start_line: std::num::NonZeroUsize::new(line).unwrap(),
                end_line: std::num::NonZeroUsize::new(line).unwrap(),
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

    #[test]
    fn reject_overbroad_root_blocks_filesystem_root() {
        let sandbox = tempfile::tempdir().unwrap();
        let policy = test_policy(sandbox.path(), sandbox.path());
        assert!(matches!(
            reject_overbroad_root(Path::new("/"), &policy).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }
}
