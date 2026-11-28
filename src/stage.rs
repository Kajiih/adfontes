//! Confined filesystem staging, version-aware slugs, `<slug>.pages.md`, PDF slicing, and compact
//! (`<= 3,600 B`) navigation manifests.

architecture_component!(Stage);

mod manifest;
mod sandbox;

use std::path::{Path, PathBuf};

pub use self::manifest::{build_document_slug, format_compact_index};
use self::manifest::{
    build_slice_filename, format_metadata_only_note, format_pdf_page_index_document,
    format_slice_summary, prepare_full_text,
};
#[cfg(test)]
pub use self::sandbox::TestSandbox;
pub use self::sandbox::{StagingPolicy, resolve_confined_staging_dir, validate_staging_dir_policy};
use self::sandbox::{read_bounded_confined_pdf, write_file_atomic};
use crate::model::{
    AdfontesError, ExtractedDocument, ExtractedPayload, PageSpan, ParsedTarget, StagedOutcome,
    StagedResult,
};
use crate::pdf::{index_pdf_bytes, slice_pdf_bytes};

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
/// ownership checks or if a PDF exceeds decompression or `CMap` bounds,
/// [`AdfontesError::Extraction`] if PDF indexing fails, or [`AdfontesError::Io`] on local
/// filesystem write failures.
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
            let prepared = prepare_full_text(format, markdown);
            let staged_path = staging_dir.join(format!("{slug}.{}", prepared.extension));
            write_file_atomic(&staged_path, prepared.content.as_bytes())?;
            StagedOutcome::FullText {
                staged_path,
                format,
                line_count: prepared.line_count,
                byte_count: prepared.byte_count,
                toc: prepared.toc,
            }
        }
        ExtractedPayload::Pdf {
            bytes,
            fallback_skips,
        } => {
            // Validate and index PDF bytes in memory before writing PDF or index files to disk so
            // corrupt, encrypted, or bomb PDFs never touch the staging directory.
            let pdf_index = index_pdf_bytes(&bytes)?;
            let staged_pdf_path = staging_dir.join(format!("{slug}.pdf"));
            write_file_atomic(&staged_pdf_path, &bytes)?;
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
                fallback_skips,
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

    let slice_name = build_slice_filename(&canonical_pdf_path, sliced.effective_span);
    let slice_pdf_path = target_dir.join(slice_name);
    write_file_atomic(&slice_pdf_path, &sliced.bytes)?;

    let byte_count = sliced.bytes.len();
    let summary = format_slice_summary(
        &canonical_pdf_path,
        &slice_pdf_path,
        sliced.effective_span,
        sliced.total_pages,
        byte_count,
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

#[cfg(test)]
mod tests {
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use super::*;
    use crate::model::{TextFormat, WorkMetadata};
    use crate::pdf::build_test_pdf;

    fn assert_owner_only_file(path: &Path) {
        #[cfg(unix)]
        {
            let mode = fs::metadata(path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        #[cfg(not(unix))]
        {
            let _ = path;
        }
    }

    #[test]
    fn stages_pdf_index_and_slices_pages_with_0600_mode() {
        let sandbox = TestSandbox::new();
        let out_dir = sandbox.path().join("run");
        let policy = sandbox.policy();
        let pdf_bytes =
            build_test_pdf(&["1 Introduction", "2 Method", "3 Derivation", "4 Conclusion"]);
        let target = ParsedTarget::parse("2409.13740v1").unwrap();
        let doc = ExtractedDocument {
            metadata: WorkMetadata::default(),
            payload: ExtractedPayload::Pdf {
                bytes: pdf_bytes.clone(),
                fallback_skips: Vec::new(),
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
        assert_owner_only_file(&slice.slice_pdf_path);
    }

    #[test]
    fn slice_staged_pdf_preserves_distinct_hash_suffixes_for_long_slugs() {
        let sandbox = TestSandbox::new();
        let out_dir = sandbox.path().join("run");
        let policy = sandbox.policy();
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
                    fallback_skips: Vec::new(),
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
                    fallback_skips: Vec::new(),
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

    #[test]
    fn honors_adfontes_disable_pdf_slice_kill_switch() {
        let sandbox = TestSandbox::new();
        let mut policy = sandbox.policy();
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
        let sandbox = TestSandbox::new();
        let policy = sandbox.policy();
        let out_dir = sandbox.path().join("corrupt-pdf");
        let doc = ExtractedDocument {
            metadata: WorkMetadata::default(),
            payload: ExtractedPayload::Pdf {
                bytes: b"%PDF-1.5\nnot actually a valid pdf".to_vec(),
                fallback_skips: Vec::new(),
            },
            raw_bytes_fetched: 33,
        };
        let target = ParsedTarget::parse("https://example.org/broken.pdf").unwrap();

        let failure = stage_document(&target, doc, Some(&out_dir), true, &policy).unwrap_err();
        let remaining_entries: Vec<_> = fs::read_dir(&out_dir).unwrap().collect();
        assert!(matches!(failure, AdfontesError::Extraction { .. }));
        assert!(remaining_entries.is_empty());
    }

    #[test]
    fn slice_staged_pdf_rejects_forbidden_output_dir_before_reading_input() {
        let sandbox = TestSandbox::new();
        let policy = sandbox.policy();
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
    fn stages_full_text_and_reports_format_metrics() {
        let sandbox = TestSandbox::new();
        let policy = sandbox.policy();
        let target = ParsedTarget::parse("RFC 9110").unwrap();
        let doc = ExtractedDocument {
            metadata: WorkMetadata::default(),
            payload: ExtractedPayload::FullText {
                format: TextFormat::PlainText,
                markdown: "First line\nSecond line".to_owned(),
            },
            raw_bytes_fetched: 22,
        };

        let staged = stage_document(&target, doc, None, false, &policy).unwrap();
        assert!(staged.compact_index.contains("CONTENT_KIND: PLAIN_TEXT"));
        assert!(staged.compact_index.contains("METRICS: 2 lines"));
    }
}
