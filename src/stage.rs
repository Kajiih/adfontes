//! Confined filesystem staging, version-aware slugs, companion `<slug>.extracted.md` and
//! `<slug>.pages.md` files, PDF slicing, and compact (`<= 3,600 B`) navigation manifests.

architecture_component!(Stage);

mod manifest;
mod sandbox;

use std::path::{Path, PathBuf};

use self::manifest::{
    PdfTextScope, PreparedPdfText, build_extracted_text_filename, build_slice_filenames,
    format_metadata_only_note, format_pdf_page_index_document, format_slice_summary,
    prepare_full_text, prepare_pdf_text_document,
};
pub use self::manifest::{build_document_slug, format_compact_index};
#[cfg(test)]
pub use self::sandbox::TestSandbox;
pub use self::sandbox::{StagingPolicy, resolve_confined_staging_dir, validate_staging_dir_policy};
use self::sandbox::{read_bounded_confined_pdf, remove_file_if_present, write_file_atomic};
use crate::model::{
    AdfontesError, ExtractedDocument, ExtractedPayload, PageSpan, ParsedTarget, PdfTextSummary,
    StagedOutcome, StagedResult,
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
    /// Readability and companion-file summary for extracted slice text.
    pub text_summary: PdfTextSummary,
    /// Compact plain-text summary suitable for CLI stdout or MCP tool responses.
    pub summary: String,
}

/// Stages an already-resolved [`ExtractedDocument`] with per-file atomic (`0600`) writes inside
/// a confined staging directory and builds its bounded (`<= 3,600 B`) navigation manifest.
///
/// For PDF payloads, validates, indexes, and prepares any companion `<slug>.extracted.md` text
/// file in memory before creating directories or writing files, and removes any stale companion
/// `<slug>.extracted.md` when re-staging a scanned PDF over an earlier readable copy.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the target staging directory fails confinement or
/// ownership checks or if a PDF exceeds decompression or `CMap` bounds,
/// [`AdfontesError::Extraction`] if PDF indexing fails, or [`AdfontesError::Io`] on local
/// filesystem write or stale-file cleanup failures.
pub fn stage_document(
    target: &ParsedTarget,
    doc: ExtractedDocument,
    output_dir: Option<&Path>,
    include_pdf_index: bool,
    policy: &StagingPolicy,
) -> Result<StagedResult, AdfontesError> {
    validate_staging_dir_policy(output_dir, policy)?;
    let slug = build_document_slug(target, &doc.metadata);

    let outcome = match doc.payload {
        ExtractedPayload::FullText { format, markdown } => {
            let prepared = prepare_full_text(format, markdown);
            let staging_dir = resolve_confined_staging_dir(output_dir, policy)?;
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
            // Validate, index, and prepare companion Markdown in memory before creating directories
            // or writing any files to disk so corrupt, encrypted, or bomb PDFs never touch disk.
            let indexed = index_pdf_bytes(&bytes)?;
            let prepared_text = prepare_pdf_text_document(
                doc.metadata.title.as_deref(),
                PdfTextScope::FullDocument {
                    total_pages: indexed.index.page_count,
                },
                &indexed.page_texts,
            );
            drop(indexed.page_texts);

            let staging_dir = resolve_confined_staging_dir(output_dir, policy)?;
            let staged_pdf_path = staging_dir.join(format!("{slug}.pdf"));
            let byte_count = bytes.len();
            write_file_atomic(&staged_pdf_path, &bytes)?;
            drop(bytes);

            let extracted_path = staging_dir.join(build_extracted_text_filename(&slug));
            let text_summary = stage_optional_pdf_text(prepared_text, &extracted_path)?;
            let page_index_path = if include_pdf_index {
                let index_path = staging_dir.join(format!("{slug}.pages.md"));
                let page_doc =
                    format_pdf_page_index_document(&doc.metadata, &indexed.index, &text_summary);
                write_file_atomic(&index_path, page_doc.as_bytes())?;
                Some(index_path)
            } else {
                None
            };
            StagedOutcome::Pdf {
                staged_pdf_path,
                page_index_path,
                byte_count,
                pdf_index: indexed.index,
                text_summary,
                fallback_skips,
            }
        }
        ExtractedPayload::MetadataOnly { reason } => {
            let note_text = format_metadata_only_note(&doc.metadata, &reason);
            let staging_dir = resolve_confined_staging_dir(output_dir, policy)?;
            let staged_note_path = staging_dir.join(format!("{slug}.metadata.md"));
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

/// Validates, slices, and writes (`0600`) a `1..=8`-page vector PDF slice and any companion
/// `.extracted.md` text file from a local confined staged PDF file.
///
/// Validates, slices, and extracts retained page text in memory before creating directories or
/// writing per-file `0600` outputs, and removes any stale `*_p*.extracted.md` when re-slicing a
/// scanned span over an earlier readable slice.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if slicing is disabled by `ADFONTES_DISABLE_PDF_SLICE`, if
/// `pdf_path` escapes allowed staging roots, traverses a symlink, is a FIFO/non-regular file, or
/// exceeds `100 MiB`, if any retained page trips decompression or `CMap` limits during slicing or
/// text extraction, or if `output_dir` fails confinement; returns [`AdfontesError::InvalidTarget`]
/// or [`AdfontesError::Extraction`] on invalid page bounds or PDF errors, or [`AdfontesError::Io`]
/// on write or stale-file cleanup failures.
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
    drop(confined_pdf.bytes);
    let prepared_text = prepare_pdf_text_document(
        None,
        PdfTextScope::Slice {
            span: sliced.effective_span,
            source_total_pages: sliced.total_pages,
        },
        &sliced.page_texts,
    );
    drop(sliced.page_texts);

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

    let filenames = build_slice_filenames(&canonical_pdf_path, sliced.effective_span);
    let slice_pdf_path = target_dir.join(&filenames.pdf);
    let byte_count = sliced.bytes.len();
    write_file_atomic(&slice_pdf_path, &sliced.bytes)?;
    drop(sliced.bytes);

    let text_summary =
        stage_optional_pdf_text(prepared_text, &target_dir.join(&filenames.extracted_text))?;
    let summary = format_slice_summary(
        &canonical_pdf_path,
        &slice_pdf_path,
        sliced.effective_span,
        sliced.total_pages,
        byte_count,
        &text_summary,
    );

    Ok(StagedPdfSlice {
        source_pdf_path: canonical_pdf_path,
        slice_pdf_path,
        total_pages: sliced.total_pages,
        effective_span: sliced.effective_span,
        byte_count,
        text_summary,
        summary,
    })
}

fn stage_optional_pdf_text(
    prepared_text: PreparedPdfText,
    dest_path: &Path,
) -> Result<PdfTextSummary, AdfontesError> {
    match prepared_text {
        PreparedPdfText::None(summary) => {
            remove_file_if_present(dest_path)?;
            Ok(summary)
        }
        PreparedPdfText::Extracted(extracted) => {
            write_file_atomic(dest_path, extracted.markdown().as_bytes())?;
            Ok(extracted.into_summary(dest_path.to_path_buf()))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use super::*;
    use crate::model::{TextFormat, WorkMetadata};
    use crate::pdf::{build_test_cmap_bomb_pdf, build_test_pdf};

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
        assert_ne!(
            slice_a.text_summary.staged_text().unwrap().path,
            slice_b.text_summary.staged_text().unwrap().path
        );
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
        assert!(matches!(failure, AdfontesError::Extraction { .. }));
        assert!(!out_dir.exists());
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

    #[test]
    fn stages_pdf_companion_extracted_text_with_0600_mode_and_exact_page_ranges() {
        let sandbox = TestSandbox::new();
        let out_dir = sandbox.path().join("pdf-text");
        let policy = sandbox.policy();
        let page1 = "Abstract: Bayesian inference under uncertain test sensitivity and specificity across cohorts.";
        let page2 = "2. Modeling a test with uncertain sensitivity and specificity in hierarchical studies.";
        let pdf_bytes = build_test_pdf(&[page1, page2]);
        let target = ParsedTarget::parse("0710.5005").unwrap();
        let doc = ExtractedDocument {
            metadata: WorkMetadata {
                title: Some("Preprint Study".to_owned()),
                ..WorkMetadata::default()
            },
            payload: ExtractedPayload::Pdf {
                bytes: pdf_bytes.clone(),
                fallback_skips: Vec::new(),
            },
            raw_bytes_fetched: pdf_bytes.len(),
        };

        let staged = stage_document(&target, doc, Some(&out_dir), true, &policy).unwrap();
        let StagedOutcome::Pdf {
            page_index_path,
            text_summary,
            ..
        } = staged.outcome
        else {
            panic!("expected staged PDF");
        };
        let staged_text = text_summary.staged_text().unwrap();
        let content = fs::read_to_string(&staged_text.path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        let page1_range = staged_text.page_line_ranges[&std::num::NonZeroU32::new(1).unwrap()];
        let page1_anchor = format!(
            "L{}-L{}",
            page1_range.start_line.get(),
            page1_range.end_line.get()
        );
        let page_index_doc = fs::read_to_string(page_index_path.unwrap()).unwrap();

        assert_eq!(lines[page1_range.start_line.get() - 1], "## Page 1");
        assert!(staged.compact_index.contains(&page1_anchor));
        assert!(page_index_doc.contains(&page1_anchor));
        assert_owner_only_file(&staged_text.path);
    }

    #[test]
    fn scanned_pdf_writes_no_empty_extracted_text_file() {
        let sandbox = TestSandbox::new();
        let out_dir = sandbox.path().join("scanned");
        let policy = sandbox.policy();
        let pdf_bytes = build_test_pdf(&["   ", ""]);
        let target = ParsedTarget::parse("https://example.org/scan.pdf").unwrap();
        let doc = ExtractedDocument {
            metadata: WorkMetadata::default(),
            payload: ExtractedPayload::Pdf {
                bytes: pdf_bytes.clone(),
                fallback_skips: Vec::new(),
            },
            raw_bytes_fetched: pdf_bytes.len(),
        };

        let staged = stage_document(&target, doc, Some(&out_dir), true, &policy).unwrap();
        let StagedOutcome::Pdf { text_summary, .. } = staged.outcome else {
            panic!("expected staged PDF");
        };
        let has_extracted_file = fs::read_dir(&out_dir).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".extracted.md")
        });

        assert_eq!(text_summary.staged_text(), None);
        assert!(!has_extracted_file);
        assert!(staged.compact_index.contains("TEXT_LAYER: NONE"));
        assert!(!staged.compact_index.contains("EXTRACTED_TEXT_PATH:"));
    }

    #[test]
    fn restaging_scanned_pdf_removes_stale_extracted_text_file() {
        let sandbox = TestSandbox::new();
        let out_dir = sandbox.path().join("restage");
        let policy = sandbox.policy();
        let target = ParsedTarget::parse("0710.5005").unwrap();
        let readable_pdf = build_test_pdf(&[
            "Readable preprint page with sufficient text to write companion Markdown.",
        ]);
        let scanned_pdf = build_test_pdf(&["   "]);

        let first = stage_document(
            &target,
            ExtractedDocument {
                metadata: WorkMetadata::default(),
                payload: ExtractedPayload::Pdf {
                    bytes: readable_pdf.clone(),
                    fallback_skips: Vec::new(),
                },
                raw_bytes_fetched: readable_pdf.len(),
            },
            Some(&out_dir),
            false,
            &policy,
        )
        .unwrap();
        let StagedOutcome::Pdf {
            text_summary: first_summary,
            ..
        } = first.outcome
        else {
            panic!("expected first staged PDF");
        };
        let stale_path = first_summary.staged_text().unwrap().path.clone();

        let second = stage_document(
            &target,
            ExtractedDocument {
                metadata: WorkMetadata::default(),
                payload: ExtractedPayload::Pdf {
                    bytes: scanned_pdf.clone(),
                    fallback_skips: Vec::new(),
                },
                raw_bytes_fetched: scanned_pdf.len(),
            },
            Some(&out_dir),
            false,
            &policy,
        )
        .unwrap();
        let StagedOutcome::Pdf {
            text_summary: second_summary,
            ..
        } = second.outcome
        else {
            panic!("expected second staged PDF");
        };

        assert_eq!(second_summary.staged_text(), None);
        assert!(!stale_path.exists());
        assert!(second.compact_index.contains("TEXT_LAYER: NONE"));
    }

    #[test]
    fn slice_staged_pdf_writes_companion_extracted_text_with_original_page_numbers_and_0600_mode() {
        let sandbox = TestSandbox::new();
        let out_dir = sandbox.path().join("slice-text");
        let policy = sandbox.policy();
        let pdf_bytes = build_test_pdf(&[
            "Page 1 intro text that is outside the requested page span for this slice.",
            "Page 2 hierarchical model derivation with sufficient prose length for readable status.",
            "Page 3 posterior predictive diagnostics with full empirical coverage tables.",
            "Page 4 concluding remarks outside the slice.",
        ]);
        let target = ParsedTarget::parse("2409.13740v1").unwrap();
        let staged = stage_document(
            &target,
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
            staged_pdf_path, ..
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
        let staged_text = slice.text_summary.staged_text().unwrap();
        let page_numbers: Vec<u32> = staged_text
            .page_line_ranges
            .keys()
            .map(|page| page.get())
            .collect();

        assert!(slice.summary.contains("SLICE_TEXT:"));
        assert!(
            slice
                .summary
                .contains("SLICE_PAGES: p.2 (L3-L5), p.3 (L7-L9)")
        );
        assert_eq!(page_numbers, vec![2, 3]);
        assert_owner_only_file(&staged_text.path);
    }

    #[test]
    fn slice_staged_pdf_omits_extracted_text_for_scanned_slice() {
        let sandbox = TestSandbox::new();
        let out_dir = sandbox.path().join("slice-scan");
        let policy = sandbox.policy();
        let pdf_bytes = build_test_pdf(&["   ", ""]);
        let target = ParsedTarget::parse("https://example.org/scan-slice.pdf").unwrap();
        let staged = stage_document(
            &target,
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
            staged_pdf_path, ..
        } = staged.outcome
        else {
            panic!("expected staged PDF");
        };

        let slice = slice_staged_pdf(
            &staged_pdf_path,
            PageSpan::new(1, 2).unwrap(),
            None,
            &policy,
        )
        .unwrap();
        let has_extracted_file = fs::read_dir(&out_dir).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".extracted.md")
        });

        assert_eq!(slice.text_summary.staged_text(), None);
        assert!(!has_extracted_file);
        assert!(!slice.summary.contains("SLICE_TEXT:"));
        assert!(slice.summary.contains("TEXT_LAYER: NONE"));
    }

    #[test]
    fn slice_staged_pdf_rejects_cmap_bomb_before_writing_any_slice_files() {
        let sandbox = TestSandbox::new();
        let source_dir = sandbox.path().join("bomb-src");
        let slice_out = sandbox.path().join("bomb-out");
        let policy = sandbox.policy();
        fs::create_dir_all(&source_dir).unwrap();
        let bomb_path = source_dir.join("bomb.pdf");
        fs::write(&bomb_path, build_test_cmap_bomb_pdf()).unwrap();

        let error = slice_staged_pdf(
            &bomb_path,
            PageSpan::new(1, 1).unwrap(),
            Some(&slice_out),
            &policy,
        )
        .unwrap_err();

        assert!(matches!(error, AdfontesError::SafetyBlock { .. }));
        assert!(!slice_out.exists());
    }

    #[test]
    fn reslicing_scanned_pdf_removes_stale_extracted_text_file() {
        let sandbox = TestSandbox::new();
        let out_dir = sandbox.path().join("reslice");
        let policy = sandbox.policy();
        let target = ParsedTarget::parse("2409.13740v1").unwrap();
        let readable_pdf = build_test_pdf(&[
            "Page 1 intro prose outside the slice span.",
            "Page 2 readable theorem statement with sufficient length for Markdown staging.",
        ]);
        let scanned_pdf = build_test_pdf(&[" ", "  "]);

        let first = stage_document(
            &target,
            ExtractedDocument {
                metadata: WorkMetadata::default(),
                payload: ExtractedPayload::Pdf {
                    bytes: readable_pdf.clone(),
                    fallback_skips: Vec::new(),
                },
                raw_bytes_fetched: readable_pdf.len(),
            },
            Some(&out_dir),
            false,
            &policy,
        )
        .unwrap();
        let StagedOutcome::Pdf {
            staged_pdf_path, ..
        } = first.outcome
        else {
            panic!("expected first staged PDF");
        };
        let first_slice = slice_staged_pdf(
            &staged_pdf_path,
            PageSpan::new(1, 2).unwrap(),
            None,
            &policy,
        )
        .unwrap();
        let stale_slice_text = first_slice.text_summary.staged_text().unwrap().path.clone();

        let _ = stage_document(
            &target,
            ExtractedDocument {
                metadata: WorkMetadata::default(),
                payload: ExtractedPayload::Pdf {
                    bytes: scanned_pdf.clone(),
                    fallback_skips: Vec::new(),
                },
                raw_bytes_fetched: scanned_pdf.len(),
            },
            Some(&out_dir),
            false,
            &policy,
        )
        .unwrap();
        let second_slice = slice_staged_pdf(
            &staged_pdf_path,
            PageSpan::new(1, 2).unwrap(),
            None,
            &policy,
        )
        .unwrap();

        assert_eq!(second_slice.text_summary.staged_text(), None);
        assert!(!stale_slice_text.exists());
        assert!(!second_slice.summary.contains("SLICE_TEXT:"));
        assert!(second_slice.summary.contains("TEXT_LAYER: NONE"));
    }
}
