//! Pure-Rust PDF page indexing and lossless vector page slicing powered by `lopdf`.
//!
//! Eliminates external C++ `poppler` requirements while ensuring sliced PDFs materialize
//! ancestor-inherited `/Pages` attributes (`Resources`, `MediaBox`, `CropBox`, `Rotate`), build a
//! fresh spec-clean flat `/Pages` tree without `lopdf::Document::delete_pages` `/Parent`-cycle
//! hangs or empty intermediate `/Pages` nodes, strip cross-page `Catalog`/`Page`/trailer keys, and
//! enforce cumulative decompression, forward-only `/ToUnicode` `CMap`, and page text decoding
//! budgets during loading, indexing, and slicing.

architecture_component!(Pdf);

mod cmap;

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::rc::Rc;

use lopdf::content::Content;
use lopdf::{
    DecompressError, Dictionary, Document, Encoding, Error as LopdfError, LoadOptions, Object,
    ObjectId, decode_text_string,
};

use self::cmap::ForwardToUnicodeCMap;
use crate::model::{
    AdfontesError, PageSpan, PdfIndex, PdfPageSummary, PdfPageText, truncate_chars,
    truncate_utf8_bytes,
};

const PDF_FORMAT_LABEL: &str = "PDF";
const KEY_FIRST: &[u8] = b"First";
const KEY_TO_UNICODE: &[u8] = b"ToUnicode";
const PDF_MAGIC: &[u8] = b"%PDF-";
const PDF_MAGIC_SCAN_WINDOW: usize = 1024;
const MAX_PDF_LOAD_DECOMPRESSED_BYTES: usize = 16 * 1024 * 1024;
const MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES: usize = 32 * 1024 * 1024;
const MAX_PDF_PAGE_DECOMPRESSED_BYTES: usize = 2 * 1024 * 1024;
const MAX_PDF_CMAP_BYTES: usize = 1024 * 1024;
const MAX_PDF_PAGE_TEXT_BYTES: usize = 512 * 1024;
const MAX_PDF_PAGE_RAW_TEXT_BYTES: usize = 2 * 1024 * 1024;
const MAX_PDF_TEXT_ARRAY_DEPTH: usize = 8;
const MAX_CMAP_AMPLIFICATION_RATIO: usize = 8;
const SIMPLE_DECODE_CHUNK_BYTES: usize = 16 * 1024;
const MAX_PDF_DOCUMENT_TEXT_BYTES: usize = 8 * 1024 * 1024;
const MAX_INDEXED_PDF_PAGES: usize = 250;
const MAX_PARENT_TREE_DEPTH: usize = 64;
const MAX_OUTLINE_ITEMS: usize = 200;
const MAX_OUTLINE_DEPTH: usize = 8;
const MAX_HEADINGS_PER_PAGE: usize = 4;
const MAX_PREVIEW_CHARS: usize = 160;
const MIN_HEADING_LEN: usize = 4;
const MAX_HEADING_LEN: usize = 110;

const INHERITABLE_PAGE_KEYS: &[&[u8]] = &[b"Resources", b"MediaBox", b"CropBox", b"Rotate"];
const STRIPPED_PAGE_KEYS: &[&[u8]] = &[b"Annots", b"StructParents", b"B", b"AA"];
const STRIPPED_CATALOG_KEYS: &[&[u8]] = &[
    b"Outlines",
    b"StructTreeRoot",
    b"MarkInfo",
    b"Names",
    b"Dests",
    b"PageLabels",
    b"Threads",
    b"AcroForm",
    b"OpenAction",
    b"AA",
];
const STRIPPED_TRAILER_KEYS: &[&[u8]] = &[
    b"Encrypt",
    b"Prev",
    b"XRefStm",
    b"DecodeParms",
    b"Filter",
    b"W",
    b"Index",
    b"Type",
    b"Length",
];

thread_local! {
    static PDF_LOAD_BUDGET_REMAINING: Cell<usize> = const { Cell::new(MAX_PDF_LOAD_DECOMPRESSED_BYTES) };
    static PDF_LOAD_SAFETY_TRIPPED: Cell<bool> = const { Cell::new(false) };
}

/// Normalized extracted text for a single PDF page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedPdfPage {
    page_number: NonZeroU32,
    char_count: usize,
    text: String,
}

impl ExtractedPdfPage {
    /// Creates an extracted page record whose [`Self::char_count`] is derived once from `text`.
    #[must_use]
    pub(crate) fn new(page_number: NonZeroU32, text: String) -> Self {
        let char_count = count_non_whitespace_chars(&text);
        Self {
            page_number,
            char_count,
            text,
        }
    }

    /// Returns the original 1-indexed page number in the source PDF.
    #[must_use]
    pub const fn page_number(&self) -> NonZeroU32 {
        self.page_number
    }

    /// Returns the non-whitespace character count of [`Self::text`].
    #[must_use]
    pub const fn char_count(&self) -> usize {
        self.char_count
    }

    /// Returns the normalized UTF-8 page text (empty when the page has no text layer).
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Combined structural index and transient per-page text emitted by [`index_pdf_bytes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedPdf {
    /// Lightweight page count, outline titles, and per-page summaries.
    pub index: PdfIndex,
    /// Normalized extracted text for each indexed page in ascending page-number order.
    pub page_texts: Vec<ExtractedPdfPage>,
}

/// Lossless vector PDF slice emitted by [`slice_pdf_bytes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlicedPdf {
    /// Total page count of the source PDF.
    pub total_pages: u32,
    /// Effective clamped 1-indexed page span present in [`Self::bytes`].
    pub effective_span: PageSpan,
    /// Validated standalone PDF bytes for the retained pages.
    pub bytes: Vec<u8>,
    /// Normalized extracted text for each retained page in ascending original page-number order.
    pub page_texts: Vec<ExtractedPdfPage>,
}

/// Returns `true` if `bytes` contains the `%PDF-` signature within its first 1,024 bytes.
#[must_use]
pub fn is_pdf_magic(bytes: &[u8]) -> bool {
    let window_len = bytes.len().min(PDF_MAGIC_SCAN_WINDOW);
    let prefix = bytes.get(..window_len).unwrap_or_default();
    prefix
        .windows(PDF_MAGIC.len())
        .any(|window| window == PDF_MAGIC)
}

/// Indexes a PDF's page count, nested outline bookmarks, bounded per-page summaries, and
/// normalized extracted page text.
///
/// Uses pure-Rust `lopdf` under strict decompression, font-CMap, and cumulative UTF-8 text byte
/// budgets without spawning external binaries.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if a PDF decompression bomb exceeds memory budgets, or
/// [`AdfontesError::Extraction`] if `bytes` is not a readable PDF or requires a user password.
pub fn index_pdf_bytes(bytes: &[u8]) -> Result<IndexedPdf, AdfontesError> {
    index_pdf_bytes_with_budget(bytes, MAX_INDEXED_PDF_PAGES, MAX_PDF_DOCUMENT_TEXT_BYTES)
}

/// Losslessly slices an inclusive [`PageSpan`] (`1..=8` pages) into a self-contained PDF and
/// extracts bounded per-page normalized text for companion `.extracted.md` staging.
///
/// Materializes inherited `/Pages` attributes (`Resources`, `MediaBox`, `CropBox`, `Rotate`) onto
/// each retained `/Page`, extracts bounded per-page text before rewriting the `/Pages` tree, wires
/// retained pages into a fresh flat `/Pages` root (avoiding `lopdf::Document::delete_pages`
/// `/Parent`-cycle hangs and empty intermediate `/Pages` nodes), strips cross-page `Catalog`/page/
/// trailer state, prunes unreachable objects, and serializes.
///
/// # Errors
///
/// Returns [`AdfontesError::InvalidTarget`] if `span` starts past the PDF's page count,
/// [`AdfontesError::SafetyBlock`] if loading or any retained page trips decompression or `CMap`
/// limits (enforcing the same fail-closed security boundary as [`index_pdf_bytes`]), or
/// [`AdfontesError::Extraction`] if the PDF is malformed or user-password-encrypted.
pub fn slice_pdf_bytes(raw_bytes: &[u8], span: PageSpan) -> Result<SlicedPdf, AdfontesError> {
    let mut doc = load_decrypted_pdf(raw_bytes)?;
    let page_ids = doc.get_pages();
    let total_pages = u32::try_from(page_ids.len()).unwrap_or(u32::MAX);
    let effective_span = span.clamp_to_total(total_pages)?;

    let mut extraction_budget = PdfExtractionBudget::new();
    let mut retained_ids = Vec::new();
    let mut page_texts = Vec::new();
    for (&page_number, &page_id) in &page_ids {
        if page_number >= effective_span.start() && page_number <= effective_span.end() {
            let Some(page_nz) = NonZeroU32::new(page_number) else {
                continue;
            };
            materialize_inherited_page_attributes(&mut doc, page_id);
            let raw_text = extraction_budget.extract_page_text(&doc, page_number, page_id)?;
            let normalized = normalize_pdf_page_text(&raw_text, MAX_PDF_PAGE_TEXT_BYTES);
            page_texts.push(ExtractedPdfPage::new(page_nz, normalized));
            strip_cross_page_page_keys(&mut doc, page_id);
            retained_ids.push(page_id);
        }
    }

    let retained_count = i64::try_from(retained_ids.len()).unwrap_or(i64::MAX);
    let kids: Vec<Object> = retained_ids
        .iter()
        .copied()
        .map(Object::Reference)
        .collect();
    let new_pages_id = doc.add_object(Dictionary::from_iter([
        ("Type", Object::Name(b"Pages".to_vec())),
        ("Kids", Object::Array(kids)),
        ("Count", Object::Integer(retained_count)),
    ]));

    for &page_id in &retained_ids {
        if let Ok(Object::Dictionary(page_mut)) = doc.get_object_mut(page_id) {
            page_mut.set("Parent", Object::Reference(new_pages_id));
        }
    }

    if let Ok(catalog) = doc.catalog_mut() {
        catalog.set("Pages", Object::Reference(new_pages_id));
        for &key in STRIPPED_CATALOG_KEYS {
            catalog.remove(key);
        }
    }

    for &key in STRIPPED_TRAILER_KEYS {
        doc.trailer.remove(key);
    }
    doc.encryption_state = None;
    doc.xref_start = 0;
    doc.bookmarks.clear();
    doc.bookmark_table.clear();

    let _ = doc.prune_objects();
    doc.renumber_objects();

    let mut out_bytes = Vec::new();
    doc.save_to(&mut out_bytes)
        .map_err(|io_error| AdfontesError::Extraction {
            format: PDF_FORMAT_LABEL,
            detail: format!("failed serializing sliced PDF: {io_error}"),
        })?;

    Ok(SlicedPdf {
        total_pages,
        effective_span,
        bytes: out_bytes,
        page_texts,
    })
}

fn index_pdf_bytes_with_budget(
    bytes: &[u8],
    max_indexed_pages: usize,
    max_document_text_bytes: usize,
) -> Result<IndexedPdf, AdfontesError> {
    let mut doc = load_decrypted_pdf(bytes)?;
    let page_ids = doc.get_pages();
    let page_count = u32::try_from(page_ids.len()).unwrap_or(u32::MAX);
    let outline_titles = extract_outline_titles(&doc);

    let mut extraction_budget = PdfExtractionBudget::new();
    let mut remaining_text_bytes = max_document_text_bytes;
    let mut pages = Vec::with_capacity(page_ids.len());
    let mut page_texts = Vec::with_capacity(page_ids.len().min(max_indexed_pages));

    for (index, (&page_number, &page_id)) in page_ids.iter().enumerate() {
        let Some(page_nz) = NonZeroU32::new(page_number) else {
            continue;
        };
        if index >= max_indexed_pages || remaining_text_bytes == 0 {
            pages.push(unindexed_page_summary(page_nz));
            continue;
        }
        materialize_inherited_page_attributes(&mut doc, page_id);
        let raw_text = extraction_budget.extract_page_text(&doc, page_number, page_id)?;
        let normalized = normalize_pdf_page_text(&raw_text, MAX_PDF_PAGE_TEXT_BYTES);
        if normalized.len() > remaining_text_bytes {
            remaining_text_bytes = 0;
            pages.push(unindexed_page_summary(page_nz));
            continue;
        }
        remaining_text_bytes = remaining_text_bytes.saturating_sub(normalized.len());
        let page = ExtractedPdfPage::new(page_nz, normalized);
        pages.push(summarize_page_text(
            page.page_number(),
            page.text(),
            page.char_count(),
        ));
        page_texts.push(page);
    }

    Ok(IndexedPdf {
        index: PdfIndex {
            page_count,
            outline_titles,
            pages,
        },
        page_texts,
    })
}

fn load_decrypted_pdf(bytes: &[u8]) -> Result<Document, AdfontesError> {
    if !is_pdf_magic(bytes) {
        return Err(AdfontesError::Extraction {
            format: PDF_FORMAT_LABEL,
            detail: "input bytes do not begin with `%PDF-` signature".to_owned(),
        });
    }

    let _budget_guard = PdfLoadBudgetGuard::new(MAX_PDF_LOAD_DECOMPRESSED_BYTES);
    let options = LoadOptions {
        max_decompressed_size: Some(MAX_PDF_LOAD_DECOMPRESSED_BYTES),
        filter: Some(filter_pdf_load_object),
        ..Default::default()
    };
    let load_result = Document::load_mem_with_options(bytes, options);
    if PDF_LOAD_SAFETY_TRIPPED.with(Cell::get) {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF object stream decompression exceeded {MAX_PDF_LOAD_DECOMPRESSED_BYTES}-byte ceiling"
            ),
        });
    }

    let doc = load_result.map_err(|pdf_error| match pdf_error {
        LopdfError::Decompress(DecompressError::MemoryLimitExceeded { limit }) => {
            AdfontesError::SafetyBlock {
                reason: format!("PDF stream decompression exceeded {limit}-byte ceiling"),
            }
        }
        other => AdfontesError::Extraction {
            format: PDF_FORMAT_LABEL,
            detail: format!("failed to parse PDF: {other}"),
        },
    })?;

    if (doc.is_encrypted() || doc.trailer.has(b"Encrypt")) && doc.encryption_state.is_none() {
        return Err(AdfontesError::Extraction {
            format: PDF_FORMAT_LABEL,
            detail: "PDF is password-encrypted and cannot be opened".to_owned(),
        });
    }
    Ok(doc)
}

// omni:ignore [positional-tuple-return] -- signature mandated by lopdf::FilterFunc
fn filter_pdf_load_object(object_id: ObjectId, object: &mut Object) -> Option<(ObjectId, Object)> {
    if PDF_LOAD_SAFETY_TRIPPED.with(Cell::get) {
        return None;
    }
    if let Object::Stream(stream) = object
        && stream.dict.has_type(b"ObjStm")
    {
        let remaining = PDF_LOAD_BUDGET_REMAINING.with(Cell::get);
        match stream.decompressed_content_with_limit(remaining) {
            Ok(data) => {
                PDF_LOAD_BUDGET_REMAINING.with(|cell| {
                    cell.set(remaining.saturating_sub(data.len()));
                });
            }
            Err(LopdfError::Decompress(DecompressError::MemoryLimitExceeded { .. })) => {
                PDF_LOAD_SAFETY_TRIPPED.with(|cell| cell.set(true));
                return None;
            }
            Err(_) => {}
        }
    }
    if matches!(object, Object::Stream(_)) {
        // Streams are always top-level indirect objects (ISO 32000-1 §7.5.7 forbids streams inside
        // ObjStm), and `lopdf` discards the returned `Object` for top-level objects while keeping
        // `object` intact; returning `Object::Null` avoids deep-cloning multi-MiB image streams.
        return Some((object_id, Object::Null));
    }
    Some((object_id, object.clone()))
}

struct PdfLoadBudgetGuard;

impl PdfLoadBudgetGuard {
    fn new(limit: usize) -> Self {
        PDF_LOAD_BUDGET_REMAINING.with(|cell| cell.set(limit));
        PDF_LOAD_SAFETY_TRIPPED.with(|cell| cell.set(false));
        Self
    }
}

impl Drop for PdfLoadBudgetGuard {
    fn drop(&mut self) {
        PDF_LOAD_BUDGET_REMAINING.with(|cell| cell.set(MAX_PDF_LOAD_DECOMPRESSED_BYTES));
        PDF_LOAD_SAFETY_TRIPPED.with(|cell| cell.set(false));
    }
}

struct PdfExtractionBudget {
    remaining_decompressed: usize,
    cached_cmaps: BTreeMap<ObjectId, Option<Rc<ForwardToUnicodeCMap>>>,
}

impl PdfExtractionBudget {
    const fn new() -> Self {
        Self {
            remaining_decompressed: MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES,
            cached_cmaps: BTreeMap::new(),
        }
    }

    fn extract_page_text(
        &mut self,
        doc: &Document,
        page_number: u32,
        page_id: ObjectId,
    ) -> Result<String, AdfontesError> {
        if self.remaining_decompressed == 0 {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF text extraction exceeded cumulative {MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES}-byte decompression ceiling"
                ),
            });
        }

        let mut fallback_dicts: Vec<(Vec<u8>, Dictionary)> = Vec::new();
        let mut encodings: BTreeMap<Vec<u8>, PageFontEncoding<'_>> = BTreeMap::new();
        if let Ok(fonts) = doc.get_page_fonts(page_id) {
            for (font_name, font) in fonts {
                let Ok(to_unicode_raw) = font.get(KEY_TO_UNICODE) else {
                    if let Some(encoding) = parse_lopdf_font_encoding(doc, font, page_number)? {
                        encodings.insert(font_name, PageFontEncoding::Lopdf(encoding));
                    }
                    continue;
                };
                if let Some(cmap) = self.load_to_unicode_cmap(doc, to_unicode_raw, page_number)? {
                    encodings.insert(font_name, PageFontEncoding::ToUnicode(cmap));
                } else {
                    let mut stripped = font.clone();
                    stripped.remove(KEY_TO_UNICODE);
                    fallback_dicts.push((font_name, stripped));
                }
            }
        }

        let content_limit = self
            .remaining_decompressed
            .min(MAX_PDF_PAGE_DECOMPRESSED_BYTES);
        let content_bytes = match doc.get_page_content_with_limit(page_id, content_limit) {
            Ok(bytes) => bytes,
            Err(LopdfError::Decompress(DecompressError::MemoryLimitExceeded { .. })) => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "PDF page {page_number} decompressed content exceeded cumulative {MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES}-byte ceiling"
                    ),
                });
            }
            Err(_) => return Ok(String::new()),
        };

        if content_bytes.len() > self.remaining_decompressed {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF page {page_number} decompressed content exceeded cumulative {MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES}-byte ceiling"
                ),
            });
        }
        self.remaining_decompressed = self
            .remaining_decompressed
            .saturating_sub(content_bytes.len());

        for (font_name, stripped) in &fallback_dicts {
            if let Some(encoding) = parse_lopdf_font_encoding(doc, stripped, page_number)? {
                encodings.insert(font_name.clone(), PageFontEncoding::Lopdf(encoding));
            }
        }

        decode_bounded_page_content(&content_bytes, &encodings, page_number)
    }

    fn load_to_unicode_cmap(
        &mut self,
        doc: &Document,
        to_unicode_raw: &Object,
        page_number: u32,
    ) -> Result<Option<Rc<ForwardToUnicodeCMap>>, AdfontesError> {
        let Ok((resolved_id, resolved_object)) = doc.dereference(to_unicode_raw) else {
            return Ok(None);
        };
        let Object::Stream(stream) = resolved_object else {
            return Ok(None);
        };
        if let Some(cmap_id) = resolved_id
            && let Some(cached) = self.cached_cmaps.get(&cmap_id)
        {
            return Ok(cached.as_ref().map(Rc::clone));
        }

        let parsed =
            parse_to_unicode_stream(stream, page_number, &mut self.remaining_decompressed)?;
        if let Some(cmap_id) = resolved_id {
            self.cached_cmaps
                .insert(cmap_id, parsed.as_ref().map(Rc::clone));
        }
        Ok(parsed)
    }
}

enum PageFontEncoding<'a> {
    ToUnicode(Rc<ForwardToUnicodeCMap>),
    Lopdf(Encoding<'a>),
}

fn parse_to_unicode_stream(
    stream: &lopdf::Stream,
    page_number: u32,
    remaining_decompressed: &mut usize,
) -> Result<Option<Rc<ForwardToUnicodeCMap>>, AdfontesError> {
    let (byte_len, decompressed) = match stream.get_plain_content_with_limit(MAX_PDF_CMAP_BYTES) {
        Ok(data) => (data.len(), Some(data)),
        Err(LopdfError::Decompress(DecompressError::MemoryLimitExceeded { .. })) => {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF page {page_number} font ToUnicode CMap exceeded {MAX_PDF_CMAP_BYTES}-byte safety ceiling"
                ),
            });
        }
        Err(_) => (stream.content.len(), None),
    };
    if byte_len > *remaining_decompressed {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "PDF page {page_number} font ToUnicode CMap decompression exceeded cumulative {MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES}-byte ceiling"
            ),
        });
    }
    *remaining_decompressed = remaining_decompressed.saturating_sub(byte_len);

    let Some(data) = decompressed else {
        return Ok(None);
    };
    Ok(ForwardToUnicodeCMap::parse(&data, page_number)?.map(Rc::new))
}

fn parse_lopdf_font_encoding<'a>(
    doc: &'a Document,
    font: &'a Dictionary,
    page_number: u32,
) -> Result<Option<Encoding<'a>>, AdfontesError> {
    match font.get_font_encoding_with_limit(doc, MAX_PDF_CMAP_BYTES) {
        Ok(encoding) => Ok(Some(encoding)),
        Err(LopdfError::Decompress(DecompressError::MemoryLimitExceeded { .. })) => {
            Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF page {page_number} font encoding stream exceeded {MAX_PDF_CMAP_BYTES}-byte safety ceiling"
                ),
            })
        }
        Err(_) => Ok(None),
    }
}

fn decode_bounded_page_content(
    content_bytes: &[u8],
    encodings: &BTreeMap<Vec<u8>, PageFontEncoding<'_>>,
    page_number: u32,
) -> Result<String, AdfontesError> {
    let mut out = String::new();
    let mut decoded_input_bytes: usize = 0;
    let mut current_encoding: Option<&PageFontEncoding<'_>> = None;

    for op_result in Content::decode_iter(content_bytes) {
        let Ok(operation) = op_result else {
            break;
        };
        match operation.operator.as_ref() {
            "Tf" => {
                current_encoding = operation
                    .operands
                    .first()
                    .and_then(|operand| operand.as_name().ok())
                    .and_then(|font_name| encodings.get(font_name));
            }
            "Tj" | "TJ" | "'" | "\"" => {
                let operator = operation.operator.as_ref();
                if matches!(operator, "'" | "\"") && !out.ends_with('\n') {
                    out.push('\n');
                }
                let operands = if operator == "\"" {
                    operation
                        .operands
                        .get(2)
                        .map_or(&[][..], std::slice::from_ref)
                } else {
                    operation.operands.as_slice()
                };
                if let Some(encoding) = current_encoding
                    && collect_bounded_text(
                        encoding,
                        operands,
                        &mut out,
                        &mut decoded_input_bytes,
                        page_number,
                        0,
                    )? == CollectStep::Truncated
                {
                    break;
                }
            }
            "T*" | "ET" if !out.ends_with('\n') => out.push('\n'),
            _ => {}
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CollectStep {
    Continue,
    Truncated,
}

fn collect_bounded_text(
    encoding: &PageFontEncoding<'_>,
    operands: &[Object],
    out: &mut String,
    decoded_input_bytes: &mut usize,
    page_number: u32,
    depth: usize,
) -> Result<CollectStep, AdfontesError> {
    for operand in operands {
        if collect_bounded_operand(
            encoding,
            operand,
            out,
            decoded_input_bytes,
            page_number,
            depth,
        )? == CollectStep::Truncated
        {
            return Ok(CollectStep::Truncated);
        }
    }
    Ok(CollectStep::Continue)
}

fn collect_bounded_operand(
    encoding: &PageFontEncoding<'_>,
    operand: &Object,
    out: &mut String,
    decoded_input_bytes: &mut usize,
    page_number: u32,
    depth: usize,
) -> Result<CollectStep, AdfontesError> {
    match operand {
        Object::String(bytes, _) => {
            decode_bounded_string(encoding, bytes, out, decoded_input_bytes, page_number)
        }
        Object::Array(items) if depth < MAX_PDF_TEXT_ARRAY_DEPTH => {
            if collect_bounded_text(
                encoding,
                items,
                out,
                decoded_input_bytes,
                page_number,
                depth + 1,
            )? == CollectStep::Truncated
            {
                return Ok(CollectStep::Truncated);
            }
            out.push(' ');
            Ok(CollectStep::Continue)
        }
        Object::Integer(kerning) if *kerning < -100 => {
            out.push(' ');
            Ok(CollectStep::Continue)
        }
        _ => Ok(CollectStep::Continue),
    }
}

fn decode_bounded_string(
    encoding: &PageFontEncoding<'_>,
    bytes: &[u8],
    out: &mut String,
    decoded_input_bytes: &mut usize,
    page_number: u32,
) -> Result<CollectStep, AdfontesError> {
    match encoding {
        PageFontEncoding::ToUnicode(cmap) => {
            let outcome = cmap.decode_bounded(bytes, out, MAX_PDF_PAGE_RAW_TEXT_BYTES);
            *decoded_input_bytes = decoded_input_bytes.saturating_add(outcome.input_bytes_consumed);
            if outcome.truncated {
                if out.len() > decoded_input_bytes.saturating_mul(MAX_CMAP_AMPLIFICATION_RATIO) {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "PDF page {page_number} text decoding amplified output past {MAX_CMAP_AMPLIFICATION_RATIO}x and {MAX_PDF_PAGE_RAW_TEXT_BYTES}-byte ceiling"
                        ),
                    });
                }
                return Ok(CollectStep::Truncated);
            }
        }
        PageFontEncoding::Lopdf(lopdf_encoding) => {
            for chunk in bytes.chunks(SIMPLE_DECODE_CHUNK_BYTES) {
                let _ = lopdf_encoding.write_to_string(chunk, out);
                *decoded_input_bytes = decoded_input_bytes.saturating_add(chunk.len());
                if out.len() > MAX_PDF_PAGE_RAW_TEXT_BYTES {
                    return Ok(CollectStep::Truncated);
                }
            }
        }
    }
    Ok(CollectStep::Continue)
}

fn materialize_inherited_page_attributes(doc: &mut Document, page_id: ObjectId) {
    let mut inherited: Vec<(&[u8], Object)> = Vec::new();
    let Ok(page) = doc.get_dictionary(page_id) else {
        return;
    };

    for &key in INHERITABLE_PAGE_KEYS {
        if page.has(key) {
            continue;
        }
        if let Some(found) = lookup_inherited_key(doc, page_id, key) {
            inherited.push((key, found));
        }
    }

    if let Ok(Object::Dictionary(page_mut)) = doc.get_object_mut(page_id) {
        for (key, value) in inherited {
            page_mut.set(key, value);
        }
    }
}

fn lookup_inherited_key(doc: &Document, start_id: ObjectId, key: &[u8]) -> Option<Object> {
    let mut visited = BTreeSet::new();
    let mut current_id = start_id;
    for _ in 0..MAX_PARENT_TREE_DEPTH {
        if !visited.insert(current_id) {
            break;
        }
        let current_node = doc.get_dictionary(current_id).ok()?;
        let parent_ref = current_node.get(b"Parent").ok()?;
        let Object::Reference(parent_id) = *parent_ref else {
            break;
        };
        let parent = doc.get_dictionary(parent_id).ok()?;
        if let Ok(value) = parent.get(key) {
            return Some(value.clone());
        }
        current_id = parent_id;
    }
    None
}

fn strip_cross_page_page_keys(doc: &mut Document, page_id: ObjectId) {
    if let Ok(Object::Dictionary(page_mut)) = doc.get_object_mut(page_id) {
        for &key in STRIPPED_PAGE_KEYS {
            page_mut.remove(key);
        }
    }
}

fn extract_outline_titles(doc: &Document) -> Vec<String> {
    let Ok(catalog) = doc.catalog() else {
        return Vec::new();
    };
    let Ok(outlines_ref) = catalog.get(b"Outlines") else {
        return Vec::new();
    };
    let Some(outlines) = resolve_dictionary(doc, outlines_ref) else {
        return Vec::new();
    };
    let Some(first_id) = outlines
        .get(KEY_FIRST)
        .ok()
        .and_then(|obj| obj.as_reference().ok())
    else {
        return Vec::new();
    };

    let mut titles = Vec::new();
    let mut visited = BTreeSet::new();
    let mut stack: Vec<(ObjectId, usize)> = vec![(first_id, 0)];

    while let Some((node_id, depth)) = stack.pop() {
        if titles.len() >= MAX_OUTLINE_ITEMS
            || depth > MAX_OUTLINE_DEPTH
            || !visited.insert(node_id)
        {
            continue;
        }
        let Ok(node) = doc.get_dictionary(node_id) else {
            continue;
        };
        if let Ok(raw_title) = node.get(b"Title")
            && let Ok((_, resolved)) = doc.dereference(raw_title)
            && let Ok(decoded) = decode_text_string(resolved)
        {
            let trimmed = decoded.trim();
            if !trimmed.is_empty() {
                titles.push(trimmed.to_owned());
            }
        }
        // Push `/Next` first so `/First` children are popped and visited immediately in pre-order.
        if let Ok(Object::Reference(next_id)) = node.get(b"Next") {
            stack.push((*next_id, depth));
        }
        if depth < MAX_OUTLINE_DEPTH
            && let Ok(Object::Reference(child_id)) = node.get(KEY_FIRST)
        {
            stack.push((*child_id, depth + 1));
        }
    }

    titles
}

fn resolve_dictionary<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Dictionary> {
    match obj {
        Object::Dictionary(table) => Some(table),
        Object::Reference(id) => doc.get_dictionary(*id).ok(),
        _ => None,
    }
}

const fn unindexed_page_summary(page_number: NonZeroU32) -> PdfPageSummary {
    PdfPageSummary {
        page_number,
        text: PdfPageText::Unindexed,
    }
}

fn normalize_pdf_page_text(raw: &str, max_bytes: usize) -> String {
    if max_bytes == 0 || raw.is_empty() {
        return String::new();
    }
    let mut expanded = String::with_capacity(raw.len().min(max_bytes));
    let mut chars = raw.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    let _ = chars.next();
                }
                expanded.push('\n');
            }
            '\x0c' => expanded.push('\n'),
            '\u{00AD}' => {}
            '\u{FB00}' => expanded.push_str("ff"),
            '\u{FB01}' => expanded.push_str("fi"),
            '\u{FB02}' => expanded.push_str("fl"),
            '\u{FB03}' => expanded.push_str("ffi"),
            '\u{FB04}' => expanded.push_str("ffl"),
            '\u{FB05}' | '\u{FB06}' => expanded.push_str("st"),
            '\n' | '\t' => expanded.push(character),
            other if other.is_control() => {}
            other => expanded.push(other),
        }
    }

    let mut normalized = String::with_capacity(expanded.len().min(max_bytes));
    let mut consecutive_blanks: usize = 0;
    for line in expanded.split('\n') {
        if normalized.len() >= max_bytes {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            if normalized.is_empty() {
                continue;
            }
            consecutive_blanks += 1;
            if consecutive_blanks <= 1 {
                normalized.push('\n');
            }
            continue;
        }
        consecutive_blanks = 0;
        if !normalized.is_empty() {
            normalized.push('\n');
        }
        let remaining = max_bytes.saturating_sub(normalized.len());
        normalized.push_str(truncate_utf8_bytes(trimmed, remaining));
    }

    let bounded = truncate_utf8_bytes(&normalized, max_bytes);
    bounded.trim_end().to_owned()
}

fn summarize_page_text(page_number: NonZeroU32, text: &str, char_count: usize) -> PdfPageSummary {
    let mut headings = Vec::new();
    let mut preview = String::new();

    for raw_line in text.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if preview.is_empty() {
            preview = truncate_chars(trimmed, MAX_PREVIEW_CHARS);
        }
        if headings.len() < MAX_HEADINGS_PER_PAGE && looks_like_pdf_heading(trimmed) {
            headings.push(truncate_chars(trimmed, MAX_HEADING_LEN));
        }
    }

    PdfPageSummary {
        page_number,
        text: PdfPageText::Indexed {
            char_count,
            headings,
            preview,
        },
    }
}

fn count_non_whitespace_chars(input: &str) -> usize {
    input
        .chars()
        .filter(|character| !character.is_whitespace())
        .count()
}

fn looks_like_pdf_heading(line: &str) -> bool {
    let len = line.chars().count();
    if !(MIN_HEADING_LEN..=MAX_HEADING_LEN).contains(&len) {
        return false;
    }
    let lower = line.to_ascii_lowercase();
    let starts_section_keyword = [
        "abstract",
        "introduction",
        "background",
        "method",
        "experiment",
        "result",
        "discussion",
        "conclusion",
        "appendix",
        "references",
        "theorem",
        "lemma",
        "proposition",
        "figure ",
        "table ",
        "algorithm ",
    ]
    .iter()
    .any(|keyword| lower.starts_with(keyword));

    starts_section_keyword || starts_like_numbered_heading(line)
}

fn starts_like_numbered_heading(line: &str) -> bool {
    if line.ends_with('.') {
        return false;
    }
    let Some((prefix, rest)) = line.split_once(char::is_whitespace) else {
        return false;
    };
    let clean_prefix = prefix.trim_end_matches('.');
    let valid_number = !clean_prefix.is_empty()
        && clean_prefix
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()));
    let starts_capitalized = rest
        .trim_start()
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_uppercase());
    valid_number && starts_capitalized
}

/// Builds a synthetic multi-page PDF with `page_texts` for cross-module tests.
#[cfg(test)]
#[must_use]
pub fn build_test_pdf(page_texts: &[&str]) -> Vec<u8> {
    tests::build_inherited_multipage_pdf(page_texts)
}

/// Builds a synthetic PDF containing a compressed `ToUnicode` `CMap` bomb for cross-module tests.
#[cfg(test)]
#[must_use]
pub fn build_test_cmap_bomb_pdf() -> Vec<u8> {
    tests::build_cmap_bomb_pdf()
}

#[cfg(test)]
mod tests {
    use lopdf::content::{Content, Operation};
    use lopdf::{Dictionary, Document, Object, Stream, StringFormat};

    use super::*;

    /// Builds a hermetic multi-page PDF whose `/MediaBox` and `/Resources` live ONLY on the
    /// parent `/Pages` node and whose `Catalog` has nested `/Outlines`, `/StructTreeRoot`,
    /// `/OpenAction`, and stale trailer `/Prev`.
    pub(super) fn build_inherited_multipage_pdf(page_texts: &[&str]) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_root_id = doc.new_object_id();

        let font_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Font".to_vec())),
            ("Subtype", Object::Name(b"Type1".to_vec())),
            ("BaseFont", Object::Name(b"Helvetica".to_vec())),
        ]));
        let resources = Dictionary::from_iter([(
            "Font",
            Object::Dictionary(Dictionary::from_iter([("F1", Object::Reference(font_id))])),
        )]);

        let mut kid_refs = Vec::new();
        for text in page_texts {
            let content = Content {
                operations: vec![
                    Operation::new("BT", vec![]),
                    Operation::new(
                        "Tf",
                        vec![Object::Name(b"F1".to_vec()), Object::Integer(12)],
                    ),
                    Operation::new("Td", vec![Object::Integer(72), Object::Integer(720)]),
                    Operation::new("Tj", vec![Object::string_literal(*text)]),
                    Operation::new("ET", vec![]),
                ],
            };
            let stream_id =
                doc.add_object(Stream::new(Dictionary::new(), content.encode().unwrap()));
            let page_id = doc.add_object(Dictionary::from_iter([
                ("Type", Object::Name(b"Page".to_vec())),
                ("Parent", Object::Reference(pages_root_id)),
                ("Contents", Object::Reference(stream_id)),
            ]));
            kid_refs.push(Object::Reference(page_id));
        }

        let pages_node = Dictionary::from_iter([
            ("Type", Object::Name(b"Pages".to_vec())),
            ("Kids", Object::Array(kid_refs.clone())),
            (
                "Count",
                Object::Integer(i64::try_from(page_texts.len()).unwrap()),
            ),
            ("Resources", Object::Dictionary(resources)),
            (
                "MediaBox",
                Object::Array(vec![
                    Object::Integer(0),
                    Object::Integer(0),
                    Object::Integer(612),
                    Object::Integer(792),
                ]),
            ),
        ]);
        doc.objects
            .insert(pages_root_id, Object::Dictionary(pages_node));

        // Indirect PDFDocEncoding title (`é` = 0xE9) nested under parent outline item.
        let child_title_ref = doc.add_object(Object::String(
            vec![b'1', b'.', b'1', b' ', b'R', 0xE9, b's', b'u', b'm', 0xE9],
            StringFormat::Literal,
        ));
        let child_outline = doc.add_object(Dictionary::from_iter([(
            "Title",
            Object::Reference(child_title_ref),
        )]));
        let outline_item = doc.add_object(Dictionary::from_iter([
            (
                "Title",
                Object::String(b"1 Introduction".to_vec(), StringFormat::Literal),
            ),
            ("First", Object::Reference(child_outline)),
            ("Last", Object::Reference(child_outline)),
            (
                "Dest",
                Object::Array(vec![kid_refs[0].clone(), Object::Name(b"Fit".to_vec())]),
            ),
        ]));
        let outlines_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Outlines".to_vec())),
            ("First", Object::Reference(outline_item)),
            ("Last", Object::Reference(outline_item)),
            ("Count", Object::Integer(2)),
        ]));
        let struct_root = doc.add_object(Dictionary::from_iter([(
            "Type",
            Object::Name(b"StructTreeRoot".to_vec()),
        )]));

        let catalog_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Catalog".to_vec())),
            ("Pages", Object::Reference(pages_root_id)),
            ("Outlines", Object::Reference(outlines_id)),
            ("StructTreeRoot", Object::Reference(struct_root)),
            (
                "OpenAction",
                Object::Array(vec![kid_refs[0].clone(), Object::Name(b"Fit".to_vec())]),
            ),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog_id));
        doc.trailer.set("Prev", Object::Integer(12345));

        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn indexes_pages_nested_outlines_and_pdfdoc_encoding_in_pure_rust() {
        let pdf_bytes = build_inherited_multipage_pdf(&[
            "1 Introduction overview",
            "2 Methods and derivations",
            "3 Results summary",
        ]);
        let indexed = index_pdf_bytes(&pdf_bytes).unwrap();
        let PdfPageText::Indexed { ref preview, .. } = indexed.index.pages[1].text else {
            panic!("expected indexed page 2");
        };
        assert_eq!(indexed.index.page_count, 3);
        assert_eq!(
            indexed.index.outline_titles,
            vec!["1 Introduction", "1.1 Résumé"]
        );
        assert!(preview.contains("2 Methods"));
        assert!(
            indexed.page_texts[1]
                .text()
                .contains("2 Methods and derivations")
        );
    }

    #[test]
    fn distinguishes_numbered_pdf_headings_from_numeric_rows_and_prose() {
        assert!(!looks_like_pdf_heading("12 34 56 78"));
        assert!(!looks_like_pdf_heading("100 samples were tested without"));
        assert!(looks_like_pdf_heading("2.1 Experimental Setup"));
    }

    #[test]
    fn slices_pdf_preserving_inherited_mediabox_and_resources() {
        let pdf_bytes = build_inherited_multipage_pdf(&[
            "Page One",
            "Page Two Target",
            "Page Three Target",
            "Page Four",
        ]);
        let span = PageSpan::new(2, 3).unwrap();
        let sliced = slice_pdf_bytes(&pdf_bytes, span).unwrap();
        let reloaded = Document::load_mem(&sliced.bytes).unwrap();
        let retained_pages = reloaded.get_pages();
        let first_retained = reloaded
            .get_dictionary(*retained_pages.get(&1).unwrap())
            .unwrap();

        assert_eq!(sliced.total_pages, 4);
        assert_eq!(retained_pages.len(), 2);
        assert!(first_retained.has(b"MediaBox"));
        assert!(first_retained.has(b"Resources"));
    }

    #[test]
    fn strips_cross_page_catalog_structures_and_stale_trailer_keys() {
        let pdf_bytes = build_inherited_multipage_pdf(&["Page One", "Page Two", "Page Three"]);
        let sliced = slice_pdf_bytes(&pdf_bytes, PageSpan::new(2, 2).unwrap()).unwrap();
        let reloaded = Document::load_mem(&sliced.bytes).unwrap();
        let catalog = reloaded.catalog().unwrap();

        assert!(!catalog.has(b"Outlines"));
        assert!(!catalog.has(b"StructTreeRoot"));
        assert!(!catalog.has(b"OpenAction"));
        assert!(!reloaded.trailer.has(b"Prev"));
    }

    #[test]
    fn slices_nested_pages_tree_with_parent_cycle_without_hanging_or_empty_nodes() {
        let mut doc = Document::with_version("1.5");
        let root_pages = doc.new_object_id();
        let left_branch = doc.new_object_id();
        let right_branch = doc.new_object_id();

        let p1 = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Page".to_vec())),
            ("Parent", Object::Reference(left_branch)),
        ]));
        let p2 = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Page".to_vec())),
            ("Parent", Object::Reference(right_branch)),
        ]));

        doc.objects.insert(
            left_branch,
            Object::Dictionary(Dictionary::from_iter([
                ("Type", Object::Name(b"Pages".to_vec())),
                ("Parent", Object::Reference(root_pages)),
                ("Kids", Object::Array(vec![Object::Reference(p1)])),
                ("Count", Object::Integer(1)),
            ])),
        );
        doc.objects.insert(
            right_branch,
            Object::Dictionary(Dictionary::from_iter([
                ("Type", Object::Name(b"Pages".to_vec())),
                ("Parent", Object::Reference(root_pages)),
                ("Kids", Object::Array(vec![Object::Reference(p2)])),
                ("Count", Object::Integer(1)),
            ])),
        );
        // Deliberately introduce a `/Parent` self-cycle on root `/Pages`.
        doc.objects.insert(
            root_pages,
            Object::Dictionary(Dictionary::from_iter([
                ("Type", Object::Name(b"Pages".to_vec())),
                ("Parent", Object::Reference(root_pages)),
                (
                    "Kids",
                    Object::Array(vec![
                        Object::Reference(left_branch),
                        Object::Reference(right_branch),
                    ]),
                ),
                ("Count", Object::Integer(2)),
                (
                    "MediaBox",
                    Object::Array(vec![
                        Object::Integer(0),
                        Object::Integer(0),
                        Object::Integer(612),
                        Object::Integer(792),
                    ]),
                ),
            ])),
        );
        let catalog = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Catalog".to_vec())),
            ("Pages", Object::Reference(root_pages)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog));

        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();

        let sliced = slice_pdf_bytes(&bytes, PageSpan::new(2, 2).unwrap()).unwrap();
        let reloaded = Document::load_mem(&sliced.bytes).unwrap();
        let page_count = reloaded.get_pages().len();
        let zero_count_pages_nodes = reloaded
            .objects
            .values()
            .filter_map(|obj| obj.as_dict().ok())
            .filter(|dict| {
                dict.get_type().is_ok_and(|kind| kind == b"Pages")
                    && dict.get(b"Count").and_then(Object::as_i64).ok() == Some(0)
            })
            .count();
        assert_eq!(page_count, 1);
        assert_eq!(zero_count_pages_nodes, 0);
    }

    #[test]
    fn rejects_non_pdf_bytes() {
        assert!(matches!(
            index_pdf_bytes(b"<html>not a pdf</html>").unwrap_err(),
            AdfontesError::Extraction { .. }
        ));
    }

    #[test]
    fn rejects_inline_encrypt_dictionary() {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Pages".to_vec())),
            ("Kids", Object::Array(vec![])),
            ("Count", Object::Integer(0)),
        ]));
        let catalog = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Catalog".to_vec())),
            ("Pages", Object::Reference(pages_id)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog));
        doc.trailer.set(
            "Encrypt",
            Object::Dictionary(Dictionary::from_iter([
                ("Filter", Object::Name(b"Standard".to_vec())),
                ("V", Object::Integer(2)),
            ])),
        );
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();

        assert!(matches!(
            index_pdf_bytes(&bytes).unwrap_err(),
            AdfontesError::Extraction { .. }
        ));
    }

    #[test]
    fn blocks_compressed_pdf_page_content_bomb() {
        let mut doc = Document::with_version("1.5");
        let pages_root = doc.new_object_id();
        let huge_payload = vec![b' '; MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES + 4096];
        let mut stream = Stream::new(Dictionary::new(), huge_payload);
        stream.compress().unwrap();
        let stream_id = doc.add_object(stream);
        let page_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Page".to_vec())),
            ("Parent", Object::Reference(pages_root)),
            ("Contents", Object::Reference(stream_id)),
        ]));
        doc.objects.insert(
            pages_root,
            Object::Dictionary(Dictionary::from_iter([
                ("Type", Object::Name(b"Pages".to_vec())),
                ("Kids", Object::Array(vec![Object::Reference(page_id)])),
                ("Count", Object::Integer(1)),
            ])),
        );
        let catalog = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Catalog".to_vec())),
            ("Pages", Object::Reference(pages_root)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog));

        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();

        assert!(matches!(
            index_pdf_bytes(&bytes).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    pub(super) fn build_cmap_bomb_pdf() -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_root = doc.new_object_id();
        let huge_cmap = vec![b' '; MAX_PDF_CMAP_BYTES + 4096];
        let mut cmap_stream = Stream::new(Dictionary::new(), huge_cmap);
        cmap_stream.compress().unwrap();
        let cmap_id = doc.add_object(cmap_stream);
        let font_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Font".to_vec())),
            ("Subtype", Object::Name(b"Type1".to_vec())),
            ("ToUnicode", Object::Reference(cmap_id)),
        ]));
        let resources_id = doc.add_object(Dictionary::from_iter([(
            "Font",
            Object::Dictionary(Dictionary::from_iter([("F1", Object::Reference(font_id))])),
        )]));
        let content_id = doc.add_object(Stream::new(
            Dictionary::new(),
            b"BT /F1 12 Tf (hi) Tj ET".to_vec(),
        ));
        let page_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Page".to_vec())),
            ("Parent", Object::Reference(pages_root)),
            ("Resources", Object::Reference(resources_id)),
            ("Contents", Object::Reference(content_id)),
        ]));
        doc.objects.insert(
            pages_root,
            Object::Dictionary(Dictionary::from_iter([
                ("Type", Object::Name(b"Pages".to_vec())),
                ("Kids", Object::Array(vec![Object::Reference(page_id)])),
                ("Count", Object::Integer(1)),
            ])),
        );
        let catalog = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Catalog".to_vec())),
            ("Pages", Object::Reference(pages_root)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog));

        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn blocks_compressed_font_to_unicode_cmap_bomb() {
        let bytes = build_cmap_bomb_pdf();

        assert!(matches!(
            index_pdf_bytes(&bytes).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            slice_pdf_bytes(&bytes, PageSpan::new(1, 1).unwrap()).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn blocks_inherited_parent_font_chained_to_unicode_cmap_bomb() {
        let mut doc = Document::with_version("1.5");
        let pages_root = doc.new_object_id();
        let huge_cmap = vec![b' '; MAX_PDF_CMAP_BYTES + 4096];
        let mut cmap_stream = Stream::new(Dictionary::new(), huge_cmap);
        cmap_stream.compress().unwrap();
        let cmap_id = doc.add_object(cmap_stream);
        let chained_cmap_ref = doc.add_object(Object::Reference(cmap_id));
        let font_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Font".to_vec())),
            ("Subtype", Object::Name(b"Type1".to_vec())),
            ("ToUnicode", Object::Reference(chained_cmap_ref)),
        ]));
        let inherited_resources = doc.add_object(Dictionary::from_iter([(
            "Font",
            Object::Dictionary(Dictionary::from_iter([("F1", Object::Reference(font_id))])),
        )]));
        let content_id = doc.add_object(Stream::new(
            Dictionary::new(),
            b"BT /F1 12 Tf (hi) Tj ET".to_vec(),
        ));
        let page_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Page".to_vec())),
            ("Parent", Object::Reference(pages_root)),
            ("Resources", Object::Dictionary(Dictionary::new())),
            ("Contents", Object::Reference(content_id)),
        ]));
        doc.objects.insert(
            pages_root,
            Object::Dictionary(Dictionary::from_iter([
                ("Type", Object::Name(b"Pages".to_vec())),
                ("Kids", Object::Array(vec![Object::Reference(page_id)])),
                ("Count", Object::Integer(1)),
                ("Resources", Object::Reference(inherited_resources)),
            ])),
        );
        let catalog = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Catalog".to_vec())),
            ("Pages", Object::Reference(pages_root)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog));

        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();

        assert!(matches!(
            index_pdf_bytes(&bytes).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn normalizes_pdf_ligatures_controls_blank_lines_and_utf8_truncation() {
        let raw = [
            "",
            "  Ef\u{FB01}cient \u{FB02}ow\u{00AD}chart\x00  ",
            "",
            "",
            "\x0cNext\u{FB03}eld  ",
            "",
        ]
        .join("\r\n");
        let normalized = normalize_pdf_page_text(&raw, 1024);
        let multibyte = normalize_pdf_page_text("αβγδε\n   ", 5);
        let expected = ["  Efficient flowchart", "", "Nextffield"].join("\n");

        assert_eq!(normalized, expected);
        assert_eq!(multibyte, "αβ");
    }

    #[test]
    fn document_text_budget_marks_overflow_and_trailing_pages_unindexed_without_partial_truncation()
    {
        // Page 0 uses 20 B of a 25 B budget (leaving 5 B). Page 1 (42 B) overflows; Page 2 ("End",
        // 3 B <= 5 B) MUST still be marked unindexed so no non-contiguous hole is created.
        let pdf_bytes = build_inherited_multipage_pdf(&[
            "Alpha page body text",
            "Beta page exceeds remaining budget cleanly",
            "End",
        ]);
        let indexed = index_pdf_bytes_with_budget(&pdf_bytes, MAX_INDEXED_PDF_PAGES, 25).unwrap();
        let page_capped = index_pdf_bytes_with_budget(&pdf_bytes, 1, 1024).unwrap();

        assert_eq!(indexed.page_texts.len(), 1);
        assert_eq!(indexed.index.pages[1].text, PdfPageText::Unindexed);
        assert_eq!(indexed.index.pages[2].text, PdfPageText::Unindexed);
        assert_eq!(page_capped.index.pages[1].text, PdfPageText::Unindexed);
    }

    #[test]
    fn whitespace_only_page_reports_zero_non_whitespace_chars_while_indexed() {
        let pdf_bytes = build_inherited_multipage_pdf(&["     ", "Readable content here"]);
        let indexed = index_pdf_bytes(&pdf_bytes).unwrap();
        let unicode_ws = normalize_pdf_page_text("\u{00A0}\u{2003} \n\u{00A0}", 1024);

        assert!(matches!(
            indexed.index.pages[0].text,
            PdfPageText::Indexed { char_count: 0, .. }
        ));
        assert!(matches!(
            indexed.index.pages[1].text,
            PdfPageText::Indexed { char_count: 19, .. }
        ));
        assert_eq!(indexed.page_texts[0].text(), "");
        assert_eq!(unicode_ws, "");
    }

    #[test]
    fn slice_pdf_bytes_extracts_retained_pages_with_original_source_page_numbers() {
        let pdf_bytes = build_inherited_multipage_pdf(&[
            "1 Introduction",
            "2 Inherited Font Derivation",
            "3 Asymptotic Analysis",
            "4 Conclusion",
        ]);
        let sliced = slice_pdf_bytes(&pdf_bytes, PageSpan::new(2, 3).unwrap()).unwrap();

        assert_eq!(sliced.page_texts.len(), 2);
        assert_eq!(sliced.page_texts[0].page_number().get(), 2);
        assert!(
            sliced.page_texts[0]
                .text()
                .contains("Inherited Font Derivation")
        );
        assert_eq!(sliced.page_texts[1].page_number().get(), 3);
    }

    fn build_to_unicode_cmap_bytes(codespace: &str, body_lines: &[&str]) -> Vec<u8> {
        let body = body_lines.join("\n");
        format!(
            concat!(
                "/CIDInit /ProcSet findresource begin\n",
                "12 dict begin\n",
                "begincmap\n",
                "/CMapName /TestCMap def\n",
                "/CMapType 2 def\n",
                "1 begincodespacerange\n",
                "{codespace}\n",
                "endcodespacerange\n",
                "{body}\n",
                "endcmap\n",
                "CMapName currentdict /CMap defineresource pop\n",
                "end\n",
                "end\n",
            ),
            codespace = codespace,
            body = body,
        )
        .into_bytes()
    }

    fn single_font_tj_ops(font_name: &str, payload: Object) -> Vec<Operation> {
        vec![
            Operation::new("BT", vec![]),
            Operation::new(
                "Tf",
                vec![
                    Object::Name(font_name.as_bytes().to_vec()),
                    Object::Integer(12),
                ],
            ),
            Operation::new("Tj", vec![payload]),
            Operation::new("ET", vec![]),
        ]
    }

    fn save_single_page_pdf(
        doc: &mut Document,
        resources_id: ObjectId,
        operations: Vec<Operation>,
    ) -> Vec<u8> {
        let pages_root = doc.new_object_id();
        let encoded_ops = Content { operations }.encode().unwrap();
        let content_id = doc.add_object(Stream::new(Dictionary::new(), encoded_ops));
        let page_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Page".to_vec())),
            ("Parent", Object::Reference(pages_root)),
            ("Resources", Object::Reference(resources_id)),
            ("Contents", Object::Reference(content_id)),
        ]));
        doc.objects.insert(
            pages_root,
            Object::Dictionary(Dictionary::from_iter([
                ("Type", Object::Name(b"Pages".to_vec())),
                ("Kids", Object::Array(vec![Object::Reference(page_id)])),
                ("Count", Object::Integer(1)),
            ])),
        );
        let catalog = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Catalog".to_vec())),
            ("Pages", Object::Reference(pages_root)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog));

        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    fn build_custom_cmap_pdf(
        cmap_body: &[u8],
        empty_filter: bool,
        encoding_hops: usize,
        operations: Vec<Operation>,
    ) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let mut stream_headers = Dictionary::new();
        if empty_filter {
            stream_headers.set("Filter", Object::Array(Vec::new()));
        }
        let cmap_id = doc.add_object(Stream::new(stream_headers, cmap_body.to_vec()));
        let mut encoding = Object::Name(b"Identity-H".to_vec());
        for _ in 0..encoding_hops {
            let ref_id = doc.add_object(encoding);
            encoding = Object::Reference(ref_id);
        }
        let font_id = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Font".to_vec())),
            ("Subtype", Object::Name(b"Type0".to_vec())),
            ("Encoding", encoding),
            ("ToUnicode", Object::Reference(cmap_id)),
        ]));
        let resources_id = doc.add_object(Dictionary::from_iter([(
            "Font",
            Object::Dictionary(Dictionary::from_iter([("F1", Object::Reference(font_id))])),
        )]));
        save_single_page_pdf(&mut doc, resources_id, operations)
    }

    fn build_shared_cmap_pages_pdf(cmap_body: &[u8], page_count: usize) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_root = doc.new_object_id();
        let cmap_id = doc.add_object(Stream::new(Dictionary::new(), cmap_body.to_vec()));
        let mut page_ids = Vec::with_capacity(page_count);
        for index in 0..page_count {
            let font_name = format!("F{index}");
            let font_id = doc.add_object(Dictionary::from_iter([
                ("Type", Object::Name(b"Font".to_vec())),
                ("Subtype", Object::Name(b"Type0".to_vec())),
                ("Encoding", Object::Name(b"Identity-H".to_vec())),
                ("ToUnicode", Object::Reference(cmap_id)),
            ]));
            let resources_id = doc.add_object(Dictionary::from_iter([(
                "Font",
                Object::Dictionary(Dictionary::from_iter([(
                    font_name.as_str(),
                    Object::Reference(font_id),
                )])),
            )]));
            let encoded_ops = Content {
                operations: single_font_tj_ops(
                    &font_name,
                    Object::String(vec![0, 0], StringFormat::Hexadecimal),
                ),
            }
            .encode()
            .unwrap();
            let content_id = doc.add_object(Stream::new(Dictionary::new(), encoded_ops));
            let page_id = doc.add_object(Dictionary::from_iter([
                ("Type", Object::Name(b"Page".to_vec())),
                ("Parent", Object::Reference(pages_root)),
                ("Resources", Object::Reference(resources_id)),
                ("Contents", Object::Reference(content_id)),
            ]));
            page_ids.push(Object::Reference(page_id));
        }
        let count = i64::try_from(page_ids.len()).unwrap();
        doc.objects.insert(
            pages_root,
            Object::Dictionary(Dictionary::from_iter([
                ("Type", Object::Name(b"Pages".to_vec())),
                ("Kids", Object::Array(page_ids)),
                ("Count", Object::Integer(count)),
            ])),
        );
        let catalog = doc.add_object(Dictionary::from_iter([
            ("Type", Object::Name(b"Catalog".to_vec())),
            ("Pages", Object::Reference(pages_root)),
        ]));
        doc.trailer.set("Root", Object::Reference(catalog));

        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn decodes_wide_bfrange_and_single_element_bfrange_array_without_reverse_map_expansion() {
        let wide_cmap = build_to_unicode_cmap_bytes(
            "<00000000> <FFFFFFFF>",
            &[
                "1 beginbfrange",
                "<00000000> <FFFFFFFF> <0041>",
                "endbfrange",
            ],
        );
        let array_target = "0041".repeat(256);
        let array_line = format!("<0000> <FFFF> [<{array_target}>]");
        let single_array_cmap = build_to_unicode_cmap_bytes(
            "<0000> <FFFF>",
            &["1 beginbfrange", &array_line, "endbfrange"],
        );
        let wide_bytes = build_custom_cmap_pdf(
            &wide_cmap,
            false,
            0,
            single_font_tj_ops(
                "F1",
                Object::String(vec![0, 0, 0, 0, 0, 0, 0, 2], StringFormat::Hexadecimal),
            ),
        );
        let array_bytes = build_custom_cmap_pdf(
            &single_array_cmap,
            false,
            0,
            single_font_tj_ops(
                "F1",
                Object::String(vec![0x00, 0x01], StringFormat::Hexadecimal),
            ),
        );

        let wide_indexed = index_pdf_bytes(&wide_bytes).unwrap();
        let wide_sliced = slice_pdf_bytes(&wide_bytes, PageSpan::new(1, 1).unwrap()).unwrap();
        let array_indexed = index_pdf_bytes(&array_bytes).unwrap();
        let array_sliced = slice_pdf_bytes(&array_bytes, PageSpan::new(1, 1).unwrap()).unwrap();

        assert_eq!(wide_indexed.page_texts[0].text(), "AC");
        assert_eq!(wide_sliced.page_texts[0].text(), "AC");
        assert!(array_indexed.page_texts[0].text().ends_with("AAB"));
        assert!(array_sliced.page_texts[0].text().ends_with("AAB"));
    }

    #[test]
    fn blocks_to_unicode_single_and_split_op_output_amplification_bombs() {
        let huge_target = "0041".repeat(256);
        let char_line = format!("<41> <{huge_target}>");
        let fanout_cmap =
            build_to_unicode_cmap_bytes("<00> <FF>", &["1 beginbfchar", &char_line, "endbfchar"]);
        let single_op_bomb = build_custom_cmap_pdf(
            &fanout_cmap,
            false,
            0,
            single_font_tj_ops("F1", Object::string_literal("A".repeat(40_000))),
        );
        let mut split_ops = single_font_tj_ops("F1", Object::string_literal("A".repeat(4_000)));
        split_ops.splice(
            3..3,
            [
                Operation::new("Tj", vec![Object::string_literal("A".repeat(4_000))]),
                Operation::new("Tj", vec![Object::string_literal("A".repeat(4_000))]),
            ],
        );
        let split_op_bomb = build_custom_cmap_pdf(&fanout_cmap, false, 0, split_ops);

        let single_index_error = index_pdf_bytes(&single_op_bomb).unwrap_err();
        let single_slice_error =
            slice_pdf_bytes(&single_op_bomb, PageSpan::new(1, 1).unwrap()).unwrap_err();
        let split_index_error = index_pdf_bytes(&split_op_bomb).unwrap_err();
        let split_slice_error =
            slice_pdf_bytes(&split_op_bomb, PageSpan::new(1, 1).unwrap()).unwrap_err();

        assert!(matches!(
            single_index_error,
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            single_slice_error,
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            split_index_error,
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            split_slice_error,
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn blocks_to_unicode_comment_smuggling_and_empty_filter_array_fanout_bombs() {
        let comment_rest = "0041".repeat(255);
        let continuation = format!(" {comment_rest}>");
        let comment_cmap = build_to_unicode_cmap_bytes(
            "<00> <FF>",
            &[
                "1 beginbfchar",
                "<41> <0041 % >",
                &continuation,
                "endbfchar",
            ],
        );
        let array_target = "0041".repeat(256);
        let array_line = format!("<0000> <FFFF> [<{array_target}>]");
        let array_cmap = build_to_unicode_cmap_bytes(
            "<0000> <FFFF>",
            &["1 beginbfrange", &array_line, "endbfrange"],
        );

        let comment_bomb = build_custom_cmap_pdf(
            &comment_cmap,
            false,
            0,
            single_font_tj_ops("F1", Object::string_literal("A".repeat(40_000))),
        );
        let filter_array_bomb = build_custom_cmap_pdf(
            &array_cmap,
            true,
            0,
            single_font_tj_ops(
                "F1",
                Object::String(vec![0u8; 40_000], StringFormat::Hexadecimal),
            ),
        );

        let comment_index_error = index_pdf_bytes(&comment_bomb).unwrap_err();
        let comment_slice_error =
            slice_pdf_bytes(&comment_bomb, PageSpan::new(1, 1).unwrap()).unwrap_err();
        let array_index_error = index_pdf_bytes(&filter_array_bomb).unwrap_err();
        let array_slice_error =
            slice_pdf_bytes(&filter_array_bomb, PageSpan::new(1, 1).unwrap()).unwrap_err();

        assert!(matches!(
            comment_index_error,
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            comment_slice_error,
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            array_index_error,
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            array_slice_error,
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn blocks_oversized_empty_filter_and_129_hop_encoding_to_unicode_bombs() {
        let oversized_raw_cmap = vec![b' '; MAX_PDF_CMAP_BYTES + 4096];
        let oversized_bomb = build_custom_cmap_pdf(
            &oversized_raw_cmap,
            true,
            0,
            single_font_tj_ops("F1", Object::string_literal("A".to_owned())),
        );
        let huge_target = "0041".repeat(256);
        let char_line = format!("<41> <{huge_target}>");
        let fanout_cmap =
            build_to_unicode_cmap_bytes("<00> <FF>", &["1 beginbfchar", &char_line, "endbfchar"]);
        let encoding_chain_bomb = build_custom_cmap_pdf(
            &fanout_cmap,
            false,
            129,
            single_font_tj_ops("F1", Object::string_literal("A".repeat(40_000))),
        );

        let oversized_index_error = index_pdf_bytes(&oversized_bomb).unwrap_err();
        let oversized_slice_error =
            slice_pdf_bytes(&oversized_bomb, PageSpan::new(1, 1).unwrap()).unwrap_err();
        let chain_index_error = index_pdf_bytes(&encoding_chain_bomb).unwrap_err();
        let chain_slice_error =
            slice_pdf_bytes(&encoding_chain_bomb, PageSpan::new(1, 1).unwrap()).unwrap_err();

        assert!(matches!(
            oversized_index_error,
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            oversized_slice_error,
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            chain_index_error,
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            chain_slice_error,
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn extracts_to_unicode_ascii_ligatures_and_caches_shared_cmap_across_pages() {
        let ligature_cmap = build_to_unicode_cmap_bytes(
            "<00> <FF>",
            &[
                "2 beginbfchar",
                "<0E> <006600660069>",
                "<41> <0041>",
                "endbfchar",
            ],
        );
        let ligature_bytes = build_custom_cmap_pdf(
            &ligature_cmap,
            false,
            0,
            single_font_tj_ops(
                "F1",
                Object::String(vec![0x41, 0x0E], StringFormat::Hexadecimal),
            ),
        );
        // Pad the single shared CMap stream to ~900 KiB (< MAX_PDF_CMAP_BYTES) across 40 pages
        // so uncached re-parsing would exceed the 32 MiB cumulative decompression budget.
        let cmap_padding = format!("% {}", "x".repeat(900 * 1024));
        let shared_cmap = build_to_unicode_cmap_bytes(
            "<0000> <FFFF>",
            &[
                &cmap_padding,
                "1 beginbfrange",
                "<0000> <FFFF> <0041>",
                "endbfrange",
            ],
        );
        let shared_bytes = build_shared_cmap_pages_pdf(&shared_cmap, 40);

        let ligature_indexed = index_pdf_bytes(&ligature_bytes).unwrap();
        let ligature_sliced =
            slice_pdf_bytes(&ligature_bytes, PageSpan::new(1, 1).unwrap()).unwrap();
        let shared_indexed = index_pdf_bytes(&shared_bytes).unwrap();

        assert_eq!(ligature_indexed.page_texts[0].text(), "Affi");
        assert_eq!(ligature_sliced.page_texts[0].text(), "Affi");
        assert_eq!(shared_indexed.page_texts.len(), 40);
        assert_eq!(shared_indexed.page_texts[39].text(), "A");
    }
}
