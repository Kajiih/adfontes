//! Pure-Rust PDF page indexing and lossless vector page slicing powered by `lopdf`.
//!
//! Eliminates external C++ `poppler` requirements while ensuring sliced PDFs materialize
//! ancestor-inherited `/Pages` attributes (`Resources`, `MediaBox`, `CropBox`, `Rotate`), build a
//! fresh spec-clean flat `/Pages` tree without `lopdf::Document::delete_pages` `/Parent`-cycle
//! hangs or empty intermediate `/Pages` nodes, strip cross-page `Catalog`/`Page`/trailer keys, and
//! enforce cumulative decompression budgets during both loading and indexing.

architecture_component!(Pdf);

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use lopdf::{
    DecompressError, Dictionary, Document, Error as LopdfError, LoadOptions, Object, ObjectId,
    decode_text_string,
};

use crate::model::{
    AdfontesError, PageSpan, PdfIndex, PdfPageSummary, PdfPageText, truncate_chars,
    truncate_utf8_bytes,
};

const PDF_FORMAT_LABEL: &str = "PDF";
const KEY_FIRST: &[u8] = b"First";
const PDF_MAGIC: &[u8] = b"%PDF-";
const PDF_MAGIC_SCAN_WINDOW: usize = 1024;
const MAX_PDF_LOAD_DECOMPRESSED_BYTES: usize = 16 * 1024 * 1024;
const MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES: usize = 32 * 1024 * 1024;
const MAX_PDF_PAGE_DECOMPRESSED_BYTES: usize = 2 * 1024 * 1024;
const MAX_PDF_CMAP_BYTES: usize = 1024 * 1024;
const MAX_PDF_PAGE_TEXT_BYTES: usize = 512 * 1024;
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

/// Lossless vector PDF slice emitted by [`slice_pdf_bytes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlicedPdf {
    /// Total page count of the source PDF.
    pub total_pages: u32,
    /// Effective clamped 1-indexed page span present in [`Self::bytes`].
    pub effective_span: PageSpan,
    /// Validated standalone PDF bytes for the retained pages.
    pub bytes: Vec<u8>,
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

/// Indexes a PDF's page count, nested outline bookmarks, and bounded per-page text hints.
///
/// Uses pure-Rust `lopdf` under strict decompression and font-CMap byte budgets without spawning
/// external binaries.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if a PDF decompression bomb exceeds memory budgets, or
/// [`AdfontesError::Extraction`] if `bytes` is not a readable PDF or requires a user password.
pub fn index_pdf_bytes(bytes: &[u8]) -> Result<PdfIndex, AdfontesError> {
    index_pdf_bytes_with_budget(bytes, MAX_INDEXED_PDF_PAGES, MAX_PDF_DOCUMENT_TEXT_BYTES)
}

/// Losslessly slices an inclusive [`PageSpan`] (`1..=8` pages) into a self-contained PDF.
///
/// Materializes inherited `/Pages` attributes (`Resources`, `MediaBox`, `CropBox`, `Rotate`) onto
/// each retained `/Page`, wires them into a fresh flat `/Pages` root (avoiding
/// `lopdf::Document::delete_pages` `/Parent`-cycle hangs and empty intermediate `/Pages` nodes),
/// strips cross-page `Catalog`/page/trailer state, prunes unreachable objects, and serializes.
///
/// # Errors
///
/// Returns [`AdfontesError::InvalidTarget`] if `span` starts past the PDF's page count,
/// [`AdfontesError::SafetyBlock`] on decompression bombs, or [`AdfontesError::Extraction`] if the PDF
/// is malformed or user-password-encrypted.
pub fn slice_pdf_bytes(raw_bytes: &[u8], span: PageSpan) -> Result<SlicedPdf, AdfontesError> {
    let mut doc = load_decrypted_pdf(raw_bytes)?;
    let page_ids = doc.get_pages();
    let total_pages = u32::try_from(page_ids.len()).unwrap_or(u32::MAX);
    let effective_span = span.clamp_to_total(total_pages)?;

    let mut retained_ids = Vec::new();
    for (&page_number, &page_id) in &page_ids {
        if page_number >= effective_span.start() && page_number <= effective_span.end() {
            materialize_inherited_page_attributes(&mut doc, page_id);
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
    })
}

fn index_pdf_bytes_with_budget(
    bytes: &[u8],
    max_indexed_pages: usize,
    max_document_text_bytes: usize,
) -> Result<PdfIndex, AdfontesError> {
    let mut doc = load_decrypted_pdf(bytes)?;
    let page_ids = doc.get_pages();
    let page_count = u32::try_from(page_ids.len()).unwrap_or(u32::MAX);
    let outline_titles = extract_outline_titles(&doc);

    let mut extraction_budget = PdfExtractionBudget::new();
    let mut remaining_text_bytes = max_document_text_bytes;
    let mut pages = Vec::with_capacity(page_ids.len());

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
        pages.push(summarize_page_text(page_nz, &normalized));
    }

    Ok(PdfIndex {
        page_count,
        outline_titles,
        pages,
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
    cmap_sizes: BTreeMap<ObjectId, usize>,
}

impl PdfExtractionBudget {
    const fn new() -> Self {
        Self {
            remaining_decompressed: MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES,
            cmap_sizes: BTreeMap::new(),
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

        charge_page_font_cmaps(
            doc,
            page_number,
            page_id,
            &mut self.remaining_decompressed,
            &mut self.cmap_sizes,
        )?;

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
        let page_budget = self
            .remaining_decompressed
            .min(MAX_PDF_PAGE_DECOMPRESSED_BYTES);
        self.remaining_decompressed = self
            .remaining_decompressed
            .saturating_sub(content_bytes.len());

        match doc.extract_text_with_limit(&[page_number], page_budget) {
            Ok(text) => Ok(text),
            Err(LopdfError::Decompress(DecompressError::MemoryLimitExceeded { .. })) => {
                Err(AdfontesError::SafetyBlock {
                    reason: format!(
                        "PDF page {page_number} text extraction exceeded {MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES}-byte ceiling"
                    ),
                })
            }
            Err(_) => Ok(String::new()),
        }
    }
}

fn charge_page_font_cmaps(
    doc: &Document,
    page_number: u32,
    page_id: ObjectId,
    remaining_decompressed: &mut usize,
    cmap_sizes: &mut BTreeMap<ObjectId, usize>,
) -> Result<(), AdfontesError> {
    let Ok(fonts) = doc.get_page_fonts(page_id) else {
        return Ok(());
    };

    for font in fonts.into_values() {
        let Ok(to_unicode_raw) = font.get(b"ToUnicode") else {
            continue;
        };
        let Ok((resolved_id, resolved_object)) = doc.dereference(to_unicode_raw) else {
            continue;
        };
        let Object::Stream(stream) = resolved_object else {
            continue;
        };
        let cmap_len = if let Some(cmap_id) = resolved_id {
            if let Some(&cached_len) = cmap_sizes.get(&cmap_id) {
                cached_len
            } else {
                let measured = measure_direct_stream_bytes(stream, page_number)?;
                cmap_sizes.insert(cmap_id, measured);
                measured
            }
        } else {
            measure_direct_stream_bytes(stream, page_number)?
        };
        if cmap_len > *remaining_decompressed {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF page {page_number} font ToUnicode CMap decompression exceeded cumulative {MAX_PDF_INDEX_TOTAL_DECOMPRESSED_BYTES}-byte ceiling"
                ),
            });
        }
        *remaining_decompressed = remaining_decompressed.saturating_sub(cmap_len);
    }
    Ok(())
}

fn measure_direct_stream_bytes(
    stream: &lopdf::Stream,
    page_number: u32,
) -> Result<usize, AdfontesError> {
    match stream.decompressed_content_with_limit(MAX_PDF_CMAP_BYTES) {
        Ok(data) => Ok(data.len()),
        Err(LopdfError::Decompress(DecompressError::MemoryLimitExceeded { .. })) => {
            Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF page {page_number} font ToUnicode CMap exceeded {MAX_PDF_CMAP_BYTES}-byte safety ceiling"
                ),
            })
        }
        Err(_) => Ok(stream.content.len()),
    }
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

fn summarize_page_text(page_number: NonZeroU32, text: &str) -> PdfPageSummary {
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
            char_count: count_non_whitespace_chars(text),
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
        let index = index_pdf_bytes(&pdf_bytes).unwrap();
        let PdfPageText::Indexed { ref preview, .. } = index.pages[1].text else {
            panic!("expected indexed page 2");
        };
        assert_eq!(index.page_count, 3);
        assert_eq!(index.outline_titles, vec!["1 Introduction", "1.1 Résumé"]);
        assert!(preview.contains("2 Methods"));
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

    #[test]
    fn blocks_compressed_font_to_unicode_cmap_bomb() {
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

        assert!(matches!(
            index_pdf_bytes(&bytes).unwrap_err(),
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

        assert!(matches!(
            indexed.pages[0].text,
            PdfPageText::Indexed { char_count: 17, .. }
        ));
        assert_eq!(indexed.pages[1].text, PdfPageText::Unindexed);
        assert_eq!(indexed.pages[2].text, PdfPageText::Unindexed);
        assert_eq!(page_capped.pages[1].text, PdfPageText::Unindexed);
    }

    #[test]
    fn whitespace_only_page_reports_zero_non_whitespace_chars_while_indexed() {
        let pdf_bytes = build_inherited_multipage_pdf(&["     ", "Readable content here"]);
        let indexed = index_pdf_bytes(&pdf_bytes).unwrap();
        let unicode_ws = normalize_pdf_page_text("\u{00A0}\u{2003} \n\u{00A0}", 1024);

        assert!(matches!(
            indexed.pages[0].text,
            PdfPageText::Indexed { char_count: 0, .. }
        ));
        assert!(matches!(
            indexed.pages[1].text,
            PdfPageText::Indexed { char_count: 19, .. }
        ));
        assert_eq!(unicode_ws, "");
    }
}
