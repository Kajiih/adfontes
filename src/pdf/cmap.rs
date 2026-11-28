//! Forward-only bounded `/ToUnicode` `CMap` parser and streaming UTF-8 decoder.
//!
//! Replaces `lopdf`'s `ToUnicodeCMap` parser on untrusted PDFs so wide `bfrange` declarations stay
//! compact intervals in `O(1)` memory without eagerly materializing a `reverse_map` `HashMap`,
//! while bounding decoded character output per call.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::model::AdfontesError;

const MAX_CMAP_CODESPACE_RANGES: usize = 256;
const MAX_CMAP_TARGET_UTF16_UNITS: usize = 256;
const MAX_CMAP_CODE_BYTES: u8 = 4;
const MAX_CMAP_CODE_HEX_DIGITS: usize = 8;
const HEX_DIGITS_PER_OCTET: usize = 2;
const HEX_DIGITS_PER_UTF16_UNIT: usize = 4;
const BITS_PER_HEX_NIBBLE: u32 = 4;
const HEX_ALPHA_OFFSET: u8 = 10;

/// Streaming outcome returned by [`ForwardToUnicodeCMap::decode_bounded`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DecodeOutcome {
    pub(super) input_bytes_consumed: usize,
    pub(super) truncated: bool,
}

/// Bounded forward-only `/ToUnicode` mapping table.
#[derive(Debug)]
pub(super) struct ForwardToUnicodeCMap {
    codespace_ranges: Vec<CodespaceRange>,
    mappings: [BTreeMap<u32, RangeEntry>; MAX_CMAP_CODE_BYTES as usize],
}

impl ForwardToUnicodeCMap {
    /// Parses a `/ToUnicode` `CMap` stream into bounded forward intervals without expanding
    /// `bfrange` definitions or building a reverse map.
    ///
    /// Returns `Ok(None)` when the stream defines no `bfchar` or `bfrange` mappings, or
    /// `Err(AdfontesError::SafetyBlock)` if codespace or target-string limits are exceeded.
    pub(super) fn parse(data: &[u8], page_number: u32) -> Result<Option<Self>, AdfontesError> {
        let mut cursor = 0usize;
        let mut state = CmapBuildState::new();

        while let Some(token) =
            next_cmap_token(data, &mut cursor, state.expects_target(), page_number)?
        {
            state.advance(token, page_number)?;
        }
        Ok(state.into_cmap())
    }

    /// Decodes a PDF string operand into `out`, stopping cleanly once `out.len() > max_out_bytes`.
    pub(super) fn decode_bounded(
        &self,
        bytes: &[u8],
        out: &mut String,
        max_out_bytes: usize,
    ) -> DecodeOutcome {
        let mut code_len: u8 = 0;
        let mut code: u32 = 0;

        for (index, &byte) in bytes.iter().enumerate() {
            if code_len == MAX_CMAP_CODE_BYTES {
                self.emit_or_replace(code, code_len, out);
                code = 0;
                code_len = 0;
                if out.len() > max_out_bytes {
                    return DecodeOutcome {
                        input_bytes_consumed: index,
                        truncated: true,
                    };
                }
            }

            code_len += 1;
            code = (code << u8::BITS) | u32::from(byte);
            if self.try_emit_matched_code(code, code_len, out) {
                code = 0;
                code_len = 0;
                if out.len() > max_out_bytes {
                    return DecodeOutcome {
                        input_bytes_consumed: index + 1,
                        truncated: true,
                    };
                }
            }
        }

        if code_len > 0 {
            self.emit_or_replace(code, code_len, out);
            if out.len() > max_out_bytes {
                return DecodeOutcome {
                    input_bytes_consumed: bytes.len(),
                    truncated: true,
                };
            }
        }

        DecodeOutcome {
            input_bytes_consumed: bytes.len(),
            truncated: false,
        }
    }

    fn try_emit_matched_code(&self, code: u32, code_len: u8, out: &mut String) -> bool {
        if self.codespace_ranges.is_empty() {
            if let Some(entry) = self.lookup_mapping(code, code_len) {
                emit_entry(entry, code, out);
                return true;
            }
            return false;
        }
        if self
            .codespace_ranges
            .iter()
            .any(|&range| matches_codespace_range(range, code, code_len))
        {
            self.emit_or_replace(code, code_len, out);
            return true;
        }
        false
    }

    fn emit_or_replace(&self, code: u32, code_len: u8, out: &mut String) {
        if let Some(entry) = self.lookup_mapping(code, code_len) {
            emit_entry(entry, code, out);
        } else {
            out.push(char::REPLACEMENT_CHARACTER);
        }
    }

    fn lookup_mapping(&self, code: u32, code_len: u8) -> Option<&RangeEntry> {
        let slot = usize::from(code_len).checked_sub(1)?;
        let map = self.mappings.get(slot)?;
        map.range(..=code)
            .next_back()
            .map(|(_, entry)| entry)
            .filter(|entry| code <= entry.end)
    }
}

struct CmapBuildState {
    section: CmapSection,
    codespace_ranges: Vec<CodespaceRange>,
    mappings: [BTreeMap<u32, RangeEntry>; MAX_CMAP_CODE_BYTES as usize],
    has_mappings: bool,
}

impl CmapBuildState {
    fn new() -> Self {
        Self {
            section: CmapSection::Outside,
            codespace_ranges: Vec::new(),
            mappings: Default::default(),
            has_mappings: false,
        }
    }

    const fn expects_target(&self) -> bool {
        matches!(
            self.section,
            CmapSection::BfChar(BfCharPhase::ExpectDst { .. })
                | CmapSection::BfRange(
                    BfRangePhase::ExpectDstOrArray { .. } | BfRangePhase::InArray { .. }
                )
        )
    }

    fn advance(&mut self, token: CmapToken<'_>, page_number: u32) -> Result<(), AdfontesError> {
        if let CmapToken::Word(word) = token {
            match word {
                b"begincodespacerange" => {
                    self.section = CmapSection::Codespace(CodespacePhase::ExpectLow);
                }
                b"beginbfchar" => self.section = CmapSection::BfChar(BfCharPhase::ExpectSrc),
                b"beginbfrange" => self.section = CmapSection::BfRange(BfRangePhase::ExpectStart),
                b"endcodespacerange" | b"endbfchar" | b"endbfrange" => {
                    self.section = CmapSection::Outside;
                }
                _ => {}
            }
            return Ok(());
        }

        match (&mut self.section, token) {
            (CmapSection::Codespace(CodespacePhase::ExpectLow), CmapToken::Hex(hex)) => {
                self.section = CmapSection::Codespace(CodespacePhase::ExpectHigh { low: hex.code });
            }
            (CmapSection::Codespace(CodespacePhase::ExpectHigh { low }), CmapToken::Hex(hex)) => {
                if let (Some(low_code), Some(high_code)) = (*low, hex.code) {
                    self.push_codespace(low_code, high_code, page_number)?;
                }
                self.section = CmapSection::Codespace(CodespacePhase::ExpectLow);
            }
            (CmapSection::BfChar(BfCharPhase::ExpectSrc), CmapToken::Hex(hex)) => {
                self.section = CmapSection::BfChar(BfCharPhase::ExpectDst { src: hex.code });
            }
            (CmapSection::BfChar(BfCharPhase::ExpectDst { src }), CmapToken::Hex(hex)) => {
                if let (Some(src_code), Some(units)) = (*src, hex.target_units) {
                    self.insert_range(
                        CmapCodeRange {
                            start: src_code.value,
                            end: src_code.value,
                            code_len: src_code.len,
                        },
                        BfTarget::Incrementing(units),
                    );
                }
                self.section = CmapSection::BfChar(BfCharPhase::ExpectSrc);
            }
            (CmapSection::BfRange(_), other) => self.advance_bfrange(other),
            _ => {}
        }
        Ok(())
    }

    fn advance_bfrange(&mut self, token: CmapToken<'_>) {
        let CmapSection::BfRange(phase) = &mut self.section else {
            return;
        };
        match (phase, token) {
            (BfRangePhase::ExpectStart, CmapToken::Hex(hex)) => {
                self.section = CmapSection::BfRange(BfRangePhase::ExpectEnd { start: hex.code });
            }
            (BfRangePhase::ExpectEnd { start }, CmapToken::Hex(hex)) => {
                let bounds = match (*start, hex.code) {
                    (Some(start_code), Some(end_code))
                        if start_code.len == end_code.len && start_code.value <= end_code.value =>
                    {
                        Some(CmapCodeRange {
                            start: start_code.value,
                            end: end_code.value,
                            code_len: start_code.len,
                        })
                    }
                    _ => None,
                };
                self.section = CmapSection::BfRange(BfRangePhase::ExpectDstOrArray { bounds });
            }
            (BfRangePhase::ExpectDstOrArray { bounds }, CmapToken::Hex(hex)) => {
                if let (Some(range), Some(units)) = (*bounds, hex.target_units) {
                    self.insert_range(range, BfTarget::Incrementing(units));
                }
                self.section = CmapSection::BfRange(BfRangePhase::ExpectStart);
            }
            (BfRangePhase::ExpectDstOrArray { bounds }, CmapToken::ArrayOpen) => {
                self.section = CmapSection::BfRange(BfRangePhase::InArray {
                    bounds: *bounds,
                    targets: Vec::new(),
                    valid: bounds.is_some(),
                });
            }
            (BfRangePhase::InArray { targets, valid, .. }, CmapToken::Hex(hex)) => {
                if let Some(units) = hex.target_units {
                    if *valid {
                        targets.push(units);
                    }
                } else {
                    *valid = false;
                }
            }
            (
                BfRangePhase::InArray {
                    bounds,
                    targets,
                    valid,
                },
                CmapToken::ArrayClose,
            ) => {
                let mut collected = std::mem::take(targets);
                if *valid
                    && !collected.is_empty()
                    && let Some(range) = *bounds
                {
                    let target = if collected.len() == 1 {
                        // Match `lopdf` 0.47.0's convention of treating single-element `bfrange`
                        // arrays (`[<0058>]`) as incrementing bases across the source range.
                        BfTarget::Incrementing(collected.swap_remove(0))
                    } else {
                        BfTarget::Array(collected)
                    };
                    self.insert_range(range, target);
                }
                self.section = CmapSection::BfRange(BfRangePhase::ExpectStart);
            }
            _ => {}
        }
    }

    fn push_codespace(
        &mut self,
        low: CmapCode,
        high: CmapCode,
        page_number: u32,
    ) -> Result<(), AdfontesError> {
        if low.len != high.len || low.value > high.value {
            return Ok(());
        }
        if self.codespace_ranges.len() >= MAX_CMAP_CODESPACE_RANGES {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "PDF page {page_number} font ToUnicode CMap exceeded {MAX_CMAP_CODESPACE_RANGES} codespace ranges"
                ),
            });
        }
        self.codespace_ranges.push(CodespaceRange {
            low: low.value,
            high: high.value,
            code_len: low.len,
        });
        Ok(())
    }

    fn insert_range(&mut self, range: CmapCodeRange, target: BfTarget) {
        if let Some(slot) = usize::from(range.code_len).checked_sub(1)
            && let Some(map) = self.mappings.get_mut(slot)
        {
            insert_range_interval(map, range.start, range.end, target);
            self.has_mappings = true;
        }
    }

    fn into_cmap(self) -> Option<ForwardToUnicodeCMap> {
        self.has_mappings.then_some(ForwardToUnicodeCMap {
            codespace_ranges: self.codespace_ranges,
            mappings: self.mappings,
        })
    }
}

enum CmapSection {
    Outside,
    Codespace(CodespacePhase),
    BfChar(BfCharPhase),
    BfRange(BfRangePhase),
}

enum CodespacePhase {
    ExpectLow,
    ExpectHigh { low: Option<CmapCode> },
}

enum BfCharPhase {
    ExpectSrc,
    ExpectDst { src: Option<CmapCode> },
}

enum BfRangePhase {
    ExpectStart,
    ExpectEnd {
        start: Option<CmapCode>,
    },
    ExpectDstOrArray {
        bounds: Option<CmapCodeRange>,
    },
    InArray {
        bounds: Option<CmapCodeRange>,
        targets: Vec<Vec<u16>>,
        valid: bool,
    },
}

enum CmapToken<'a> {
    Word(&'a [u8]),
    ArrayOpen,
    ArrayClose,
    Hex(HexToken),
    Other,
}

struct HexToken {
    code: Option<CmapCode>,
    target_units: Option<Vec<u16>>,
}

#[derive(Debug, Clone, Copy)]
struct CmapCode {
    value: u32,
    len: u8,
}

#[derive(Debug, Clone, Copy)]
struct CmapCodeRange {
    start: u32,
    end: u32,
    code_len: u8,
}

#[derive(Debug, Clone, Copy)]
struct CodespaceRange {
    low: u32,
    high: u32,
    code_len: u8,
}

#[derive(Debug)]
struct RangeEntry {
    end: u32,
    orig_start: u32,
    target: Rc<BfTarget>,
}

#[derive(Debug)]
enum BfTarget {
    Incrementing(Vec<u16>),
    Array(Vec<Vec<u16>>),
}

fn emit_entry(entry: &RangeEntry, code: u32, out: &mut String) {
    match entry.target.as_ref() {
        BfTarget::Incrementing(units) => {
            let offset =
                u16::try_from((code - entry.orig_start) & u32::from(u16::MAX)).unwrap_or(0);
            emit_utf16_with_last_offset(units, offset, out);
        }
        BfTarget::Array(targets) => {
            let target = usize::try_from(code - entry.orig_start)
                .ok()
                .and_then(|idx| targets.get(idx));
            if let Some(units) = target {
                emit_utf16_with_last_offset(units, 0, out);
            } else {
                out.push(char::REPLACEMENT_CHARACTER);
            }
        }
    }
}

fn emit_utf16_with_last_offset(units: &[u16], last_offset: u16, out: &mut String) {
    let Some((&last, prefix)) = units.split_last() else {
        return;
    };
    let adjusted_last = last.wrapping_add(last_offset);
    let iter = prefix.iter().copied().chain(std::iter::once(adjusted_last));
    for decoded in char::decode_utf16(iter) {
        out.push(decoded.unwrap_or(char::REPLACEMENT_CHARACTER));
    }
}

fn matches_codespace_range(range: CodespaceRange, code: u32, code_len: u8) -> bool {
    if range.code_len != code_len {
        return false;
    }
    let start_slot = usize::from(MAX_CMAP_CODE_BYTES - code_len);
    let low_bytes = range.low.to_be_bytes();
    let high_bytes = range.high.to_be_bytes();
    let code_bytes = code.to_be_bytes();
    (start_slot..usize::from(MAX_CMAP_CODE_BYTES))
        .all(|idx| low_bytes[idx] <= code_bytes[idx] && code_bytes[idx] <= high_bytes[idx])
}

fn insert_range_interval(
    map: &mut BTreeMap<u32, RangeEntry>,
    start: u32,
    end: u32,
    target: BfTarget,
) {
    let mut right_tail: Option<(u32, RangeEntry)> = None;
    if let Some((_, prev_entry)) = map.range_mut(..start).next_back()
        && prev_entry.end >= start
    {
        if prev_entry.end > end {
            right_tail = Some((
                end + 1,
                RangeEntry {
                    end: prev_entry.end,
                    orig_start: prev_entry.orig_start,
                    target: Rc::clone(&prev_entry.target),
                },
            ));
        }
        prev_entry.end = start - 1;
    }

    let covered_keys: Vec<u32> = map.range(start..=end).map(|(&key, _)| key).collect();
    for key in covered_keys {
        if let Some(removed) = map.remove(&key)
            && removed.end > end
        {
            right_tail = Some((end + 1, removed));
        }
    }

    map.insert(
        start,
        RangeEntry {
            end,
            orig_start: start,
            target: Rc::new(target),
        },
    );
    if let Some((tail_start, tail_entry)) = right_tail {
        map.insert(tail_start, tail_entry);
    }
}

fn next_cmap_token<'a>(
    data: &'a [u8],
    cursor: &mut usize,
    collect_target: bool,
    page_number: u32,
) -> Result<Option<CmapToken<'a>>, AdfontesError> {
    skip_cmap_whitespace_and_comments(data, cursor);
    let Some(&byte) = data.get(*cursor) else {
        return Ok(None);
    };
    let token = match byte {
        b'[' => {
            *cursor += 1;
            CmapToken::ArrayOpen
        }
        b']' => {
            *cursor += 1;
            CmapToken::ArrayClose
        }
        b'(' => {
            skip_cmap_literal_string(data, cursor);
            CmapToken::Other
        }
        b'<' => {
            *cursor += 1;
            if data.get(*cursor) == Some(&b'<') {
                *cursor += 1;
                CmapToken::Other
            } else {
                CmapToken::Hex(scan_cmap_hex_token(
                    data,
                    cursor,
                    collect_target,
                    page_number,
                )?)
            }
        }
        b'>' => {
            *cursor += 1;
            if data.get(*cursor) == Some(&b'>') {
                *cursor += 1;
            }
            CmapToken::Other
        }
        other if is_pdf_delimiter(other) => {
            *cursor += 1;
            CmapToken::Other
        }
        _ => {
            let start = *cursor;
            while let Some(&next) = data.get(*cursor) {
                if is_pdf_whitespace(next) || is_pdf_delimiter(next) {
                    break;
                }
                *cursor += 1;
            }
            CmapToken::Word(&data[start..*cursor])
        }
    };
    Ok(Some(token))
}

fn skip_cmap_whitespace_and_comments(data: &[u8], cursor: &mut usize) {
    while let Some(&byte) = data.get(*cursor) {
        if is_pdf_whitespace(byte) {
            *cursor += 1;
        } else if byte == b'%' {
            *cursor += 1;
            skip_cmap_comment(data, cursor);
        } else {
            break;
        }
    }
}

fn skip_cmap_literal_string(data: &[u8], cursor: &mut usize) {
    *cursor += 1;
    let mut paren_depth: usize = 1;
    while let Some(&byte) = data.get(*cursor) {
        *cursor += 1;
        match byte {
            b'\\' => {
                if *cursor < data.len() {
                    *cursor += 1;
                }
            }
            b'(' => paren_depth += 1,
            b')' => {
                paren_depth -= 1;
                if paren_depth == 0 {
                    break;
                }
            }
            _ => {}
        }
    }
}

fn scan_cmap_hex_token(
    data: &[u8],
    cursor: &mut usize,
    collect_target: bool,
    page_number: u32,
) -> Result<HexToken, AdfontesError> {
    let mut digit_count: usize = 0;
    let mut raw_code: u32 = 0;
    let mut valid_code_hex = true;
    let mut current_unit: u16 = 0;
    let mut unit_nibbles: usize = 0;
    let mut target_units = collect_target.then(Vec::new);
    let mut valid_target = collect_target;

    while let Some(&byte) = data.get(*cursor) {
        *cursor += 1;
        if byte == b'>' {
            break;
        }
        if let Some(nibble) = hex_nibble(byte) {
            if digit_count < MAX_CMAP_CODE_HEX_DIGITS {
                raw_code = (raw_code << BITS_PER_HEX_NIBBLE) | u32::from(nibble);
            }
            digit_count += 1;
            if let Some(units) = target_units.as_mut() {
                current_unit = (current_unit << BITS_PER_HEX_NIBBLE) | u16::from(nibble);
                unit_nibbles += 1;
                if unit_nibbles == HEX_DIGITS_PER_UTF16_UNIT {
                    if units.len() >= MAX_CMAP_TARGET_UTF16_UNITS {
                        return Err(AdfontesError::SafetyBlock {
                            reason: format!(
                                "PDF page {page_number} font ToUnicode CMap target exceeded {MAX_CMAP_TARGET_UTF16_UNITS}-unit ceiling"
                            ),
                        });
                    }
                    units.push(current_unit);
                    current_unit = 0;
                    unit_nibbles = 0;
                }
            }
        } else if byte == b'%' {
            valid_code_hex = false;
            skip_cmap_comment(data, cursor);
        } else if !is_pdf_whitespace(byte) {
            valid_code_hex = false;
            valid_target = false;
        }
    }
    if unit_nibbles > 0 {
        valid_target = false;
    }

    let code = if valid_code_hex
        && digit_count > 0
        && digit_count <= MAX_CMAP_CODE_HEX_DIGITS
        && digit_count.is_multiple_of(HEX_DIGITS_PER_OCTET)
    {
        u8::try_from(digit_count / HEX_DIGITS_PER_OCTET)
            .ok()
            .map(|len| CmapCode {
                value: raw_code,
                len,
            })
    } else {
        None
    };
    let target_units = target_units.filter(|units| valid_target && !units.is_empty());
    Ok(HexToken { code, target_units })
}

fn skip_cmap_comment(data: &[u8], cursor: &mut usize) {
    while let Some(&comment) = data.get(*cursor) {
        *cursor += 1;
        if matches!(comment, b'\n' | b'\r') {
            break;
        }
    }
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' | b'A'..=b'F' => Some(byte.to_ascii_lowercase() - b'a' + HEX_ALPHA_OFFSET),
        _ => None,
    }
}

const fn is_pdf_whitespace(byte: u8) -> bool {
    matches!(byte, b'\x00' | b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
}

const fn is_pdf_delimiter(byte: u8) -> bool {
    matches!(
        byte,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_cmap(codespace: &str, body_lines: &[&str]) -> Vec<u8> {
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

    #[test]
    fn parses_wide_4byte_bfrange_in_constant_memory_and_decodes_offsets() {
        let cmap_bytes = build_cmap(
            "<00000000> <FFFFFFFF>",
            &[
                "1 beginbfrange",
                "<00000000> <FFFFFFFF> <0041>",
                "endbfrange",
            ],
        );
        let cmap = ForwardToUnicodeCMap::parse(&cmap_bytes, 1)
            .unwrap()
            .unwrap();
        let mut out = String::new();
        let outcome = cmap.decode_bounded(&[0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 25], &mut out, 64);

        assert_eq!(out, "ACZ");
        assert!(!outcome.truncated);
        assert_eq!(outcome.input_bytes_consumed, 12);
    }

    #[test]
    fn later_bfrange_and_bfchar_override_earlier_ranges_without_losing_tail_offsets() {
        let cmap_bytes = build_cmap(
            "<00> <FF>",
            &[
                "2 beginbfrange",
                "<10> <20> <0041>",
                "<12> <14> <0061>",
                "endbfrange",
                "1 beginbfchar",
                "<16> <007A>",
                "endbfchar",
            ],
        );
        let cmap = ForwardToUnicodeCMap::parse(&cmap_bytes, 1)
            .unwrap()
            .unwrap();
        let mut out = String::new();
        let _ = cmap.decode_bounded(&[0x10, 0x11, 0x12, 0x13, 0x15, 0x16, 0x17], &mut out, 64);

        assert_eq!(out, "ABabFzH");
    }

    #[test]
    fn decodes_surrogate_pairs_ligatures_arrays_and_comment_smuggled_targets() {
        let cmap_bytes = build_cmap(
            "<00> <FF>",
            &[
                "3 beginbfchar",
                "<01> <D83DDE00>",
                "<02> <006600660069>",
                "<03> <0041 % >",
                " 0042>",
                "endbfchar",
                "2 beginbfrange",
                "<10> <11> [<0058>]",
                "<20> <21> [<0030> <0031>]",
                "endbfrange",
            ],
        );
        let cmap = ForwardToUnicodeCMap::parse(&cmap_bytes, 1)
            .unwrap()
            .unwrap();
        let mut out = String::new();
        let _ = cmap.decode_bounded(&[0x01, 0x02, 0x03, 0x10, 0x11, 0x20, 0x21], &mut out, 64);

        assert_eq!(out, "\u{1F600}ffiABXY01");
    }

    #[test]
    fn rejects_excessive_codespace_ranges_and_oversized_target_strings() {
        let many_ranges = std::iter::repeat_n("<00> <FF>", MAX_CMAP_CODESPACE_RANGES + 1)
            .collect::<Vec<_>>()
            .join("\n");
        let count_line = format!("{} begincodespacerange", MAX_CMAP_CODESPACE_RANGES + 1);
        let codespace_bomb = [count_line.as_str(), &many_ranges, "endcodespacerange"].join("\n");
        let huge_target = "0041".repeat(MAX_CMAP_TARGET_UTF16_UNITS + 1);
        let target_line = format!("<41> <{huge_target}>");
        let target_bomb = build_cmap("<00> <FF>", &["1 beginbfchar", &target_line, "endbfchar"]);

        let codespace_error =
            ForwardToUnicodeCMap::parse(codespace_bomb.as_bytes(), 1).unwrap_err();
        let target_error = ForwardToUnicodeCMap::parse(&target_bomb, 1).unwrap_err();

        assert!(matches!(codespace_error, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(target_error, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn decode_bounded_truncates_with_exact_consumed_input_byte_count() {
        let fanout = "0041".repeat(16);
        let entry = format!("<41> <{fanout}>");
        let cmap_bytes = build_cmap("<00> <FF>", &["1 beginbfchar", &entry, "endbfchar"]);
        let cmap = ForwardToUnicodeCMap::parse(&cmap_bytes, 1)
            .unwrap()
            .unwrap();
        let mut out = String::new();
        let outcome = cmap.decode_bounded(&[0x41; 10], &mut out, 20);

        assert!(outcome.truncated);
        assert_eq!(outcome.input_bytes_consumed, 2);
        assert_eq!(out.len(), 32);
    }
}
