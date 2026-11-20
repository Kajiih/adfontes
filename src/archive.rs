//! Bounded pure-Rust decompression and `tar` archive extraction helpers.
//!
//! Isolates `flate2` (`miniz_oxide` `rust_backend`) and `tar` behind strict decompression
//! ceilings, entry-count caps, duplicate-path rejection, and path-traversal guards so both
//! HTTP wire decoding (`Transport`) and `arXiv` TeX archive unpacking (`Extract`) share one
//! auditable safety boundary.

architecture_component!(Archive);

use std::collections::BTreeSet;
use std::io::{Cursor, Read};
use std::path::{Component, Path};

use flate2::read::{DeflateDecoder, MultiGzDecoder, ZlibDecoder};
use tar::{Archive as TarArchive, EntryType};

use crate::model::AdfontesError;

/// Default maximum uncompressed size allowed for `arXiv` TeX archives (`40 MiB`).
pub const DEFAULT_MAX_ARCHIVE_BYTES: usize = 40 * 1024 * 1024;

/// Default maximum regular file count allowed inside a `tar` archive.
pub const DEFAULT_MAX_TAR_FILES: usize = 250;

const ARCHIVE_FORMAT_LABEL: &str = "Archive";
const INVALID_UTF8_PATH_ERROR: &str = "tar entry path is not valid UTF-8";
const GZIP_MAGIC: [u8; 2] = [0x1F, 0x8B];
const XZ_MAGIC: [u8; 6] = [0xFD, b'7', b'z', b'X', b'Z', 0x00];
const BZIP2_MAGIC: [u8; 3] = *b"BZh";
const READ_CHUNK_SIZE: usize = 16 * 1024;
const TAR_MAX_ENTRY_MULTIPLIER: usize = 4;
const TAR_MIN_INSPECTED_ENTRIES_CAP: usize = 1024;

/// A single validated regular file extracted in memory from a `tar` archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedTarFile {
    /// Normalized relative POSIX path (no leading `/`, no `..`, no backslashes).
    pub path: String,
    /// Uncompressed file contents.
    pub bytes: Vec<u8>,
}

/// Returns `true` when `bytes` starts with the RFC 1952 gzip magic bytes (`1F 8B`).
#[must_use]
pub fn is_gzip_magic(bytes: &[u8]) -> bool {
    bytes.starts_with(&GZIP_MAGIC)
}

/// Returns the unsupported archive format label if `bytes` begins with `.xz` or `.bz2` magic.
#[must_use]
pub fn rejected_archive_format(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&XZ_MAGIC) {
        Some("xz")
    } else if bytes.starts_with(&BZIP2_MAGIC) {
        Some("bzip2")
    } else {
        None
    }
}

/// Decompresses a single- or multi-member gzip stream while enforcing a strict `max_bytes` cap.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the decompressed output exceeds `max_bytes`, or
/// [`AdfontesError::Extraction`] if the gzip stream is malformed or truncated.
pub fn safe_gunzip_bytes(compressed: &[u8], max_bytes: usize) -> Result<Vec<u8>, AdfontesError> {
    let decoder = MultiGzDecoder::new(compressed);
    read_bounded(decoder, max_bytes, "gzip")
}

/// Decompresses an HTTP `Content-Encoding: deflate` stream (RFC 1950 zlib or RFC 1951 raw
/// deflate) while enforcing a strict `max_bytes` cap.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the decompressed output exceeds `max_bytes`, or
/// [`AdfontesError::Extraction`] if neither zlib nor raw deflate decoding succeeds.
pub fn safe_inflate_bytes(compressed: &[u8], max_bytes: usize) -> Result<Vec<u8>, AdfontesError> {
    match read_bounded(ZlibDecoder::new(compressed), max_bytes, "deflate") {
        Ok(bytes) => Ok(bytes),
        Err(safety @ AdfontesError::SafetyBlock { .. }) => Err(safety),
        Err(_) => read_bounded(DeflateDecoder::new(compressed), max_bytes, "raw deflate"),
    }
}

/// Safely unpacks an uncompressed or gzip-compressed `tar` archive into memory.
///
/// Rejects `.xz`/`.bz2` archives, oversized uncompressed tar streams, symlinks, hardlinks,
/// special device or sparse entries, malformed V7 extension headers, excessive directory/global
/// entry storms, duplicate normalized paths, absolute paths, parent-directory traversal (`..`),
/// excessive file counts, and cumulative byte bombs.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] when any security or resource ceiling is tripped, or
/// [`AdfontesError::Extraction`] when the `tar` header/stream is malformed.
pub fn safe_extract_tar_bytes(
    raw_bytes: &[u8],
    max_total_bytes: usize,
    max_files: usize,
) -> Result<Vec<ExtractedTarFile>, AdfontesError> {
    if let Some(format_name) = rejected_archive_format(raw_bytes) {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("unsupported compressed archive format `{format_name}` is blocked"),
        });
    }

    let decompressed_owner;
    let tar_slice: &[u8] = if is_gzip_magic(raw_bytes) {
        decompressed_owner = safe_gunzip_bytes(raw_bytes, max_total_bytes)?;
        &decompressed_owner
    } else {
        if raw_bytes.len() > max_total_bytes {
            return Err(AdfontesError::SafetyBlock {
                reason: format!("uncompressed tar archive exceeded {max_total_bytes}-byte limit"),
            });
        }
        raw_bytes
    };

    let mut archive = TarArchive::new(Cursor::new(tar_slice));
    let entries = archive
        .entries()
        .map_err(|io_error| AdfontesError::Extraction {
            format: ARCHIVE_FORMAT_LABEL,
            detail: format!("failed to read tar entries: {io_error}"),
        })?;

    let max_inspected_entries = max_files
        .saturating_mul(TAR_MAX_ENTRY_MULTIPLIER)
        .max(TAR_MIN_INSPECTED_ENTRIES_CAP);
    let mut inspected_entries: usize = 0;
    let mut extracted = Vec::new();
    let mut seen_paths = BTreeSet::new();
    let mut cumulative_bytes: usize = 0;

    for entry_result in entries {
        inspected_entries = inspected_entries.saturating_add(1);
        if inspected_entries > max_inspected_entries {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "tar archive exceeded maximum inspected entry count of {max_inspected_entries}"
                ),
            });
        }

        let entry = entry_result.map_err(|io_error| AdfontesError::Extraction {
            format: ARCHIVE_FORMAT_LABEL,
            detail: format!("corrupt tar entry: {io_error}"),
        })?;
        let entry_type = entry.header().entry_type();
        match entry_type {
            EntryType::Directory | EntryType::XGlobalHeader => continue,
            EntryType::Regular | EntryType::Continuous => {}
            other => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!("disallowed tar entry type `{other:?}` in archive"),
                });
            }
        }

        let raw_path_bytes = entry.path_bytes();
        let raw_path_text = std::str::from_utf8(raw_path_bytes.as_ref()).map_err(|_| {
            AdfontesError::SafetyBlock {
                reason: INVALID_UTF8_PATH_ERROR.to_owned(),
            }
        })?;
        if raw_path_text.ends_with('/') {
            if entry.size() == 0 {
                continue;
            }
            return Err(AdfontesError::SafetyBlock {
                reason: format!("non-empty tar regular entry `{raw_path_text}` ends with `/`"),
            });
        }

        if extracted.len() >= max_files {
            return Err(AdfontesError::SafetyBlock {
                reason: format!("tar archive exceeded maximum file count of {max_files}"),
            });
        }

        let normalized_path = sanitize_tar_entry_path(raw_path_text)?;
        if !seen_paths.insert(normalized_path.clone()) {
            return Err(AdfontesError::SafetyBlock {
                reason: format!("duplicate normalized tar entry path `{normalized_path}`"),
            });
        }

        let remaining_budget = max_total_bytes.saturating_sub(cumulative_bytes);
        if entry.size() > remaining_budget as u64 {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "tar entry `{normalized_path}` declared size exceeds remaining {remaining_budget}-byte budget"
                ),
            });
        }

        let file_bytes = read_bounded(entry, remaining_budget, "tar entry")?;
        cumulative_bytes = cumulative_bytes.saturating_add(file_bytes.len());
        extracted.push(ExtractedTarFile {
            path: normalized_path,
            bytes: file_bytes,
        });
    }

    Ok(extracted)
}

fn read_bounded<R: Read>(
    mut reader: R,
    max_bytes: usize,
    stream_label: &str,
) -> Result<Vec<u8>, AdfontesError> {
    let mut output = Vec::new();
    let mut chunk = [0u8; READ_CHUNK_SIZE];
    while output.len() < max_bytes {
        let remaining = max_bytes - output.len();
        let to_read = remaining.min(READ_CHUNK_SIZE);
        let Some(target) = chunk.get_mut(..to_read) else {
            break;
        };
        let bytes_read = reader
            .read(target)
            .map_err(|io_error| AdfontesError::Extraction {
                format: ARCHIVE_FORMAT_LABEL,
                detail: format!("{stream_label} read failed: {io_error}"),
            })?;
        if bytes_read == 0 {
            return Ok(output);
        }
        if let Some(slice) = target.get(..bytes_read) {
            output.extend_from_slice(slice);
        }
    }

    let mut overflow_probe = [0u8; 1];
    let probe_read =
        reader
            .read(&mut overflow_probe)
            .map_err(|io_error| AdfontesError::Extraction {
                format: ARCHIVE_FORMAT_LABEL,
                detail: format!("{stream_label} read failed: {io_error}"),
            })?;
    if probe_read > 0 {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("{stream_label} decompressed output exceeded {max_bytes}-byte limit"),
        });
    }
    Ok(output)
}

fn sanitize_tar_entry_path(path_text: &str) -> Result<String, AdfontesError> {
    if path_text.is_empty() || path_text.contains('\\') || path_text.chars().any(char::is_control) {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("disallowed characters in tar entry path `{path_text}`"),
        });
    }

    let mut clean_parts = Vec::new();
    for component in Path::new(path_text).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => {
                let segment = part.to_str().ok_or_else(|| AdfontesError::SafetyBlock {
                    reason: "non-UTF-8 segment in tar entry path".to_owned(),
                })?;
                if segment.is_empty() {
                    return Err(AdfontesError::SafetyBlock {
                        reason: "empty segment in tar entry path".to_owned(),
                    });
                }
                clean_parts.push(segment);
            }
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!("tar entry path `{path_text}` escapes archive root"),
                });
            }
        }
    }

    if clean_parts.is_empty() {
        return Err(AdfontesError::SafetyBlock {
            reason: "tar entry path resolved to empty name".to_owned(),
        });
    }
    Ok(clean_parts.join("/"))
}

#[cfg(test)]
pub(crate) fn test_gzip_bytes(input: &[u8]) -> Vec<u8> {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input).unwrap();
    encoder.finish().unwrap()
}

#[cfg(test)]
pub(crate) fn test_zlib_bytes(input: &[u8]) -> Vec<u8> {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::ZlibEncoder;

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(input).unwrap();
    encoder.finish().unwrap()
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::DeflateEncoder;
    use rstest::rstest;
    use tar::{Builder, Header};

    use super::*;

    fn build_tar_bytes(entries: &[(&str, EntryType, &[u8])]) -> Vec<u8> {
        let mut builder = Builder::new(Vec::new());
        for &(name, entry_type, body) in entries {
            let mut header = Header::new_gnu();
            header.set_entry_type(entry_type);
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            if entry_type == EntryType::Symlink || entry_type == EntryType::Link {
                builder
                    .append_link(&mut header, name, "/etc/passwd")
                    .unwrap();
            } else {
                builder.append_data(&mut header, name, body).unwrap();
            }
        }
        builder.into_inner().unwrap()
    }

    fn build_raw_tar_entry(name_bytes: &[u8], typeflag: u8, body: &[u8]) -> Vec<u8> {
        let mut block = [0u8; 512];
        let copy_len = name_bytes.len().min(100);
        block[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
        block[100..108].copy_from_slice(b"0000644\0");
        block[108..116].copy_from_slice(b"0000000\0");
        block[116..124].copy_from_slice(b"0000000\0");
        let size_octal = format!("{:011o}\0", body.len());
        block[124..136].copy_from_slice(size_octal.as_bytes());
        block[136..148].copy_from_slice(b"00000000000\0");
        block[148..156].copy_from_slice(b"        ");
        block[156] = typeflag;
        block[257..263].copy_from_slice(b"ustar\0");
        block[263..265].copy_from_slice(b"00");
        let checksum: u32 = block.iter().map(|&byte| u32::from(byte)).sum();
        let checksum_octal = format!("{checksum:06o}\0 ");
        block[148..156].copy_from_slice(checksum_octal.as_bytes());

        let mut out = Vec::new();
        out.extend_from_slice(&block);
        out.extend_from_slice(body);
        let pad = (512 - (body.len() % 512)) % 512;
        out.extend(std::iter::repeat_n(0u8, pad));
        out.extend(std::iter::repeat_n(0u8, 1024));
        out
    }

    #[test]
    fn gunzips_exact_limit_and_blocks_one_byte_over_limit() {
        let exact = test_gzip_bytes(&[b'A'; 64]);
        let over = test_gzip_bytes(&[b'A'; 65]);
        assert_eq!(safe_gunzip_bytes(&exact, 64).unwrap().len(), 64);
        assert!(matches!(
            safe_gunzip_bytes(&over, 64).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn gunzips_multi_member_stream_and_handles_max_usize_limit() {
        let mut combined = test_gzip_bytes(b"first-");
        combined.extend_from_slice(&test_gzip_bytes(b"second"));
        assert_eq!(
            safe_gunzip_bytes(&combined, usize::MAX).unwrap(),
            b"first-second"
        );
    }

    #[test]
    fn inflates_zlib_and_raw_deflate_and_blocks_raw_deflate_bomb() {
        let zlib_bytes = test_zlib_bytes(b"zlib-payload");

        let mut raw_enc = DeflateEncoder::new(Vec::new(), Compression::default());
        raw_enc.write_all(b"raw-deflate-payload").unwrap();
        let raw_bytes = raw_enc.finish().unwrap();

        let mut raw_bomb = DeflateEncoder::new(Vec::new(), Compression::default());
        raw_bomb.write_all(&vec![b'Z'; 4096]).unwrap();
        let bomb_bytes = raw_bomb.finish().unwrap();

        assert_eq!(
            safe_inflate_bytes(&zlib_bytes, 128).unwrap(),
            b"zlib-payload"
        );
        assert_eq!(
            safe_inflate_bytes(&raw_bytes, 128).unwrap(),
            b"raw-deflate-payload"
        );
        assert!(matches!(
            safe_inflate_bytes(&bomb_bytes, 256).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn extracts_clean_gzipped_tar_archive_including_zero_byte_trailing_slash_stub() {
        let mut tar_bytes =
            build_tar_bytes(&[("./main.tex", EntryType::Regular, b"\\section{Intro}")]);
        tar_bytes.truncate(tar_bytes.len() - 1024);
        tar_bytes.extend_from_slice(&build_raw_tar_entry(b"figures/", b'0', b""));
        let gz_tar = test_gzip_bytes(&tar_bytes);

        let files = safe_extract_tar_bytes(&gz_tar, 4096, 10).unwrap();
        assert_eq!(
            files,
            vec![ExtractedTarFile {
                path: "main.tex".to_owned(),
                bytes: b"\\section{Intro}".to_vec(),
            }]
        );
    }

    #[rstest]
    #[case(b"../escape.tex")]
    #[case(b"a/../../escape.tex")]
    #[case(b"/etc/passwd")]
    #[case(b"dir\\evil.tex")]
    #[case(b".")]
    #[case(b"./")]
    #[case(b"non_empty_dir/")]
    #[case(&[0xFF, b'.', b't', b'e', b'x'])]
    fn blocks_malicious_raw_tar_paths(#[case] raw_name: &[u8]) {
        let archive = build_raw_tar_entry(raw_name, b'0', b"payload");
        assert!(matches!(
            safe_extract_tar_bytes(&archive, 4096, 10).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[rstest]
    #[case(EntryType::Symlink)]
    #[case(EntryType::Link)]
    fn blocks_disallowed_link_tar_entries(#[case] entry_type: EntryType) {
        let archive = build_tar_bytes(&[("evil", entry_type, b"")]);
        assert!(matches!(
            safe_extract_tar_bytes(&archive, 4096, 10).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn blocks_duplicate_normalized_paths_in_tar_archive() {
        let archive = build_tar_bytes(&[
            ("main.tex", EntryType::Regular, b"first"),
            ("./main.tex", EntryType::Regular, b"second"),
        ]);
        assert!(matches!(
            safe_extract_tar_bytes(&archive, 4096, 10).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn blocks_unsupported_compression_magic_and_oversized_uncompressed_tar() {
        let raw_tar = build_tar_bytes(&[("main.tex", EntryType::Regular, &[b'X'; 600])]);
        assert!(matches!(
            safe_extract_tar_bytes(&raw_tar, 512, 10).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            safe_extract_tar_bytes(&[0xFD, b'7', b'z', b'X', b'Z', 0x00, 0x01], 4096, 10)
                .unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            safe_extract_tar_bytes(b"BZh91AY&SY", 4096, 10).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn blocks_tar_file_count_and_cumulative_byte_limits() {
        let too_many = build_tar_bytes(&[
            ("a.tex", EntryType::Regular, b"1"),
            ("b.tex", EntryType::Regular, b"2"),
            ("c.tex", EntryType::Regular, b"3"),
        ]);
        let too_large = build_tar_bytes(&[
            ("a.tex", EntryType::Regular, &[b'X'; 300]),
            ("b.tex", EntryType::Regular, &[b'Y'; 300]),
        ]);
        assert!(matches!(
            safe_extract_tar_bytes(&too_many, 16 * 1024, 2).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
        assert!(matches!(
            safe_extract_tar_bytes(&too_large, 500, 10).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }
}
