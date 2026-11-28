//! Pure, synchronous HTTP response media-type classification and content extraction.
//!
//! Operates strictly on buffered [`HttpResponse`] values with zero network I/O, async runtime, or
//! frontier-state dependencies.

use crate::extract::{
    JatsExtraction, XmlDocumentExtraction, extract_html_to_markdown, extract_markdown_toc,
    extract_text_title, extract_xml_document, normalize_markdown_spacing,
};
use crate::model::{
    AdfontesError, ExtractedDocument, ExtractedPayload, HttpUrl, TextFormat, UnavailableReason,
    WorkMetadata,
};
use crate::pdf::is_pdf_magic;
use crate::transport::HttpResponse;

pub(super) const MIME_PLAIN_TEXT: &str = "text/plain";
pub(super) const MIN_FULLTEXT_CHARS: usize = 160;

const PDF_MAGIC_PREFIX: &[u8] = b"%PDF-";
const MIN_OA_LANDING_HTML_CHARS: usize = 2_500;
const MIN_OA_LANDING_HEADINGLESS_CHARS: usize = 5_000;
const MIN_OA_PAYWALL_OVERRIDE_CHARS: usize = 6_000;
const MIN_OA_BODY_HEADINGS: usize = 2;
const MIN_OA_PAYWALL_OVERRIDE_HEADINGS: usize = 3;
const MAX_ARXIV_STUB_CHECK_LEN: usize = 2_500;

const PAYWALL_MARKERS: &[&str] = &[
    "buy this article",
    "purchase access",
    "rent this article",
    "institutional access",
    "access through your institution",
    "sign in to access",
    "subscribe to read",
    "get full access",
    "purchase pdf",
    "paywall",
];

const NON_BODY_SECTION_HEADINGS: &[&str] = &[
    "abstract",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "author information",
    "supplementary information",
    "metrics",
    "rights and permissions",
    "about this article",
];

/// Extracted content from an open-access landing page HTTP 200 response.
#[derive(Debug)]
pub(super) enum LandingExtraction {
    FullText {
        format: TextFormat,
        title: Option<String>,
        markdown: String,
        source_url: HttpUrl,
        raw_bytes_fetched: usize,
    },
    Pdf {
        bytes: Vec<u8>,
        source_url: HttpUrl,
        raw_bytes_fetched: usize,
    },
}

/// Extracted HTML body from an `arXiv` / `ar5iv` HTML endpoint.
#[derive(Debug)]
pub(super) struct ArxivHtmlText {
    pub(super) title: Option<String>,
    pub(super) markdown: String,
    pub(super) source_url: HttpUrl,
    pub(super) raw_bytes_fetched: usize,
}

/// Classifies and extracts a direct HTTP 200 web response into an [`ExtractedDocument`].
pub(super) fn dispatch_http_response(
    response: HttpResponse,
) -> Result<ExtractedDocument, AdfontesError> {
    let raw_len = response.body.len();
    let mime = response.mime_type();
    let host_label = response.final_url.host_str().to_owned();
    let mut metadata = WorkMetadata {
        venue_or_source: (!host_label.is_empty()).then_some(host_label),
        source_url: Some(response.final_url.clone()),
        ..WorkMetadata::default()
    };

    if is_pdf_response(&response) {
        return Ok(ExtractedDocument {
            metadata,
            payload: ExtractedPayload::Pdf {
                bytes: response.body,
            },
            raw_bytes_fetched: raw_len,
        });
    }

    if is_markdown_mime(&mime)
        || (is_plain_text_mime(&mime) && looks_like_markdown_path(response.final_url.path()))
    {
        let text = response.decode_text();
        let markdown = normalize_markdown_spacing(&text);
        metadata.title = extract_text_title(&markdown);
        return Ok(ExtractedDocument {
            metadata,
            payload: ExtractedPayload::FullText {
                format: TextFormat::Markdown,
                markdown,
            },
            raw_bytes_fetched: raw_len,
        });
    }

    if is_html_mime(&mime) {
        let extracted = extract_html_to_markdown(&response.decode_text());
        metadata.title = extracted.title;
        if extracted.markdown.trim().is_empty() {
            return Ok(ExtractedDocument {
                metadata,
                payload: ExtractedPayload::MetadataOnly {
                    reason: UnavailableReason::UpstreamUnavailable {
                        detail: "HTML page yielded no readable body text".to_owned(),
                    },
                },
                raw_bytes_fetched: raw_len,
            });
        }
        return Ok(ExtractedDocument {
            metadata,
            payload: ExtractedPayload::FullText {
                format: TextFormat::Html,
                markdown: extracted.markdown,
            },
            raw_bytes_fetched: raw_len,
        });
    }

    if is_plain_text_mime(&mime) {
        let text = response.decode_text();
        let normalized = normalize_markdown_spacing(&text);
        metadata.title = extract_text_title(&normalized);
        return Ok(ExtractedDocument {
            metadata,
            payload: ExtractedPayload::FullText {
                format: TextFormat::PlainText,
                markdown: normalized,
            },
            raw_bytes_fetched: raw_len,
        });
    }

    if let Some(xml_doc) = try_dispatch_xml_response(&response, &mime, &metadata)? {
        return Ok(xml_doc);
    }

    Ok(ExtractedDocument {
        metadata,
        payload: ExtractedPayload::MetadataOnly {
            reason: UnavailableReason::UnsupportedContentType {
                mime: if mime.is_empty() {
                    "application/octet-stream".to_owned()
                } else {
                    mime
                },
            },
        },
        raw_bytes_fetched: raw_len,
    })
}

/// Extracts substantive full text or PDF bytes from an open-access landing response.
pub(super) fn extract_oa_landing_response(resp: HttpResponse) -> Option<LandingExtraction> {
    if is_pdf_response(&resp) {
        let raw_bytes_fetched = resp.body.len();
        return Some(LandingExtraction::Pdf {
            bytes: resp.body,
            source_url: resp.final_url,
            raw_bytes_fetched,
        });
    }
    let mime = resp.mime_type();
    if is_markdown_mime(&mime) {
        let markdown = normalize_markdown_spacing(&resp.decode_text());
        if markdown.len() >= MIN_FULLTEXT_CHARS {
            return Some(LandingExtraction::FullText {
                format: TextFormat::Markdown,
                title: None,
                markdown,
                source_url: resp.final_url,
                raw_bytes_fetched: resp.body.len(),
            });
        }
    } else if is_html_mime(&mime) {
        let extracted = extract_html_to_markdown(&resp.decode_text());
        if is_substantive_oa_html(&extracted.markdown) {
            return Some(LandingExtraction::FullText {
                format: TextFormat::Html,
                title: extracted.title,
                markdown: extracted.markdown,
                source_url: resp.final_url,
                raw_bytes_fetched: resp.body.len(),
            });
        }
    }
    None
}

/// Validates and extracts HTML full text from an `arXiv` / `ar5iv` HTTP 200 response.
///
/// Returns an explanatory skip reason (`Err(String)`) when the endpoint redirected to `/abs/`,
/// returned a non-HTML MIME type, served an HTML-unavailable notice stub, or produced too little
/// body text.
pub(super) fn extract_arxiv_html_response(resp: HttpResponse) -> Result<ArxivHtmlText, String> {
    if is_arxiv_abs_url(&resp.final_url) {
        return Err("redirected to /abs/".to_owned());
    }
    let mime = resp.mime_type();
    if !is_html_mime(&mime) {
        return Err(format!("unexpected MIME `{mime}`"));
    }
    let decoded = resp.decode_text();
    let extracted = extract_html_to_markdown(&decoded);
    if extracted.markdown.len() < MAX_ARXIV_STUB_CHECK_LEN
        && looks_like_arxiv_no_html_stub(&extracted.markdown)
    {
        return Err("HTML unavailable notice".to_owned());
    }
    if extracted.markdown.len() < MIN_FULLTEXT_CHARS {
        return Err("extracted body too short".to_owned());
    }
    Ok(ArxivHtmlText {
        title: extracted.title,
        markdown: extracted.markdown,
        source_url: resp.final_url,
        raw_bytes_fetched: resp.body.len(),
    })
}

/// Returns `true` when `resp` carries a verified PDF MIME/magic payload.
pub(super) fn is_pdf_response(resp: &HttpResponse) -> bool {
    let mime = resp.mime_type();
    if matches!(
        mime.as_str(),
        "application/pdf" | "application/x-pdf" | "application/acrobat"
    ) {
        return is_pdf_magic(&resp.body);
    }
    if is_html_mime(&mime)
        || is_markdown_mime(&mime)
        || is_plain_text_mime(&mime)
        || is_xml_mime(&mime)
        || matches!(
            mime.as_str(),
            "application/json" | "application/vnd.citationstyles.csl+json"
        )
    {
        return false;
    }
    starts_with_pdf_magic(&resp.body)
}

/// Returns `true` for plain-text and reStructuredText media types.
pub(super) fn is_plain_text_mime(mime: &str) -> bool {
    matches!(
        mime,
        MIME_PLAIN_TEXT | "text/x-rst" | "text/restructuredtext"
    )
}

/// Returns `true` when a URL path points at an `arXiv` `/abs/` abstract landing page.
pub(super) fn is_arxiv_abs_url(url: &HttpUrl) -> bool {
    let path = url.path().to_ascii_lowercase();
    path.starts_with("/abs/")
}

fn try_dispatch_xml_response(
    response: &HttpResponse,
    mime: &str,
    base_metadata: &WorkMetadata,
) -> Result<Option<ExtractedDocument>, AdfontesError> {
    if !is_xml_mime(mime) {
        return Ok(None);
    }
    let raw_len = response.body.len();
    let xml = response.decode_text();
    let mut metadata = base_metadata.clone();
    match extract_xml_document(&xml) {
        Ok(XmlDocumentExtraction::Jats(JatsExtraction::FullText {
            title,
            authors,
            year,
            abstract_text,
            markdown,
        })) => {
            metadata.title = title;
            metadata.authors = authors;
            metadata.year = year;
            metadata.abstract_text = abstract_text;
            Ok(Some(ExtractedDocument {
                metadata,
                payload: ExtractedPayload::FullText {
                    format: TextFormat::JatsXml,
                    markdown,
                },
                raw_bytes_fetched: raw_len,
            }))
        }
        Ok(XmlDocumentExtraction::Jats(JatsExtraction::BodyRestricted {
            title,
            authors,
            year,
            abstract_text,
        })) => {
            metadata.title = title;
            metadata.authors = authors;
            metadata.year = year;
            metadata.abstract_text = abstract_text;
            Ok(Some(ExtractedDocument {
                metadata,
                payload: ExtractedPayload::MetadataOnly {
                    reason: UnavailableReason::BodyRestricted,
                },
                raw_bytes_fetched: raw_len,
            }))
        }
        Ok(XmlDocumentExtraction::Rfc(extracted)) if !extracted.markdown.is_empty() => {
            metadata.title = extracted.title;
            Ok(Some(ExtractedDocument {
                metadata,
                payload: ExtractedPayload::FullText {
                    format: TextFormat::RfcXml,
                    markdown: extracted.markdown,
                },
                raw_bytes_fetched: raw_len,
            }))
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => Err(safety),
        Ok(XmlDocumentExtraction::Rfc(_) | XmlDocumentExtraction::UnsupportedRoot) | Err(_) => {
            Ok(None)
        }
    }
}

fn starts_with_pdf_magic(bytes: &[u8]) -> bool {
    let trimmed = bytes
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .and_then(|start| bytes.get(start..))
        .unwrap_or_default();
    let Some(rest) = trimmed.strip_prefix(PDF_MAGIC_PREFIX) else {
        return false;
    };
    // ISO 32000-1 §7.5.2 header line: `%PDF-<major>.<minor>` followed by EOL (rejecting TeX
    // comments such as `%PDF-1.7 compatibility note`).
    let Some((major, after_major)) = rest.split_first() else {
        return false;
    };
    let Some((dot, after_dot)) = after_major.split_first() else {
        return false;
    };
    let Some((minor, after_version)) = after_dot.split_first() else {
        return false;
    };
    if !major.is_ascii_digit() || *dot != b'.' || !minor.is_ascii_digit() {
        return false;
    }
    after_version
        .iter()
        .find(|&&byte| byte != b' ' && byte != b'\t')
        .is_some_and(|&byte| matches!(byte, b'\r' | b'\n'))
}

fn is_substantive_oa_html(markdown: &str) -> bool {
    let char_count = markdown.chars().count();
    if char_count < MIN_OA_LANDING_HTML_CHARS {
        return false;
    }
    let body_headings = extract_markdown_toc(markdown)
        .iter()
        .filter(|entry| {
            let normalized = entry.heading.trim().to_ascii_lowercase();
            !NON_BODY_SECTION_HEADINGS
                .iter()
                .any(|ignored| normalized == *ignored)
        })
        .count();
    let lower = markdown.to_ascii_lowercase();
    let has_paywall = PAYWALL_MARKERS.iter().any(|marker| lower.contains(marker));
    if has_paywall {
        body_headings >= MIN_OA_PAYWALL_OVERRIDE_HEADINGS
            && char_count >= MIN_OA_PAYWALL_OVERRIDE_CHARS
    } else {
        body_headings >= MIN_OA_BODY_HEADINGS || char_count >= MIN_OA_LANDING_HEADINGLESS_CHARS
    }
}

fn looks_like_arxiv_no_html_stub(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("html is not available for the source")
        || lower.contains("no html version available")
}

fn is_html_mime(mime: &str) -> bool {
    matches!(mime, "text/html" | "application/xhtml+xml")
}

fn is_markdown_mime(mime: &str) -> bool {
    matches!(mime, "text/markdown" | "text/x-markdown")
}

fn is_xml_mime(mime: &str) -> bool {
    matches!(
        mime,
        "application/xml" | "text/xml" | "application/jats+xml" | "application/rfc+xml"
    )
}

fn looks_like_markdown_path(path: &str) -> bool {
    let segment = path.rsplit('/').next().unwrap_or("");
    let has_markdown_extension = segment.rsplit_once('.').is_some_and(|(stem, extension)| {
        !stem.is_empty()
            && (extension.eq_ignore_ascii_case("md") || extension.eq_ignore_ascii_case("markdown"))
    });
    has_markdown_extension
        || segment.eq_ignore_ascii_case("llms.txt")
        || segment.eq_ignore_ascii_case("llms-full.txt")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn synthetic_response(url: &str, content_type: Option<&str>, body: &[u8]) -> HttpResponse {
        HttpResponse {
            status: 200,
            final_url: HttpUrl::parse(url).unwrap(),
            redirect_chain: Vec::new(),
            content_type: content_type.map(str::to_owned),
            headers: BTreeMap::new(),
            body: body.to_vec(),
        }
    }

    #[test]
    fn pdf_response_detection_distinguishes_magic_headers_from_tex_comments_and_html() {
        let valid_binary = synthetic_response(
            "https://example.org/paper",
            Some("application/octet-stream"),
            b"%PDF-1.7\n%binary",
        );
        let tex_comment = synthetic_response(
            "https://example.org/src",
            Some("application/x-eprint"),
            b"%PDF-1.7 compatibility note\n\\documentclass{article}",
        );
        let html_mention = synthetic_response(
            "https://example.org/doc.html",
            Some("text/html"),
            b"%PDF-1.7\n<html><body>not a pdf</body></html>",
        );

        assert!(is_pdf_response(&valid_binary));
        assert!(!is_pdf_response(&tex_comment));
        assert!(!is_pdf_response(&html_mention));
    }

    #[test]
    fn arxiv_html_response_rejects_abs_redirects_and_unavailable_stubs() {
        let abs_resp = synthetic_response(
            "https://arxiv.org/abs/2409.13740",
            Some("text/html"),
            b"<article><p>Abstract only</p></article>",
        );
        let stub_resp = synthetic_response(
            "https://arxiv.org/html/2409.13740",
            Some("text/html"),
            b"<article><p>HTML is not available for the source of this paper.</p></article>",
        );
        let body = "Substantive mathematical derivation and analysis. ".repeat(10);
        let good_resp = synthetic_response(
            "https://arxiv.org/html/2409.13740",
            Some("text/html"),
            format!("<article><h1>Exact Bounds</h1><p>{body}</p></article>").as_bytes(),
        );

        let abs_result = extract_arxiv_html_response(abs_resp);
        let stub_result = extract_arxiv_html_response(stub_resp);
        let good_result = extract_arxiv_html_response(good_resp).unwrap();

        assert_eq!(abs_result.unwrap_err(), "redirected to /abs/");
        assert_eq!(stub_result.unwrap_err(), "HTML unavailable notice");
        assert_eq!(good_result.title.as_deref(), Some("Exact Bounds"));
    }

    #[test]
    fn oa_landing_extraction_rejects_paywall_stubs_and_accepts_multi_section_html() {
        let abstract_pad = "Abstract details about clinical trial design. ".repeat(40);
        let paywall_resp = synthetic_response(
            "https://journal.example.org/article",
            Some("text/html"),
            format!(
                "<article><h1>Study</h1><h2>Abstract</h2><p>{abstract_pad}</p><p>Buy this article via institutional access.</p></article>"
            )
            .as_bytes(),
        );
        let section_body =
            "Full open-access empirical derivations, tables, and proofs across cohorts. "
                .repeat(35);
        let full_resp = synthetic_response(
            "https://journal.example.org/open-article",
            Some("text/html"),
            format!(
                "<article><h1>Open Study</h1><h2>Introduction</h2><p>{section_body}</p><h2>Methods</h2><p>{section_body}</p></article>"
            )
            .as_bytes(),
        );

        assert!(extract_oa_landing_response(paywall_resp).is_none());
        assert!(matches!(
            extract_oa_landing_response(full_resp),
            Some(LandingExtraction::FullText {
                format: TextFormat::Html,
                ..
            })
        ));
    }
}
