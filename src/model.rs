//! Validated scholarly identifiers, HTTP URLs, PDF page spans, and document domain models.
//!
//! External strings are parsed once at the boundary into invariant-preserving value types so
//! downstream transport, extraction, resolution, and staging layers never re-parse raw strings.

architecture_component!(Model);

use std::borrow::Cow;
use std::fmt;
use std::net::IpAddr;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::str::FromStr;

use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, percent_decode_str, utf8_percent_encode};
use thiserror::Error;
use url::{Host, Url};

/// Maximum number of PDF pages allowed in a single [`PageSpan`] slice.
pub const MAX_PDF_SLICE_PAGES: u32 = 8;

/// Maximum PDF page count recommended for direct whole-file multimodal `read_file` inspection.
pub const DIRECT_PDF_READ_MAX_PAGES: u32 = 20;

const ARXIV_SCHEME_PREFIX: &str = "arxiv:";
const DOI_SCHEME_PREFIX: &str = "doi:";
const DOI_DIRECTORY_PREFIX: &str = "10.";
const PMCID_SCHEME_PREFIX: &str = "pmcid:";
const PMC_PREFIX: &str = "pmc";
const PMID_SCHEME_PREFIX: &str = "pmid:";
const RFC_PREFIX: &str = "rfc";
const PDF_EXTENSION: &str = ".pdf";
const RFC_RANGE_ERROR: &str = "RFC number must be between 1 and 99999";
const YYMM_MONTH_TENS_INDEX: usize = 2;
const YYMM_MONTH_ONES_INDEX: usize = 3;

const ARXIV_LEGACY_ARCHIVE_MAX_LEN: usize = 24;
const ARXIV_LEGACY_DIGIT_COUNT: usize = 7;
const ARXIV_YYMM_DIGIT_COUNT: usize = 4;
const ARXIV_MIN_SEQUENCE_DIGITS: usize = 4;
const ARXIV_MAX_SEQUENCE_DIGITS: usize = 5;
const ARXIV_MAX_VERSION_DIGITS: usize = 3;
const CALENDAR_MIN_MONTH: u8 = 1;
const CALENDAR_MAX_MONTH: u8 = 12;

const DOI_MIN_REGISTRANT_DIGITS: usize = 4;
const DOI_MAX_REGISTRANT_DIGITS: usize = 9;
const DOI_MAX_REGISTRANT_SEGMENTS: usize = 4;
const DOI_MAX_TOTAL_LEN: usize = 256;
const PREPRINT_DOI_PREFIX: &str = "10.1101";
const PREPRINT_CONTENT_PREFIX: &str = "/content/10.1101/";
const PREPRINT_MAX_VERSION_DIGITS: usize = 3;
const PREPRINT_VIEW_SUFFIXES: &[&str] = &[
    ".full.pdf+html",
    ".full.pdf",
    ".full-text",
    ".abstract",
    ".figures-only",
    ".article-metrics",
    ".article-info",
    ".full",
    ".pdf",
];
const PMID_MAX_DIGITS: usize = 12;
const PMCID_MAX_DIGITS: usize = 12;
const RFC_MAX_DIGITS: usize = 5;
const RFC_MAX_NUMBER: u32 = 99_999;
const GITHUB_BLOB_MIN_SEGMENTS: usize = 5;
const GITHUB_BLOB_TAIL_INDEX: usize = 3;
const DECIMAL_BASE: u8 = 10;
const DOI_PATH_ENCODED: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'/')
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~')
    .remove(b':')
    .remove(b'(')
    .remove(b')')
    .remove(b';')
    .remove(b',')
    .remove(b'+');

/// Domain errors returned across `adfontes` subsystems.
#[derive(Debug, Error)]
pub enum AdfontesError {
    /// Input target, URL, or page span violated boundary validation rules.
    #[error("invalid target `{input}`: {reason}")]
    InvalidTarget {
        /// Raw caller-supplied value.
        input: String,
        /// Human-readable validation failure reason.
        reason: String,
    },

    /// Request, payload, or staging path violated a security or resource-budget boundary.
    #[error("safety block: {reason}")]
    SafetyBlock {
        /// Explanation of the security or resource boundary that tripped.
        reason: String,
    },

    /// Upstream HTTP or network transport error.
    #[error("transport failure for `{url}`: {detail}")]
    Transport {
        /// Target URL being requested.
        url: String,
        /// Underlying failure message.
        detail: String,
    },

    /// Document parsing or PDF slicing failure.
    #[error("{format} extraction failed: {detail}")]
    Extraction {
        /// Source format name (`HTML`, `XML`, `PDF`, `TeX`, `RFC`).
        format: &'static str,
        /// Detailed parser failure reason.
        detail: String,
    },

    /// Local filesystem staging failure.
    #[error("filesystem error while {context}: {source}")]
    Io {
        /// Operation description.
        context: String,
        /// Underlying OS I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Validated scholarly work identifier resolvable through structured retrieval waterfalls.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScholarlyId {
    /// Canonical `arXiv` e-print identifier (`2409.13740v1` or `math/0211159v1`).
    Arxiv(ArxivId),
    /// Digital Object Identifier (`10.xxxx/...`).
    Doi(Doi),
    /// PubMed Central open-access identifier (`PMC1234567`).
    PmcId(PmcId),
    /// PubMed numeric identifier (`12345678`).
    Pmid(Pmid),
    /// IETF Request for Comments number (`RFC 9110`).
    Rfc(RfcNumber),
}

impl ScholarlyId {
    /// Parses a bare scholarly identifier or canonical scholarly URL into a [`ScholarlyId`].
    ///
    /// Delegates to [`ParsedTarget::parse`] so security blocks ([`AdfontesError::SafetyBlock`])
    /// are preserved, and rejects non-scholarly HTTP(S) URLs as [`AdfontesError::InvalidTarget`].
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::SafetyBlock`] for disallowed schemes or credentials, and
    /// [`AdfontesError::InvalidTarget`] when `raw` is not a recognized scholarly identifier or URL.
    pub fn parse(raw: &str) -> Result<Self, AdfontesError> {
        match ParsedTarget::parse(raw)? {
            ParsedTarget::Scholarly(id) => Ok(id),
            ParsedTarget::Url(_) => Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "expected a scholarly identifier (arXiv, DOI, PMCID, PMID, or RFC)"
                    .to_owned(),
            }),
        }
    }

    /// Returns an opaque canonical equivalence key for cycle detection and alias deduplication.
    ///
    /// Folds `arXiv` versions via [`ArxivId::base_id`] and ASCII-lowercases DOI Basic Latin
    /// characters per ISO 26324.
    #[must_use]
    pub fn canonical_key(&self) -> ScholarlyIdKey {
        match self {
            Self::Arxiv(id) => {
                ScholarlyIdKey(ScholarlyIdKeyKind::Arxiv(id.base_id().to_ascii_lowercase()))
            }
            Self::Doi(doi) => {
                ScholarlyIdKey(ScholarlyIdKeyKind::Doi(doi.as_str().to_ascii_lowercase()))
            }
            Self::PmcId(pmc) => ScholarlyIdKey(ScholarlyIdKeyKind::PmcId(pmc.as_str().to_owned())),
            Self::Pmid(pmid) => ScholarlyIdKey(ScholarlyIdKeyKind::Pmid(pmid.as_str().to_owned())),
            Self::Rfc(rfc) => ScholarlyIdKey(ScholarlyIdKeyKind::Rfc(*rfc)),
        }
    }
}

impl fmt::Display for ScholarlyId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arxiv(id) => write!(formatter, "arXiv:{id}"),
            Self::Doi(doi) => formatter.write_str(doi.as_str()),
            Self::PmcId(pmc) => formatter.write_str(pmc.as_str()),
            Self::Pmid(pmid) => write!(formatter, "PMID:{pmid}"),
            Self::Rfc(rfc) => write!(formatter, "{rfc}"),
        }
    }
}

/// Opaque canonical equivalence key for deduplicating [`ScholarlyId`] aliases across a work.
///
/// Constructed exclusively via [`ScholarlyId::canonical_key`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScholarlyIdKey(ScholarlyIdKeyKind);

/// Validated scholarly identifier or HTTP(S) URL parsed at the input boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedTarget {
    /// Canonical scholarly work identifier (`arXiv`, `DOI`, `PMCID`, `PMID`, `RFC`).
    Scholarly(ScholarlyId),
    /// Validated `http://` or `https://` web URL.
    Url(HttpUrl),
}

impl ParsedTarget {
    /// Parses a user- or agent-supplied target string into a strongly typed [`ParsedTarget`].
    ///
    /// Canonical scholarly URLs (`arxiv.org`, `ar5iv`, `doi.org`, DataCite `10.48550/arXiv.*`,
    /// `bioRxiv` / `medRxiv` `10.1101` preprints, Europe PMC / NCBI `PMC*`, PubMed, and IETF RFC
    /// Editor / Datatracker) are classified into their [`ScholarlyId`] variant so they benefit
    /// from full multi-format waterfalls.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] or [`AdfontesError::SafetyBlock`] if `raw` is empty,
    /// contains control characters, uses a non-HTTP(S) URI scheme, embeds credentials, or fails
    /// syntax validation for its detected identifier family.
    pub fn parse(raw: &str) -> Result<Self, AdfontesError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "target must not be empty".to_owned(),
            });
        }
        if trimmed.chars().any(char::is_control) {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "target must not contain control characters".to_owned(),
            });
        }

        if looks_like_url_with_scheme(trimmed) {
            let http_url = HttpUrl::parse(trimmed)?;
            return Ok(classify_known_scholarly_url(&http_url)
                .map_or(Self::Url(http_url), Self::Scholarly));
        }

        if strip_ascii_prefix_ignore_case(trimmed, ARXIV_SCHEME_PREFIX).is_some() {
            return ArxivId::parse(trimmed).map(|id| Self::Scholarly(ScholarlyId::Arxiv(id)));
        }
        if strip_ascii_prefix_ignore_case(trimmed, DOI_SCHEME_PREFIX).is_some() {
            return parse_doi_or_arxiv_datacite(trimmed).map(Self::Scholarly);
        }
        if strip_ascii_prefix_ignore_case(trimmed, PMCID_SCHEME_PREFIX).is_some() {
            return PmcId::parse(trimmed).map(|id| Self::Scholarly(ScholarlyId::PmcId(id)));
        }
        if strip_ascii_prefix_ignore_case(trimmed, PMID_SCHEME_PREFIX).is_some() {
            return Pmid::parse(trimmed).map(|id| Self::Scholarly(ScholarlyId::Pmid(id)));
        }
        if strip_ascii_prefix_ignore_case(trimmed, PMC_PREFIX).is_some() {
            return PmcId::parse(trimmed).map(|id| Self::Scholarly(ScholarlyId::PmcId(id)));
        }
        if strip_ascii_prefix_ignore_case(trimmed, RFC_PREFIX).is_some() {
            return RfcNumber::parse(trimmed).map(|id| Self::Scholarly(ScholarlyId::Rfc(id)));
        }
        if trimmed.starts_with(DOI_DIRECTORY_PREFIX) {
            return parse_doi_or_arxiv_datacite(trimmed).map(Self::Scholarly);
        }
        if looks_like_non_scholarly_uri_scheme(trimmed) {
            let http_url = HttpUrl::parse(trimmed)?;
            return Ok(classify_known_scholarly_url(&http_url)
                .map_or(Self::Url(http_url), Self::Scholarly));
        }
        if let Ok(arxiv) = ArxivId::parse(trimmed) {
            return Ok(Self::Scholarly(ScholarlyId::Arxiv(arxiv)));
        }

        Err(AdfontesError::InvalidTarget {
            input: raw.to_owned(),
            reason: "expected an arXiv ID, DOI, PMCID, PMID, RFC number, or http(s):// URL"
                .to_owned(),
        })
    }
}

/// Validated `arXiv` identifier (`YYMM.NNNN[N][vV]` or `<archive>/YYMMNNN[vV]`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArxivId(String);

impl ArxivId {
    /// Parses a bare or `arXiv:`-prefixed identifier.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] if `raw` is not a valid modern or legacy `arXiv` ID.
    pub fn parse(raw: &str) -> Result<Self, AdfontesError> {
        let candidate = strip_ascii_prefix_ignore_case(raw.trim(), ARXIV_SCHEME_PREFIX)
            .unwrap_or_else(|| raw.trim())
            .trim();
        let candidate =
            strip_ascii_suffix_ignore_case(candidate, PDF_EXTENSION).unwrap_or(candidate);

        if let Some((archive, number_and_version)) = candidate.split_once('/') {
            let canonical_archive = canonicalize_legacy_arxiv_archive(archive, raw)?;
            let split = split_arxiv_version(number_and_version, raw)?;
            if split.digits.len() != ARXIV_LEGACY_DIGIT_COUNT
                || !split.digits.bytes().all(|byte| byte.is_ascii_digit())
            {
                return Err(AdfontesError::InvalidTarget {
                    input: raw.to_owned(),
                    reason: "legacy arXiv ID must have 7 digits after `<archive>/`".to_owned(),
                });
            }
            validate_yymm_prefix(split.digits, raw)?;
            let base = format!("{canonical_archive}/{}", split.digits);
            return Ok(Self::from_base_and_version(base, split.version));
        }

        let Some((year_month, remainder)) = candidate.split_once('.') else {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "modern arXiv ID must match YYMM.NNNN[N][vV]".to_owned(),
            });
        };
        let split = split_arxiv_version(remainder, raw)?;
        let valid_sequence_len =
            (ARXIV_MIN_SEQUENCE_DIGITS..=ARXIV_MAX_SEQUENCE_DIGITS).contains(&split.digits.len());
        if year_month.len() != ARXIV_YYMM_DIGIT_COUNT
            || !year_month.bytes().all(|byte| byte.is_ascii_digit())
            || !valid_sequence_len
            || !split.digits.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "modern arXiv ID must have 4-digit YYMM and 4-to-5-digit sequence"
                    .to_owned(),
            });
        }
        validate_yymm_prefix(year_month, raw)?;
        let base = format!("{year_month}.{}", split.digits);
        Ok(Self::from_base_and_version(base, split.version))
    }

    fn from_base_and_version(base: String, version: Option<u32>) -> Self {
        match version {
            Some(number) => Self(format!("{base}v{number}")),
            None => Self(base),
        }
    }

    /// Returns the canonical `arXiv` identifier string including any explicit `vN` suffix.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the versionless `arXiv` identifier (`2409.13740` or `math/0211159`).
    #[must_use]
    pub fn base_id(&self) -> &str {
        match self.0.rsplit_once('v') {
            Some((base, version))
                if !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit()) =>
            {
                base
            }
            _ => &self.0,
        }
    }
}

macro_rules! impl_from_str {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl FromStr for $ty {
                type Err = AdfontesError;

                fn from_str(raw: &str) -> Result<Self, Self::Err> {
                    Self::parse(raw)
                }
            }
        )+
    };
}

macro_rules! impl_display_as_str {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl fmt::Display for $ty {
                fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str(self.as_str())
                }
            }
        )+
    };
}

impl_from_str!(
    ParsedTarget,
    ScholarlyId,
    ArxivId,
    Doi,
    PmcId,
    Pmid,
    RfcNumber,
    HttpUrl
);
impl_display_as_str!(ArxivId, Doi, PmcId, Pmid, HttpUrl);

/// Validated Digital Object Identifier (`10.<registrant>/<suffix>`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Doi(String);

impl Doi {
    /// Parses and normalizes a bare or `doi:`-prefixed DOI string.
    ///
    /// Rejects non-UTF-8 percent-encoded sequences, path-traversal segments (`.` / `..`),
    /// backslashes, and query/fragment injection characters (`?` / `#`).
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] if the registrant prefix or suffix is malformed.
    pub fn parse(raw: &str) -> Result<Self, AdfontesError> {
        let trimmed = strip_ascii_prefix_ignore_case(raw.trim(), DOI_SCHEME_PREFIX)
            .unwrap_or_else(|| raw.trim())
            .trim();
        let decoded = percent_decode_utf8(trimmed).ok_or_else(|| AdfontesError::InvalidTarget {
            input: raw.to_owned(),
            reason: "percent-encoded identifier is not valid UTF-8".to_owned(),
        })?;
        let cleaned = strip_trailing_sentence_punctuation(&decoded);
        if cleaned.len() > DOI_MAX_TOTAL_LEN {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "DOI exceeds maximum supported length".to_owned(),
            });
        }
        let Some(after_directory) = cleaned.strip_prefix(DOI_DIRECTORY_PREFIX) else {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "DOI must begin with `10.`".to_owned(),
            });
        };
        let Some((registrant, suffix)) = after_directory.split_once('/') else {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "DOI must contain `/` separating registrant and suffix".to_owned(),
            });
        };
        let registrant_segments: Vec<&str> = registrant.split('.').collect();
        let registrant_parts_valid = !registrant_segments.is_empty()
            && registrant_segments.len() <= DOI_MAX_REGISTRANT_SEGMENTS
            && registrant_segments.iter().all(|segment| {
                !segment.is_empty()
                    && segment.len() <= DOI_MAX_REGISTRANT_DIGITS
                    && segment.bytes().all(|byte| byte.is_ascii_digit())
            })
            && registrant_segments.first().is_some_and(|first| {
                (DOI_MIN_REGISTRANT_DIGITS..=DOI_MAX_REGISTRANT_DIGITS).contains(&first.len())
            });
        let suffix_has_forbidden_chars = suffix.is_empty()
            || suffix.chars().any(|character| {
                character.is_whitespace()
                    || character.is_control()
                    || matches!(character, '\\' | '?' | '#')
            });
        let suffix_has_dot_segment = suffix
            .split('/')
            .any(|segment| segment == "." || segment == "..");
        if !registrant_parts_valid || suffix_has_forbidden_chars || suffix_has_dot_segment {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "DOI registrant or suffix is invalid".to_owned(),
            });
        }
        Ok(Self(format!("{DOI_DIRECTORY_PREFIX}{registrant}/{suffix}")))
    }

    /// Returns the normalized DOI string (`10.xxxx/...`).
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the DOI safely percent-encoded for interpolation into HTTP API URL paths.
    ///
    /// Preserves `/` separating the registrant and suffix while encoding characters that could
    /// alter URL syntax (`%`, `<`, `>`, `"`, `{`, `}`, `|`, `^`, `` ` ``, non-ASCII).
    #[must_use]
    pub fn percent_encoded_path(&self) -> String {
        utf8_percent_encode(&self.0, DOI_PATH_ENCODED).to_string()
    }
}

/// Validated PubMed Central identifier (`PMC<digits>`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PmcId(String);

impl PmcId {
    /// Parses a `PMC<digits>` identifier (case-insensitive prefix, 1–12 digits, no leading zero).
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] if the identifier lacks a `PMC` prefix or valid digits.
    pub fn parse(raw: &str) -> Result<Self, AdfontesError> {
        let trimmed = raw.trim();
        let without_scheme = strip_ascii_prefix_ignore_case(trimmed, PMCID_SCHEME_PREFIX)
            .unwrap_or(trimmed)
            .trim();
        let digits =
            strip_ascii_prefix_ignore_case(without_scheme, PMC_PREFIX).ok_or_else(|| {
                AdfontesError::InvalidTarget {
                    input: raw.to_owned(),
                    reason: "PMCID must start with `PMC`".to_owned(),
                }
            })?;
        if !is_positive_ascii_digits(digits, Some(PMCID_MAX_DIGITS)) {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "PMCID must contain 1-to-12 non-zero-leading digits after `PMC`".to_owned(),
            });
        }
        Ok(Self(format!("PMC{digits}")))
    }

    /// Returns the canonical uppercase `PMC<digits>` string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Validated PubMed identifier (`<digits>`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pmid(String);

impl Pmid {
    /// Parses a `PMID` numeric string (with optional `pmid:` prefix, 1–12 digits, no leading zero).
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] if `raw` does not consist of 1–12 valid ASCII digits.
    pub fn parse(raw: &str) -> Result<Self, AdfontesError> {
        let digits = strip_ascii_prefix_ignore_case(raw.trim(), PMID_SCHEME_PREFIX)
            .unwrap_or_else(|| raw.trim())
            .trim();
        if !is_positive_ascii_digits(digits, Some(PMID_MAX_DIGITS)) {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "PMID must be a positive 1-to-12-digit integer without leading zeros"
                    .to_owned(),
            });
        }
        Ok(Self(digits.to_owned()))
    }

    /// Returns the numeric PMID string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Validated positive IETF RFC number (`RFC 1`..`RFC 99999`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RfcNumber(NonZeroU32);

impl RfcNumber {
    /// Constructs a validated [`RfcNumber`] from a positive integer (`1..=99_999`).
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] if `number` is zero or exceeds 5 digits.
    pub fn new(number: u32) -> Result<Self, AdfontesError> {
        let parsed = NonZeroU32::new(number)
            .filter(|value| value.get() <= RFC_MAX_NUMBER)
            .ok_or_else(|| AdfontesError::InvalidTarget {
                input: number.to_string(),
                reason: RFC_RANGE_ERROR.to_owned(),
            })?;
        Ok(Self(parsed))
    }

    /// Parses `RFC 9110`, `rfc9110`, `rfc-9110`, or `rfc:9110`.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] if the `RFC` prefix is missing or the number is invalid.
    pub fn parse(raw: &str) -> Result<Self, AdfontesError> {
        let trimmed = raw.trim();
        let after_prefix =
            strip_ascii_prefix_ignore_case(trimmed, RFC_PREFIX).ok_or_else(|| {
                AdfontesError::InvalidTarget {
                    input: raw.to_owned(),
                    reason: "RFC identifier must begin with `RFC`".to_owned(),
                }
            })?;
        let digits = after_prefix
            .strip_prefix([' ', '-', ':'])
            .unwrap_or(after_prefix);
        if !is_positive_ascii_digits(digits, Some(RFC_MAX_DIGITS)) {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "RFC number must be a 1-to-5-digit positive integer".to_owned(),
            });
        }
        let parsed = digits
            .parse::<u32>()
            .ok()
            .and_then(|value| Self::new(value).ok())
            .ok_or_else(|| AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: RFC_RANGE_ERROR.to_owned(),
            })?;
        Ok(parsed)
    }

    /// Returns the numeric RFC number.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl fmt::Display for RfcNumber {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "RFC {}", self.0.get())
    }
}

/// Typed host extracted from a validated [`HttpUrl`] without IPv6 bracket ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UrlHost<'a> {
    /// ASCII/punycode DNS domain name (without trailing FQDN root dot).
    Domain(&'a str),
    /// Literal IPv4 or IPv6 address (already parsed; never wrapped in `[...]`).
    Ip(IpAddr),
}

/// Validated `http://` or `https://` URL without embedded credentials or port `0`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HttpUrl(Url);

impl HttpUrl {
    /// Parses and validates an `http://` or `https://` URL.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::SafetyBlock`] for non-HTTP(S) schemes or embedded userinfo, and
    /// [`AdfontesError::InvalidTarget`] for syntactically malformed URLs, port `0`, or missing hosts.
    pub fn parse(raw: &str) -> Result<Self, AdfontesError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.chars().any(char::is_control) {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "URL must be non-empty and free of control characters".to_owned(),
            });
        }
        let mut parsed =
            Url::parse(trimmed).map_err(|parse_error| AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: format!("malformed URL: {parse_error}"),
            })?;
        let scheme = parsed.scheme();
        if scheme != "http" && scheme != "https" {
            return Err(AdfontesError::SafetyBlock {
                reason: format!(
                    "disallowed URL scheme `{scheme}` (only http and https are allowed)"
                ),
            });
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(AdfontesError::SafetyBlock {
                reason: "URLs containing userinfo credentials (`user:pass@host`) are forbidden"
                    .to_owned(),
            });
        }
        if parsed.port() == Some(0) {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "URL port 0 is not permitted".to_owned(),
            });
        }
        if parsed.host_str().is_none_or(str::is_empty) {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "URL must include a non-empty hostname".to_owned(),
            });
        }
        if let Some(Host::Domain(domain)) = parsed.host() {
            let without_root_dot = domain.strip_suffix('.').unwrap_or(domain);
            if without_root_dot.is_empty() || without_root_dot.split('.').any(str::is_empty) {
                return Err(AdfontesError::InvalidTarget {
                    input: raw.to_owned(),
                    reason: "URL hostname contains empty domain labels".to_owned(),
                });
            }
        }
        parsed.set_fragment(None);
        Ok(Self(parsed))
    }

    /// Returns the normalized URL string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Returns `"http"` or `"https"`.
    #[must_use]
    pub fn scheme(&self) -> &str {
        self.0.scheme()
    }

    /// Returns the raw URL host string (IPv6 addresses include brackets per RFC 3986).
    #[must_use]
    pub fn host_str(&self) -> &str {
        self.0.host_str().unwrap_or_default()
    }

    /// Returns the parsed [`UrlHost`] (either an unbracketed [`IpAddr`] or a normalized domain
    /// name stripped of trailing FQDN root dots) for SSRF checks and socket dialing.
    #[must_use]
    pub fn host(&self) -> UrlHost<'_> {
        match self.0.host() {
            Some(Host::Ipv4(address)) => UrlHost::Ip(IpAddr::V4(address)),
            Some(Host::Ipv6(address)) => UrlHost::Ip(IpAddr::V6(address)),
            Some(Host::Domain(domain)) => UrlHost::Domain(domain.trim_end_matches('.')),
            None => UrlHost::Domain(""),
        }
    }

    /// Returns the explicit or scheme-default port (`80` for HTTP, `443` for HTTPS).
    #[must_use]
    pub fn port_or_default(&self) -> u16 {
        self.0.port_or_known_default().unwrap_or(443)
    }

    /// Returns the URL path component.
    #[must_use]
    pub fn path(&self) -> &str {
        self.0.path()
    }

    /// Returns a copy of this URL with additional `application/x-www-form-urlencoded` query pairs
    /// appended (encoding spaces as `+` and literal `+` as `%2B`).
    #[must_use]
    pub fn with_query_pairs<'a>(
        &self,
        pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Self {
        let mut iter = pairs.into_iter();
        let Some((first_key, first_value)) = iter.next() else {
            return self.clone();
        };
        let mut url = self.0.clone();
        {
            let mut query = url.query_pairs_mut();
            query.append_pair(first_key, first_value);
            for (key, value) in iter {
                query.append_pair(key, value);
            }
        }
        Self(url)
    }

    /// Resolves a relative or absolute redirect `Location` reference against `self`.
    ///
    /// Rejects empty values and control characters prior to WHATWG URL resolution so embedded
    /// `\r`, `\n`, `\t`, or `\0` bytes cannot be silently stripped or percent-encoded.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] or [`AdfontesError::SafetyBlock`] if `location` is
    /// empty, contains control characters, or resolves to a disallowed scheme or userinfo URL.
    pub fn join_redirect(&self, location: &str) -> Result<Self, AdfontesError> {
        if location.chars().any(char::is_control) {
            return Err(AdfontesError::InvalidTarget {
                input: location.to_owned(),
                reason: "redirect Location must not contain control characters".to_owned(),
            });
        }
        let trimmed = location.trim();
        if trimmed.is_empty() {
            return Err(AdfontesError::InvalidTarget {
                input: location.to_owned(),
                reason: "redirect Location must not be empty".to_owned(),
            });
        }
        let joined = self
            .0
            .join(trimmed)
            .map_err(|parse_error| AdfontesError::InvalidTarget {
                input: location.to_owned(),
                reason: format!("invalid redirect Location: {parse_error}"),
            })?;
        Self::parse(joined.as_str())
    }

    /// Rewrites `https://github.com/<owner>/<repo>/blob/<ref>/<path>` to `raw.githubusercontent.com`.
    ///
    /// Strips any query string (`?plain=1`) and requires standard default HTTPS/HTTP ports and
    /// non-empty path segments so malformed `//` paths are not rewritten.
    #[must_use]
    pub fn to_raw_github_url(&self) -> Option<Self> {
        if self.0.port().is_some() {
            return None;
        }
        let UrlHost::Domain(domain) = self.host() else {
            return None;
        };
        if !domain.eq_ignore_ascii_case("github.com")
            && !domain.eq_ignore_ascii_case("www.github.com")
        {
            return None;
        }
        let segments: Vec<&str> = self.0.path_segments()?.collect();
        if segments.len() < GITHUB_BLOB_MIN_SEGMENTS
            || segments.iter().any(|segment| segment.is_empty())
            || *segments.get(2)? != "blob"
        {
            return None;
        }
        let owner = segments.first()?;
        let repo = segments.get(1)?;
        let tail = segments.get(GITHUB_BLOB_TAIL_INDEX..)?.join("/");
        Self::parse(&format!(
            "https://raw.githubusercontent.com/{owner}/{repo}/{tail}"
        ))
        .ok()
    }
}

/// Validated inclusive 1-indexed PDF page span (`1 <= start <= end`, `count <= MAX_PDF_SLICE_PAGES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageSpan {
    start: NonZeroU32,
    end: NonZeroU32,
}

impl PageSpan {
    /// Constructs a validated inclusive page span `[start, end]`.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] if `start == 0`, `end < start`, or the span exceeds
    /// [`MAX_PDF_SLICE_PAGES`] (`8` pages).
    pub fn new(start: u32, end: u32) -> Result<Self, AdfontesError> {
        let raw_repr = format!("{start}-{end}");
        let Some(start_page) = NonZeroU32::new(start) else {
            return Err(AdfontesError::InvalidTarget {
                input: raw_repr,
                reason: "start_page must be >= 1".to_owned(),
            });
        };
        let Some(end_page) = NonZeroU32::new(end) else {
            return Err(AdfontesError::InvalidTarget {
                input: raw_repr,
                reason: "end_page must be >= 1".to_owned(),
            });
        };
        if end < start {
            return Err(AdfontesError::InvalidTarget {
                input: raw_repr,
                reason: "end_page must be >= start_page".to_owned(),
            });
        }
        let count = end - start + 1;
        if count > MAX_PDF_SLICE_PAGES {
            return Err(AdfontesError::InvalidTarget {
                input: raw_repr,
                reason: format!(
                    "page span requests {count} pages, exceeding the {MAX_PDF_SLICE_PAGES}-page cap"
                ),
            });
        }
        Ok(Self {
            start: start_page,
            end: end_page,
        })
    }

    /// Parses a CLI page range specification (`"4"` or `"4-8"`).
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] if `spec` is not an ASCII positive integer or
    /// `start-end` range within [`MAX_PDF_SLICE_PAGES`].
    pub fn parse_spec(spec: &str) -> Result<Self, AdfontesError> {
        let trimmed = spec.trim();
        if let Some((left, right)) = trimmed.split_once('-') {
            let start = parse_strict_positive_u32(left.trim(), spec, "page range start")?;
            let end = parse_strict_positive_u32(right.trim(), spec, "page range end")?;
            Self::new(start, end)
        } else {
            let single = parse_strict_positive_u32(trimmed, spec, "page number")?;
            Self::new(single, single)
        }
    }

    /// Clamps `end` to `total_pages` when `start <= total_pages`, failing if `start > total_pages`.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::InvalidTarget`] if `total_pages == 0` or `self.start() > total_pages`.
    pub fn clamp_to_total(self, total_pages: u32) -> Result<Self, AdfontesError> {
        if total_pages == 0 || self.start.get() > total_pages {
            return Err(AdfontesError::InvalidTarget {
                input: self.to_string(),
                reason: format!(
                    "start_page {} exceeds PDF page count {total_pages}",
                    self.start.get()
                ),
            });
        }
        Self::new(self.start.get(), self.end.get().min(total_pages))
    }

    /// Returns the 1-indexed inclusive start page.
    #[must_use]
    pub const fn start(self) -> u32 {
        self.start.get()
    }

    /// Returns the 1-indexed inclusive end page.
    #[must_use]
    pub const fn end(self) -> u32 {
        self.end.get()
    }
}

impl fmt::Display for PageSpan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start == self.end {
            write!(formatter, "{}", self.start.get())
        } else {
            write!(formatter, "{}-{}", self.start.get(), self.end.get())
        }
    }
}

impl FromStr for PageSpan {
    type Err = AdfontesError;

    fn from_str(spec: &str) -> Result<Self, Self::Err> {
        Self::parse_spec(spec)
    }
}

/// Digital text source format from which normalized Markdown was extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextFormat {
    /// Origin-served Markdown (`text/markdown`, `.md`, or `/llms.txt`).
    Markdown,
    /// Structured HTML5 (`LaTeXML`, `ar5iv`, technical essay, or HTML specification).
    Html,
    /// NISO JATS archival XML (`Europe PMC` `fullTextXML`).
    JatsXml,
    /// IETF RFCXML v3 (`rfc<N>.xml`).
    RfcXml,
    /// Paginated plain-text IETF RFC (`rfc<N>.txt` with `\x0c` headers stripped).
    RfcText,
    /// Flattened `arXiv` LaTeX e-print source archive (`.tar.gz` / `.gz`).
    Latex,
    /// Plain UTF-8 text or reStructuredText (`PEP`, `.rst`, `.txt`).
    PlainText,
}

impl TextFormat {
    /// Returns the compact uppercase wire label emitted in `CONTENT_KIND:`.
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Markdown => "MARKDOWN",
            Self::Html => "HTML_MARKDOWN",
            Self::JatsXml => "JATS_XML_MARKDOWN",
            Self::RfcXml => "RFCXML_MARKDOWN",
            Self::RfcText => "RFC_TEXT_MARKDOWN",
            Self::Latex => "LATEX_SOURCE_MARKDOWN",
            Self::PlainText => "PLAIN_TEXT",
        }
    }
}

/// Reason why full text or open-access PDF bytes could not be retrieved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnavailableReason {
    /// Work is closed-access with no legal open-access full-text or PDF location.
    NoOpenAccessCopy,
    /// Provider returned bibliographic front-matter (`<front>`) while restricting `<body>`.
    BodyRestricted,
    /// Endpoint returned an unsupported binary MIME type.
    UnsupportedContentType {
        /// Upstream `Content-Type` media type.
        mime: String,
    },
    /// Upstream full-text endpoints failed while bibliographic metadata succeeded.
    UpstreamUnavailable {
        /// Summary of upstream failure statuses.
        detail: String,
    },
}

impl fmt::Display for UnavailableReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoOpenAccessCopy => {
                formatter.write_str("no legal open-access full text or PDF copy is available")
            }
            Self::BodyRestricted => formatter.write_str(
                "repository returned bibliographic front-matter without open-access body text",
            ),
            Self::UnsupportedContentType { mime } => {
                write!(formatter, "unsupported content type `{mime}`")
            }
            Self::UpstreamUnavailable { detail } => {
                write!(
                    formatter,
                    "open-access full-text endpoints unavailable ({detail})"
                )
            }
        }
    }
}

/// Bibliographic provenance metadata attached to a resolved work.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkMetadata {
    /// Work title.
    pub title: Option<String>,
    /// Ordered author display names.
    pub authors: Vec<String>,
    /// Publication or preprint year.
    pub year: Option<u16>,
    /// Journal, conference, standards series, or host domain.
    pub venue_or_source: Option<String>,
    /// Canonical identifier (`arXiv:2409.13740v1`, `10.1038/...`, `PMC...`, `RFC 9110`).
    pub canonical_id: Option<String>,
    /// Final resolved source URL from which the payload or metadata was retrieved.
    pub source_url: Option<HttpUrl>,
    /// Abstract text when provided by bibliographic metadata feeds.
    pub abstract_text: Option<String>,
    /// Provenance or status warnings (for example IETF `Obsoleted by RFC 9110`).
    pub notes: Vec<String>,
}

/// Line-numbered Markdown heading entry in a staged document's Table of Contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TocEntry {
    /// Markdown heading level (`1..=6`).
    pub level: u8,
    /// Sanitized single-line heading title.
    pub heading: String,
    /// 1-indexed start line of the section in the staged file.
    pub start_line: NonZeroUsize,
    /// 1-indexed inclusive end line of the section in the staged file.
    pub end_line: NonZeroUsize,
}

/// Extracted text summary state for a single PDF page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PdfPageText {
    /// Page was inspected for text extraction (may still have `char_count == 0` when scanned).
    Indexed {
        /// Non-whitespace extracted character count on this page.
        char_count: usize,
        /// Detected section/figure/table heading lines on this page.
        headings: Vec<String>,
        /// Short first-line text preview for page orientation.
        preview: String,
    },
    /// Page exceeded indexing caps (page limit or cumulative text budget); inspect via `slice_pdf`.
    Unindexed,
}

/// Lightweight per-page text hint summary for a staged PDF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdfPageSummary {
    /// 1-indexed page number.
    pub page_number: NonZeroU32,
    /// Text extraction state and orientation hints for this page.
    pub text: PdfPageText,
}

/// Structural index over a staged PDF document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdfIndex {
    /// Total number of pages in the PDF.
    pub page_count: u32,
    /// Top-level PDF outline/bookmark titles when present.
    pub outline_titles: Vec<String>,
    /// Per-page text hints across the document.
    pub pages: Vec<PdfPageSummary>,
}

/// Structured full-text endpoint failure recorded prior to falling back to a binary PDF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackSkip {
    /// Short upstream endpoint label (for example `"arxiv-html"`, `"ar5iv"`, or `"epmc-xml"`).
    pub endpoint: &'static str,
    /// Concise reason the structured endpoint could not supply full text.
    pub reason: String,
}

/// Disjoint retrieval outcome produced by scholarly resolution and extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractedPayload {
    /// Structured digital full text normalized to clean UTF-8 Markdown or plain text.
    FullText {
        /// Underlying digital source format.
        format: TextFormat,
        /// Normalized Markdown or text body.
        markdown: String,
    },
    /// Open-access PDF binary when structured digital text is unavailable.
    Pdf {
        /// Validated `%PDF-` document bytes.
        bytes: Vec<u8>,
        /// Structured full-text endpoints skipped before falling back to PDF.
        fallback_skips: Vec<FallbackSkip>,
    },
    /// Closed-access or metadata-only record with explicit unavailability reason.
    MetadataOnly {
        /// Structured reason why full text could not be staged.
        reason: UnavailableReason,
    },
}

/// Complete in-memory resolution result prior to local filesystem staging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedDocument {
    /// Bibliographic metadata and provenance notes.
    pub metadata: WorkMetadata,
    /// Disjoint full-text, PDF, or metadata-only payload.
    pub payload: ExtractedPayload,
    /// Total upstream payload bytes fetched for the winning representation.
    pub raw_bytes_fetched: usize,
}

/// Disjoint on-disk staging outcome paired with navigation metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StagedOutcome {
    /// Normalized Markdown or text file staged for `read_file` line-window reads.
    FullText {
        /// Absolute path to the staged `.md` or `.txt` file.
        staged_path: PathBuf,
        /// Source format label.
        format: TextFormat,
        /// Total 1-indexed line count.
        line_count: usize,
        /// UTF-8 byte length on disk.
        byte_count: usize,
        /// Extracted section Table of Contents.
        toc: Vec<TocEntry>,
    },
    /// Validated PDF staged alongside an optional `<slug>.pages.md` page map.
    Pdf {
        /// Absolute path to the staged `.pdf` file.
        staged_pdf_path: PathBuf,
        /// Absolute path to the generated `<slug>.pages.md` index when enabled.
        page_index_path: Option<PathBuf>,
        /// PDF byte length on disk.
        byte_count: usize,
        /// Extracted page counts and per-page heading hints.
        pdf_index: PdfIndex,
        /// Structured full-text endpoints skipped before falling back to PDF.
        fallback_skips: Vec<FallbackSkip>,
    },
    /// Bibliographic metadata note staged when no open-access full text exists.
    MetadataOnly {
        /// Absolute path to the staged `.metadata.md` summary note.
        staged_note_path: PathBuf,
        /// Unavailability classification.
        reason: UnavailableReason,
    },
}

/// Final staged result returned by the `Stage` pipeline to CLI and MCP callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedResult {
    /// Bibliographic provenance metadata.
    pub metadata: WorkMetadata,
    /// Variant-specific staged paths and structural index.
    pub outcome: StagedOutcome,
    /// Bounded (`<= 3,600 B`) compact navigation manifest printed to stdout / MCP tool output.
    pub compact_index: String,
}

/// Reclassifies a known scholarly landing/PDF URL into its canonical [`ScholarlyId`].
#[must_use]
pub fn classify_known_scholarly_url(url: &HttpUrl) -> Option<ScholarlyId> {
    let UrlHost::Domain(domain) = url.host() else {
        return None;
    };
    let host = domain.to_ascii_lowercase();
    let path = url.path();

    classify_arxiv_url(&host, path)
        .or_else(|| classify_doi_url(&host, path))
        .or_else(|| classify_preprint_doi_url(&host, path))
        .or_else(|| classify_biomedical_url(&host, path))
        .or_else(|| classify_rfc_url(&host, path))
}

/// Truncates a string to at most `max_chars` Unicode scalar values, appending `…`.
#[must_use]
pub fn truncate_chars(input: &str, max_chars: usize) -> String {
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

/// Truncates a UTF-8 string slice to at most `max_bytes` bytes on a valid char boundary.
#[must_use]
pub fn truncate_utf8_bytes(input: &str, max_bytes: usize) -> &str {
    if input.len() <= max_bytes {
        return input;
    }
    let mut boundary = max_bytes;
    while boundary > 0 && !input.is_char_boundary(boundary) {
        boundary -= 1;
    }
    input.get(..boundary).unwrap_or_default()
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ScholarlyIdKeyKind {
    Arxiv(String),
    Doi(String),
    PmcId(String),
    Pmid(String),
    Rfc(RfcNumber),
}

struct ArxivVersionSplit<'a> {
    digits: &'a str,
    version: Option<u32>,
}

fn classify_arxiv_url(host: &str, path: &str) -> Option<ScholarlyId> {
    if !matches!(
        host,
        "arxiv.org" | "www.arxiv.org" | "export.arxiv.org" | "ar5iv.labs.arxiv.org" | "ar5iv.org"
    ) {
        return None;
    }
    for prefix in ["/abs/", "/pdf/", "/html/", "/src/", "/e-print/"] {
        if let Some(rest) = strip_ascii_prefix_ignore_case(path, prefix) {
            let trimmed = rest.trim_end_matches('/');
            let decoded = percent_decode_utf8(trimmed)?;
            if let Ok(arxiv) = ArxivId::parse(&decoded) {
                return Some(ScholarlyId::Arxiv(arxiv));
            }
        }
    }
    None
}

fn classify_doi_url(host: &str, path: &str) -> Option<ScholarlyId> {
    if !matches!(host, "doi.org" | "www.doi.org" | "dx.doi.org") {
        return None;
    }
    let raw_doi = path.trim_matches('/');
    parse_doi_or_arxiv_datacite(raw_doi).ok()
}

fn classify_preprint_doi_url(host: &str, path: &str) -> Option<ScholarlyId> {
    if !matches!(
        host,
        "biorxiv.org" | "www.biorxiv.org" | "medrxiv.org" | "www.medrxiv.org"
    ) {
        return None;
    }
    let rest = strip_ascii_prefix_ignore_case(path, PREPRINT_CONTENT_PREFIX)?;
    let decoded = percent_decode_utf8(rest.trim_end_matches('/'))?;
    if decoded.is_empty() || decoded.contains('/') {
        return None;
    }
    let without_view = PREPRINT_VIEW_SUFFIXES
        .iter()
        .find_map(|suffix| strip_ascii_suffix_ignore_case(&decoded, suffix))
        .unwrap_or(&decoded);
    let base_suffix = match without_view.rsplit_once(['v', 'V']) {
        Some((prefix, version)) => {
            if !is_positive_ascii_digits(version, Some(PREPRINT_MAX_VERSION_DIGITS)) {
                return None;
            }
            prefix
        }
        None => without_view,
    };
    let valid_numeric_segments = !base_suffix.is_empty()
        && base_suffix.split('.').all(|segment| {
            !segment.is_empty() && segment.bytes().all(|byte| byte.is_ascii_digit())
        });
    if !valid_numeric_segments {
        return None;
    }
    Doi::parse(&format!("{PREPRINT_DOI_PREFIX}/{base_suffix}"))
        .ok()
        .map(ScholarlyId::Doi)
}

fn classify_biomedical_url(host: &str, path: &str) -> Option<ScholarlyId> {
    if matches!(
        host,
        "pmc.ncbi.nlm.nih.gov" | "www.ncbi.nlm.nih.gov" | "ncbi.nlm.nih.gov"
    ) {
        for segment in path.split('/') {
            if let Ok(pmc) = PmcId::parse(segment) {
                return Some(ScholarlyId::PmcId(pmc));
            }
        }
        if let Some(rest) = strip_ascii_prefix_ignore_case(path, "/pubmed/")
            && let Some(first) = rest.split('/').next()
            && let Ok(pmid) = Pmid::parse(first)
        {
            return Some(ScholarlyId::Pmid(pmid));
        }
    }

    if host == "pubmed.ncbi.nlm.nih.gov"
        && let Some(first) = path.trim_matches('/').split('/').next()
        && let Ok(pmid) = Pmid::parse(first)
    {
        return Some(ScholarlyId::Pmid(pmid));
    }

    if matches!(host, "europepmc.org" | "www.europepmc.org") {
        let trimmed = path.trim_matches('/');
        if let Some(rest) = strip_ascii_prefix_ignore_case(trimmed, "article/pmc/")
            && let Some(first) = rest.split('/').next()
        {
            let digits = strip_ascii_prefix_ignore_case(first, PMC_PREFIX).unwrap_or(first);
            if let Ok(pmc) = PmcId::parse(&format!("PMC{digits}")) {
                return Some(ScholarlyId::PmcId(pmc));
            }
        }
        if let Some(rest) = strip_ascii_prefix_ignore_case(trimmed, "articles/")
            && let Some(first) = rest.split('/').next()
            && let Ok(pmc) = PmcId::parse(first)
        {
            return Some(ScholarlyId::PmcId(pmc));
        }
        if let Some(rest) = strip_ascii_prefix_ignore_case(trimmed, "article/med/")
            && let Some(first) = rest.split('/').next()
            && let Ok(pmid) = Pmid::parse(first)
        {
            return Some(ScholarlyId::Pmid(pmid));
        }
    }

    None
}

fn classify_rfc_url(host: &str, path: &str) -> Option<ScholarlyId> {
    let trimmed = path.trim_end_matches('/');
    let candidate = if matches!(host, "www.rfc-editor.org" | "rfc-editor.org") {
        strip_ascii_prefix_ignore_case(trimmed, "/rfc/pdfrfc/")
            .or_else(|| strip_ascii_prefix_ignore_case(trimmed, "/rfc/"))
            .or_else(|| strip_ascii_prefix_ignore_case(trimmed, "/info/"))?
    } else if host == "datatracker.ietf.org" {
        strip_ascii_prefix_ignore_case(trimmed, "/doc/html/")
            .or_else(|| strip_ascii_prefix_ignore_case(trimmed, "/doc/"))?
    } else {
        return None;
    };

    if candidate.contains('/') {
        return None;
    }

    let base = strip_ascii_suffix_ignore_case(candidate, ".txt.pdf")
        .or_else(|| strip_ascii_suffix_ignore_case(candidate, ".txt"))
        .or_else(|| strip_ascii_suffix_ignore_case(candidate, ".xml"))
        .or_else(|| strip_ascii_suffix_ignore_case(candidate, ".html"))
        .or_else(|| strip_ascii_suffix_ignore_case(candidate, ".htm"))
        .or_else(|| strip_ascii_suffix_ignore_case(candidate, PDF_EXTENSION))
        .unwrap_or(candidate);

    RfcNumber::parse(base).ok().map(ScholarlyId::Rfc)
}

fn parse_doi_or_arxiv_datacite(raw: &str) -> Result<ScholarlyId, AdfontesError> {
    let doi = Doi::parse(raw)?;
    if let Some(arxiv_suffix) = strip_ascii_prefix_ignore_case(doi.as_str(), "10.48550/arxiv.") {
        let candidate = arxiv_suffix.trim_end_matches('/');
        if let Ok(arxiv) = ArxivId::parse(candidate) {
            return Ok(ScholarlyId::Arxiv(arxiv));
        }
    }
    Ok(ScholarlyId::Doi(doi))
}

fn looks_like_url_with_scheme(input: &str) -> bool {
    let Some((scheme, _)) = input.split_once("://") else {
        return false;
    };
    is_valid_rfc3986_scheme(scheme)
}

fn looks_like_non_scholarly_uri_scheme(input: &str) -> bool {
    let Some((scheme, _)) = input.split_once(':') else {
        return false;
    };
    is_valid_rfc3986_scheme(scheme)
}

fn is_valid_rfc3986_scheme(scheme: &str) -> bool {
    let mut chars = scheme.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && chars.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
        })
}

fn strip_ascii_prefix_ignore_case<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    let head = input.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| input.get(prefix.len()..))
        .flatten()
}

fn strip_ascii_suffix_ignore_case<'a>(input: &'a str, suffix: &str) -> Option<&'a str> {
    let prefix_len = input.len().checked_sub(suffix.len())?;
    let tail = input.get(prefix_len..)?;
    tail.eq_ignore_ascii_case(suffix)
        .then(|| input.get(..prefix_len))
        .flatten()
}

fn canonicalize_legacy_arxiv_archive(archive: &str, raw: &str) -> Result<String, AdfontesError> {
    if archive.is_empty() || archive.len() > ARXIV_LEGACY_ARCHIVE_MAX_LEN {
        return Err(AdfontesError::InvalidTarget {
            input: raw.to_owned(),
            reason: "invalid legacy arXiv archive prefix".to_owned(),
        });
    }
    let (primary, subject) = match archive.split_once('.') {
        Some((left, right)) => (left, Some(right)),
        None => (archive, None),
    };
    let primary_valid = is_hyphenated_ascii_letters(primary);
    let subject_valid = subject.is_none_or(is_hyphenated_ascii_letters);
    if !primary_valid || !subject_valid {
        return Err(AdfontesError::InvalidTarget {
            input: raw.to_owned(),
            reason: "legacy arXiv archive must use alphabetic segments without path traversal"
                .to_owned(),
        });
    }
    let canonical_primary = primary.to_ascii_lowercase();
    Ok(match subject {
        Some(class) => format!("{canonical_primary}.{class}"),
        None => canonical_primary,
    })
}

fn is_hyphenated_ascii_letters(segment: &str) -> bool {
    !segment.is_empty()
        && segment
            .split('-')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_alphabetic()))
}

fn validate_yymm_prefix(digits: &str, raw: &str) -> Result<(), AdfontesError> {
    let bytes = digits.as_bytes();
    let Some((&tens, &ones)) = bytes
        .get(YYMM_MONTH_TENS_INDEX)
        .zip(bytes.get(YYMM_MONTH_ONES_INDEX))
    else {
        return Err(AdfontesError::InvalidTarget {
            input: raw.to_owned(),
            reason: "arXiv YYMM prefix must contain 4 ASCII digits".to_owned(),
        });
    };
    let month = (tens - b'0') * DECIMAL_BASE + (ones - b'0');
    if !(CALENDAR_MIN_MONTH..=CALENDAR_MAX_MONTH).contains(&month) {
        return Err(AdfontesError::InvalidTarget {
            input: raw.to_owned(),
            reason: format!("arXiv month `{month:02}` must be between 01 and 12"),
        });
    }
    Ok(())
}

fn split_arxiv_version<'a>(
    segment: &'a str,
    raw: &str,
) -> Result<ArxivVersionSplit<'a>, AdfontesError> {
    if let Some((digits, version_digits)) = segment.rsplit_once(['v', 'V']) {
        if !is_positive_ascii_digits(version_digits, Some(ARXIV_MAX_VERSION_DIGITS)) {
            return Err(AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "arXiv version suffix must be a positive integer (`v1`, `v2`, ...)"
                    .to_owned(),
            });
        }
        let version = version_digits
            .parse::<u32>()
            .ok()
            .filter(|&number| number > 0)
            .ok_or_else(|| AdfontesError::InvalidTarget {
                input: raw.to_owned(),
                reason: "arXiv version number must be >= 1".to_owned(),
            })?;
        Ok(ArxivVersionSplit {
            digits,
            version: Some(version),
        })
    } else {
        Ok(ArxivVersionSplit {
            digits: segment,
            version: None,
        })
    }
}

fn parse_strict_positive_u32(
    digits: &str,
    raw: &str,
    field_name: &str,
) -> Result<u32, AdfontesError> {
    if !is_positive_ascii_digits(digits, None) {
        return Err(AdfontesError::InvalidTarget {
            input: raw.to_owned(),
            reason: format!("{field_name} must be a positive ASCII integer"),
        });
    }
    digits
        .parse::<u32>()
        .map_err(|_| AdfontesError::InvalidTarget {
            input: raw.to_owned(),
            reason: format!("{field_name} overflows u32"),
        })
}

fn is_positive_ascii_digits(digits: &str, max_len: Option<usize>) -> bool {
    !digits.is_empty()
        && max_len.is_none_or(|limit| digits.len() <= limit)
        && !digits.starts_with('0')
        && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn percent_decode_utf8(input: &str) -> Option<String> {
    percent_decode_str(input)
        .decode_utf8()
        .ok()
        .map(Cow::into_owned)
}

fn strip_trailing_sentence_punctuation(input: &str) -> &str {
    let mut current = input;
    loop {
        if let Some(stripped) = current.strip_suffix(['.', ',', ';']) {
            current = stripped;
            continue;
        }
        if let Some(stripped) = current.strip_suffix(')')
            && !trailing_paren_is_matched(stripped)
        {
            current = stripped;
            continue;
        }
        return current;
    }
}

fn trailing_paren_is_matched(prefix_without_closing_paren: &str) -> bool {
    let mut open_depth: usize = 0;
    for byte in prefix_without_closing_paren.bytes() {
        if byte == b'(' {
            open_depth += 1;
        } else if byte == b')' {
            open_depth = open_depth.saturating_sub(1);
        }
    }
    open_depth > 0
}

#[cfg(test)]
mod tests {
    use std::net::{Ipv4Addr, Ipv6Addr};

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("2409.13740", "2409.13740")]
    #[case("arXiv:2409.13740v2", "2409.13740v2")]
    #[case("https://arxiv.org/pdf/2411.00640v1.PDF", "2411.00640v1")]
    #[case("https://ar5iv.labs.arxiv.org/html/2402.09727", "2402.09727")]
    #[case("https://doi.org/10.48550%2FarXiv.2409.13740v1/", "2409.13740v1")]
    #[case("10.48550/arXiv.2409.13740", "2409.13740")]
    #[case("https://arxiv.org/abs/math%2F0211159v1", "math/0211159v1")]
    #[case("MATH/0211159v1", "math/0211159v1")]
    fn parses_arxiv_targets_and_datacite_aliases(#[case] raw: &str, #[case] expected_full: &str) {
        let parsed = ParsedTarget::parse(raw).unwrap();
        let expected = ArxivId::parse(expected_full).unwrap();
        assert_eq!(
            parsed,
            ParsedTarget::Scholarly(ScholarlyId::Arxiv(expected.clone()))
        );
        assert_eq!(expected.as_str(), expected_full);
    }

    #[rstest]
    #[case("2409.13740", "2409.13740")]
    #[case("2409.13740v2", "2409.13740")]
    #[case("math/0211159v1", "math/0211159")]
    #[case("math.GT/0211159v12", "math.GT/0211159")]
    fn extracts_versionless_arxiv_base_id(#[case] raw: &str, #[case] expected_base: &str) {
        let id = ArxivId::parse(raw).unwrap();
        assert_eq!(id.base_id(), expected_base);
    }

    #[test]
    fn scholarly_id_canonical_key_folds_arxiv_versions_and_case() {
        let modern_v1 = ScholarlyId::parse("2409.13740v1").unwrap().canonical_key();
        let modern_v2 = ScholarlyId::parse("2409.13740v2").unwrap().canonical_key();
        let legacy_upper = ScholarlyId::parse("math.GT/0211159v1")
            .unwrap()
            .canonical_key();
        let legacy_lower = ScholarlyId::parse("math.gt/0211159")
            .unwrap()
            .canonical_key();
        let doi_upper = ScholarlyId::parse("10.1038/S41586-025-10072-4")
            .unwrap()
            .canonical_key();
        let doi_lower = ScholarlyId::parse("10.1038/s41586-025-10072-4")
            .unwrap()
            .canonical_key();
        let pmc = ScholarlyId::parse("PMC10016948").unwrap().canonical_key();

        assert_eq!(modern_v1, modern_v2);
        assert_eq!(legacy_upper, legacy_lower);
        assert_eq!(doi_upper, doi_lower);
        assert_ne!(doi_lower, pmc);
    }

    #[rstest]
    #[case("arXiv:2409.13740v1")]
    #[case("10.1111/rssc.12435")]
    #[case("PMC10016948")]
    #[case("PMID:38491022")]
    #[case("RFC 9110")]
    fn scholarly_id_display_round_trips_through_parse(#[case] canonical: &str) {
        let parsed = ScholarlyId::parse(canonical).unwrap();
        assert_eq!(parsed.to_string(), canonical);
        assert_eq!(ScholarlyId::parse(&parsed.to_string()).unwrap(), parsed);
    }

    #[test]
    fn scholarly_id_parse_rejects_plain_urls_and_preserves_safety_blocks() {
        let plain_url_error = ScholarlyId::parse("https://example.org/paper.html").unwrap_err();
        let scheme_error = ScholarlyId::parse("file:///etc/passwd").unwrap_err();
        let userinfo_error =
            ScholarlyId::parse("https://user:pass@arxiv.org/abs/2409.13740").unwrap_err();

        assert!(matches!(
            plain_url_error,
            AdfontesError::InvalidTarget { .. }
        ));
        assert!(matches!(scheme_error, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(userinfo_error, AdfontesError::SafetyBlock { .. }));
    }

    #[rstest]
    #[case("../0211159")]
    #[case("./0211159")]
    #[case("-/0211159")]
    #[case("..../0211159")]
    #[case("https://doi.org/10.48550/arXiv...%2F0211159")]
    #[case("9999.9999")]
    #[case("0000.0000")]
    #[case("2413.12345")]
    #[case("2409.13740v0")]
    #[case("arxiv:arxiv:2409.13740")]
    fn rejects_invalid_and_traversal_arxiv_ids(#[case] raw: &str) {
        assert!(ArxivId::parse(raw).is_err());
    }

    #[rstest]
    #[case("10.1038/s41586-025-10072-4.", "10.1038/s41586-025-10072-4")]
    #[case("10.1038/s41586-025-10072-4.)", "10.1038/s41586-025-10072-4")]
    #[case("10.1038/s41586-025-10072-4))", "10.1038/s41586-025-10072-4")]
    #[case("10.1002/1097-0142(1999)", "10.1002/1097-0142(1999)")]
    #[case(
        "https://doi.org/10.1145%2F3539618.3591920/",
        "10.1145/3539618.3591920"
    )]
    #[case(
        "doi:10.1002/(SICI)1097-0258(19980430)17:8<857::AID-SIM777>3.0.CO;2-E",
        "10.1002/(SICI)1097-0258(19980430)17:8<857::AID-SIM777>3.0.CO;2-E"
    )]
    fn parses_percent_encoded_and_sici_dois(#[case] raw: &str, #[case] expected: &str) {
        let parsed = ParsedTarget::parse(raw).unwrap();
        let ParsedTarget::Scholarly(ScholarlyId::Doi(doi)) = parsed else {
            panic!("expected Doi variant for {raw}");
        };
        assert_eq!(doi.as_str(), expected);
    }

    #[rstest]
    #[case("10.1038/foo%2F..%2F..%2Fadmin")]
    #[case("10.1038/foo%5Cbar")]
    #[case("10.1038/foo%3Femail=evil")]
    #[case("10.1038/foo%23frag")]
    #[case("10.1038/bad%FFutf8")]
    #[case("doi:doi:10.1038/foo")]
    fn rejects_injected_or_malformed_dois(#[case] raw: &str) {
        assert!(Doi::parse(raw).is_err());
    }

    #[test]
    fn percent_encodes_sici_doi_for_api_paths() {
        let doi = Doi::parse("10.1002/(SICI)17:8<857>3.0").unwrap();
        assert_eq!(doi.percent_encoded_path(), "10.1002/(SICI)17:8%3C857%3E3.0");
    }

    #[rstest]
    #[case("pmc10912345", "PMC10912345")]
    #[case("https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/", "PMC7654321")]
    #[case("https://www.europepmc.org/article/pmc/pmc998877/pdf", "PMC998877")]
    #[case("https://europepmc.org/articles/PMC112233", "PMC112233")]
    fn parses_pmcid_variants(#[case] raw: &str, #[case] expected: &str) {
        let parsed = ParsedTarget::parse(raw).unwrap();
        let ParsedTarget::Scholarly(ScholarlyId::PmcId(pmc)) = parsed else {
            panic!("expected PmcId variant for {raw}");
        };
        assert_eq!(pmc.as_str(), expected);
    }

    #[rstest]
    #[case("PMC0")]
    #[case("PMC 123")]
    #[case("PMC0123")]
    #[case("pmcid:pmcid:PMC123")]
    fn rejects_invalid_pmcids(#[case] raw: &str) {
        assert!(PmcId::parse(raw).is_err());
    }

    #[rstest]
    #[case("pmid:38491022", "38491022")]
    #[case("https://pubmed.ncbi.nlm.nih.gov/38491022/", "38491022")]
    #[case("https://www.ncbi.nlm.nih.gov/pubmed/38491022", "38491022")]
    #[case("https://europepmc.org/article/med/38491022/abstract", "38491022")]
    fn parses_pmid_variants(#[case] raw: &str, #[case] expected: &str) {
        let parsed = ParsedTarget::parse(raw).unwrap();
        let ParsedTarget::Scholarly(ScholarlyId::Pmid(pmid)) = parsed else {
            panic!("expected Pmid variant for {raw}");
        };
        assert_eq!(pmid.as_str(), expected);
    }

    #[rstest]
    #[case("0")]
    #[case("001234")]
    #[case("1234567890123")]
    #[case("pmid:pmid:12345")]
    fn rejects_invalid_pmids(#[case] raw: &str) {
        assert!(Pmid::parse(raw).is_err());
    }

    #[rstest]
    #[case("RFC 9110", 9110)]
    #[case("rfc-8650", 8650)]
    #[case("https://www.rfc-editor.org/rfc/rfc9110.xml", 9110)]
    #[case("https://www.rfc-editor.org/rfc/pdfrfc/rfc9110.txt.pdf", 9110)]
    #[case("https://datatracker.ietf.org/doc/html/rfc2119", 2119)]
    fn parses_rfc_variants(#[case] raw: &str, #[case] expected_number: u32) {
        let parsed = ParsedTarget::parse(raw).unwrap();
        let ParsedTarget::Scholarly(ScholarlyId::Rfc(rfc)) = parsed else {
            panic!("expected Rfc variant for {raw}");
        };
        assert_eq!(rfc.get(), expected_number);
    }

    #[rstest]
    #[case("-9110")]
    #[case("RFC0")]
    #[case("RFC---9110")]
    #[case("RFC100000")]
    fn rejects_invalid_rfc_numbers(#[case] raw: &str) {
        assert!(RfcNumber::parse(raw).is_err());
    }

    #[rstest]
    #[case(
        "https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v2.full",
        "10.1101/2024.06.03.24308385"
    )]
    #[case(
        "https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v1.full.pdf+html",
        "10.1101/2024.06.03.24308385"
    )]
    #[case(
        "https://www.biorxiv.org/content/10.1101/2023.01.15.524123v3.full.pdf",
        "10.1101/2023.01.15.524123"
    )]
    #[case(
        "https://biorxiv.org/content/10.1101/123456v1.abstract",
        "10.1101/123456"
    )]
    #[case(
        "https://medrxiv.org/content/10.1101/2024.06.03.24308385",
        "10.1101/2024.06.03.24308385"
    )]
    #[case(
        "https://www.biorxiv.org/content/10.1101/2023.01.15.524123%762.full/",
        "10.1101/2023.01.15.524123"
    )]
    fn classifies_biorxiv_and_medrxiv_preprint_doi_urls(
        #[case] raw: &str,
        #[case] expected_doi: &str,
    ) {
        let parsed = ParsedTarget::parse(raw).unwrap();
        let ParsedTarget::Scholarly(ScholarlyId::Doi(doi)) = parsed else {
            panic!("expected Doi variant for {raw}");
        };
        assert_eq!(doi.as_str(), expected_doi);
    }

    #[rstest]
    #[case("https://www.rfc-editor.org/errata/rfc9110")]
    #[case("https://datatracker.ietf.org/doc/rfc9110/history/")]
    #[case("https://example.org/research/article.html")]
    #[case("https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v0.full")]
    #[case("https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v01.full")]
    #[case("https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v1000.full")]
    #[case("https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v1.full.full")]
    #[case("https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v1/supplementary-material")]
    #[case("https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v1%2Fextra")]
    #[case("https://www.biorxiv.org/content/10.1101/early/2024/06/03/24308385")]
    #[case("https://www.biorxiv.org/content/10.1101/2024..03.24308385v1")]
    #[case("https://www.biorxiv.org/content/10.1101/not-a-doi")]
    #[case("https://www.medrxiv.org/content/early/recent")]
    fn keeps_non_canonical_urls_as_url_variant(#[case] raw: &str) {
        let parsed = ParsedTarget::parse(raw).unwrap();
        assert!(matches!(parsed, ParsedTarget::Url(_)));
    }

    #[test]
    fn exposes_typed_unbracketed_hosts_for_ipv4_ipv6_and_domains() {
        let ipv6 = HttpUrl::parse("http://[::1]:8080/paper.pdf").unwrap();
        let ipv4 = HttpUrl::parse("http://127.0.0.1:8080/paper.pdf").unwrap();
        let fqdn = HttpUrl::parse("https://arxiv.org./abs/2409.13740").unwrap();
        assert_eq!(ipv6.host(), UrlHost::Ip(IpAddr::V6(Ipv6Addr::LOCALHOST)));
        assert_eq!(ipv4.host(), UrlHost::Ip(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert_eq!(fqdn.host(), UrlHost::Domain("arxiv.org"));
    }

    #[test]
    fn rewrites_github_blob_urls_and_rejects_malformed_paths() {
        let blob = HttpUrl::parse(
            "https://github.com/python/peps/blob/main/peps/pep-0695.rst?plain=1#abstract",
        )
        .unwrap();
        let tree = HttpUrl::parse("https://github.com/python/peps/tree/main/peps").unwrap();
        let double_slash =
            HttpUrl::parse("https://github.com/owner/repo//blob/main/file.md").unwrap();
        let custom_port =
            HttpUrl::parse("https://github.com:8443/owner/repo/blob/main/file.md").unwrap();
        assert_eq!(
            blob.to_raw_github_url().unwrap().as_str(),
            "https://raw.githubusercontent.com/python/peps/main/peps/pep-0695.rst"
        );
        assert_eq!(tree.to_raw_github_url(), None);
        assert_eq!(double_slash.to_raw_github_url(), None);
        assert_eq!(custom_port.to_raw_github_url(), None);
    }

    #[test]
    fn join_redirect_resolves_relative_paths() {
        let base = HttpUrl::parse("https://example.org/papers/index.html").unwrap();
        assert_eq!(
            base.join_redirect("../pdf/main.pdf").unwrap().as_str(),
            "https://example.org/pdf/main.pdf"
        );
    }

    #[rstest]
    #[case("")]
    #[case("https://evil\r\n.example.org/p")]
    #[case("/foo\x00bar")]
    fn join_redirect_rejects_empty_and_control_locations(#[case] location: &str) {
        let base = HttpUrl::parse("https://example.org/papers/index.html").unwrap();
        assert!(matches!(
            base.join_redirect(location).unwrap_err(),
            AdfontesError::InvalidTarget { .. }
        ));
    }

    #[test]
    fn join_redirect_blocks_unsafe_scheme_downgrade() {
        let base = HttpUrl::parse("https://example.org/papers/index.html").unwrap();
        assert!(matches!(
            base.join_redirect("file:///etc/passwd").unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[rstest]
    #[case("file:///etc/passwd")]
    #[case("file:/etc/passwd")]
    #[case("javascript:alert(1)")]
    #[case("data:text/html,hello")]
    #[case("ftp://example.com/paper.pdf")]
    #[case("https://user:secret@example.com/paper.pdf")]
    #[case("https://user@example.com/paper.pdf")]
    #[case("https://:secret@example.com/paper.pdf")]
    fn blocks_unsafe_schemes_and_credentials(#[case] raw: &str) {
        let failure = ParsedTarget::parse(raw).unwrap_err();
        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn rejects_zero_port_urls() {
        assert!(HttpUrl::parse("http://example.org:0/paper.pdf").is_err());
    }

    #[rstest]
    #[case(
        "https://api.example.org/v2/works",
        &[],
        "https://api.example.org/v2/works"
    )]
    #[case(
        "https://api.example.org/v2/works",
        &[("q", "hello world"), ("frag", "a#b\nc")],
        "https://api.example.org/v2/works?q=hello+world&frag=a%23b%0Ac"
    )]
    #[case(
        "https://api.example.org/v2/works?select=id",
        &[("mailto", "a+b@uni.edu"), ("tag", "q&r=1")],
        "https://api.example.org/v2/works?select=id&mailto=a%2Bb%40uni.edu&tag=q%26r%3D1"
    )]
    fn appends_form_urlencoded_query_pairs_without_dangling_question_mark(
        #[case] raw_base: &str,
        #[case] pairs: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let base = HttpUrl::parse(raw_base).unwrap();
        let updated = base.with_query_pairs(pairs.iter().copied());
        assert_eq!(updated.as_str(), expected);
        assert_eq!(updated.0.fragment(), None);
    }

    #[rstest]
    #[case("0-2")]
    #[case("5-3")]
    #[case("1-9")]
    #[case("+1-+3")]
    #[case("abc")]
    fn rejects_invalid_page_spans(#[case] spec: &str) {
        let failure = PageSpan::parse_spec(spec).unwrap_err();
        assert!(matches!(failure, AdfontesError::InvalidTarget { .. }));
    }

    #[test]
    fn clamps_valid_page_span_to_document_length() {
        let span = PageSpan::parse_spec("18-25")
            .unwrap()
            .clamp_to_total(20)
            .unwrap();
        assert_eq!(span.start(), 18);
        assert_eq!(span.end(), 20);
        assert_eq!(span.to_string(), "18-20");
    }

    #[test]
    fn rejects_page_span_exceeding_document_length() {
        let span = PageSpan::parse_spec("5").unwrap();
        assert!(span.clamp_to_total(0).is_err());
        assert!(span.clamp_to_total(4).is_err());
    }

    #[rstest]
    #[case("abcdef", 0, "…")]
    #[case("abcdef", 6, "abcdef")]
    #[case("abcdef", 3, "abc…")]
    #[case("αβγδε", 2, "αβ…")]
    fn truncates_strings_by_unicode_scalar_count(
        #[case] input: &str,
        #[case] max_chars: usize,
        #[case] expected: &str,
    ) {
        assert_eq!(truncate_chars(input, max_chars), expected);
    }
}
