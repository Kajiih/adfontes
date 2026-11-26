//! Keyless open-access scholarly retrieval waterfalls and web/spec content dispatch.
//!
//! Resolves [`ParsedTarget`] identifiers (`arXiv`, `DOI`, `PMCID`, `PMID`, `RFC`, and web/GitHub
//! URLs) via preference-ordered open-access sources that favor structured digital text (`Markdown`,
//! `LaTeXML` HTML, `JATS` XML, `RFCXML`, flattened `TeX`) ahead of binary PDFs and return honest
//! [`ExtractedPayload::MetadataOnly`] outcomes when no legal open-access copy exists.

architecture_component!(Resolve);

mod providers;

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use self::providers::{JsonDoc, OaCandidates};
use crate::extract::{
    JatsExtraction, XmlDocumentExtraction, clean_rfc_text, extract_arxiv_source_archive,
    extract_html_to_markdown, extract_jats_xml, extract_markdown_toc, extract_rfc_xml,
    extract_text_title, extract_xml_document, normalize_markdown_spacing, parse_arxiv_atom_feed,
};
use crate::model::{
    AdfontesError, ArxivId, Doi, ExtractedDocument, ExtractedPayload, HttpUrl, ParsedTarget, PmcId,
    Pmid, RfcNumber, TextFormat, UnavailableReason, WorkMetadata, classify_known_scholarly_url,
};
use crate::pdf::is_pdf_magic;
use crate::transport::{HttpClient, HttpResponse};

const ACCEPT_HTML: &str = "text/html, application/xhtml+xml;q=0.9";
const ACCEPT_XML: &str = "application/xml, text/xml;q=0.9";
const ACCEPT_JSON: &str = "application/json";
const ACCEPT_CSL_JSON: &str = "application/vnd.citationstyles.csl+json, application/json;q=0.8";
const ACCEPT_PDF: &str = "application/pdf, */*;q=0.1";
const ACCEPT_WEB_PREFERRED: &str =
    "text/markdown, text/html;q=0.9, application/pdf;q=0.8, text/plain;q=0.7, */*;q=0.5";
const MIME_PLAIN_TEXT: &str = "text/plain";
const PDF_MAGIC_PREFIX: &[u8] = b"%PDF-";

const HTTP_OK: u16 = 200;
const MAX_RECLASSIFICATION_DEPTH: u8 = 3;
const MAX_WATERFALL_HTTP_REQUESTS: usize = 24;
const MAX_WATERFALL_DURATION: Duration = Duration::from_secs(90);
const MIN_FULLTEXT_CHARS: usize = 160;
const MIN_OA_LANDING_HTML_CHARS: usize = 2_500;
const MIN_OA_LANDING_HEADINGLESS_CHARS: usize = 5_000;
const MIN_OA_PAYWALL_OVERRIDE_CHARS: usize = 6_000;
const MIN_OA_BODY_HEADINGS: usize = 2;
const MIN_OA_PAYWALL_OVERRIDE_HEADINGS: usize = 3;
const MAX_ARXIV_STUB_CHECK_LEN: usize = 2_500;
const MAX_OA_CANDIDATES: usize = 6;
const MAX_EMAIL_LOCAL_LEN: usize = 64;
const MAX_EMAIL_TOTAL_LEN: usize = 254;

const PLACEHOLDER_EMAIL_DOMAINS: &[&str] =
    &["example.com", "example.org", "example.net", "test.com"];
const RESERVED_EMAIL_TLDS: &[&str] = &["example", "invalid", "test", "localhost"];
const PLACEHOLDER_EMAIL_LOCALS: &[&str] = &["test", "user", "placeholder", "your_email", "email"];

const GENERIC_REPOSITORY_VENUES: &[&str] = &["arXiv", "Europe PMC", "PubMed"];

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

/// Base service URLs used by scholarly retrieval waterfalls.
///
/// Defaults to official HTTPS upstream endpoints in production and can be pointed at local
/// loopback fixtures in hermetic tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolverEndpoints {
    /// Base origin for `arxiv.org` (`/html/`, `/pdf/`).
    pub arxiv_org: String,
    /// Base origin for `export.arxiv.org` (`/api/query`, `/src/`).
    pub export_arxiv_org: String,
    /// Base origin for `ar5iv.labs.arxiv.org` (`/html/`).
    pub ar5iv: String,
    /// Base origin for `doi.org` (CSL-JSON content negotiation).
    pub doi_org: String,
    /// Base origin for `api.openalex.org`.
    pub openalex: String,
    /// Base origin for `api.unpaywall.org`.
    pub unpaywall: String,
    /// Base origin for `www.ebi.ac.uk/europepmc`.
    pub europe_pmc: String,
    /// Base origin for `datatracker.ietf.org`.
    pub ietf_datatracker: String,
    /// Base origin for `www.rfc-editor.org`.
    pub rfc_editor: String,
}

impl Default for ResolverEndpoints {
    fn default() -> Self {
        Self::production()
    }
}

impl ResolverEndpoints {
    /// Returns the default production HTTPS endpoints for all scholarly repositories.
    #[must_use]
    pub fn production() -> Self {
        Self {
            arxiv_org: "https://arxiv.org".to_owned(),
            export_arxiv_org: "https://export.arxiv.org".to_owned(),
            ar5iv: "https://ar5iv.labs.arxiv.org".to_owned(),
            doi_org: "https://doi.org".to_owned(),
            openalex: "https://api.openalex.org".to_owned(),
            unpaywall: "https://api.unpaywall.org".to_owned(),
            europe_pmc: "https://www.ebi.ac.uk/europepmc".to_owned(),
            ietf_datatracker: "https://datatracker.ietf.org".to_owned(),
            rfc_editor: "https://www.rfc-editor.org".to_owned(),
        }
    }
}

/// Runtime configuration for scholarly resolution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolverConfig {
    /// Optional contact email (`ADFONTES_CONTACT_EMAIL`) for polite-pool OpenAlex and Unpaywall.
    pub contact_email: Option<String>,
    /// Upstream repository base URLs.
    pub endpoints: ResolverEndpoints,
}

impl ResolverConfig {
    /// Builds a [`ResolverConfig`] from process environment variables once at startup.
    #[must_use]
    pub fn from_env() -> Self {
        let contact_email = std::env::var("ADFONTES_CONTACT_EMAIL")
            .ok()
            .map(|raw| raw.trim().to_owned())
            .filter(|email| is_usable_contact_email(email));
        Self {
            contact_email,
            endpoints: ResolverEndpoints::production(),
        }
    }

    /// Returns the configured email only if it passes non-placeholder syntactic validation.
    #[must_use]
    pub fn usable_contact_email(&self) -> Option<&str> {
        self.contact_email
            .as_deref()
            .map(str::trim)
            .filter(|email| is_usable_contact_email(email))
    }
}

/// Resolves any validated [`ParsedTarget`] through its keyless open-access retrieval waterfall.
///
/// # Errors
///
/// Propagates [`AdfontesError::SafetyBlock`] immediately if any transport, archive, or XML safety
/// ceiling trips, or returns [`AdfontesError::Transport`] when all upstream endpoints fail without
/// yielding even bibliographic metadata.
pub async fn resolve_target(
    target: &ParsedTarget,
    client: &HttpClient,
    config: &ResolverConfig,
) -> Result<ExtractedDocument, AdfontesError> {
    let mut visited = BTreeSet::new();
    let budget = ResolveBudget::new(client);
    resolve_target_inner(target, &budget, config, 0, &mut visited).await
}

struct ResolveBudget<'a> {
    client: &'a HttpClient,
    started_at: Instant,
    requests_made: AtomicUsize,
    last_transport_error: Mutex<Option<String>>,
}

impl<'a> ResolveBudget<'a> {
    fn new(client: &'a HttpClient) -> Self {
        Self {
            client,
            started_at: Instant::now(),
            requests_made: AtomicUsize::new(0),
            last_transport_error: Mutex::new(None),
        }
    }

    fn is_exhausted(&self) -> bool {
        self.requests_made.load(Ordering::Acquire) >= MAX_WATERFALL_HTTP_REQUESTS
            || self.started_at.elapsed() >= MAX_WATERFALL_DURATION
    }

    async fn get(
        &self,
        url: &HttpUrl,
        accept: Option<&str>,
    ) -> Result<HttpResponse, AdfontesError> {
        if self.started_at.elapsed() >= MAX_WATERFALL_DURATION {
            let detail = format!(
                "waterfall deadline exceeded ({}s)",
                MAX_WATERFALL_DURATION.as_secs()
            );
            self.record_transport_error_if_unset(&detail);
            return Err(AdfontesError::Transport {
                url: url.as_str().to_owned(),
                detail,
            });
        }
        if self
            .requests_made
            .try_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < MAX_WATERFALL_HTTP_REQUESTS).then_some(count + 1)
            })
            .is_err()
        {
            let detail = format!(
                "waterfall HTTP request budget exhausted ({MAX_WATERFALL_HTTP_REQUESTS} requests)"
            );
            self.record_transport_error_if_unset(&detail);
            return Err(AdfontesError::Transport {
                url: url.as_str().to_owned(),
                detail,
            });
        }
        let result = self.client.get(url, accept, &[]).await;
        if let Err(AdfontesError::Transport { ref detail, .. }) = result {
            self.record_transport_error(detail);
        }
        result
    }

    fn append_last_transport_error(&self, summary: &str) -> String {
        let guard = self
            .last_transport_error
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        match *guard {
            Some(ref detail) if !detail.is_empty() => format!("{summary} ({detail})"),
            _ => summary.to_owned(),
        }
    }

    fn record_transport_error(&self, detail: &str) {
        let mut guard = self
            .last_transport_error
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *guard = Some(detail.to_owned());
    }

    fn record_transport_error_if_unset(&self, detail: &str) {
        let mut guard = self
            .last_transport_error
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if guard.is_none() {
            *guard = Some(detail.to_owned());
        }
    }
}

async fn resolve_target_inner(
    target: &ParsedTarget,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
    depth: u8,
    visited: &mut BTreeSet<String>,
) -> Result<ExtractedDocument, AdfontesError> {
    visited.insert(target_visit_key(target));
    match target {
        ParsedTarget::Arxiv(id) => resolve_arxiv(id, budget, config).await,
        ParsedTarget::Doi(doi) => resolve_doi(doi, budget, config, depth, visited).await,
        ParsedTarget::PmcId(pmcid) => resolve_pmcid(pmcid, budget, config).await,
        ParsedTarget::Pmid(pmid) => resolve_pmid(pmid, budget, config, depth, visited).await,
        ParsedTarget::Rfc(rfc) => resolve_rfc(*rfc, budget, config).await,
        ParsedTarget::Url(url) => resolve_url(url, budget, config, depth, visited).await,
    }
}

async fn resolve_arxiv(
    id: &ArxivId,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<ExtractedDocument, AdfontesError> {
    let mut metadata = WorkMetadata {
        canonical_id: Some(format!("arXiv:{id}")),
        venue_or_source: Some("arXiv".to_owned()),
        ..WorkMetadata::default()
    };
    let mut failures = Vec::new();

    // 1. Metadata via Atom API (gracefully survive rate-limiting / HTTP 429 / 503).
    fetch_arxiv_atom_metadata(id, budget, config, &mut metadata, &mut failures).await?;

    // 2. Official arXiv LaTeXML HTML (`arxiv.org/html/<id>`).
    if let Some(doc) = try_arxiv_html_endpoint(
        &format!(
            "{}/html/{}",
            config.endpoints.arxiv_org.trim_end_matches('/'),
            id.as_str()
        ),
        budget,
        &mut metadata,
        &mut failures,
        "arxiv-html",
    )
    .await?
    {
        return Ok(doc);
    }

    // 3. ar5iv HTML mirror (`ar5iv.labs.arxiv.org/html/<id>`), rejecting redirects to `/abs/`.
    if let Some(doc) = try_arxiv_html_endpoint(
        &format!(
            "{}/html/{}",
            config.endpoints.ar5iv.trim_end_matches('/'),
            id.as_str()
        ),
        budget,
        &mut metadata,
        &mut failures,
        "ar5iv",
    )
    .await?
    {
        return Ok(doc);
    }

    // 4. LaTeX source archive (`export.arxiv.org/src/<id>`).
    if let Some(doc) =
        try_arxiv_source_endpoint(id, budget, config, &mut metadata, &mut failures).await?
    {
        return Ok(doc);
    }

    // 5. Fallback to canonical arXiv PDF (`arxiv.org/pdf/<id>.pdf`).
    let pdf_url = HttpUrl::parse(&format!(
        "{}/pdf/{}.pdf",
        config.endpoints.arxiv_org.trim_end_matches('/'),
        id.as_str()
    ))?;
    if let Some(doc) =
        try_arxiv_pdf_endpoint(&pdf_url, budget, &mut metadata, &mut failures).await?
    {
        return Ok(doc);
    }

    if metadata.title.is_some() || metadata.abstract_text.is_some() {
        metadata.source_url = HttpUrl::parse(&format!(
            "{}/abs/{}",
            config.endpoints.arxiv_org.trim_end_matches('/'),
            id.as_str()
        ))
        .ok();
        return Ok(ExtractedDocument {
            metadata,
            payload: ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::UpstreamUnavailable {
                    detail: failures.join("; "),
                },
            },
            raw_bytes_fetched: 0,
        });
    }

    Err(AdfontesError::Transport {
        url: pdf_url.as_str().to_owned(),
        detail: format!(
            "all arXiv retrieval endpoints failed ({})",
            failures.join("; ")
        ),
    })
}

async fn fetch_arxiv_atom_metadata(
    id: &ArxivId,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
    metadata: &mut WorkMetadata,
    failures: &mut Vec<String>,
) -> Result<(), AdfontesError> {
    let atom_base_url = HttpUrl::parse(&format!(
        "{}/api/query",
        config.endpoints.export_arxiv_org.trim_end_matches('/'),
    ))?;
    let atom_url = atom_base_url.with_query_pairs([("id_list", id.as_str())]);
    match budget.get(&atom_url, Some(ACCEPT_XML)).await {
        Ok(resp) if resp.status == HTTP_OK => {
            let xml = resp.decode_text();
            match parse_arxiv_atom_feed(&xml) {
                Ok(entry) => {
                    metadata.title = entry.title;
                    metadata.authors = entry.authors;
                    metadata.year = entry.year;
                    metadata.abstract_text = entry.abstract_text;
                }
                Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
                Err(other) => failures.push(format!("atom: {other}")),
            }
        }
        Ok(resp) => failures.push(format!("atom: HTTP {}", resp.status)),
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Err(other) => failures.push(format!("atom: {other}")),
    }
    Ok(())
}

async fn try_arxiv_html_endpoint(
    raw_url: &str,
    budget: &ResolveBudget<'_>,
    metadata: &mut WorkMetadata,
    failures: &mut Vec<String>,
    label: &str,
) -> Result<Option<ExtractedDocument>, AdfontesError> {
    let url = HttpUrl::parse(raw_url)?;
    match budget.get(&url, Some(ACCEPT_HTML)).await {
        Ok(resp) if resp.status == HTTP_OK => {
            if is_arxiv_abs_url(&resp.final_url) {
                failures.push(format!("{label}: redirected to /abs/"));
                return Ok(None);
            }
            let mime = resp.mime_type();
            if !is_html_mime(&mime) {
                failures.push(format!("{label}: unexpected MIME `{mime}`"));
                return Ok(None);
            }
            let decoded = resp.decode_text();
            let extracted = extract_html_to_markdown(&decoded);
            if extracted.markdown.len() < MAX_ARXIV_STUB_CHECK_LEN
                && looks_like_arxiv_no_html_stub(&extracted.markdown)
            {
                failures.push(format!("{label}: HTML unavailable notice"));
                return Ok(None);
            }
            if extracted.markdown.len() < MIN_FULLTEXT_CHARS {
                failures.push(format!("{label}: extracted body too short"));
                return Ok(None);
            }
            if metadata.title.is_none() {
                metadata.title = extracted.title;
            }
            metadata.source_url = Some(resp.final_url);
            Ok(Some(ExtractedDocument {
                metadata: metadata.clone(),
                payload: ExtractedPayload::FullText {
                    format: TextFormat::Html,
                    markdown: extracted.markdown,
                },
                raw_bytes_fetched: resp.body.len(),
            }))
        }
        Ok(resp) => {
            failures.push(format!("{label}: HTTP {}", resp.status));
            Ok(None)
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => Err(safety),
        Err(other) => {
            failures.push(format!("{label}: {other}"));
            Ok(None)
        }
    }
}

async fn try_arxiv_source_endpoint(
    id: &ArxivId,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
    metadata: &mut WorkMetadata,
    failures: &mut Vec<String>,
) -> Result<Option<ExtractedDocument>, AdfontesError> {
    let src_url = HttpUrl::parse(&format!(
        "{}/src/{}",
        config.endpoints.export_arxiv_org.trim_end_matches('/'),
        id.as_str()
    ))?;
    match budget.get(&src_url, None).await {
        Ok(resp) if resp.status == HTTP_OK => {
            if is_pdf_response(&resp) {
                let byte_len = resp.body.len();
                metadata.source_url = Some(resp.final_url);
                return Ok(Some(ExtractedDocument {
                    metadata: metadata.clone(),
                    payload: ExtractedPayload::Pdf { bytes: resp.body },
                    raw_bytes_fetched: byte_len,
                }));
            }
            match extract_arxiv_source_archive(&resp.body) {
                Ok(extracted) if extracted.markdown.len() >= MIN_FULLTEXT_CHARS => {
                    if metadata.title.is_none() {
                        metadata.title = extracted.title;
                    }
                    metadata.source_url = Some(resp.final_url);
                    Ok(Some(ExtractedDocument {
                        metadata: metadata.clone(),
                        payload: ExtractedPayload::FullText {
                            format: TextFormat::Latex,
                            markdown: extracted.markdown,
                        },
                        raw_bytes_fetched: resp.body.len(),
                    }))
                }
                Ok(_) => {
                    failures.push("src: extracted TeX was too short".to_owned());
                    Ok(None)
                }
                Err(safety @ AdfontesError::SafetyBlock { .. }) => Err(safety),
                Err(other) => {
                    failures.push(format!("src: {other}"));
                    Ok(None)
                }
            }
        }
        Ok(resp) => {
            failures.push(format!("src: HTTP {}", resp.status));
            Ok(None)
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => Err(safety),
        Err(other) => {
            failures.push(format!("src: {other}"));
            Ok(None)
        }
    }
}

async fn try_arxiv_pdf_endpoint(
    pdf_url: &HttpUrl,
    budget: &ResolveBudget<'_>,
    metadata: &mut WorkMetadata,
    failures: &mut Vec<String>,
) -> Result<Option<ExtractedDocument>, AdfontesError> {
    match budget.get(pdf_url, Some(ACCEPT_PDF)).await {
        Ok(resp) if resp.status == HTTP_OK && is_pdf_response(&resp) => {
            let byte_len = resp.body.len();
            metadata.source_url = Some(resp.final_url);
            Ok(Some(ExtractedDocument {
                metadata: metadata.clone(),
                payload: ExtractedPayload::Pdf { bytes: resp.body },
                raw_bytes_fetched: byte_len,
            }))
        }
        Ok(resp) => {
            failures.push(format!("pdf: HTTP {}", resp.status));
            Ok(None)
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => Err(safety),
        Err(other) => {
            failures.push(format!("pdf: {other}"));
            Ok(None)
        }
    }
}

async fn resolve_pmid(
    pmid: &Pmid,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
    depth: u8,
    visited: &mut BTreeSet<String>,
) -> Result<ExtractedDocument, AdfontesError> {
    let base = config.endpoints.europe_pmc.trim_end_matches('/');
    let search_url = HttpUrl::parse(&format!(
        "{base}/webservices/rest/search?query=EXT_ID:{}%20AND%20SRC:MED&format=json&resultType=core",
        pmid.as_str()
    ))?;
    let mut metadata = WorkMetadata {
        canonical_id: Some(format!("PMID:{}", pmid.as_str())),
        venue_or_source: Some("PubMed".to_owned()),
        source_url: HttpUrl::parse(&format!(
            "https://pubmed.ncbi.nlm.nih.gov/{}/",
            pmid.as_str()
        ))
        .ok(),
        ..WorkMetadata::default()
    };
    let mut oa_pdf_urls = Vec::new();

    let Some(search_doc) = fetch_optional_json(&search_url, ACCEPT_JSON, budget).await? else {
        return Err(AdfontesError::Transport {
            url: search_url.as_str().to_owned(),
            detail: budget.append_last_transport_error("Europe PMC PMID lookup failed"),
        });
    };
    let Some(ids) = search_doc.extract_europe_pmc_record(&mut metadata, &mut oa_pdf_urls) else {
        return Err(AdfontesError::Transport {
            url: search_url.as_str().to_owned(),
            detail: format!("PMID {} not found in Europe PMC", pmid.as_str()),
        });
    };

    let mut saw_body_restricted = false;

    if let Some(pmcid) = ids.pmcid
        && visited.insert(target_visit_key(&ParsedTarget::PmcId(pmcid.clone())))
    {
        match resolve_pmcid(&pmcid, budget, config).await {
            Ok(pmc_doc) if !matches!(pmc_doc.payload, ExtractedPayload::MetadataOnly { .. }) => {
                return Ok(merge_caller_priority_metadata(pmc_doc, metadata));
            }
            Ok(pmc_doc) => {
                saw_body_restricted |= is_body_restricted_doc(&pmc_doc);
                merge_missing_metadata(&mut metadata, pmc_doc.metadata);
            }
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            Err(_) => {}
        }
    }

    if depth < MAX_RECLASSIFICATION_DEPTH
        && !budget.is_exhausted()
        && let Some(parsed_target) = ids.follow_up_target
        && !visited.contains(&target_visit_key(&parsed_target))
    {
        match Box::pin(resolve_target_inner(
            &parsed_target,
            budget,
            config,
            depth + 1,
            visited,
        ))
        .await
        {
            Ok(doc) if !matches!(doc.payload, ExtractedPayload::MetadataOnly { .. }) => {
                return Ok(merge_caller_priority_metadata(doc, metadata));
            }
            Ok(doc) => {
                saw_body_restricted |= is_body_restricted_doc(&doc);
                merge_missing_metadata(&mut metadata, doc.metadata);
            }
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            Err(_) => {}
        }
    }

    if let Some(pdf_doc) = try_oa_pdf_urls(&oa_pdf_urls, &mut metadata, budget).await? {
        return Ok(pdf_doc);
    }

    let reason = if saw_body_restricted {
        UnavailableReason::BodyRestricted
    } else {
        UnavailableReason::NoOpenAccessCopy
    };
    Ok(ExtractedDocument {
        metadata,
        payload: ExtractedPayload::MetadataOnly { reason },
        raw_bytes_fetched: 0,
    })
}

async fn resolve_doi(
    doi: &Doi,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
    depth: u8,
    visited: &mut BTreeSet<String>,
) -> Result<ExtractedDocument, AdfontesError> {
    let encoded_doi = doi.percent_encoded_path();
    let doi_canonical_url = HttpUrl::parse(&format!(
        "{}/{encoded_doi}",
        config.endpoints.doi_org.trim_end_matches('/')
    ))?;
    let mut metadata = WorkMetadata {
        canonical_id: Some(doi.as_str().to_owned()),
        source_url: Some(doi_canonical_url.clone()),
        ..WorkMetadata::default()
    };
    let mut oa_candidates = OaCandidates::default();

    let openalex_url = build_openalex_doi_url(config, &encoded_doi)?;
    let unpaywall_url = config
        .usable_contact_email()
        .map(|email| {
            HttpUrl::parse(&format!(
                "{}/v2/{encoded_doi}",
                config.endpoints.unpaywall.trim_end_matches('/')
            ))
            .map(|base| base.with_query_pairs([("email", email)]))
        })
        .transpose()?;

    // Run independent metadata legs concurrently, cancelling siblings immediately on fatal errors
    // such as `SafetyBlock`, while preserving deterministic 1 -> 2 -> 3 merges.
    let (csl_opt, oa_opt, upw_opt) = tokio::try_join!(
        fetch_optional_json(&doi_canonical_url, ACCEPT_CSL_JSON, budget),
        fetch_optional_json(&openalex_url, ACCEPT_JSON, budget),
        async {
            match unpaywall_url.as_ref() {
                Some(url) => fetch_optional_json(url, ACCEPT_JSON, budget).await,
                None => Ok(None),
            }
        },
    )?;
    let metadata_found = csl_opt.is_some() || oa_opt.is_some() || upw_opt.is_some();

    // 1. CSL-JSON content negotiation via `doi.org`.
    if let Some(csl) = csl_opt {
        csl.merge_csl_work(&mut metadata);
    }

    // 2. OpenAlex work record (authors, venue, inverted-index abstract, PMCID, OA locations).
    if let Some(oa_work) = oa_opt {
        oa_work.merge_openalex_work(&mut metadata, &mut oa_candidates);
    }

    // 3. Optional Unpaywall OA lookup when a real non-placeholder contact email is configured.
    if let Some(upw_work) = upw_opt {
        upw_work.merge_unpaywall_work(&mut metadata, &mut oa_candidates);
    }

    let mut saw_body_restricted = false;

    // 4. Structured full-text preference: discovered PMCID -> Europe PMC JATS XML.
    if let Some(pmcid) = oa_candidates.pmcid()
        && visited.insert(target_visit_key(&ParsedTarget::PmcId(pmcid.clone())))
    {
        match resolve_pmcid(pmcid, budget, config).await {
            Ok(pmc_doc) if !matches!(pmc_doc.payload, ExtractedPayload::MetadataOnly { .. }) => {
                return Ok(merge_caller_priority_metadata(pmc_doc, metadata));
            }
            Ok(pmc_doc) => {
                saw_body_restricted |= is_body_restricted_doc(&pmc_doc);
                merge_missing_metadata(&mut metadata, pmc_doc.metadata);
            }
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            Err(_) => {}
        }
    }

    // 5. Try scholarly repository landing URLs (`arXiv` / `PMCID`), then verified OA PDFs, then OA HTML.
    if let Some(doc) = try_oa_candidates(
        &oa_candidates,
        &mut metadata,
        budget,
        config,
        depth,
        visited,
        &mut saw_body_restricted,
    )
    .await?
    {
        return Ok(doc);
    }

    if metadata_found {
        let reason = if saw_body_restricted {
            UnavailableReason::BodyRestricted
        } else {
            UnavailableReason::NoOpenAccessCopy
        };
        return Ok(ExtractedDocument {
            metadata,
            payload: ExtractedPayload::MetadataOnly { reason },
            raw_bytes_fetched: 0,
        });
    }

    Err(AdfontesError::Transport {
        url: doi_canonical_url.as_str().to_owned(),
        detail: budget
            .append_last_transport_error("DOI resolution failed across doi.org and OpenAlex"),
    })
}

async fn try_oa_candidates(
    candidates: &OaCandidates,
    metadata: &mut WorkMetadata,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
    depth: u8,
    visited: &mut BTreeSet<String>,
    saw_body_restricted: &mut bool,
) -> Result<Option<ExtractedDocument>, AdfontesError> {
    // 1. Prefer OA landing links that point directly to arXiv or PMC structured repositories.
    if depth < MAX_RECLASSIFICATION_DEPTH {
        let mut attempted_targets = 0usize;
        for landing in candidates.landing_urls() {
            if attempted_targets >= MAX_OA_CANDIDATES || budget.is_exhausted() {
                break;
            }
            if let Some(target @ (ParsedTarget::Arxiv(_) | ParsedTarget::PmcId(_))) =
                classify_known_scholarly_url(landing)
            {
                let key = target_visit_key(&target);
                if visited.contains(&key) {
                    continue;
                }
                attempted_targets += 1;
                match Box::pin(resolve_target_inner(
                    &target,
                    budget,
                    config,
                    depth + 1,
                    visited,
                ))
                .await
                {
                    Ok(doc) if !matches!(doc.payload, ExtractedPayload::MetadataOnly { .. }) => {
                        return Ok(Some(merge_caller_priority_metadata(doc, metadata.clone())));
                    }
                    Ok(doc) => {
                        *saw_body_restricted |= is_body_restricted_doc(&doc);
                        merge_missing_metadata(metadata, doc.metadata);
                    }
                    Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
                    Err(_) => {}
                }
            }
        }
    }

    // 2. Try OA PDF URLs while verifying real PDF MIME/magic (rejecting HTML paywall pages).
    if let Some(pdf_doc) = try_oa_pdf_urls(candidates.pdf_urls(), metadata, budget).await? {
        return Ok(Some(pdf_doc));
    }

    // 3. Try OA landing pages only when they yield PDF, Markdown, or substantive non-paywall HTML.
    try_oa_landing_pages(
        candidates.landing_urls(),
        metadata,
        budget,
        config,
        depth,
        visited,
        saw_body_restricted,
    )
    .await
}

async fn try_oa_landing_pages(
    landing_urls: &[HttpUrl],
    metadata: &mut WorkMetadata,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
    depth: u8,
    visited: &mut BTreeSet<String>,
    saw_body_restricted: &mut bool,
) -> Result<Option<ExtractedDocument>, AdfontesError> {
    for landing_url in landing_urls.iter().take(MAX_OA_CANDIDATES) {
        if budget.is_exhausted() {
            break;
        }
        match budget.get(landing_url, Some(ACCEPT_WEB_PREFERRED)).await {
            Ok(resp) => {
                if depth < MAX_RECLASSIFICATION_DEPTH
                    && let Some(target @ (ParsedTarget::Arxiv(_) | ParsedTarget::PmcId(_))) =
                        classify_known_scholarly_url(&resp.final_url)
                    && !visited.contains(&target_visit_key(&target))
                {
                    match Box::pin(resolve_target_inner(
                        &target,
                        budget,
                        config,
                        depth + 1,
                        visited,
                    ))
                    .await
                    {
                        Ok(doc)
                            if !matches!(doc.payload, ExtractedPayload::MetadataOnly { .. }) =>
                        {
                            return Ok(Some(merge_caller_priority_metadata(doc, metadata.clone())));
                        }
                        Ok(doc) => {
                            *saw_body_restricted |= is_body_restricted_doc(&doc);
                            merge_missing_metadata(metadata, doc.metadata);
                            continue;
                        }
                        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
                        Err(_) => continue,
                    }
                }
                if resp.status != HTTP_OK || is_arxiv_abs_url(&resp.final_url) {
                    continue;
                }
                if let Some(doc) = extract_oa_landing_response(resp, metadata) {
                    return Ok(Some(doc));
                }
            }
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            Err(_) => {}
        }
    }
    Ok(None)
}

fn extract_oa_landing_response(
    resp: HttpResponse,
    metadata: &mut WorkMetadata,
) -> Option<ExtractedDocument> {
    if is_pdf_response(&resp) {
        let byte_len = resp.body.len();
        metadata.source_url = Some(resp.final_url);
        return Some(ExtractedDocument {
            metadata: metadata.clone(),
            payload: ExtractedPayload::Pdf { bytes: resp.body },
            raw_bytes_fetched: byte_len,
        });
    }
    let mime = resp.mime_type();
    if is_markdown_mime(&mime) {
        let markdown = normalize_markdown_spacing(&resp.decode_text());
        if markdown.len() >= MIN_FULLTEXT_CHARS {
            metadata.source_url = Some(resp.final_url);
            return Some(ExtractedDocument {
                metadata: metadata.clone(),
                payload: ExtractedPayload::FullText {
                    format: TextFormat::Markdown,
                    markdown,
                },
                raw_bytes_fetched: resp.body.len(),
            });
        }
    } else if is_html_mime(&mime) {
        let extracted = extract_html_to_markdown(&resp.decode_text());
        if is_substantive_oa_html(&extracted.markdown) {
            if metadata.title.is_none() {
                metadata.title = extracted.title;
            }
            metadata.source_url = Some(resp.final_url);
            return Some(ExtractedDocument {
                metadata: metadata.clone(),
                payload: ExtractedPayload::FullText {
                    format: TextFormat::Html,
                    markdown: extracted.markdown,
                },
                raw_bytes_fetched: resp.body.len(),
            });
        }
    }
    None
}

async fn resolve_pmcid(
    pmcid: &PmcId,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<ExtractedDocument, AdfontesError> {
    let base = config.endpoints.europe_pmc.trim_end_matches('/');
    let search_url = HttpUrl::parse(&format!(
        "{base}/webservices/rest/search?query=PMCID:{}&format=json&resultType=core",
        pmcid.as_str()
    ))?;
    let mut metadata = WorkMetadata {
        canonical_id: Some(pmcid.as_str().to_owned()),
        venue_or_source: Some("Europe PMC".to_owned()),
        ..WorkMetadata::default()
    };
    let mut oa_pdf_urls = Vec::new();

    let mut saw_metadata = if let Some(search_doc) =
        fetch_optional_json(&search_url, ACCEPT_JSON, budget).await?
        && search_doc
            .extract_europe_pmc_record(&mut metadata, &mut oa_pdf_urls)
            .is_some()
    {
        true
    } else {
        false
    };

    let xml_url = HttpUrl::parse(&format!(
        "{base}/webservices/rest/{}/fullTextXML",
        pmcid.as_str()
    ))?;
    let mut body_restricted = false;
    match budget.get(&xml_url, Some(ACCEPT_XML)).await {
        Ok(resp) if resp.status == HTTP_OK => {
            let xml = resp.decode_text();
            match extract_jats_xml(&xml) {
                Ok(JatsExtraction::FullText {
                    title,
                    authors,
                    year,
                    abstract_text,
                    markdown,
                }) => {
                    fill_metadata_if_missing(&mut metadata, title, authors, year, abstract_text);
                    metadata.source_url = Some(resp.final_url);
                    return Ok(ExtractedDocument {
                        metadata,
                        payload: ExtractedPayload::FullText {
                            format: TextFormat::JatsXml,
                            markdown,
                        },
                        raw_bytes_fetched: resp.body.len(),
                    });
                }
                Ok(JatsExtraction::BodyRestricted {
                    title,
                    authors,
                    year,
                    abstract_text,
                }) => {
                    fill_metadata_if_missing(&mut metadata, title, authors, year, abstract_text);
                    metadata.source_url = Some(resp.final_url);
                    saw_metadata = true;
                    body_restricted = true;
                }
                Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
                Err(_) => {}
            }
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Ok(_) | Err(_) => {}
    }

    if let Some(pdf_doc) = try_oa_pdf_urls(&oa_pdf_urls, &mut metadata, budget).await? {
        return Ok(pdf_doc);
    }

    if saw_metadata {
        let reason = if body_restricted {
            UnavailableReason::BodyRestricted
        } else {
            UnavailableReason::NoOpenAccessCopy
        };
        return Ok(ExtractedDocument {
            metadata,
            payload: ExtractedPayload::MetadataOnly { reason },
            raw_bytes_fetched: 0,
        });
    }

    Err(AdfontesError::Transport {
        url: xml_url.as_str().to_owned(),
        detail: budget
            .append_last_transport_error("Europe PMC returned neither metadata nor fullTextXML"),
    })
}

async fn try_oa_pdf_urls(
    pdf_urls: &[HttpUrl],
    metadata: &mut WorkMetadata,
    budget: &ResolveBudget<'_>,
) -> Result<Option<ExtractedDocument>, AdfontesError> {
    for pdf_url in pdf_urls.iter().take(MAX_OA_CANDIDATES) {
        if budget.is_exhausted() {
            break;
        }
        match budget.get(pdf_url, Some(ACCEPT_PDF)).await {
            Ok(resp) if resp.status == HTTP_OK && is_pdf_response(&resp) => {
                let byte_len = resp.body.len();
                metadata.source_url = Some(resp.final_url);
                return Ok(Some(ExtractedDocument {
                    metadata: metadata.clone(),
                    payload: ExtractedPayload::Pdf { bytes: resp.body },
                    raw_bytes_fetched: byte_len,
                }));
            }
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            Ok(_) | Err(_) => {}
        }
    }
    Ok(None)
}

async fn resolve_rfc(
    rfc: RfcNumber,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<ExtractedDocument, AdfontesError> {
    let rfc_number = rfc.get();
    let mut metadata = WorkMetadata {
        canonical_id: Some(rfc.to_string()),
        venue_or_source: Some("IETF".to_owned()),
        ..WorkMetadata::default()
    };

    // 1. IETF Datatracker metadata & obsoleted-by / updated-by status notes.
    let tracker_url = HttpUrl::parse(&format!(
        "{}/doc/rfc{rfc_number}/doc.json",
        config.endpoints.ietf_datatracker.trim_end_matches('/')
    ))?;
    let saw_metadata = fetch_optional_json(&tracker_url, ACCEPT_JSON, budget)
        .await?
        .is_some_and(|doc_meta| {
            doc_meta.merge_ietf_doc(&mut metadata);
            true
        });

    // 2. Structured RFCXML v3 (`rfc<N>.xml`).
    let xml_url = HttpUrl::parse(&format!(
        "{}/rfc/rfc{rfc_number}.xml",
        config.endpoints.rfc_editor.trim_end_matches('/')
    ))?;
    match budget.get(&xml_url, Some(ACCEPT_XML)).await {
        Ok(resp) if resp.status == HTTP_OK => {
            let xml = resp.decode_text();
            match extract_rfc_xml(&xml) {
                Ok(extracted) if !extracted.markdown.is_empty() => {
                    if metadata.title.is_none() {
                        metadata.title = extracted.title;
                    }
                    metadata.source_url = Some(resp.final_url);
                    return Ok(ExtractedDocument {
                        metadata,
                        payload: ExtractedPayload::FullText {
                            format: TextFormat::RfcXml,
                            markdown: extracted.markdown,
                        },
                        raw_bytes_fetched: resp.body.len(),
                    });
                }
                Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
                Ok(_) | Err(_) => {}
            }
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Ok(_) | Err(_) => {}
    }

    // 3. Paginated plain-text RFC (`rfc<N>.txt`) with `\x0c` stripping and MIME check.
    let txt_url = HttpUrl::parse(&format!(
        "{}/rfc/rfc{rfc_number}.txt",
        config.endpoints.rfc_editor.trim_end_matches('/')
    ))?;
    let txt_failure_detail = match budget.get(&txt_url, Some(MIME_PLAIN_TEXT)).await {
        Ok(resp) if resp.status == HTTP_OK => {
            let mime = resp.mime_type();
            if is_plain_text_mime(&mime) || mime.is_empty() {
                let extracted = clean_rfc_text(&resp.decode_text());
                if metadata.title.is_none() {
                    metadata.title = extracted.title;
                }
                metadata.source_url = Some(resp.final_url);
                return Ok(ExtractedDocument {
                    metadata,
                    payload: ExtractedPayload::FullText {
                        format: TextFormat::RfcText,
                        markdown: extracted.markdown,
                    },
                    raw_bytes_fetched: resp.body.len(),
                });
            }
            format!("RFC text endpoint returned unexpected MIME `{mime}`")
        }
        Ok(resp) => format!("RFC text endpoint returned HTTP {}", resp.status),
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Err(other) => format!("RFC text endpoint failed: {other}"),
    };

    if saw_metadata {
        metadata.source_url = HttpUrl::parse(&format!(
            "{}/doc/rfc{rfc_number}/",
            config.endpoints.ietf_datatracker.trim_end_matches('/')
        ))
        .ok();
        return Ok(ExtractedDocument {
            metadata,
            payload: ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::UpstreamUnavailable {
                    detail: txt_failure_detail,
                },
            },
            raw_bytes_fetched: 0,
        });
    }

    Err(AdfontesError::Transport {
        url: txt_url.as_str().to_owned(),
        detail: txt_failure_detail,
    })
}

async fn resolve_url(
    url: &HttpUrl,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
    depth: u8,
    visited: &mut BTreeSet<String>,
) -> Result<ExtractedDocument, AdfontesError> {
    budget.client.validate_url_policy(url)?;
    let response = if let Some(raw_github) = url.to_raw_github_url() {
        visited.insert(format!("url:{}", raw_github.as_str()));
        match budget.get(&raw_github, Some(ACCEPT_WEB_PREFERRED)).await {
            Ok(resp) if resp.status == HTTP_OK => resp,
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            _ => budget.get(url, Some(ACCEPT_WEB_PREFERRED)).await?,
        }
    } else {
        budget.get(url, Some(ACCEPT_WEB_PREFERRED)).await?
    };

    // Reclassify intermediate and final post-redirect scholarly URLs BEFORE rejecting non-200
    // statuses (e.g., grounding redirect -> doi.org -> 403 publisher, or -> arxiv.org/html 404).
    if depth < MAX_RECLASSIFICATION_DEPTH {
        for hop in response
            .redirect_chain
            .iter()
            .chain(std::iter::once(&response.final_url))
        {
            if hop != url
                && let Some(reclassified) = classify_known_scholarly_url(hop)
                && !matches!(reclassified, ParsedTarget::Url(_))
                && !visited.contains(&target_visit_key(&reclassified))
            {
                return Box::pin(resolve_target_inner(
                    &reclassified,
                    budget,
                    config,
                    depth + 1,
                    visited,
                ))
                .await;
            }
        }
    }

    // Rewrite post-redirect GitHub `/blob/` URLs to `raw.githubusercontent.com` before falling back.
    let response = if depth < MAX_RECLASSIFICATION_DEPTH
        && response.final_url != *url
        && let Some(raw_github) = response.final_url.to_raw_github_url()
        && visited.insert(format!("url:{}", raw_github.as_str()))
    {
        match budget.get(&raw_github, Some(ACCEPT_WEB_PREFERRED)).await {
            Ok(raw_resp) if raw_resp.status == HTTP_OK => raw_resp,
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            Ok(_) | Err(_) => response,
        }
    } else {
        response
    };

    if response.status != HTTP_OK {
        return Err(AdfontesError::Transport {
            url: response.final_url.as_str().to_owned(),
            detail: format!("HTTP {}", response.status),
        });
    }

    dispatch_http_response(response)
}

fn dispatch_http_response(response: HttpResponse) -> Result<ExtractedDocument, AdfontesError> {
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

    if let Some(xml_doc) = try_dispatch_xml_response(&response, &mime, &mut metadata)? {
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

fn try_dispatch_xml_response(
    response: &HttpResponse,
    mime: &str,
    metadata: &mut WorkMetadata,
) -> Result<Option<ExtractedDocument>, AdfontesError> {
    if !is_xml_mime(mime) {
        return Ok(None);
    }
    let raw_len = response.body.len();
    let xml = response.decode_text();
    match extract_xml_document(&xml) {
        Ok(XmlDocumentExtraction::Jats(JatsExtraction::FullText {
            title,
            authors,
            year,
            abstract_text,
            markdown,
        })) => {
            fill_metadata_if_missing(metadata, title, authors, year, abstract_text);
            Ok(Some(ExtractedDocument {
                metadata: metadata.clone(),
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
            fill_metadata_if_missing(metadata, title, authors, year, abstract_text);
            Ok(Some(ExtractedDocument {
                metadata: metadata.clone(),
                payload: ExtractedPayload::MetadataOnly {
                    reason: UnavailableReason::BodyRestricted,
                },
                raw_bytes_fetched: raw_len,
            }))
        }
        Ok(XmlDocumentExtraction::Rfc(extracted)) if !extracted.markdown.is_empty() => {
            if metadata.title.is_none() {
                metadata.title = extracted.title;
            }
            Ok(Some(ExtractedDocument {
                metadata: metadata.clone(),
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

async fn fetch_optional_json(
    url: &HttpUrl,
    accept: &str,
    budget: &ResolveBudget<'_>,
) -> Result<Option<JsonDoc>, AdfontesError> {
    match budget.get(url, Some(accept)).await {
        Ok(resp) if resp.status == HTTP_OK => Ok(JsonDoc::parse(&resp.body)),
        Err(safety @ AdfontesError::SafetyBlock { .. }) => Err(safety),
        Ok(_) | Err(_) => Ok(None),
    }
}

fn build_openalex_doi_url(
    config: &ResolverConfig,
    encoded_doi: &str,
) -> Result<HttpUrl, AdfontesError> {
    let base = config.endpoints.openalex.trim_end_matches('/');
    let url = HttpUrl::parse(&format!("{base}/works/https://doi.org/{encoded_doi}"))?;
    Ok(url.with_query_pairs(config.usable_contact_email().map(|email| ("mailto", email))))
}

fn merge_caller_priority_metadata(
    mut mirror_doc: ExtractedDocument,
    mut caller_metadata: WorkMetadata,
) -> ExtractedDocument {
    let mirror_id = mirror_doc.metadata.canonical_id.clone();
    let mirror_source_url = mirror_doc.metadata.source_url.clone();
    merge_missing_metadata(&mut caller_metadata, mirror_doc.metadata);
    if let Some(url) = mirror_source_url {
        caller_metadata.source_url = Some(url);
    }
    if let Some(id) = mirror_id
        && caller_metadata.canonical_id.as_deref() != Some(id.as_str())
    {
        let note = format!("Full text via {id}");
        if !caller_metadata.notes.contains(&note) {
            caller_metadata.notes.push(note);
        }
    }
    mirror_doc.metadata = caller_metadata;
    mirror_doc
}

fn merge_missing_metadata(target: &mut WorkMetadata, extra: WorkMetadata) {
    fill_metadata_if_missing(
        target,
        extra.title,
        extra.authors,
        extra.year,
        extra.abstract_text,
    );
    let target_is_generic = target
        .venue_or_source
        .as_deref()
        .is_none_or(is_generic_repository_venue);
    if target_is_generic
        && let Some(extra_venue) = extra.venue_or_source
        && (target.venue_or_source.is_none() || !is_generic_repository_venue(&extra_venue))
    {
        target.venue_or_source = Some(extra_venue);
    }
    if target.canonical_id.is_none() {
        target.canonical_id = extra.canonical_id;
    }
    for note in extra.notes {
        if !target.notes.contains(&note) {
            target.notes.push(note);
        }
    }
}

fn fill_metadata_if_missing(
    target: &mut WorkMetadata,
    title: Option<String>,
    authors: Vec<String>,
    year: Option<u16>,
    abstract_text: Option<String>,
) {
    if target.title.is_none() {
        target.title = title;
    }
    if target.authors.is_empty() {
        target.authors = authors;
    }
    if target.year.is_none() {
        target.year = year;
    }
    if target.abstract_text.is_none() {
        target.abstract_text = abstract_text;
    }
}

fn is_generic_repository_venue(venue: &str) -> bool {
    GENERIC_REPOSITORY_VENUES
        .iter()
        .any(|generic| venue.eq_ignore_ascii_case(generic))
}

fn target_visit_key(target: &ParsedTarget) -> String {
    match target {
        ParsedTarget::Arxiv(id) => format!("arxiv:{}", id.as_str().to_ascii_lowercase()),
        ParsedTarget::Doi(doi) => format!("doi:{}", doi.as_str().to_ascii_lowercase()),
        ParsedTarget::PmcId(pmc) => format!("pmc:{}", pmc.as_str().to_ascii_uppercase()),
        ParsedTarget::Pmid(pmid) => format!("pmid:{}", pmid.as_str()),
        ParsedTarget::Rfc(rfc) => format!("rfc:{}", rfc.get()),
        ParsedTarget::Url(url) => format!("url:{}", url.as_str()),
    }
}

const fn is_body_restricted_doc(doc: &ExtractedDocument) -> bool {
    matches!(
        doc.payload,
        ExtractedPayload::MetadataOnly {
            reason: UnavailableReason::BodyRestricted
        }
    )
}

fn is_pdf_response(resp: &HttpResponse) -> bool {
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
            ACCEPT_JSON | "application/vnd.citationstyles.csl+json"
        )
    {
        return false;
    }
    starts_with_pdf_magic(&resp.body)
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

fn is_arxiv_abs_url(url: &HttpUrl) -> bool {
    let path = url.path().to_ascii_lowercase();
    path.starts_with("/abs/")
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

fn is_plain_text_mime(mime: &str) -> bool {
    matches!(
        mime,
        MIME_PLAIN_TEXT | "text/x-rst" | "text/restructuredtext"
    )
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

fn is_usable_contact_email(raw: &str) -> bool {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.len() > MAX_EMAIL_TOTAL_LEN
        || trimmed
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
    {
        return false;
    }
    let Some((local, domain)) = trimmed.split_once('@') else {
        return false;
    };
    if local.is_empty()
        || local.len() > MAX_EMAIL_LOCAL_LEN
        || domain.is_empty()
        || domain.contains('@')
    {
        return false;
    }
    let lower_local = local.to_ascii_lowercase();
    if PLACEHOLDER_EMAIL_LOCALS.contains(&lower_local.as_str()) {
        return false;
    }
    is_valid_contact_email_domain(&domain.to_ascii_lowercase())
}

fn is_valid_contact_email_domain(lower_domain: &str) -> bool {
    let labels: Vec<&str> = lower_domain.split('.').collect();
    if labels.len() < 2 {
        return false;
    }
    for label in &labels {
        if label.is_empty()
            || label.starts_with('-')
            || label.ends_with('-')
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return false;
        }
    }
    let Some(&tld) = labels.last() else {
        return false;
    };
    if tld.len() < 2 || !tld.bytes().all(|byte| byte.is_ascii_alphabetic()) {
        return false;
    }
    if RESERVED_EMAIL_TLDS.contains(&tld) {
        return false;
    }
    !PLACEHOLDER_EMAIL_DOMAINS.iter().any(|placeholder| {
        lower_domain == *placeholder || lower_domain.ends_with(&format!(".{placeholder}"))
    })
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead as _, BufReader, Write as _};
    use std::net::{SocketAddr, TcpListener};
    use std::sync::{Arc, Condvar};
    use std::thread;

    use indoc::indoc;
    use rstest::rstest;
    use serde_json::json;
    use wiremock::matchers::{any, method, path, path_regex};
    use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

    use super::*;
    use crate::transport::TransportConfig;

    fn resolver_config_for(base_url: &str, contact_email: Option<&str>) -> ResolverConfig {
        ResolverConfig {
            contact_email: contact_email.map(str::to_owned),
            endpoints: ResolverEndpoints {
                arxiv_org: base_url.to_owned(),
                export_arxiv_org: base_url.to_owned(),
                ar5iv: format!("{base_url}/ar5iv"),
                doi_org: format!("{base_url}/doi"),
                openalex: format!("{base_url}/openalex"),
                unpaywall: format!("{base_url}/unpaywall"),
                europe_pmc: format!("{base_url}/epmc"),
                ietf_datatracker: format!("{base_url}/datatracker"),
                rfc_editor: format!("{base_url}/rfceditor"),
            },
        }
    }

    fn http_client_for(addrs: &[SocketAddr]) -> HttpClient {
        let mut config = TransportConfig::default();
        config.allowed_loopback_addrs.extend(addrs.iter().copied());
        HttpClient::new(config)
    }

    fn spawn_concurrent_doi_barrier_server(
        expected_legs: usize,
    ) -> (SocketAddr, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let barrier = Arc::new((Mutex::new(0_usize), Condvar::new()));

        let handle = thread::spawn(move || {
            let mut workers = Vec::with_capacity(expected_legs);
            for _ in 0..expected_legs {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let leg_barrier = Arc::clone(&barrier);
                workers.push(thread::spawn(move || {
                    let mut reader = BufReader::new(&stream);
                    let mut request_line = String::new();
                    let _ = reader.read_line(&mut request_line);
                    let req_path = request_line
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("/")
                        .to_owned();

                    let (lock, cvar) = &*leg_barrier;
                    {
                        let mut count = lock.lock().unwrap();
                        *count += 1;
                        drop(count);
                        cvar.notify_all();
                    }
                    let arrived = cvar
                        .wait_timeout_while(
                            lock.lock().unwrap(),
                            Duration::from_secs(2),
                            |pending| *pending < expected_legs,
                        )
                        .is_ok_and(|(_guard, timeout)| !timeout.timed_out());

                    let (status, content_type, body) = if !arrived {
                        (504, "text/plain", b"sequential deadlock".as_slice())
                    } else if req_path.starts_with("/doi/") {
                        (
                            200,
                            "application/vnd.citationstyles.csl+json",
                            br#"{"title":"Concurrent CSL Title","issued":{"date-parts":[[2026]]}}"#
                                .as_slice(),
                        )
                    } else if req_path.starts_with("/openalex/") {
                        (
                            200,
                            "application/json",
                            br#"{"title":"Fallback OA Title","abstract_inverted_index":{"Parallel":[0],"legs":[1],"pass.":[2]},"best_oa_location":null,"locations":[]}"#
                                .as_slice(),
                        )
                    } else if req_path.starts_with("/unpaywall/") {
                        (
                            200,
                            "application/json",
                            br#"{"title":"Fallback Unpaywall Title","journal_name":"Unpaywall Journal","year":2025,"best_oa_location":null,"oa_locations":[]}"#
                                .as_slice(),
                        )
                    } else {
                        (404, "text/plain", b"not found".as_slice())
                    };
                    let header = [
                        format!("HTTP/1.1 {status} OK"),
                        format!("Content-Type: {content_type}"),
                        format!("Content-Length: {}", body.len()),
                        "Connection: close".to_owned(),
                        String::new(),
                        String::new(),
                    ]
                    .join("\r\n");
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(body);
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        (addr, handle)
    }

    #[rstest]
    #[case("researcher@university.edu", true)]
    #[case("user@example.com", false)]
    #[case("alice@sub.example.com", false)]
    #[case("alice@host.invalid", false)]
    #[case("alice@box.localhost", false)]
    #[case("alice@app.test", false)]
    #[case("alice@bad/domain.org", false)]
    #[case("test@real.org", false)]
    #[case("no-at-sign.org", false)]
    #[case("bad email@university.edu", false)]
    fn validates_contact_email_against_placeholders(#[case] email: &str, #[case] expected: bool) {
        assert_eq!(is_usable_contact_email(email), expected);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn arxiv_survives_atom_429_rejects_ar5iv_abs_redirect_and_extracts_tex_source() {
        let tex = indoc! {r"
            \documentclass{article}
            \title{Surviving Rate Limits}
            \begin{document}
            \section{Theorem}
            This synthetic TeX source provides full derivations for the theorem with sufficient
            content length to pass the substantive body check and verify equation preservation.
            \begin{equation}
            g(f) = \mu f - \frac{1}{2} \sigma^2 f^2
            \end{equation}
            \end{document}
        "};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/api/query"))
            .respond_with(ResponseTemplate::new(429).set_body_string("Rate exceeded."))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/ar5iv/html/"))
            .respond_with(ResponseTemplate::new(302).insert_header("Location", "/abs/2409.13740"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/abs/"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw("<article><p>Abstract only</p></article>", "text/html"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/src/"))
            .respond_with(ResponseTemplate::new(200).set_body_string(tex))
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("2409.13740").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();
        let ExtractedPayload::FullText { format, markdown } = doc.payload else {
            panic!("expected LaTeX full text");
        };
        assert_eq!(format, TextFormat::Latex);
        assert_eq!(doc.metadata.title.as_deref(), Some("Surviving Rate Limits"));
        assert!(markdown.contains(r"g(f) = \mu f - \frac{1}{2} \sigma^2 f^2"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn closed_access_doi_returns_honest_metadata_only_with_inverted_abstract() {
        let csl = br#"{"title":"Closed Access Study","author":[{"given":"Ada","family":"Lovelace"}],"issued":{"date-parts":[[2026]]},"container-title":"Journal of Science"}"#;
        let openalex = br#"{"abstract_inverted_index":{"Empirical":[0],"findings":[1],"hold.":[2]},"best_oa_location":null,"locations":[]}"#;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/doi/"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(csl.as_slice(), "application/vnd.citationstyles.csl+json"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(openalex.as_slice(), "application/json"),
            )
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("10.1000/closed-paper").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();
        assert_eq!(
            doc.payload,
            ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::NoOpenAccessCopy
            }
        );
        assert_eq!(doc.metadata.title.as_deref(), Some("Closed Access Study"));
        assert_eq!(
            doc.metadata.abstract_text.as_deref(),
            Some("Empirical findings hold.")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn europe_pmc_bodyless_jats_reports_body_restricted() {
        let jats_front_only = indoc! {r"
            <article>
              <front><article-meta><title-group><article-title>Embargoed Trial</article-title></title-group></article-meta></front>
            </article>
        "};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("fullTextXML"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(jats_front_only, "application/xml"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "resultList": {"result": [{"title": "Embargoed Trial", "pubYear": "2025"}]}
            })))
            .with_priority(10)
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("PMC9999999").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();
        assert_eq!(
            doc.payload,
            ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::BodyRestricted
            }
        );
        assert_eq!(doc.metadata.title.as_deref(), Some("Embargoed Trial"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rfc_waterfall_attaches_obsoleted_by_notes_and_falls_back_from_xml_to_txt() {
        let tracker_json = json!({
            "title": "HTTP/1.1 Semantics",
            "time": "2014-06-01T00:00:00Z",
            "std_level": "Proposed Standard",
            "obsoleted_by": ["/doc/rfc9110/"],
            "updated_by": []
        });
        let rfc_txt = indoc! {r"
            Internet Engineering Task Force (IETF)                        R. Fielding
            Request for Comments: 7231                                      June 2014

                                   HTTP/1.1 Semantics

            Abstract

               Defines HTTP/1.1 semantics.
        "};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"/doc\.json$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(tracker_json))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"\.xml$"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string(rfc_txt))
            .with_priority(10)
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("RFC 7231").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();
        let ExtractedPayload::FullText { format, markdown } = doc.payload else {
            panic!("expected RFC text fallback");
        };
        assert_eq!(format, TextFormat::RfcText);
        assert_eq!(doc.metadata.notes, vec!["Obsoleted by RFC 9110"]);
        assert!(markdown.contains("Defines HTTP/1.1 semantics."));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn url_dispatch_handles_markdown_and_unsupported_binary_mime() {
        let server = MockServer::start().await;
        let md = indoc! {r"
            # Native Markdown

            Direct body.
        "};
        Mock::given(method("GET"))
            .and(path("/doc.md"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(md, "text/markdown; charset=utf-8"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/blob.bin"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(vec![0, 1, 2, 3], "application/octet-stream"),
            )
            .mount(&server)
            .await;
        let client = http_client_for(&[*server.address()]);
        let config = resolver_config_for(&server.uri(), None);

        let md_target = ParsedTarget::parse(&format!("{}/doc.md", server.uri())).unwrap();
        let bin_target = ParsedTarget::parse(&format!("{}/blob.bin", server.uri())).unwrap();
        let md_doc = resolve_target(&md_target, &client, &config).await.unwrap();
        let bin_doc = resolve_target(&bin_target, &client, &config).await.unwrap();

        assert!(matches!(
            md_doc.payload,
            ExtractedPayload::FullText {
                format: TextFormat::Markdown,
                ..
            }
        ));
        assert!(matches!(
            bin_doc.payload,
            ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::UnsupportedContentType { .. }
            }
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn doi_propagates_safety_block_from_pmcid_xml_dtd_attack() {
        let openalex = br#"{"title":"Target Study","ids":{"pmcid":"https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/"},"locations":[]}"#;
        let xxe_xml = b"<!DOCTYPE article [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><article><body/></article>";
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(openalex.as_slice(), "application/json"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("fullTextXML"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(xxe_xml.as_slice(), "application/xml"),
            )
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("10.1000/xxe-pmc").unwrap();
        let failure = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap_err();
        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn html_and_tex_mentioning_pdf_magic_are_not_misclassified_as_pdf() {
        let html = b"<article><h1>PDF Format Guide</h1><p>Every valid PDF starts with the header %PDF-1.7 followed by binary markers.</p></article>";
        let tex = indoc! {r"
            %PDF-1.7 compatibility note in a TeX comment
            \documentclass{article}
            \title{PDF Spec Notes}
            \begin{document}
            \section{Overview}
            This LaTeX paper discusses the %PDF-1.7 specification in detail with enough prose
            content to exceed the minimum length threshold for extracted scientific documents,
            ensuring both section headings and full explanatory paragraphs are preserved intact.
            \end{document}
        "};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/guide.html"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(html.as_slice(), "text/html"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/src/"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(tex.as_bytes(), "application/x-eprint"),
            )
            .mount(&server)
            .await;
        let client = http_client_for(&[*server.address()]);
        let config = resolver_config_for(&server.uri(), None);

        let web_target = ParsedTarget::parse(&format!("{}/guide.html", server.uri())).unwrap();
        let arxiv_target = ParsedTarget::parse("2409.13740").unwrap();
        let web_doc = resolve_target(&web_target, &client, &config).await.unwrap();
        let arxiv_doc = resolve_target(&arxiv_target, &client, &config)
            .await
            .unwrap();

        assert!(matches!(
            web_doc.payload,
            ExtractedPayload::FullText {
                format: TextFormat::Html,
                ..
            }
        ));
        assert!(matches!(
            arxiv_doc.payload,
            ExtractedPayload::FullText {
                format: TextFormat::Latex,
                ..
            }
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn paywalled_oa_landing_html_returns_metadata_only_not_full_text() {
        let abstract_pad =
            "Abstract details about clinical trial endpoints and study design. ".repeat(35);
        let paywall_html = format!(
            "<article><h1>Paywalled Journal Study</h1><h2>Abstract</h2><p>{abstract_pad}</p><p>Buy this article or sign in to access via institutional access.</p></article>"
        );
        let paywall_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paywall"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(paywall_html.into_bytes(), "text/html"),
            )
            .mount(&paywall_server)
            .await;

        let paywall_url = format!("{}/paywall", paywall_server.uri());
        let oa_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "title": "Paywalled Study",
                "primary_location": {"source": {"display_name": "Nature"}},
                "best_oa_location": {"is_oa": true, "landing_page_url": paywall_url}
            })))
            .mount(&oa_server)
            .await;

        let config = resolver_config_for(&oa_server.uri(), None);
        let client = http_client_for(&[*paywall_server.address(), *oa_server.address()]);

        let target = ParsedTarget::parse("10.1038/paywall-test").unwrap();
        let doc = resolve_target(&target, &client, &config).await.unwrap();
        assert_eq!(
            doc.payload,
            ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::NoOpenAccessCopy
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pmid_falls_back_from_body_restricted_pmcid_to_doi_and_preserves_metadata() {
        let jats_front_only = b"<article><front><article-meta><title-group><article-title>Trial</article-title></title-group></article-meta></front></article>";
        let tex = indoc! {r"
            \documentclass{article}
            \title{Preprint Full Text}
            \begin{document}
            \section{Results}
            Full derivation and statistical results from the open-access preprint mirror with
            sufficient body length to satisfy full-text validation thresholds completely.
            \end{document}
        "};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(|req: &WiremockRequest| req.url.as_str().contains("EXT_ID:38123456"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "resultList": {
                    "result": [{
                        "title": "Published Trial",
                        "journalTitle": "NEJM",
                        "pubYear": "2025",
                        "pmcid": "PMC1111111",
                        "doi": "10.48550/arXiv.2409.13740"
                    }]
                }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("PMC1111111/fullTextXML"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(jats_front_only.as_slice(), "application/xml"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/src/2409\.13740"))
            .respond_with(ResponseTemplate::new(200).set_body_string(tex))
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("PMID:38123456").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();
        assert!(matches!(
            doc.payload,
            ExtractedPayload::FullText {
                format: TextFormat::Latex,
                ..
            }
        ));
        assert_eq!(doc.metadata.canonical_id.as_deref(), Some("PMID:38123456"));
        assert_eq!(doc.metadata.venue_or_source.as_deref(), Some("NEJM"));
        assert!(
            doc.metadata
                .notes
                .iter()
                .any(|note| note.contains("arXiv:2409.13740"))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn xml_dispatch_handles_namespaced_jats_and_body_restricted_response() {
        let namespaced_jats = indoc! {r#"
            <?xml version="1.0" encoding="UTF-8"?>
            <!-- leading comment -->
            <jats:article xmlns:jats="http://jats.nlm.nih.gov">
              <jats:front><jats:article-meta><jats:title-group><jats:article-title>Namespaced Trial</jats:article-title></jats:title-group></jats:article-meta></jats:front>
            </jats:article>
        "#};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/paper.xml"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(namespaced_jats, "application/jats+xml"),
            )
            .mount(&server)
            .await;
        let target = ParsedTarget::parse(&format!("{}/paper.xml", server.uri())).unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();
        assert_eq!(
            doc.payload,
            ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::BodyRestricted
            }
        );
        assert_eq!(doc.metadata.title.as_deref(), Some("Namespaced Trial"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn doi_deduplicates_pmcid_and_preserves_body_restricted_status() {
        let openalex = br#"{"title":"Restricted Study","ids":{"pmcid":"https://pmc.ncbi.nlm.nih.gov/articles/PMC9999999/"},"locations":[{"is_oa":true,"landing_page_url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC9999999/"}]}"#;
        let front_only = b"<article><front><article-meta><title-group><article-title>Restricted Study</article-title></title-group></article-meta></front></article>";

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(openalex.as_slice(), "application/json"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("PMC9999999/fullTextXML"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(front_only.as_slice(), "application/xml"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("10.1000/restricted-pmc").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();

        server.verify().await;
        assert_eq!(
            doc.payload,
            ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::BodyRestricted
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn doi_surfaces_unsupported_proxy_error_instead_of_generic_failure() {
        let transport = TransportConfig {
            proxy: crate::transport::ProxyConfig {
                all_proxy: Some("socks5://127.0.0.1:1080".to_owned()),
                ..crate::transport::ProxyConfig::none()
            },
            ..TransportConfig::default()
        };
        let client = HttpClient::new(transport);
        let config = ResolverConfig {
            contact_email: None,
            endpoints: ResolverEndpoints {
                doi_org: "https://example.org/doi".to_owned(),
                openalex: "https://example.org/openalex".to_owned(),
                ..ResolverEndpoints::production()
            },
        };
        let target = ParsedTarget::parse("10.1000/proxy-check").unwrap();
        let failure = resolve_target(&target, &client, &config).await.unwrap_err();
        assert!(
            matches!(failure, AdfontesError::Transport { ref detail, .. } if detail.contains("socks5"))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn doi_caps_total_waterfall_http_requests_within_budget() {
        let openalex = indoc! {r#"{
            "title": "Many Mirrors",
            "locations": [
                {"is_oa": true, "landing_page_url": "https://arxiv.org/abs/2401.00001"},
                {"is_oa": true, "landing_page_url": "https://arxiv.org/abs/2401.00002"},
                {"is_oa": true, "landing_page_url": "https://arxiv.org/abs/2401.00003"},
                {"is_oa": true, "landing_page_url": "https://arxiv.org/abs/2401.00004"},
                {"is_oa": true, "landing_page_url": "https://arxiv.org/abs/2401.00005"},
                {"is_oa": true, "landing_page_url": "https://arxiv.org/abs/2401.00006"}
            ]
        }"#};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(openalex, "application/json"))
            .mount(&server)
            .await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(404))
            .with_priority(10)
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("10.1000/budget-cap").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();

        let recorded = server.received_requests().await.unwrap();
        assert!(recorded.len() <= MAX_WATERFALL_HTTP_REQUESTS);
        assert!(matches!(doc.payload, ExtractedPayload::MetadataOnly { .. }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn doi_includes_encoded_contact_email_in_openalex_and_unpaywall_queries() {
        let unpaywall_body = indoc! {r#"{
            "title": "Unpaywall Routed Title",
            "is_oa": false
        }"#};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .and(|req: &WiremockRequest| {
                req.url.query() == Some("mailto=alice%2Btag%40university.edu")
            })
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/unpaywall/"))
            .and(|req: &WiremockRequest| {
                req.url.query() == Some("email=alice%2Btag%40university.edu")
            })
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(unpaywall_body, "application/json"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("10.1000/contact-check").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), Some("alice+tag@university.edu")),
        )
        .await
        .unwrap();

        server.verify().await;
        assert_eq!(
            doc.metadata.title.as_deref(),
            Some("Unpaywall Routed Title")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn doi_executes_metadata_legs_concurrently_with_deterministic_merge() {
        let (addr, server) = spawn_concurrent_doi_barrier_server(2);
        let client = http_client_for(&[addr]);
        let base_url = format!("http://{addr}");
        let config = ResolverConfig {
            contact_email: None,
            endpoints: ResolverEndpoints {
                doi_org: format!("{base_url}/doi"),
                openalex: format!("{base_url}/openalex"),
                ..ResolverEndpoints::production()
            },
        };
        let target = ParsedTarget::parse("10.1000/concurrent-legs").unwrap();
        let doc = resolve_target(&target, &client, &config).await.unwrap();
        server.join().unwrap();

        assert_eq!(doc.metadata.title.as_deref(), Some("Concurrent CSL Title"));
        assert_eq!(
            doc.metadata.abstract_text.as_deref(),
            Some("Parallel legs pass.")
        );
        assert_eq!(
            doc.payload,
            ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::NoOpenAccessCopy
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn doi_executes_three_concurrent_legs_preserving_unpaywall_priority() {
        let (addr, server) = spawn_concurrent_doi_barrier_server(3);
        let client = http_client_for(&[addr]);
        let base_url = format!("http://{addr}");
        let config = resolver_config_for(&base_url, Some("researcher@university.edu"));
        let target = ParsedTarget::parse("10.1000/three-legs").unwrap();
        let doc = resolve_target(&target, &client, &config).await.unwrap();
        server.join().unwrap();

        assert_eq!(doc.metadata.title.as_deref(), Some("Concurrent CSL Title"));
        assert_eq!(doc.metadata.year, Some(2026));
        assert_eq!(
            doc.metadata.abstract_text.as_deref(),
            Some("Parallel legs pass.")
        );
        assert_eq!(
            doc.metadata.venue_or_source.as_deref(),
            Some("Unpaywall Journal")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn doi_cancels_slow_sibling_leg_immediately_on_safety_block() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({}))
                    .set_delay(Duration::from_secs(5)),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/doi/"))
            .respond_with(
                ResponseTemplate::new(302).insert_header("Location", "http://10.0.0.1/evil"),
            )
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("10.1000/safety-short-circuit").unwrap();
        let client = http_client_for(&[*server.address()]);
        let config = resolver_config_for(&server.uri(), None);
        let started = Instant::now();
        let failure = resolve_target(&target, &client, &config).await.unwrap_err();
        let elapsed = started.elapsed();

        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
        assert!(
            elapsed < Duration::from_secs(2),
            "slow sibling delayed SafetyBlock fail-fast: {elapsed:?}"
        );
    }
}
