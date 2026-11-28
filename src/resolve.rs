//! Keyless open-access scholarly retrieval waterfalls and web/spec content dispatch.
//!
//! Resolves [`ParsedTarget`] identifiers (`arXiv`, `DOI`, `PMCID`, `PMID`, `RFC`, and web/GitHub
//! URLs) via a unified [`session::WorkResolution`] frontier that favors structured digital text
//! (`Markdown`, `LaTeXML` HTML, `JATS` XML, `RFCXML`, flattened `TeX`) ahead of binary PDFs and
//! returns honest [`crate::model::ExtractedPayload::MetadataOnly`] outcomes when no legal open-access copy
//! exists.

architecture_component!(Resolve);

mod content;
mod providers;
mod session;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use self::content::{
    LandingExtraction, MIME_PLAIN_TEXT, MIN_FULLTEXT_CHARS, dispatch_http_response,
    extract_arxiv_html_response, extract_oa_landing_response, is_arxiv_abs_url, is_pdf_response,
    is_plain_text_mime,
};
use self::providers::{JsonDoc, ProviderDiscovery};
use self::session::{
    AliasEnqueue, DeferredFallback, FrontierEntry, ResolvedPdf, ResolvedText, UnavailableTrace,
    WorkResolution, is_structured_repository,
};
use crate::extract::{
    JatsExtraction, clean_rfc_text, extract_arxiv_source_archive, extract_jats_xml,
    extract_rfc_xml, parse_arxiv_atom_feed,
};
use crate::model::{
    AdfontesError, ArxivId, Doi, ExtractedDocument, HttpUrl, ParsedTarget, PmcId, Pmid, RfcNumber,
    ScholarlyId, TextFormat, WorkMetadata, classify_known_scholarly_url,
};
use crate::transport::{HttpClient, HttpResponse};

const ACCEPT_HTML: &str = "text/html, application/xhtml+xml;q=0.9";
const ACCEPT_XML: &str = "application/xml, text/xml;q=0.9";
const ACCEPT_JSON: &str = "application/json";
const ACCEPT_CSL_JSON: &str = "application/vnd.citationstyles.csl+json, application/json;q=0.8";
const ACCEPT_PDF: &str = "application/pdf, */*;q=0.1";
const ACCEPT_WEB_PREFERRED: &str =
    "text/markdown, text/html;q=0.9, application/pdf;q=0.8, text/plain;q=0.7, */*;q=0.5";

const ENDPOINT_ARXIV_ATOM: &str = "atom";
const ENDPOINT_ARXIV_HTML: &str = "arxiv-html";
const ENDPOINT_AR5IV: &str = "ar5iv";
const ENDPOINT_ARXIV_SRC: &str = "src";
const ENDPOINT_EPMC_XML: &str = "epmc-xml";
const ENDPOINT_EPMC_PMID: &str = "epmc-pmid";
const ENDPOINT_RFC_XML: &str = "rfc-xml";
const ENDPOINT_RFC_TXT: &str = "rfc-txt";
const ENDPOINT_OA_PDF: &str = "oa-pdf";

const HTTP_OK: u16 = 200;
const MAX_WATERFALL_HTTP_REQUESTS: usize = 24;
const MAX_WATERFALL_DURATION: Duration = Duration::from_secs(90);
const MAX_EMAIL_LOCAL_LEN: usize = 64;
const MAX_EMAIL_TOTAL_LEN: usize = 254;

const PLACEHOLDER_EMAIL_DOMAINS: &[&str] =
    &["example.com", "example.org", "example.net", "test.com"];
const RESERVED_EMAIL_TLDS: &[&str] = &["example", "invalid", "test", "localhost"];
const PLACEHOLDER_EMAIL_LOCALS: &[&str] = &["test", "user", "placeholder", "your_email", "email"];

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
    let budget = ResolveBudget::new(client);
    match target {
        ParsedTarget::Scholarly(id) => resolve_work(id, &[], &budget, config).await,
        ParsedTarget::Url(url) => resolve_url(url, &budget, config).await,
    }
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

async fn resolve_url(
    url: &HttpUrl,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<ExtractedDocument, AdfontesError> {
    budget.client.validate_url_policy(url)?;
    let initial_raw_github = url.to_raw_github_url();
    let response = if let Some(ref raw_github) = initial_raw_github {
        match budget.get(raw_github, Some(ACCEPT_WEB_PREFERRED)).await {
            Ok(resp) if resp.status == HTTP_OK => resp,
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            _ => budget.get(url, Some(ACCEPT_WEB_PREFERRED)).await?,
        }
    } else {
        budget.get(url, Some(ACCEPT_WEB_PREFERRED)).await?
    };

    // Reclassify intermediate and final post-redirect scholarly URLs BEFORE rejecting non-200
    // statuses (e.g., grounding redirect -> doi.org -> 403 publisher, or -> arxiv.org/html 404).
    let mut reclassified_ids = Vec::new();
    for hop in response
        .redirect_chain
        .iter()
        .chain(std::iter::once(&response.final_url))
    {
        if hop != url
            && let Some(reclassified) = classify_known_scholarly_url(hop)
            && !reclassified_ids.iter().any(|existing: &ScholarlyId| {
                existing.canonical_key() == reclassified.canonical_key()
            })
        {
            reclassified_ids.push(reclassified);
        }
    }
    if let Some((primary, rest)) = reclassified_ids.split_first() {
        return resolve_work(primary, rest, budget, config).await;
    }

    // Rewrite post-redirect GitHub `/blob/` URLs to `raw.githubusercontent.com` before falling back.
    let response = if response.final_url != *url
        && let Some(raw_github) = response.final_url.to_raw_github_url()
        && initial_raw_github.as_ref() != Some(&raw_github)
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

async fn resolve_work(
    caller_id: &ScholarlyId,
    seed_aliases: &[ScholarlyId],
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<ExtractedDocument, AdfontesError> {
    let initial_url = initial_caller_source_url(caller_id, config);
    let mut session = WorkResolution::new(caller_id.clone(), initial_url, seed_aliases);

    loop {
        while let Some(entry) = session.pop_next_frontier() {
            if budget.is_exhausted() {
                break;
            }
            if let Some(full_text) =
                expand_frontier_entry(&entry, &mut session, budget, config).await?
            {
                return Ok(session.finish_full_text(full_text));
            }
        }
        if let Some(deferred) = session.take_deferred_fallback() {
            match deferred {
                DeferredFallback::FullText(full_text) => {
                    return Ok(session.finish_full_text(full_text));
                }
                DeferredFallback::Pdf(pdf) => return Ok(session.finish_pdf(pdf)),
            }
        }
        if budget.is_exhausted() {
            break;
        }
        if let Some(pdf) = try_oa_pdf_pool(&mut session, budget).await? {
            return Ok(session.finish_pdf(pdf));
        }
        if budget.is_exhausted() {
            break;
        }
        match try_oa_landing_pool(&mut session, budget).await? {
            LandingPoolOutcome::FullText(full_text) => {
                return Ok(session.finish_full_text(full_text));
            }
            LandingPoolOutcome::Pdf(pdf) => return Ok(session.finish_pdf(pdf)),
            LandingPoolOutcome::DiscoveredAlias => {}
            LandingPoolOutcome::Exhausted => break,
        }
    }

    let fallback_url = fallback_metadata_source_url(caller_id, config);
    match session.finish_unavailable(fallback_url) {
        Ok(doc) => Ok(doc),
        Err(trace) => Err(build_caller_transport_error(
            caller_id, &trace, budget, config,
        )),
    }
}

async fn expand_frontier_entry(
    entry: &FrontierEntry,
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<Option<ResolvedText>, AdfontesError> {
    match &entry.id {
        ScholarlyId::Arxiv(id) => expand_arxiv(id, entry.depth, session, budget, config).await,
        ScholarlyId::PmcId(pmcid) => {
            expand_pmcid(pmcid, entry.depth, session, budget, config).await
        }
        ScholarlyId::Pmid(pmid) => {
            expand_pmid(pmid, entry.depth, session, budget, config).await?;
            Ok(None)
        }
        ScholarlyId::Doi(doi) => {
            expand_doi(doi, entry.depth, session, budget, config).await?;
            Ok(None)
        }
        ScholarlyId::Rfc(rfc) => expand_rfc(*rfc, session, budget, config).await,
    }
}

async fn expand_arxiv(
    id: &ArxivId,
    depth: u8,
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<Option<ResolvedText>, AdfontesError> {
    let origin = ScholarlyId::Arxiv(id.clone());

    // 1. Metadata & optional journal DOI via Atom API.
    fetch_arxiv_atom_metadata(id, depth, session, budget, config).await?;

    // 2. Official arXiv LaTeXML HTML (`arxiv.org/html/<id>`).
    if budget.is_exhausted() {
        return Ok(None);
    }
    let arxiv_html_url = format!(
        "{}/html/{}",
        config.endpoints.arxiv_org.trim_end_matches('/'),
        id.as_str()
    );
    if let Some(full_text) = try_arxiv_html_endpoint(
        &arxiv_html_url,
        &origin,
        ENDPOINT_ARXIV_HTML,
        session,
        budget,
    )
    .await?
    {
        return Ok(Some(full_text));
    }

    // 3. ar5iv HTML mirror (`ar5iv.labs.arxiv.org/html/<id>`).
    if budget.is_exhausted() {
        return Ok(None);
    }
    let ar5iv_url = format!(
        "{}/html/{}",
        config.endpoints.ar5iv.trim_end_matches('/'),
        id.as_str()
    );
    if let Some(full_text) =
        try_arxiv_html_endpoint(&ar5iv_url, &origin, ENDPOINT_AR5IV, session, budget).await?
    {
        return Ok(Some(full_text));
    }

    // 4. LaTeX source archive (`export.arxiv.org/src/<id>`).
    if budget.is_exhausted() {
        return Ok(None);
    }
    if let Some(full_text) = try_arxiv_source_endpoint(id, &origin, session, budget, config).await?
    {
        return Ok(Some(full_text));
    }

    Ok(None)
}

async fn fetch_arxiv_atom_metadata(
    id: &ArxivId,
    depth: u8,
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
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
                    if entry.title.is_some() || entry.abstract_text.is_some() {
                        session.mark_metadata_seen();
                    }
                    session.merge_fields_if_missing(
                        entry.title,
                        entry.authors,
                        entry.year,
                        entry.abstract_text,
                    );
                    if let Some(raw_doi) = entry.doi
                        && let Ok(alias) = ScholarlyId::parse(&raw_doi)
                    {
                        session.enqueue_alias(alias, depth.saturating_add(1));
                    }
                }
                Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
                Err(other) => {
                    session.record_endpoint_failure(ENDPOINT_ARXIV_ATOM, other.to_string());
                }
            }
        }
        Ok(resp) => {
            session.record_endpoint_failure(ENDPOINT_ARXIV_ATOM, format!("HTTP {}", resp.status));
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Err(other) => {
            session.record_endpoint_failure(ENDPOINT_ARXIV_ATOM, other.to_string());
        }
    }
    Ok(())
}

async fn try_arxiv_html_endpoint(
    raw_url: &str,
    origin: &ScholarlyId,
    label: &'static str,
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
) -> Result<Option<ResolvedText>, AdfontesError> {
    let url = HttpUrl::parse(raw_url)?;
    match budget.get(&url, Some(ACCEPT_HTML)).await {
        Ok(resp) if resp.status == HTTP_OK => match extract_arxiv_html_response(resp) {
            Ok(extracted) => {
                session.merge_fields_if_missing(extracted.title, Vec::new(), None, None);
                Ok(Some(ResolvedText {
                    origin: origin.clone(),
                    format: TextFormat::Html,
                    markdown: extracted.markdown,
                    source_url: extracted.source_url,
                    raw_bytes_fetched: extracted.raw_bytes_fetched,
                }))
            }
            Err(reason) => {
                session.record_structured_skip(label, reason);
                Ok(None)
            }
        },
        Ok(resp) => {
            session.record_structured_skip(label, format!("HTTP {}", resp.status));
            Ok(None)
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => Err(safety),
        Err(other) => {
            session.record_structured_skip(label, concise_error_detail(&other));
            Ok(None)
        }
    }
}

async fn try_arxiv_source_endpoint(
    id: &ArxivId,
    origin: &ScholarlyId,
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<Option<ResolvedText>, AdfontesError> {
    let src_url = HttpUrl::parse(&format!(
        "{}/src/{}",
        config.endpoints.export_arxiv_org.trim_end_matches('/'),
        id.as_str()
    ))?;
    let mut cached_pdf = false;
    match budget.get(&src_url, None).await {
        Ok(resp) if resp.status == HTTP_OK => {
            if is_pdf_response(&resp) {
                let raw_bytes_fetched = resp.body.len();
                session.record_structured_skip(ENDPOINT_ARXIV_SRC, "PDF-only archive");
                session.record_deferred_fallback(DeferredFallback::Pdf(ResolvedPdf {
                    origin: origin.clone(),
                    bytes: resp.body,
                    source_url: resp.final_url,
                    raw_bytes_fetched,
                }));
                cached_pdf = true;
            } else {
                match extract_arxiv_source_archive(&resp.body) {
                    Ok(extracted) if extracted.markdown.len() >= MIN_FULLTEXT_CHARS => {
                        session.merge_fields_if_missing(extracted.title, Vec::new(), None, None);
                        return Ok(Some(ResolvedText {
                            origin: origin.clone(),
                            format: TextFormat::Latex,
                            markdown: extracted.markdown,
                            source_url: resp.final_url,
                            raw_bytes_fetched: resp.body.len(),
                        }));
                    }
                    Ok(_) => session
                        .record_structured_skip(ENDPOINT_ARXIV_SRC, "extracted TeX was too short"),
                    Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
                    Err(other) => {
                        session.record_structured_skip(
                            ENDPOINT_ARXIV_SRC,
                            concise_error_detail(&other),
                        );
                    }
                }
            }
        }
        Ok(resp) => {
            session.record_structured_skip(ENDPOINT_ARXIV_SRC, format!("HTTP {}", resp.status));
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Err(other) => {
            session.record_structured_skip(ENDPOINT_ARXIV_SRC, concise_error_detail(&other));
        }
    }

    if !cached_pdf {
        let pdf_url = HttpUrl::parse(&format!(
            "{}/pdf/{}.pdf",
            config.endpoints.arxiv_org.trim_end_matches('/'),
            id.as_str()
        ))?;
        session.push_priority_pdf(pdf_url, origin.clone());
    }
    Ok(None)
}

async fn expand_pmcid(
    pmcid: &PmcId,
    depth: u8,
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<Option<ResolvedText>, AdfontesError> {
    let origin = ScholarlyId::PmcId(pmcid.clone());
    let base = config.endpoints.europe_pmc.trim_end_matches('/');
    let search_url = HttpUrl::parse(&format!(
        "{base}/webservices/rest/search?query=PMCID:{}&format=json&resultType=core",
        pmcid.as_str()
    ))?;

    if let Some(search_doc) = fetch_optional_json(&search_url, ACCEPT_JSON, budget).await? {
        let mut metadata = WorkMetadata::default();
        let mut discovery = ProviderDiscovery::default();
        if search_doc.extract_europe_pmc_record(&mut metadata, &mut discovery) {
            session.mark_metadata_seen();
            session.mark_europe_pmc_core_fetched();
            session.merge_metadata(metadata);
            session.ingest_discovery(discovery, &origin, depth);
        }
    }

    if budget.is_exhausted() {
        return Ok(None);
    }
    let xml_url = HttpUrl::parse(&format!(
        "{base}/webservices/rest/{}/fullTextXML",
        pmcid.as_str()
    ))?;
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
                    session.mark_metadata_seen();
                    session.merge_fields_if_missing(title, authors, year, abstract_text);
                    return Ok(Some(ResolvedText {
                        origin,
                        format: TextFormat::JatsXml,
                        markdown,
                        source_url: resp.final_url,
                        raw_bytes_fetched: resp.body.len(),
                    }));
                }
                Ok(JatsExtraction::BodyRestricted {
                    title,
                    authors,
                    year,
                    abstract_text,
                }) => {
                    session.mark_metadata_seen();
                    session.mark_body_restricted();
                    session.merge_fields_if_missing(title, authors, year, abstract_text);
                    session.set_source_url_if_caller_depth(depth, resp.final_url);
                    session.record_structured_skip(ENDPOINT_EPMC_XML, "body restricted");
                }
                Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
                Err(other) => {
                    session.record_structured_skip(ENDPOINT_EPMC_XML, concise_error_detail(&other));
                }
            }
        }
        Ok(resp) => {
            session.record_structured_skip(ENDPOINT_EPMC_XML, format!("HTTP {}", resp.status));
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Err(other) => {
            session.record_structured_skip(ENDPOINT_EPMC_XML, concise_error_detail(&other));
        }
    }
    Ok(None)
}

async fn expand_pmid(
    pmid: &Pmid,
    depth: u8,
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<(), AdfontesError> {
    if depth > 0 && session.has_europe_pmc_core() {
        return Ok(());
    }
    let origin = ScholarlyId::Pmid(pmid.clone());
    let base = config.endpoints.europe_pmc.trim_end_matches('/');
    let search_url = HttpUrl::parse(&format!(
        "{base}/webservices/rest/search?query=EXT_ID:{}%20AND%20SRC:MED&format=json&resultType=core",
        pmid.as_str()
    ))?;

    let Some(search_doc) = fetch_optional_json(&search_url, ACCEPT_JSON, budget).await? else {
        return Ok(());
    };
    let mut metadata = WorkMetadata::default();
    let mut discovery = ProviderDiscovery::default();
    if search_doc.extract_europe_pmc_record(&mut metadata, &mut discovery) {
        session.mark_metadata_seen();
        session.mark_europe_pmc_core_fetched();
        session.merge_metadata(metadata);
        session.ingest_discovery(discovery, &origin, depth);
    } else if depth == 0 {
        session.record_endpoint_failure(
            ENDPOINT_EPMC_PMID,
            format!("PMID {} not found in Europe PMC", pmid.as_str()),
        );
    }
    Ok(())
}

async fn expand_doi(
    doi: &Doi,
    depth: u8,
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<(), AdfontesError> {
    let origin = ScholarlyId::Doi(doi.clone());
    let encoded_doi = doi.percent_encoded_path();
    let doi_canonical_url = HttpUrl::parse(&format!(
        "{}/{encoded_doi}",
        config.endpoints.doi_org.trim_end_matches('/')
    ))?;
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

    if csl_opt.is_none() && oa_opt.is_none() && upw_opt.is_none() {
        return Ok(());
    }
    session.mark_metadata_seen();

    let mut metadata = WorkMetadata::default();
    let mut discovery = ProviderDiscovery::default();
    if let Some(csl) = csl_opt {
        csl.merge_csl_work(&mut metadata);
    }
    if let Some(oa_work) = oa_opt {
        oa_work.merge_openalex_work(&mut metadata, &mut discovery);
    }
    if let Some(upw_work) = upw_opt {
        upw_work.merge_unpaywall_work(&mut metadata, &mut discovery);
    }

    session.merge_metadata(metadata);
    session.ingest_discovery(discovery, &origin, depth);
    Ok(())
}

async fn expand_rfc(
    rfc: RfcNumber,
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> Result<Option<ResolvedText>, AdfontesError> {
    let origin = ScholarlyId::Rfc(rfc);
    let rfc_number = rfc.get();

    // 1. IETF Datatracker metadata & obsoleted-by / updated-by status notes.
    let tracker_url = HttpUrl::parse(&format!(
        "{}/doc/rfc{rfc_number}/doc.json",
        config.endpoints.ietf_datatracker.trim_end_matches('/')
    ))?;
    if let Some(doc_meta) = fetch_optional_json(&tracker_url, ACCEPT_JSON, budget).await? {
        let mut metadata = WorkMetadata::default();
        doc_meta.merge_ietf_doc(&mut metadata);
        session.mark_metadata_seen();
        session.merge_metadata(metadata);
    }

    // 2. Structured RFCXML v3 (`rfc<N>.xml`).
    if budget.is_exhausted() {
        return Ok(None);
    }
    let xml_url = HttpUrl::parse(&format!(
        "{}/rfc/rfc{rfc_number}.xml",
        config.endpoints.rfc_editor.trim_end_matches('/')
    ))?;
    match budget.get(&xml_url, Some(ACCEPT_XML)).await {
        Ok(resp) if resp.status == HTTP_OK => {
            let xml = resp.decode_text();
            match extract_rfc_xml(&xml) {
                Ok(extracted) if !extracted.markdown.is_empty() => {
                    session.merge_fields_if_missing(extracted.title, Vec::new(), None, None);
                    return Ok(Some(ResolvedText {
                        origin,
                        format: TextFormat::RfcXml,
                        markdown: extracted.markdown,
                        source_url: resp.final_url,
                        raw_bytes_fetched: resp.body.len(),
                    }));
                }
                Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
                Ok(_) => session.record_structured_skip(ENDPOINT_RFC_XML, "empty RFCXML body"),
                Err(other) => {
                    session.record_structured_skip(ENDPOINT_RFC_XML, concise_error_detail(&other));
                }
            }
        }
        Ok(resp) => {
            session.record_structured_skip(ENDPOINT_RFC_XML, format!("HTTP {}", resp.status));
        }
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Err(other) => {
            session.record_structured_skip(ENDPOINT_RFC_XML, concise_error_detail(&other));
        }
    }

    // 3. Paginated plain-text RFC (`rfc<N>.txt`) with `\x0c` stripping and MIME check.
    if budget.is_exhausted() {
        return Ok(None);
    }
    let txt_url = HttpUrl::parse(&format!(
        "{}/rfc/rfc{rfc_number}.txt",
        config.endpoints.rfc_editor.trim_end_matches('/')
    ))?;
    match budget.get(&txt_url, Some(MIME_PLAIN_TEXT)).await {
        Ok(resp) if resp.status == HTTP_OK => {
            let mime = resp.mime_type();
            if is_plain_text_mime(&mime) || mime.is_empty() {
                let extracted = clean_rfc_text(&resp.decode_text());
                session.merge_fields_if_missing(extracted.title, Vec::new(), None, None);
                return Ok(Some(ResolvedText {
                    origin,
                    format: TextFormat::RfcText,
                    markdown: extracted.markdown,
                    source_url: resp.final_url,
                    raw_bytes_fetched: resp.body.len(),
                }));
            }
            session.record_endpoint_failure(
                ENDPOINT_RFC_TXT,
                format!("RFC text endpoint returned unexpected MIME `{mime}`"),
            );
        }
        Ok(resp) => session.record_endpoint_failure(
            ENDPOINT_RFC_TXT,
            format!("RFC text endpoint returned HTTP {}", resp.status),
        ),
        Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
        Err(other) => session.record_endpoint_failure(
            ENDPOINT_RFC_TXT,
            format!("RFC text endpoint failed: {other}"),
        ),
    }
    Ok(None)
}

async fn try_oa_pdf_pool(
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
) -> Result<Option<ResolvedPdf>, AdfontesError> {
    while !budget.is_exhausted()
        && let Some(candidate) = session.pop_next_pdf()
    {
        match budget.get(&candidate.url, Some(ACCEPT_PDF)).await {
            Ok(resp) if resp.status == HTTP_OK && is_pdf_response(&resp) => {
                let raw_bytes_fetched = resp.body.len();
                return Ok(Some(ResolvedPdf {
                    origin: candidate.origin,
                    bytes: resp.body,
                    source_url: resp.final_url,
                    raw_bytes_fetched,
                }));
            }
            Ok(resp) if resp.status == HTTP_OK => {
                session.record_endpoint_failure(ENDPOINT_OA_PDF, "not a PDF");
            }
            Ok(resp) => {
                session.record_endpoint_failure(ENDPOINT_OA_PDF, format!("HTTP {}", resp.status));
            }
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            Err(other) => {
                session.record_endpoint_failure(ENDPOINT_OA_PDF, other.to_string());
            }
        }
    }
    Ok(None)
}

enum LandingPoolOutcome {
    FullText(ResolvedText),
    Pdf(ResolvedPdf),
    DiscoveredAlias,
    Exhausted,
}

async fn try_oa_landing_pool(
    session: &mut WorkResolution,
    budget: &ResolveBudget<'_>,
) -> Result<LandingPoolOutcome, AdfontesError> {
    while !budget.is_exhausted()
        && let Some(candidate) = session.pop_next_landing()
    {
        match budget.get(&candidate.url, Some(ACCEPT_WEB_PREFERRED)).await {
            Ok(resp) => {
                if let Some(outcome) =
                    process_oa_landing_response(session, candidate.origin, candidate.depth, resp)
                {
                    return Ok(outcome);
                }
            }
            Err(safety @ AdfontesError::SafetyBlock { .. }) => return Err(safety),
            Err(_) => {}
        }
    }
    Ok(LandingPoolOutcome::Exhausted)
}

fn process_oa_landing_response(
    session: &mut WorkResolution,
    origin: ScholarlyId,
    depth: u8,
    resp: HttpResponse,
) -> Option<LandingPoolOutcome> {
    let mut discovered_new_alias = false;
    let mut handled_structured_repo = false;
    for hop in resp
        .redirect_chain
        .iter()
        .chain(std::iter::once(&resp.final_url))
    {
        if let Some(alias) = classify_known_scholarly_url(hop) {
            let structured = is_structured_repository(&alias);
            let outcome = session.enqueue_alias(alias, depth.saturating_add(1));
            if outcome == AliasEnqueue::Enqueued {
                discovered_new_alias = true;
            }
            if structured && outcome != AliasEnqueue::DepthExceeded {
                handled_structured_repo = true;
            }
        }
    }
    let skip_html = handled_structured_repo || is_arxiv_abs_url(&resp.final_url);
    let extracted = extract_non_stub_landing(resp, origin, session, skip_html);
    if discovered_new_alias {
        if let Some(fallback) = extracted {
            session.record_deferred_fallback(fallback);
        }
        return Some(LandingPoolOutcome::DiscoveredAlias);
    }
    match extracted? {
        DeferredFallback::FullText(text) => Some(LandingPoolOutcome::FullText(text)),
        DeferredFallback::Pdf(pdf) => Some(LandingPoolOutcome::Pdf(pdf)),
    }
}

fn extract_non_stub_landing(
    resp: HttpResponse,
    origin: ScholarlyId,
    session: &mut WorkResolution,
    skip_html: bool,
) -> Option<DeferredFallback> {
    if resp.status != HTTP_OK {
        return None;
    }
    if skip_html {
        return is_pdf_response(&resp).then(|| {
            let raw_bytes_fetched = resp.body.len();
            DeferredFallback::Pdf(ResolvedPdf {
                origin,
                bytes: resp.body,
                source_url: resp.final_url,
                raw_bytes_fetched,
            })
        });
    }
    match extract_oa_landing_response(resp)? {
        LandingExtraction::FullText {
            format,
            title,
            markdown,
            source_url,
            raw_bytes_fetched,
        } => {
            session.merge_fields_if_missing(title, Vec::new(), None, None);
            Some(DeferredFallback::FullText(ResolvedText {
                origin,
                format,
                markdown,
                source_url,
                raw_bytes_fetched,
            }))
        }
        LandingExtraction::Pdf {
            bytes,
            source_url,
            raw_bytes_fetched,
        } => Some(DeferredFallback::Pdf(ResolvedPdf {
            origin,
            bytes,
            source_url,
            raw_bytes_fetched,
        })),
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

fn initial_caller_source_url(caller_id: &ScholarlyId, config: &ResolverConfig) -> Option<HttpUrl> {
    match caller_id {
        ScholarlyId::Doi(doi) => HttpUrl::parse(&format!(
            "{}/{}",
            config.endpoints.doi_org.trim_end_matches('/'),
            doi.percent_encoded_path()
        ))
        .ok(),
        ScholarlyId::Pmid(pmid) => HttpUrl::parse(&format!(
            "https://pubmed.ncbi.nlm.nih.gov/{}/",
            pmid.as_str()
        ))
        .ok(),
        ScholarlyId::Arxiv(_) | ScholarlyId::PmcId(_) | ScholarlyId::Rfc(_) => None,
    }
}

fn fallback_metadata_source_url(
    caller_id: &ScholarlyId,
    config: &ResolverConfig,
) -> Option<HttpUrl> {
    match caller_id {
        ScholarlyId::Arxiv(id) => HttpUrl::parse(&format!(
            "{}/abs/{}",
            config.endpoints.arxiv_org.trim_end_matches('/'),
            id.as_str()
        ))
        .ok(),
        ScholarlyId::PmcId(pmcid) => HttpUrl::parse(&format!(
            "https://europepmc.org/article/PMC/{}",
            pmcid.as_str()
        ))
        .ok(),
        ScholarlyId::Rfc(rfc) => HttpUrl::parse(&format!(
            "{}/doc/rfc{}/",
            config.endpoints.ietf_datatracker.trim_end_matches('/'),
            rfc.get()
        ))
        .ok(),
        ScholarlyId::Doi(_) | ScholarlyId::Pmid(_) => None,
    }
}

fn build_caller_transport_error(
    caller_id: &ScholarlyId,
    trace: &UnavailableTrace,
    budget: &ResolveBudget<'_>,
    config: &ResolverConfig,
) -> AdfontesError {
    match caller_id {
        ScholarlyId::Arxiv(id) => {
            let url = format!(
                "{}/pdf/{}.pdf",
                config.endpoints.arxiv_org.trim_end_matches('/'),
                id.as_str()
            );
            let detail = if trace.failure_summary.is_empty() {
                budget.append_last_transport_error("all arXiv retrieval endpoints failed")
            } else {
                format!(
                    "all arXiv retrieval endpoints failed ({})",
                    trace.failure_summary
                )
            };
            AdfontesError::Transport { url, detail }
        }
        ScholarlyId::Doi(doi) => {
            let url = format!(
                "{}/{}",
                config.endpoints.doi_org.trim_end_matches('/'),
                doi.percent_encoded_path()
            );
            AdfontesError::Transport {
                url,
                detail: budget.append_last_transport_error(
                    "DOI resolution failed across doi.org and OpenAlex",
                ),
            }
        }
        ScholarlyId::PmcId(pmcid) => {
            let url = format!(
                "{}/webservices/rest/{}/fullTextXML",
                config.endpoints.europe_pmc.trim_end_matches('/'),
                pmcid.as_str()
            );
            AdfontesError::Transport {
                url,
                detail: budget.append_last_transport_error(
                    "Europe PMC returned neither metadata nor fullTextXML",
                ),
            }
        }
        ScholarlyId::Pmid(pmid) => {
            let url = format!(
                "{}/webservices/rest/search?query=EXT_ID:{}%20AND%20SRC:MED&format=json&resultType=core",
                config.endpoints.europe_pmc.trim_end_matches('/'),
                pmid.as_str()
            );
            let detail = if trace.failure_summary.is_empty() {
                budget.append_last_transport_error("Europe PMC PMID lookup failed")
            } else {
                trace.failure_summary.clone()
            };
            AdfontesError::Transport { url, detail }
        }
        ScholarlyId::Rfc(rfc) => {
            let url = format!(
                "{}/rfc/rfc{}.txt",
                config.endpoints.rfc_editor.trim_end_matches('/'),
                rfc.get()
            );
            let detail = if trace.failure_summary.is_empty() {
                budget.append_last_transport_error("all RFC retrieval endpoints failed")
            } else {
                trace.failure_summary.clone()
            };
            AdfontesError::Transport { url, detail }
        }
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

fn concise_error_detail(error: &AdfontesError) -> String {
    match error {
        AdfontesError::Transport { detail, .. } | AdfontesError::Extraction { detail, .. } => {
            detail.clone()
        }
        AdfontesError::InvalidTarget { reason, .. } | AdfontesError::SafetyBlock { reason } => {
            reason.clone()
        }
        AdfontesError::Io { .. } => error.to_string(),
    }
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
    use crate::model::{ExtractedPayload, UnavailableReason};
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
    async fn pmcid_follows_europe_pmc_doi_to_open_access_pdf_when_jats_body_restricted() {
        let jats_front_only = indoc! {r"
            <article>
              <front><article-meta><title-group><article-title>JRSS-C Trial</article-title></title-group></article-meta></front>
            </article>
        "};
        let server = MockServer::start().await;
        let pdf_url = format!("{}/oa/rssc.pdf", server.uri());
        Mock::given(method("GET"))
            .and(|req: &WiremockRequest| req.url.as_str().contains("PMCID:PMC10016948"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "resultList": {
                    "result": [{
                        "title": "JRSS-C Trial",
                        "journalTitle": "JRSS-C",
                        "pubYear": "2023",
                        "pmcid": "PMC10016948",
                        "doi": "10.1111/rssc.12435"
                    }]
                }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("PMC10016948/fullTextXML"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(jats_front_only, "application/xml"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "best_oa_location": {"is_oa": true, "pdf_url": pdf_url}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/oa/rssc.pdf"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(b"%PDF-1.7\n%mock-pdf\n".as_slice(), "application/pdf"),
            )
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("PMC10016948").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();

        let ExtractedPayload::Pdf { fallback_skips, .. } = doc.payload else {
            panic!("expected PDF payload");
        };
        assert_eq!(doc.metadata.canonical_id.as_deref(), Some("PMC10016948"));
        assert_eq!(doc.metadata.venue_or_source.as_deref(), Some("JRSS-C"));
        assert!(
            doc.metadata
                .notes
                .iter()
                .any(|note| note == "PDF via 10.1111/rssc.12435")
        );
        assert!(
            fallback_skips
                .iter()
                .any(|skip| skip.endpoint == ENDPOINT_EPMC_XML && skip.reason == "body restricted")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pmcid_prefers_discovered_arxiv_structured_text_over_europe_pmc_pdf() {
        let tex = indoc! {r"
            \documentclass{article}
            \title{Structured ArXiv Mirror}
            \begin{document}
            \section{Results}
            Complete derivations from the structured arXiv source mirror that should always beat
            a binary Europe PMC PDF when both open-access copies exist for the same scholarly work.
            \end{document}
        "};
        let server = MockServer::start().await;
        let epmc_pdf_url = format!("{}/epmc/paper.pdf", server.uri());
        Mock::given(method("GET"))
            .and(|req: &WiremockRequest| req.url.as_str().contains("PMCID:PMC2222222"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "resultList": {
                    "result": [{
                        "title": "Structured ArXiv Mirror",
                        "doi": "10.48550/arXiv.2409.13740",
                        "fullTextUrlList": {
                            "fullTextUrl": [{
                                "documentStyle": "pdf",
                                "availabilityCode": "OA",
                                "url": epmc_pdf_url
                            }]
                        }
                    }]
                }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("PMC2222222/fullTextXML"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/src/2409\.13740"))
            .respond_with(ResponseTemplate::new(200).set_body_string(tex))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/epmc/paper.pdf"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(b"%PDF-1.7\n%should-not-win\n".as_slice(), "application/pdf"),
            )
            .expect(0)
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("PMC2222222").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();

        server.verify().await;
        assert!(matches!(
            doc.payload,
            ExtractedPayload::FullText {
                format: TextFormat::Latex,
                ..
            }
        ));
        assert!(
            doc.metadata
                .notes
                .iter()
                .any(|note| note == "Full text via arXiv:2409.13740")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pdf_only_arxiv_follows_atom_doi_to_pmcid_jats_before_falling_back_to_cached_pdf() {
        let atom = indoc! {r#"
            <?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
              <entry>
                <title>Journal Published Preprint</title>
                <published>2024-01-01T00:00:00Z</published>
                <arxiv:doi>10.1000/jats-mirror</arxiv:doi>
              </entry>
            </feed>
        "#};
        let jats = indoc! {r"
            <article>
              <front><article-meta><title-group><article-title>Journal Published Preprint</article-title></title-group></article-meta></front>
              <body><sec><title>Introduction</title><p>Full structured JATS text discovered through the arXiv Atom journal DOI and OpenAlex PMCID link, preventing unnecessary PDF fallback.</p></sec></body>
            </article>
        "};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/api/query"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(atom, "application/xml"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/src/2401\.00099"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(b"%PDF-1.7\n%pdf-only-src\n".as_slice(), "application/pdf"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "ids": {"pmcid": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3333333/"}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("PMC3333333/fullTextXML"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(jats, "application/xml"))
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("2401.00099").unwrap();
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
                format: TextFormat::JatsXml,
                ..
            }
        ));
        assert!(
            doc.metadata
                .notes
                .iter()
                .any(|note| note == "Full text via PMC3333333")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pdf_only_arxiv_reuses_cached_src_pdf_without_refetching_pdf_endpoint() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/src/0710\.5005"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(b"%PDF-1.7\n%cached-src-pdf\n".as_slice(), "application/pdf"),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/pdf/0710\.5005\.pdf"))
            .respond_with(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("https://ar5iv.labs.arxiv.org/html/0710.5005").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[*server.address()]),
            &resolver_config_for(&server.uri(), None),
        )
        .await
        .unwrap();

        server.verify().await;
        let ExtractedPayload::Pdf { fallback_skips, .. } = doc.payload else {
            panic!("expected PDF payload");
        };
        assert_eq!(
            doc.metadata.canonical_id.as_deref(),
            Some("arXiv:0710.5005")
        );
        assert!(
            fallback_skips
                .iter()
                .any(|skip| skip.endpoint == ENDPOINT_ARXIV_HTML)
        );
        assert!(
            fallback_skips
                .iter()
                .any(|skip| skip.endpoint == ENDPOINT_AR5IV)
        );
        assert!(
            fallback_skips.iter().any(
                |skip| skip.endpoint == ENDPOINT_ARXIV_SRC && skip.reason == "PDF-only archive"
            )
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn medrxiv_url_resolves_through_doi_waterfall_to_pmcid_jats() {
        let jats = indoc! {r"
            <article>
              <front><article-meta><title-group><article-title>MedRxiv Preprint</article-title></title-group></article-meta></front>
              <body><sec><title>Methods</title><p>Cohort analysis recovered from Europe PMC JATS XML after classifying the medRxiv URL as its Crossref 10.1101 DOI.</p></sec></body>
            </article>
        "};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex(
                r"^/openalex/works/https://doi\.org/10\.1101/2024\.06\.03\.24308385",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "ids": {"pmcid": "PMC4444444"}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("PMC4444444/fullTextXML"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(jats, "application/xml"))
            .mount(&server)
            .await;

        let target = ParsedTarget::parse(
            "https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v2.full",
        )
        .unwrap();
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
                format: TextFormat::JatsXml,
                ..
            }
        ));
        assert_eq!(
            doc.metadata.canonical_id.as_deref(),
            Some("10.1101/2024.06.03.24308385")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn openalex_arxiv_pdf_url_upgrades_to_structured_arxiv_tex_source() {
        let tex = indoc! {r"
            \documentclass{article}
            \title{Upgraded From PDF URL}
            \begin{document}
            \section{Proof}
            OpenAlex listed only an arXiv PDF link, which the session upgraded into a structured
            arXiv frontier entry so LaTeX source won over the binary PDF mirror.
            \end{document}
        "};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "title": "Upgraded From PDF URL",
                "best_oa_location": {
                    "is_oa": true,
                    "pdf_url": "https://arxiv.org/pdf/2409.13740.pdf"
                }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/src/2409\.13740"))
            .respond_with(ResponseTemplate::new(200).set_body_string(tex))
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("10.1000/arxiv-pdf-upgrade").unwrap();
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
        assert!(
            doc.metadata
                .notes
                .iter()
                .any(|note| note == "Full text via arXiv:2409.13740")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn arxiv_metadata_only_preserves_abs_source_url_when_alias_pmcid_is_body_restricted() {
        let atom = indoc! {r#"
            <?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
              <entry>
                <title>Embargoed Preprint</title>
                <published>2024-01-01T00:00:00Z</published>
                <arxiv:doi>10.1000/embargoed</arxiv:doi>
              </entry>
            </feed>
        "#};
        let front_only = b"<article><front><article-meta><title-group><article-title>Embargoed Preprint</article-title></title-group></article-meta></front></article>";
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/api/query"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(atom, "application/xml"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "ids": {"pmcid": "PMC5555555"}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("PMC5555555/fullTextXML"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(front_only.as_slice(), "application/xml"),
            )
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("2401.00077").unwrap();
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
        assert_eq!(
            doc.metadata.source_url.as_ref().map(HttpUrl::path),
            Some("/abs/2401.00077")
        );
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
    async fn oa_pool_propagates_ssrf_safety_block_fail_closed() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "title": "SSRF Test",
                "best_oa_location": {
                    "is_oa": true,
                    "pdf_url": "http://169.254.169.254/latest/meta-data"
                }
            })))
            .mount(&server)
            .await;

        let target = ParsedTarget::parse("10.1000/ssrf-pool").unwrap();
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
    async fn missing_pmid_reports_not_found_and_still_allows_seed_aliases() {
        let tex = indoc! {r"
            \documentclass{article}
            \title{Seeded Preprint}
            \begin{document}
            \section{Body}
            Resolved via seed alias even when Europe PMC returned an empty PMID resultList for
            the primary PubMed identifier in a multi-hop redirect chain.
            \end{document}
        "};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_regex("webservices/rest/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "resultList": {"result": []}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/src/2409\.13740"))
            .respond_with(ResponseTemplate::new(200).set_body_string(tex))
            .mount(&server)
            .await;

        let client = http_client_for(&[*server.address()]);
        let config = resolver_config_for(&server.uri(), None);
        let missing_target = ParsedTarget::parse("PMID:99999999").unwrap();
        let missing_error = resolve_target(&missing_target, &client, &config)
            .await
            .unwrap_err();
        let budget = ResolveBudget::new(&client);
        let pmid = ScholarlyId::parse("PMID:99999999").unwrap();
        let seed = ScholarlyId::parse("2409.13740").unwrap();
        let seeded_doc = resolve_work(&pmid, &[seed], &budget, &config)
            .await
            .unwrap();

        assert!(matches!(
            missing_error,
            AdfontesError::Transport { ref detail, .. }
                if detail.contains("PMID 99999999 not found in Europe PMC")
        ));
        assert!(matches!(
            seeded_doc.payload,
            ExtractedPayload::FullText {
                format: TextFormat::Latex,
                ..
            }
        ));
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
    async fn doi_and_arxiv_surface_unsupported_proxy_error_instead_of_generic_failure() {
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
                arxiv_org: "https://example.org/arxiv".to_owned(),
                export_arxiv_org: "https://example.org/export".to_owned(),
                ar5iv: "https://example.org/ar5iv".to_owned(),
                doi_org: "https://example.org/doi".to_owned(),
                openalex: "https://example.org/openalex".to_owned(),
                ..ResolverEndpoints::production()
            },
        };
        let doi_target = ParsedTarget::parse("10.1000/proxy-check").unwrap();
        let arxiv_target = ParsedTarget::parse("2409.13740").unwrap();
        let doi_failure = resolve_target(&doi_target, &client, &config)
            .await
            .unwrap_err();
        let arxiv_failure = resolve_target(&arxiv_target, &client, &config)
            .await
            .unwrap_err();

        assert!(
            matches!(doi_failure, AdfontesError::Transport { ref detail, .. } if detail.contains("socks5"))
        );
        assert!(
            matches!(arxiv_failure, AdfontesError::Transport { ref detail, .. } if detail.contains("socks5"))
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
    async fn pmcid_and_doi_achieve_identical_pdf_recovery_without_redundant_epmc_search() {
        let jats_front_only = indoc! {r"
            <article>
              <front><article-meta><title-group><article-title>JRSS-C Trial</article-title></title-group></article-meta></front>
            </article>
        "};
        let server = MockServer::start().await;
        let pdf_url = format!("{}/oa/rssc.pdf", server.uri());
        Mock::given(method("GET"))
            .and(path_regex("^/epmc/webservices/rest/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "resultList": {
                    "result": [{
                        "title": "JRSS-C Trial",
                        "journalTitle": "JRSS-C",
                        "pubYear": "2023",
                        "pmcid": "PMC10016948",
                        "pmid": "36998643",
                        "doi": "10.1111/rssc.12435"
                    }]
                }
            })))
            .expect(2)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("PMC10016948/fullTextXML"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(jats_front_only, "application/xml"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex("^/openalex/"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "ids": {
                    "pmcid": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10016948/",
                    "pmid": "https://pubmed.ncbi.nlm.nih.gov/36998643"
                },
                "best_oa_location": {"is_oa": true, "pdf_url": pdf_url}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/oa/rssc.pdf"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(b"%PDF-1.7\n%mock-pdf\n".as_slice(), "application/pdf"),
            )
            .mount(&server)
            .await;

        let client = http_client_for(&[*server.address()]);
        let config = resolver_config_for(&server.uri(), None);
        let pmc_target = ParsedTarget::parse("PMC10016948").unwrap();
        let doi_target = ParsedTarget::parse("10.1111/rssc.12435").unwrap();
        let pmc_doc = resolve_target(&pmc_target, &client, &config).await.unwrap();
        let doi_doc = resolve_target(&doi_target, &client, &config).await.unwrap();

        server.verify().await;
        assert!(matches!(pmc_doc.payload, ExtractedPayload::Pdf { .. }));
        assert!(matches!(doi_doc.payload, ExtractedPayload::Pdf { .. }));
        assert_eq!(pmc_doc.metadata.source_url, doi_doc.metadata.source_url);
        assert!(
            pmc_doc
                .metadata
                .notes
                .iter()
                .any(|note| note == "PDF via 10.1111/rssc.12435")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn doi_executes_metadata_legs_concurrently_with_deterministic_merge() {
        let (addr, server) = spawn_concurrent_doi_barrier_server(2);
        let target = ParsedTarget::parse("10.1000/concurrent-two").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[addr]),
            &resolver_config_for(&format!("http://{addr}"), None),
        )
        .await
        .unwrap();
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
        let target = ParsedTarget::parse("10.1000/concurrent-three").unwrap();
        let doc = resolve_target(
            &target,
            &http_client_for(&[addr]),
            &resolver_config_for(&format!("http://{addr}"), Some("researcher@university.edu")),
        )
        .await
        .unwrap();
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

    #[tokio::test(flavor = "current_thread")]
    async fn landing_pool_recovers_open_access_html_and_pdf_across_distinct_candidates() {
        let body_pad = "Empirical hierarchical estimation converges cleanly with tight posterior calibration. ".repeat(38);
        let html_article = format!(
            "<article><h1>Open Access Monograph</h1><h2>1 Introduction</h2><p>{body_pad}</p><h2>2 Results</h2><p>Convergence holds across all strata.</p></article>"
        );
        let server = MockServer::start().await;
        let html_url = format!("{}/repo/article.html", server.uri());
        let pdf_url = format!("{}/repo/paper.pdf", server.uri());
        Mock::given(method("GET"))
            .and(path("/openalex/works/https://doi.org/10.1000/landing-html"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "title": "Landing HTML",
                "best_oa_location": {"is_oa": true, "landing_page_url": html_url}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/repo/article.html"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(html_article.into_bytes(), "text/html"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/openalex/works/https://doi.org/10.1000/landing-pdf"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "title": "Landing PDF",
                "best_oa_location": {"is_oa": true, "landing_page_url": pdf_url}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/repo/paper.pdf"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                b"%PDF-1.7\n%mock-landing-pdf\n".as_slice(),
                "application/pdf",
            ))
            .mount(&server)
            .await;

        let client = http_client_for(&[*server.address()]);
        let config = resolver_config_for(&server.uri(), None);
        let html_doc = resolve_target(
            &ParsedTarget::parse("10.1000/landing-html").unwrap(),
            &client,
            &config,
        )
        .await
        .unwrap();
        let pdf_doc = resolve_target(
            &ParsedTarget::parse("10.1000/landing-pdf").unwrap(),
            &client,
            &config,
        )
        .await
        .unwrap();

        assert!(matches!(
            html_doc.payload,
            ExtractedPayload::FullText {
                format: TextFormat::Html,
                ..
            }
        ));
        assert!(matches!(pdf_doc.payload, ExtractedPayload::Pdf { .. }));
    }

    #[test]
    fn landing_redirect_to_alias_defers_open_access_html_instead_of_discarding_body() {
        let caller = ScholarlyId::parse("10.1000/outer-work").unwrap();
        let mut session = WorkResolution::new(caller.clone(), None, &[]);
        let _ = session.pop_next_frontier();
        let body_pad =
            "Complete open-access empirical results and theoretical proofs across cohorts. "
                .repeat(38);
        let html_article = format!(
            "<article><h1>Deferred Article</h1><h2>1 Setup</h2><p>{body_pad}</p><h2>2 Proofs</h2><p>All theorems follow directly.</p></article>"
        );
        let response = HttpResponse {
            status: HTTP_OK,
            content_type: Some("text/html; charset=utf-8".to_owned()),
            headers: std::collections::BTreeMap::new(),
            body: html_article.into_bytes(),
            final_url: HttpUrl::parse("https://publisher.example.org/open-article").unwrap(),
            redirect_chain: vec![HttpUrl::parse("https://doi.org/10.1000/inner-doi").unwrap()],
        };

        let outcome = process_oa_landing_response(&mut session, caller, 0, response);
        let discovered = session.pop_next_frontier().unwrap();
        let deferred = session.take_deferred_fallback().unwrap();

        assert!(matches!(outcome, Some(LandingPoolOutcome::DiscoveredAlias)));
        assert_eq!(discovered.id.to_string(), "10.1000/inner-doi");
        assert!(
            matches!(deferred, DeferredFallback::FullText(ref text) if text.markdown.contains("All theorems follow directly."))
        );
    }

    #[test]
    fn concise_error_detail_preserves_transport_and_extraction_causes() {
        let transport = AdfontesError::Transport {
            url: "https://ar5iv.labs.arxiv.org/html/0710.5005".to_owned(),
            detail: "request timed out".to_owned(),
        };
        let extraction = AdfontesError::Extraction {
            format: "TeX",
            detail: "archive contained no .tex files".to_owned(),
        };

        assert_eq!(concise_error_detail(&transport), "request timed out");
        assert_eq!(
            concise_error_detail(&extraction),
            "archive contained no .tex files"
        );
    }
}
