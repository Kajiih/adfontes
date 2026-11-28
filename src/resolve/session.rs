//! Pure, synchronous scholarly work-resolution frontier and fallback pool.
//!
//! Coordinates alias discovery, structured-first queue ordering, deduplicated open-access PDF and
//! landing fallback pools, bibliographic metadata merging, and endpoint failure tracking with zero
//! network I/O, async runtime, or manifest wire-format dependencies.

use std::collections::{BTreeSet, VecDeque};

use super::providers::{DiscoveryParts, ProviderDiscovery};
use crate::model::{
    ExtractedDocument, ExtractedPayload, HttpUrl, ScholarlyId, ScholarlyIdKey, TextFormat,
    UnavailableReason, WorkMetadata, classify_known_scholarly_url,
};

const MAX_RECLASSIFICATION_DEPTH: u8 = 3;
const MAX_OA_CANDIDATES: usize = 6;

const GENERIC_REPOSITORY_VENUES: &[&str] = &["arXiv", "Europe PMC", "PubMed", "IETF"];

/// Scheduled scholarly identifier paired with its discovery hop depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FrontierEntry {
    pub(super) id: ScholarlyId,
    pub(super) depth: u8,
}

/// Candidate open-access PDF URL paired with the scholarly identifier that discovered it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PooledPdf {
    pub(super) url: HttpUrl,
    pub(super) origin: ScholarlyId,
}

/// Candidate open-access landing URL paired with its origin identifier and hop depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PooledLanding {
    pub(super) url: HttpUrl,
    pub(super) origin: ScholarlyId,
    pub(super) depth: u8,
}

/// Classification of attempting to enqueue a discovered [`ScholarlyId`] on the frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AliasEnqueue {
    /// Unvisited identifier within [`MAX_RECLASSIFICATION_DEPTH`] queued for expansion.
    Enqueued,
    /// Identifier's canonical key was already seen in this session.
    AlreadyVisited,
    /// Identifier was skipped solely because `depth > MAX_RECLASSIFICATION_DEPTH`.
    DepthExceeded,
}

/// Verified structured full-text payload ready to finalize against caller metadata.
#[derive(Debug)]
pub(super) struct ResolvedText {
    pub(super) origin: ScholarlyId,
    pub(super) format: TextFormat,
    pub(super) markdown: String,
    pub(super) source_url: HttpUrl,
    pub(super) raw_bytes_fetched: usize,
}

/// Verified binary PDF payload ready to finalize against caller metadata.
#[derive(Debug)]
pub(super) struct ResolvedPdf {
    pub(super) origin: ScholarlyId,
    pub(super) bytes: Vec<u8>,
    pub(super) source_url: HttpUrl,
    pub(super) raw_bytes_fetched: usize,
}

/// Payload already fetched during discovery or landing expansion, deferred until any newly
/// discovered structured aliases finish running.
#[derive(Debug)]
pub(super) enum DeferredFallback {
    FullText(ResolvedText),
    Pdf(ResolvedPdf),
}

/// Failure trace returned when neither full text nor bibliographic metadata could be retrieved.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct UnavailableTrace {
    pub(super) failure_summary: String,
}

/// Pure work-resolution state machine spanning all equivalent identifiers for one scholarly work.
#[derive(Debug)]
pub(super) struct WorkResolution {
    caller_id: ScholarlyId,
    caller_key: ScholarlyIdKey,
    metadata: WorkMetadata,
    saw_metadata: bool,
    saw_body_restricted: bool,
    europe_pmc_core_fetched: bool,
    visited: BTreeSet<ScholarlyIdKey>,
    structured_queue: VecDeque<FrontierEntry>,
    hub_queue: VecDeque<FrontierEntry>,
    oa_pool: OaFallbackPool,
    deferred_fallback: Option<DeferredFallback>,
    trace: ResolutionTrace,
}

impl WorkResolution {
    pub(super) fn new(
        caller_id: ScholarlyId,
        initial_source_url: Option<HttpUrl>,
        seed_aliases: &[ScholarlyId],
    ) -> Self {
        let caller_key = caller_id.canonical_key();
        let metadata = WorkMetadata {
            canonical_id: Some(caller_id.to_string()),
            venue_or_source: default_venue(&caller_id).map(str::to_owned),
            source_url: initial_source_url,
            ..WorkMetadata::default()
        };
        let mut session = Self {
            caller_id: caller_id.clone(),
            caller_key,
            metadata,
            saw_metadata: false,
            saw_body_restricted: false,
            europe_pmc_core_fetched: false,
            visited: BTreeSet::new(),
            structured_queue: VecDeque::new(),
            hub_queue: VecDeque::new(),
            oa_pool: OaFallbackPool::default(),
            deferred_fallback: None,
            trace: ResolutionTrace::default(),
        };
        session.enqueue_alias(caller_id, 0);
        for alias in seed_aliases {
            session.enqueue_alias(alias.clone(), 1);
        }
        session
    }

    pub(super) fn enqueue_alias(&mut self, id: ScholarlyId, depth: u8) -> AliasEnqueue {
        let key = id.canonical_key();
        if self.visited.contains(&key) {
            return AliasEnqueue::AlreadyVisited;
        }
        if depth > MAX_RECLASSIFICATION_DEPTH {
            return AliasEnqueue::DepthExceeded;
        }
        self.visited.insert(key);
        let entry = FrontierEntry { id, depth };
        if is_structured_repository(&entry.id) {
            self.structured_queue.push_back(entry);
        } else {
            self.hub_queue.push_back(entry);
        }
        AliasEnqueue::Enqueued
    }

    pub(super) fn pop_next_frontier(&mut self) -> Option<FrontierEntry> {
        self.structured_queue
            .pop_front()
            .or_else(|| self.hub_queue.pop_front())
    }

    pub(super) fn ingest_discovery(
        &mut self,
        discovery: ProviderDiscovery,
        origin: &ScholarlyId,
        depth: u8,
    ) {
        let next_depth = depth.saturating_add(1);
        let DiscoveryParts {
            aliases,
            pdf_urls,
            landing_urls,
        } = discovery.into_parts();

        for alias in aliases {
            self.enqueue_alias(alias, next_depth);
        }
        let mut pdf_added = 0usize;
        for pdf_url in pdf_urls {
            if let Some(alias) = classify_known_scholarly_url(&pdf_url) {
                self.enqueue_alias(alias, next_depth);
            }
            if pdf_added < MAX_OA_CANDIDATES && self.oa_pool.push_pdf(pdf_url, origin.clone()) {
                pdf_added += 1;
            }
        }
        let mut landing_added = 0usize;
        for landing_url in landing_urls {
            if let Some(alias) = classify_known_scholarly_url(&landing_url) {
                let structured = is_structured_repository(&alias);
                let outcome = self.enqueue_alias(alias, next_depth);
                if structured && outcome != AliasEnqueue::DepthExceeded {
                    continue;
                }
            }
            if landing_added < MAX_OA_CANDIDATES
                && self
                    .oa_pool
                    .push_landing(landing_url, origin.clone(), depth)
            {
                landing_added += 1;
            }
        }
    }

    pub(super) fn push_priority_pdf(&mut self, url: HttpUrl, origin: ScholarlyId) {
        self.oa_pool.push_priority_pdf(url, origin);
    }

    pub(super) fn pop_next_pdf(&mut self) -> Option<PooledPdf> {
        self.oa_pool.pop_next_pdf()
    }

    pub(super) fn pop_next_landing(&mut self) -> Option<PooledLanding> {
        self.oa_pool.pop_next_landing()
    }

    pub(super) fn record_deferred_fallback(&mut self, fallback: DeferredFallback) {
        match fallback {
            DeferredFallback::FullText(_) => {
                if !matches!(self.deferred_fallback, Some(DeferredFallback::FullText(_))) {
                    self.deferred_fallback = Some(fallback);
                }
            }
            DeferredFallback::Pdf(_) => {
                if self.deferred_fallback.is_none() {
                    self.deferred_fallback = Some(fallback);
                }
            }
        }
    }

    pub(super) const fn take_deferred_fallback(&mut self) -> Option<DeferredFallback> {
        self.deferred_fallback.take()
    }

    pub(super) const fn mark_metadata_seen(&mut self) {
        self.saw_metadata = true;
    }

    pub(super) const fn mark_body_restricted(&mut self) {
        self.saw_body_restricted = true;
    }

    pub(super) const fn mark_europe_pmc_core_fetched(&mut self) {
        self.europe_pmc_core_fetched = true;
    }

    pub(super) const fn has_europe_pmc_core(&self) -> bool {
        self.europe_pmc_core_fetched
    }

    pub(super) fn merge_metadata(&mut self, extra: WorkMetadata) {
        merge_missing_metadata(&mut self.metadata, extra);
    }

    pub(super) fn merge_fields_if_missing(
        &mut self,
        title: Option<String>,
        authors: Vec<String>,
        year: Option<u16>,
        abstract_text: Option<String>,
    ) {
        fill_metadata_if_missing(&mut self.metadata, title, authors, year, abstract_text);
    }

    pub(super) fn set_source_url_if_caller_depth(&mut self, depth: u8, url: HttpUrl) {
        if depth == 0 {
            self.metadata.source_url = Some(url);
        }
    }

    pub(super) fn record_endpoint_failure(
        &mut self,
        endpoint: &'static str,
        reason: impl Into<String>,
    ) {
        self.trace.record_endpoint_failure(endpoint, reason);
    }

    pub(super) fn finish_full_text(mut self, resolved: ResolvedText) -> ExtractedDocument {
        self.metadata.source_url = Some(resolved.source_url);
        if resolved.origin.canonical_key() != self.caller_key {
            let note = format!("Full text via {}", resolved.origin);
            if !self.metadata.notes.contains(&note) {
                self.metadata.notes.push(note);
            }
        }
        ExtractedDocument {
            metadata: self.metadata,
            payload: ExtractedPayload::FullText {
                format: resolved.format,
                markdown: resolved.markdown,
            },
            raw_bytes_fetched: resolved.raw_bytes_fetched,
        }
    }

    pub(super) fn finish_pdf(mut self, resolved: ResolvedPdf) -> ExtractedDocument {
        self.metadata.source_url = Some(resolved.source_url);
        if resolved.origin.canonical_key() != self.caller_key {
            let note = format!("PDF via {}", resolved.origin);
            if !self.metadata.notes.contains(&note) {
                self.metadata.notes.push(note);
            }
        }
        ExtractedDocument {
            metadata: self.metadata,
            payload: ExtractedPayload::Pdf {
                bytes: resolved.bytes,
            },
            raw_bytes_fetched: resolved.raw_bytes_fetched,
        }
    }

    pub(super) fn finish_unavailable(
        mut self,
        fallback_source_url: Option<HttpUrl>,
    ) -> Result<ExtractedDocument, UnavailableTrace> {
        let failure_summary = self.trace.failure_summary();
        if self.saw_metadata {
            if self.metadata.source_url.is_none() {
                self.metadata.source_url = fallback_source_url;
            }
            let reason = if self.saw_body_restricted {
                UnavailableReason::BodyRestricted
            } else if !failure_summary.is_empty()
                && matches!(self.caller_id, ScholarlyId::Arxiv(_) | ScholarlyId::Rfc(_))
            {
                UnavailableReason::UpstreamUnavailable {
                    detail: failure_summary,
                }
            } else {
                UnavailableReason::NoOpenAccessCopy
            };
            return Ok(ExtractedDocument {
                metadata: self.metadata,
                payload: ExtractedPayload::MetadataOnly { reason },
                raw_bytes_fetched: 0,
            });
        }

        Err(UnavailableTrace { failure_summary })
    }
}

/// Returns `true` when `id` resolves directly against a structured full-text repository.
pub(super) const fn is_structured_repository(id: &ScholarlyId) -> bool {
    matches!(
        id,
        ScholarlyId::Arxiv(_) | ScholarlyId::PmcId(_) | ScholarlyId::Rfc(_)
    )
}

#[derive(Debug, Default)]
struct OaFallbackPool {
    priority_pdfs: VecDeque<PooledPdf>,
    pdf_urls: VecDeque<PooledPdf>,
    landing_urls: VecDeque<PooledLanding>,
    seen_pdfs: BTreeSet<String>,
    seen_landings: BTreeSet<String>,
}

impl OaFallbackPool {
    fn push_priority_pdf(&mut self, url: HttpUrl, origin: ScholarlyId) {
        if self.seen_pdfs.insert(url.as_str().to_owned()) {
            self.priority_pdfs.push_back(PooledPdf { url, origin });
            return;
        }
        if let Some(pos) = self
            .pdf_urls
            .iter()
            .position(|candidate| candidate.url == url)
            && let Some(promoted) = self.pdf_urls.remove(pos)
        {
            self.priority_pdfs.push_back(promoted);
        }
    }

    fn push_pdf(&mut self, url: HttpUrl, origin: ScholarlyId) -> bool {
        if self.seen_pdfs.insert(url.as_str().to_owned()) {
            self.pdf_urls.push_back(PooledPdf { url, origin });
            return true;
        }
        false
    }

    fn push_landing(&mut self, url: HttpUrl, origin: ScholarlyId, depth: u8) -> bool {
        if self.seen_landings.insert(url.as_str().to_owned()) {
            self.landing_urls
                .push_back(PooledLanding { url, origin, depth });
            return true;
        }
        false
    }

    fn pop_next_pdf(&mut self) -> Option<PooledPdf> {
        self.priority_pdfs
            .pop_front()
            .or_else(|| self.pdf_urls.pop_front())
    }

    fn pop_next_landing(&mut self) -> Option<PooledLanding> {
        self.landing_urls.pop_front()
    }
}

#[derive(Debug, Default)]
struct ResolutionTrace {
    endpoint_failures: Vec<String>,
}

impl ResolutionTrace {
    fn record_endpoint_failure(&mut self, endpoint: &'static str, reason: impl Into<String>) {
        self.endpoint_failures
            .push(format!("{endpoint}: {}", reason.into()));
    }

    fn failure_summary(&self) -> String {
        self.endpoint_failures.join("; ")
    }
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

const fn default_venue(id: &ScholarlyId) -> Option<&'static str> {
    match id {
        ScholarlyId::Arxiv(_) => Some("arXiv"),
        ScholarlyId::PmcId(_) => Some("Europe PMC"),
        ScholarlyId::Pmid(_) => Some("PubMed"),
        ScholarlyId::Rfc(_) => Some("IETF"),
        ScholarlyId::Doi(_) => None,
    }
}

fn is_generic_repository_venue(venue: &str) -> bool {
    GENERIC_REPOSITORY_VENUES
        .iter()
        .any(|generic| venue.eq_ignore_ascii_case(generic))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontier_prioritizes_structured_repositories_ahead_of_hubs_and_deduplicates_keys() {
        let caller = ScholarlyId::parse("10.1111/rssc.12435").unwrap();
        let mut session = WorkResolution::new(caller, None, &[]);
        let first = session.pop_next_frontier().unwrap();
        let enqueued_pmid = session.enqueue_alias(ScholarlyId::parse("PMID:36998643").unwrap(), 1);
        let enqueued_pmc = session.enqueue_alias(ScholarlyId::parse("PMC10016948").unwrap(), 1);
        let enqueued_dup =
            session.enqueue_alias(ScholarlyId::parse("10.1111/RSSC.12435").unwrap(), 1);
        let popped = [
            first.id.to_string(),
            session.pop_next_frontier().unwrap().id.to_string(),
            session.pop_next_frontier().unwrap().id.to_string(),
        ];

        assert_eq!(enqueued_pmid, AliasEnqueue::Enqueued);
        assert_eq!(enqueued_pmc, AliasEnqueue::Enqueued);
        assert_eq!(enqueued_dup, AliasEnqueue::AlreadyVisited);
        assert_eq!(
            popped,
            ["10.1111/rssc.12435", "PMC10016948", "PMID:36998643"]
        );
    }

    #[test]
    fn priority_pdfs_preserve_fifo_order_bypass_oa_cap_and_never_reprobe_attempted_urls() {
        let caller = ScholarlyId::parse("10.1000/many-mirrors").unwrap();
        let arxiv_a = ScholarlyId::parse("2401.00001").unwrap();
        let arxiv_b = ScholarlyId::parse("2401.00002").unwrap();
        let mut session = WorkResolution::new(caller.clone(), None, &[]);

        let mut first_batch = ProviderDiscovery::default();
        first_batch.push_pdf("https://mirror.example.org/p1.pdf");
        first_batch.push_pdf("https://mirror.example.org/p2.pdf");
        first_batch.push_pdf("https://mirror.example.org/p3.pdf");
        first_batch.push_pdf("https://mirror.example.org/p4.pdf");
        first_batch.push_pdf("https://mirror.example.org/p5.pdf");
        first_batch.push_pdf("https://mirror.example.org/p6.pdf");
        first_batch.push_pdf("https://mirror.example.org/p7.pdf");
        session.ingest_discovery(first_batch, &caller, 0);

        let mut epmc_batch = ProviderDiscovery::default();
        epmc_batch.push_pdf("https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC1");
        session.ingest_discovery(epmc_batch, &caller, 1);

        let promoted = HttpUrl::parse("https://mirror.example.org/p6.pdf").unwrap();
        let extra_priority = HttpUrl::parse("https://arxiv.org/pdf/2401.00002.pdf").unwrap();
        session.push_priority_pdf(promoted.clone(), arxiv_a.clone());
        session.push_priority_pdf(extra_priority.clone(), arxiv_b);

        let first = session.pop_next_pdf().unwrap();
        let second = session.pop_next_pdf().unwrap();
        let popped_regular = [
            session.pop_next_pdf().unwrap().url.as_str().to_owned(),
            session.pop_next_pdf().unwrap().url.as_str().to_owned(),
            session.pop_next_pdf().unwrap().url.as_str().to_owned(),
            session.pop_next_pdf().unwrap().url.as_str().to_owned(),
            session.pop_next_pdf().unwrap().url.as_str().to_owned(),
            session.pop_next_pdf().unwrap().url.as_str().to_owned(),
        ];
        session.push_priority_pdf(promoted, arxiv_a);

        assert_eq!(first.url.as_str(), "https://mirror.example.org/p6.pdf");
        assert_eq!(second.url, extra_priority);
        assert_eq!(
            popped_regular[5],
            "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC1"
        );
        assert!(session.pop_next_pdf().is_none());
    }

    #[test]
    fn ingest_discovery_classifies_structured_pdfs_and_keeps_doi_landings_eligible() {
        let caller = ScholarlyId::parse("PMC10016948").unwrap();
        let mut session = WorkResolution::new(caller.clone(), None, &[]);
        let _ = session.pop_next_frontier();

        let mut discovery = ProviderDiscovery::default();
        discovery.push_pdf("https://arxiv.org/pdf/2409.13740.pdf");
        discovery.push_landing("https://arxiv.org/abs/2409.13740");
        discovery
            .push_landing("https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v2.full");
        session.ingest_discovery(discovery, &caller, 0);

        let first_alias = session.pop_next_frontier().unwrap();
        let second_alias = session.pop_next_frontier().unwrap();
        let landing = session.pop_next_landing().unwrap();

        assert_eq!(first_alias.id.to_string(), "arXiv:2409.13740");
        assert_eq!(second_alias.id.to_string(), "10.1101/2024.06.03.24308385");
        assert_eq!(
            landing.url.as_str(),
            "https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v2.full"
        );
        assert!(session.pop_next_landing().is_none());
    }

    #[test]
    fn venue_upgrade_and_attribution_suppress_self_case_differences() {
        let caller = ScholarlyId::parse("10.1111/RSSC.12435").unwrap();
        let mut session = WorkResolution::new(caller, None, &[]);
        session.merge_metadata(WorkMetadata {
            venue_or_source: Some("Europe PMC".to_owned()),
            ..WorkMetadata::default()
        });
        session.merge_metadata(WorkMetadata {
            venue_or_source: Some("JRSS-C".to_owned()),
            ..WorkMetadata::default()
        });
        let same_doi_lower = ScholarlyId::parse("10.1111/rssc.12435").unwrap();
        let doc = session.finish_pdf(ResolvedPdf {
            origin: same_doi_lower,
            bytes: b"%PDF-1.7\n".to_vec(),
            source_url: HttpUrl::parse("https://journal.example.org/paper.pdf").unwrap(),
            raw_bytes_fetched: 9,
        });

        assert_eq!(doc.metadata.venue_or_source.as_deref(), Some("JRSS-C"));
        assert_eq!(doc.metadata.notes, Vec::<String>::new());
    }

    #[test]
    fn finish_unavailable_distinguishes_metadata_only_from_transport_trace() {
        let arxiv = ScholarlyId::parse("2409.13740").unwrap();
        let mut with_meta = WorkResolution::new(arxiv.clone(), None, &[]);
        with_meta.mark_metadata_seen();
        with_meta.record_endpoint_failure("arxiv-html", "HTTP 404");
        let fallback_url = HttpUrl::parse("https://arxiv.org/abs/2409.13740").unwrap();
        let meta_doc = with_meta
            .finish_unavailable(Some(fallback_url.clone()))
            .unwrap();

        let mut without_meta = WorkResolution::new(arxiv, None, &[]);
        without_meta.record_endpoint_failure("atom", "HTTP 503");
        let trace_error = without_meta.finish_unavailable(None).unwrap_err();

        assert_eq!(meta_doc.metadata.source_url, Some(fallback_url));
        assert_eq!(
            meta_doc.payload,
            ExtractedPayload::MetadataOnly {
                reason: UnavailableReason::UpstreamUnavailable {
                    detail: "arxiv-html: HTTP 404".to_owned(),
                }
            }
        );
        assert_eq!(trace_error.failure_summary, "atom: HTTP 503");
    }

    #[test]
    fn ingest_discovery_enqueues_preprint_doi_aliases_from_pdf_urls() {
        let caller = ScholarlyId::parse("10.1000/published-journal").unwrap();
        let mut session = WorkResolution::new(caller.clone(), None, &[]);
        let _ = session.pop_next_frontier();

        let mut discovery = ProviderDiscovery::default();
        discovery
            .push_pdf("https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v2.full.pdf");
        session.ingest_discovery(discovery, &caller, 0);

        let discovered = session.pop_next_frontier().unwrap();
        let pooled_pdf = session.pop_next_pdf().unwrap();

        assert_eq!(discovered.id.to_string(), "10.1101/2024.06.03.24308385");
        assert_eq!(
            pooled_pdf.url.as_str(),
            "https://www.medrxiv.org/content/10.1101/2024.06.03.24308385v2.full.pdf"
        );
    }

    #[test]
    fn deferred_fallback_prefers_full_text_over_pdf_and_consumes_once() {
        let caller = ScholarlyId::parse("10.1000/deferred-check").unwrap();
        let mut session = WorkResolution::new(caller.clone(), None, &[]);
        session.record_deferred_fallback(DeferredFallback::Pdf(ResolvedPdf {
            origin: caller.clone(),
            bytes: b"%PDF-1.7\nfirst".to_vec(),
            source_url: HttpUrl::parse("https://mirror.example.org/first.pdf").unwrap(),
            raw_bytes_fetched: 14,
        }));
        session.record_deferred_fallback(DeferredFallback::Pdf(ResolvedPdf {
            origin: caller.clone(),
            bytes: b"%PDF-1.7\nsecond".to_vec(),
            source_url: HttpUrl::parse("https://mirror.example.org/second.pdf").unwrap(),
            raw_bytes_fetched: 15,
        }));
        session.record_deferred_fallback(DeferredFallback::FullText(ResolvedText {
            origin: caller,
            format: TextFormat::Html,
            markdown: "# Deferred Article".to_owned(),
            source_url: HttpUrl::parse("https://mirror.example.org/article.html").unwrap(),
            raw_bytes_fetched: 18,
        }));

        let taken = session.take_deferred_fallback().unwrap();
        let after = session.take_deferred_fallback();

        assert!(matches!(
            taken,
            DeferredFallback::FullText(ref text)
                if text.source_url.as_str() == "https://mirror.example.org/article.html"
        ));
        assert!(after.is_none());
    }

    #[test]
    fn depth_exceeded_structured_landing_remains_in_pool_while_visited_is_skipped() {
        let caller = ScholarlyId::parse("10.1000/depth-check").unwrap();
        let visited_pmc = ScholarlyId::parse("PMC1111111").unwrap();
        let mut session = WorkResolution::new(caller.clone(), None, &[visited_pmc]);

        let mut discovery = ProviderDiscovery::default();
        discovery.push_landing("https://pmc.ncbi.nlm.nih.gov/articles/PMC1111111/");
        discovery.push_landing("https://pmc.ncbi.nlm.nih.gov/articles/PMC2222222/");
        session.ingest_discovery(discovery, &caller, MAX_RECLASSIFICATION_DEPTH);

        let pooled = session.pop_next_landing().unwrap();

        assert_eq!(
            pooled.url.as_str(),
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC2222222/"
        );
        assert!(session.pop_next_landing().is_none());
    }
}
