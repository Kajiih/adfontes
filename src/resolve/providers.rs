//! Bounded JSON cursors and metadata mergers for scholarly provider APIs.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::extract::{clean_html_fragment, sanitize_inline_text};
use crate::model::{HttpUrl, ParsedTarget, PmcId, RfcNumber, WorkMetadata};

const MAX_METADATA_JSON_BYTES: usize = 4 * 1024 * 1024;
const MAX_INVERTED_ABSTRACT_WORDS: usize = 4_000;
const MAX_INVERTED_ABSTRACT_ELEMENTS: usize = MAX_INVERTED_ABSTRACT_WORDS * 4;
const MAX_INVERTED_ABSTRACT_BYTES: usize = 64 * 1024;
const YEAR_PREFIX_LEN: usize = 4;

const FIELD_TITLE: &str = "title";
const FIELD_ABSTRACT: &str = "abstract";
const FIELD_AUTHOR: &str = "author";
const FIELD_DISPLAY_NAME: &str = "display_name";
const FIELD_IS_OA: &str = "is_oa";
const FIELD_PMCID: &str = "pmcid";
const FIELD_BEST_OA_LOCATION: &str = "best_oa_location";
const FIELD_PDF_URL: &str = "pdf_url";
const FIELD_LANDING_PAGE_URL: &str = "landing_page_url";
const FIELD_UNPAYWALL_PDF: &str = "url_for_pdf";
const FIELD_UNPAYWALL_LANDING: &str = "url_for_landing_page";

/// Deduplicated open-access repository, PDF, and landing URL candidates collected across providers.
#[derive(Debug, Default)]
pub(super) struct OaCandidates {
    pmcid: Option<PmcId>,
    pdf_urls: Vec<HttpUrl>,
    landing_urls: Vec<HttpUrl>,
    seen: BTreeSet<String>,
}

impl OaCandidates {
    pub(super) const fn pmcid(&self) -> Option<&PmcId> {
        self.pmcid.as_ref()
    }

    pub(super) fn pdf_urls(&self) -> &[HttpUrl] {
        &self.pdf_urls
    }

    pub(super) fn landing_urls(&self) -> &[HttpUrl] {
        &self.landing_urls
    }

    fn push_pdf(&mut self, raw: &str) {
        if let Ok(parsed) = HttpUrl::parse(raw)
            && self.seen.insert(parsed.as_str().to_owned())
        {
            self.pdf_urls.push(parsed);
        }
    }

    fn push_landing(&mut self, raw: &str) {
        if let Ok(parsed) = HttpUrl::parse(raw)
            && self.seen.insert(parsed.as_str().to_owned())
        {
            self.landing_urls.push(parsed);
        }
    }
}

/// Identifiers extracted from the first Europe PMC record for follow-up waterfalls.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct EuropePmcIds {
    pub(super) pmcid: Option<PmcId>,
    pub(super) follow_up_target: Option<ParsedTarget>,
}

/// Size-bounded, parsed provider JSON payload encapsulating [`serde_json::Value`] inside `providers`.
#[derive(Debug)]
pub(super) struct JsonDoc(Value);

impl JsonDoc {
    pub(super) fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() > MAX_METADATA_JSON_BYTES {
            return None;
        }
        serde_json::from_slice(bytes).ok().map(Self)
    }

    pub(super) fn merge_csl_work(&self, metadata: &mut WorkMetadata) {
        let root = self.cursor();
        set_clean_if_none(&mut metadata.title, root.key(FIELD_TITLE).str());
        if metadata.authors.is_empty() {
            push_given_family_authors(root.key(FIELD_AUTHOR), &mut metadata.authors);
        }
        if metadata.year.is_none() {
            metadata.year = csl_first_year(root.key("issued"))
                .or_else(|| csl_first_year(root.key("published-print")))
                .or_else(|| csl_first_year(root.key("published-online")));
        }
        set_clean_if_none(
            &mut metadata.venue_or_source,
            root.key("container-title").string_or_first(),
        );
        if metadata.abstract_text.is_none() {
            metadata.abstract_text = root
                .key(FIELD_ABSTRACT)
                .str()
                .map(clean_html_fragment)
                .filter(|val| !val.is_empty());
        }
    }

    pub(super) fn merge_openalex_work(
        &self,
        metadata: &mut WorkMetadata,
        candidates: &mut OaCandidates,
    ) {
        let root = self.cursor();
        set_clean_if_none(&mut metadata.title, root.key(FIELD_TITLE).str());
        if metadata.authors.is_empty() {
            for authorship in root.key("authorships").array() {
                if let Some(name) = authorship
                    .key(FIELD_AUTHOR)
                    .key(FIELD_DISPLAY_NAME)
                    .clean_str()
                {
                    metadata.authors.push(name);
                }
            }
        }
        if metadata.year.is_none() {
            metadata.year = root.key("publication_year").year();
        }
        set_clean_if_none(
            &mut metadata.venue_or_source,
            root.key("primary_location")
                .key("source")
                .key(FIELD_DISPLAY_NAME)
                .str(),
        );
        if metadata.abstract_text.is_none() {
            metadata.abstract_text =
                reconstruct_inverted_abstract(root.key("abstract_inverted_index"));
        }
        if candidates.pmcid.is_none()
            && let Some(raw_pmc) = root.key("ids").key(FIELD_PMCID).str()
        {
            candidates.pmcid = extract_pmcid_from_value(raw_pmc);
        }

        let best = root.key(FIELD_BEST_OA_LOCATION);
        if best.key(FIELD_IS_OA).bool() != Some(false) {
            collect_location(best, FIELD_PDF_URL, FIELD_LANDING_PAGE_URL, candidates);
        }
        for loc in root.key("locations").array() {
            if loc.key(FIELD_IS_OA).bool() == Some(true) {
                collect_location(loc, FIELD_PDF_URL, FIELD_LANDING_PAGE_URL, candidates);
            }
        }
    }

    pub(super) fn merge_unpaywall_work(
        &self,
        metadata: &mut WorkMetadata,
        candidates: &mut OaCandidates,
    ) {
        let root = self.cursor();
        set_clean_if_none(&mut metadata.title, root.key(FIELD_TITLE).str());
        if metadata.authors.is_empty() {
            push_given_family_authors(root.key("z_authors"), &mut metadata.authors);
        }
        if metadata.year.is_none() {
            metadata.year = root.key("year").year();
        }
        set_clean_if_none(
            &mut metadata.venue_or_source,
            root.key("journal_name").str(),
        );
        if root.key(FIELD_IS_OA).bool() == Some(false) {
            return;
        }
        let best = root.key(FIELD_BEST_OA_LOCATION);
        if best.key(FIELD_IS_OA).bool() != Some(false) {
            collect_location(
                best,
                FIELD_UNPAYWALL_PDF,
                FIELD_UNPAYWALL_LANDING,
                candidates,
            );
        }
        for loc in root.key("oa_locations").array() {
            if loc.key(FIELD_IS_OA).bool() != Some(false) {
                collect_location(
                    loc,
                    FIELD_UNPAYWALL_PDF,
                    FIELD_UNPAYWALL_LANDING,
                    candidates,
                );
            }
        }
    }

    pub(super) fn extract_europe_pmc_record(
        &self,
        metadata: &mut WorkMetadata,
        oa_pdf_urls: &mut Vec<HttpUrl>,
    ) -> Option<EuropePmcIds> {
        let record = self.cursor().key("resultList").key("result").first();
        record.0.filter(|val| val.is_object())?;

        set_clean_if_none(&mut metadata.title, record.key(FIELD_TITLE).str());
        if metadata.authors.is_empty() {
            for author in record.key("authorList").key(FIELD_AUTHOR).array() {
                if let Some(name) = author.key("fullName").clean_str() {
                    metadata.authors.push(name);
                }
            }
            if metadata.authors.is_empty()
                && let Some(authors_summary) = record.key("authorString").str()
            {
                for part in authors_summary.split(',') {
                    let cleaned = sanitize_inline_text(part.trim_end_matches('.'));
                    if !cleaned.is_empty() {
                        metadata.authors.push(cleaned);
                    }
                }
            }
        }
        if metadata.year.is_none() {
            metadata.year = record.key("pubYear").year();
        }
        if let Some(journal) = record
            .key("journalInfo")
            .key("journal")
            .key(FIELD_TITLE)
            .clean_str()
            .or_else(|| record.key("journalTitle").clean_str())
        {
            metadata.venue_or_source = Some(journal);
        }
        if metadata.abstract_text.is_none() {
            metadata.abstract_text = record
                .key("abstractText")
                .str()
                .map(clean_html_fragment)
                .filter(|val| !val.is_empty());
        }
        let mut seen_pdf_urls = BTreeSet::new();
        for entry in record.key("fullTextUrlList").key("fullTextUrl").array() {
            let is_pdf = entry
                .key("documentStyle")
                .str()
                .is_some_and(|style| style.eq_ignore_ascii_case("pdf"));
            let is_open = entry.key("availabilityCode").str().is_some_and(|code| {
                code.eq_ignore_ascii_case("oa") || code.eq_ignore_ascii_case("f")
            });
            if is_pdf
                && is_open
                && let Some(raw_url) = entry.key("url").str()
                && let Ok(parsed) = HttpUrl::parse(raw_url)
                && seen_pdf_urls.insert(parsed.as_str().to_owned())
            {
                oa_pdf_urls.push(parsed);
            }
        }

        let pmcid = record
            .key(FIELD_PMCID)
            .str()
            .and_then(extract_pmcid_from_value);
        let follow_up_target = record
            .key("doi")
            .str()
            .and_then(|raw_doi| ParsedTarget::parse(raw_doi).ok())
            .filter(|target| !matches!(target, ParsedTarget::Url(_)));
        Some(EuropePmcIds {
            pmcid,
            follow_up_target,
        })
    }

    pub(super) fn merge_ietf_doc(&self, metadata: &mut WorkMetadata) {
        let root = self.cursor();
        set_clean_if_none(&mut metadata.title, root.key(FIELD_TITLE).str());
        if metadata.authors.is_empty() {
            for author in root.key("authors").array() {
                if let Some(name) = author.key("name").clean_str() {
                    metadata.authors.push(name);
                }
            }
        }
        if metadata.year.is_none() {
            metadata.year = root.key("time").year();
        }
        set_clean_if_none(&mut metadata.abstract_text, root.key(FIELD_ABSTRACT).str());
        if let Some(level) = root.key("std_level").clean_str() {
            metadata.venue_or_source = Some(format!("IETF ({level})"));
        }
        append_ietf_relation_note(
            root.key("obsoleted_by"),
            "Obsoleted by",
            &mut metadata.notes,
        );
        append_ietf_relation_note(root.key("updated_by"), "Updated by", &mut metadata.notes);
    }

    const fn cursor(&self) -> JsonCursor<'_> {
        JsonCursor(Some(&self.0))
    }
}

#[derive(Clone, Copy)]
struct JsonCursor<'a>(Option<&'a Value>);

impl<'a> JsonCursor<'a> {
    fn key(self, name: &str) -> Self {
        Self(self.0.and_then(|val| val.get(name)))
    }

    fn first(self) -> Self {
        Self(
            self.0
                .and_then(Value::as_array)
                .and_then(|items| items.first()),
        )
    }

    fn clean_str(self) -> Option<String> {
        self.str()
            .map(sanitize_inline_text)
            .filter(|val| !val.is_empty())
    }

    fn str(self) -> Option<&'a str> {
        self.0.and_then(Value::as_str)
    }

    fn bool(self) -> Option<bool> {
        self.0.and_then(Value::as_bool)
    }

    fn year(self) -> Option<u16> {
        match self.0? {
            Value::Number(number) => number.as_u64().and_then(|val| u16::try_from(val).ok()),
            Value::String(raw) => raw
                .trim()
                .get(..YEAR_PREFIX_LEN)
                .filter(|prefix| prefix.bytes().all(|byte| byte.is_ascii_digit()))
                .and_then(|prefix| prefix.parse::<u16>().ok()),
            _ => None,
        }
    }

    fn array(self) -> impl Iterator<Item = Self> + 'a {
        self.0
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|item| Self(Some(item)))
    }

    fn string_or_first(self) -> Option<&'a str> {
        match self.0? {
            Value::String(val) => Some(val.as_str()),
            Value::Array(items) => items.first().and_then(Value::as_str),
            _ => None,
        }
    }
}

fn push_given_family_authors(list: JsonCursor<'_>, out: &mut Vec<String>) {
    for author in list.array() {
        let given = author.key("given").clean_str().unwrap_or_default();
        let family = author.key("family").clean_str().unwrap_or_default();
        let name = if !given.is_empty() || !family.is_empty() {
            format!("{given} {family}").trim().to_owned()
        } else {
            author.key("literal").clean_str().unwrap_or_default()
        };
        if !name.is_empty() {
            out.push(name);
        }
    }
}

fn csl_first_year(field: JsonCursor<'_>) -> Option<u16> {
    field.key("date-parts").first().first().year()
}

fn collect_location(
    loc: JsonCursor<'_>,
    pdf_key: &str,
    landing_key: &str,
    candidates: &mut OaCandidates,
) {
    if let Some(pdf_url) = loc.key(pdf_key).str() {
        candidates.push_pdf(pdf_url);
    }
    if let Some(landing_url) = loc.key(landing_key).str() {
        candidates.push_landing(landing_url);
    }
}

fn set_clean_if_none(slot: &mut Option<String>, raw: Option<&str>) {
    if slot.is_none() {
        *slot = raw.map(sanitize_inline_text).filter(|val| !val.is_empty());
    }
}

fn extract_pmcid_from_value(raw: &str) -> Option<PmcId> {
    let trimmed = raw.trim().trim_end_matches('/');
    if let Ok(direct) = PmcId::parse(trimmed) {
        return Some(direct);
    }
    let tail = trimmed.rsplit('/').next()?;
    PmcId::parse(tail).ok()
}

fn append_ietf_relation_note(list: JsonCursor<'_>, label: &str, notes: &mut Vec<String>) {
    let formatted: Vec<String> = list
        .array()
        .filter_map(JsonCursor::str)
        .map(format_ietf_doc_slug)
        .filter(|val| !val.is_empty())
        .collect();
    if !formatted.is_empty() {
        notes.push(format!("{label} {}", formatted.join(", ")));
    }
}

fn format_ietf_doc_slug(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches('/');
    let leaf = trimmed.rsplit('/').next().unwrap_or(trimmed);
    if let Ok(rfc) = RfcNumber::parse(leaf) {
        return rfc.to_string();
    }
    sanitize_inline_text(leaf)
}

fn reconstruct_inverted_abstract(cursor: JsonCursor<'_>) -> Option<String> {
    let entries = cursor.0.and_then(Value::as_object)?;
    let mut positioned: BTreeMap<usize, &str> = BTreeMap::new();
    let mut scanned_elements = 0usize;

    'words: for (word, positions) in entries {
        if scanned_elements >= MAX_INVERTED_ABSTRACT_ELEMENTS {
            break;
        }
        scanned_elements += 1;
        for item in JsonCursor(Some(positions)).array() {
            if scanned_elements >= MAX_INVERTED_ABSTRACT_ELEMENTS {
                break 'words;
            }
            scanned_elements += 1;
            if let Some(pos) = item
                .0
                .and_then(Value::as_u64)
                .and_then(|raw| usize::try_from(raw).ok())
                .filter(|&pos| pos < MAX_INVERTED_ABSTRACT_WORDS)
            {
                positioned.entry(pos).or_insert(word.as_str());
            }
        }
    }

    if positioned.is_empty() {
        return None;
    }
    let mut out = String::new();
    for word in positioned.into_values() {
        let clean_word = sanitize_inline_text(word);
        if clean_word.is_empty() {
            continue;
        }
        let sep = usize::from(!out.is_empty());
        if out
            .len()
            .saturating_add(sep)
            .saturating_add(clean_word.len())
            > MAX_INVERTED_ABSTRACT_BYTES
        {
            break;
        }
        if sep == 1 {
            out.push(' ');
        }
        out.push_str(&clean_word);
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openalex_inverted_abstract_deduplicates_positions_and_preserves_prefix_order() {
        let mut meta = WorkMetadata::default();
        let mut candidates = OaCandidates::default();
        JsonDoc::parse(
            br#"{"abstract_inverted_index":{"zebra":[0,0,0,5000],"alpha":[2],"middle":[1]}}"#,
        )
        .unwrap()
        .merge_openalex_work(&mut meta, &mut candidates);
        assert_eq!(meta.abstract_text.as_deref(), Some("zebra middle alpha"));
    }

    #[test]
    fn openalex_inverted_abstract_caps_elements_and_preserves_title() {
        let out_of_range = std::iter::repeat_n("5000", 20_000)
            .collect::<Vec<_>>()
            .join(",");
        let flood_payload = format!(
            r#"{{"title":"Bounded Study","abstract_inverted_index":{{"flood":[{out_of_range}],"ignored":[1],"bad":-1}}}}"#
        );
        let mut flood_meta = WorkMetadata::default();
        let mut flood_candidates = OaCandidates::default();
        JsonDoc::parse(flood_payload.as_bytes())
            .unwrap()
            .merge_openalex_work(&mut flood_meta, &mut flood_candidates);

        let mid_prefix = std::iter::repeat_n("5000", 15_997)
            .collect::<Vec<_>>()
            .join(",");
        let mid_payload = format!(
            r#"{{"abstract_inverted_index":{{"kept":[0],"pad":[{mid_prefix}],"tail":[1]}}}}"#
        );
        let mut mid_meta = WorkMetadata::default();
        let mut mid_candidates = OaCandidates::default();
        JsonDoc::parse(mid_payload.as_bytes())
            .unwrap()
            .merge_openalex_work(&mut mid_meta, &mut mid_candidates);

        assert_eq!(flood_meta.title.as_deref(), Some("Bounded Study"));
        assert_eq!(flood_meta.abstract_text, None);
        assert_eq!(mid_meta.abstract_text.as_deref(), Some("kept"));
    }

    #[test]
    fn openalex_inverted_abstract_caps_empty_keys_and_ignores_non_map_shapes() {
        let many_keys = (0..17_000usize)
            .map(|idx| format!(r#""k{idx}":[]"#))
            .collect::<Vec<_>>()
            .join(",");
        let key_payload = format!(r#"{{"abstract_inverted_index":{{{many_keys},"tail":[0]}}}}"#);
        let mut key_meta = WorkMetadata::default();
        let mut key_candidates = OaCandidates::default();
        JsonDoc::parse(key_payload.as_bytes())
            .unwrap()
            .merge_openalex_work(&mut key_meta, &mut key_candidates);

        let mut bad_meta = WorkMetadata::default();
        let mut bad_candidates = OaCandidates::default();
        JsonDoc::parse(
            br#"{"title":"Non-map Abstract","abstract_inverted_index":[1,2,3],"publication_year":-1}"#,
        )
        .unwrap()
        .merge_openalex_work(&mut bad_meta, &mut bad_candidates);

        assert_eq!(key_meta.abstract_text, None);
        assert_eq!(bad_meta.title.as_deref(), Some("Non-map Abstract"));
        assert_eq!(bad_meta.abstract_text, None);
        assert_eq!(bad_meta.year, None);
    }

    #[test]
    fn csl_and_ietf_degrade_leniently_on_invalid_years_and_nulls() {
        let mut csl_meta = WorkMetadata::default();
        JsonDoc::parse(
            br#"{"issued":{"date-parts":[null,[-5]]},"published-print":{"date-parts":[["2025.0"]]},"abstract":"<jats:p>Significance at <i>p</i> &lt; 0.05 and <b>n</b> &gt; 10.</jats:p>"}"#,
        )
        .unwrap()
        .merge_csl_work(&mut csl_meta);

        let mut ietf_meta = WorkMetadata::default();
        JsonDoc::parse(
            br#"{"title":"HTTP Semantics","obsoleted_by":[null,"/doc/rfc9110/"],"updated_by":[null]}"#,
        )
        .unwrap()
        .merge_ietf_doc(&mut ietf_meta);

        assert_eq!(csl_meta.year, Some(2025));
        assert_eq!(
            csl_meta.abstract_text.as_deref(),
            Some("Significance at p < 0.05 and n > 10.")
        );
        assert_eq!(ietf_meta.notes, vec!["Obsoleted by RFC 9110"]);
    }

    #[test]
    fn europe_pmc_degrades_leniently_on_null_authors_bad_year_and_trailing_hits() {
        let mut epmc_meta = WorkMetadata::default();
        let mut oa_pdfs = Vec::new();
        let doc = JsonDoc::parse(
            br#"{"resultList":{"result":[{"authorList":{"author":null},"authorString":"Ada Lovelace.","pubYear":-1,"abstractText":"<p>First result.</p><p>Second &lt; 0.01.</p>"},{"pubYear":{"unexpected":1}}]}}"#,
        )
        .unwrap();
        doc.extract_europe_pmc_record(&mut epmc_meta, &mut oa_pdfs)
            .unwrap();

        assert_eq!(epmc_meta.authors, vec!["Ada Lovelace"]);
        assert_eq!(epmc_meta.year, None);
        assert_eq!(
            epmc_meta.abstract_text.as_deref(),
            Some("First result. Second < 0.01.")
        );
    }

    #[test]
    fn isolates_malformed_sibling_fields_and_enforces_4_mib_cap() {
        let mut csl_meta = WorkMetadata::default();
        JsonDoc::parse(
            br#"{"title":"Fault Isolated CSL","author":"not-an-array","issued":{"date-parts":"bad"},"abstract":42,"published-print":{"date-parts":[[2024]]}}"#,
        )
        .unwrap()
        .merge_csl_work(&mut csl_meta);

        let mut oa_meta = WorkMetadata::default();
        let mut oa_candidates = OaCandidates::default();
        JsonDoc::parse(
            br#"{"title":"Fault Isolated OA","authorships":123,"locations":"oops","abstract_inverted_index":{"Clean":[0],"abstract":[1]}}"#,
        )
        .unwrap()
        .merge_openalex_work(&mut oa_meta, &mut oa_candidates);

        let oversized = vec![b' '; MAX_METADATA_JSON_BYTES + 1];
        assert!(JsonDoc::parse(&oversized).is_none());
        assert_eq!(csl_meta.title.as_deref(), Some("Fault Isolated CSL"));
        assert_eq!(csl_meta.year, Some(2024));
        assert_eq!(oa_meta.abstract_text.as_deref(), Some("Clean abstract"));
    }

    #[test]
    fn openalex_inverted_abstract_bounds_total_output_bytes() {
        const BIG_WORD_BYTES: usize = 4 * 1024;
        let big_word = "a".repeat(BIG_WORD_BYTES);
        let positions = (0..MAX_INVERTED_ABSTRACT_WORDS)
            .map(|pos| pos.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let payload = format!(r#"{{"abstract_inverted_index":{{"{big_word}":[{positions}]}}}}"#);
        let mut meta = WorkMetadata::default();
        let mut candidates = OaCandidates::default();
        JsonDoc::parse(payload.as_bytes())
            .unwrap()
            .merge_openalex_work(&mut meta, &mut candidates);

        let abstract_text = meta.abstract_text.expect("abstract prefix must fit budget");
        assert_ne!(abstract_text, "");
        assert!(abstract_text.len() <= MAX_INVERTED_ABSTRACT_BYTES);
    }

    #[test]
    fn europe_pmc_falls_back_on_blank_journal_title_and_rejects_signed_year_strings() {
        let mut epmc_meta = WorkMetadata::default();
        let mut oa_pdfs = Vec::new();
        let doc = JsonDoc::parse(
            br#"{"resultList":{"result":[{"journalInfo":{"journal":{"title":"   "}},"journalTitle":"Fallback Journal","pubYear":"+2025"}]}}"#,
        )
        .unwrap();
        doc.extract_europe_pmc_record(&mut epmc_meta, &mut oa_pdfs)
            .unwrap();

        assert_eq!(
            epmc_meta.venue_or_source.as_deref(),
            Some("Fallback Journal")
        );
        assert_eq!(epmc_meta.year, None);
    }
}
