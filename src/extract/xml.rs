//! Safe zero-I/O XML extractors for Europe PMC NISO JATS, IETF RFCXML v3, and `arXiv` Atom.
//!
//! Pre-scans XML outside comments, processing instructions, and CDATA sections to block
//! `<!ENTITY`, `<!ELEMENT`, `<!ATTLIST`, `<!NOTATION`, and internal DTD subsets
//! (`<!DOCTYPE ... [`) while allowing standard external `<!DOCTYPE article PUBLIC ...>` prologs
//! parsed iteratively via `roxmltree` (`entity_resolver: None`, `nodes_limit: 250_000`).

use std::fmt::Write as _;

use roxmltree::{Document, Node, ParsingOptions};

use super::title::{
    ExtractedText, normalize_markdown_spacing, push_fenced_code_block, push_inline_code,
    push_markdown_blockquote, push_markdown_heading, render_gfm_table, sanitize_inline_text,
};
use crate::model::AdfontesError;

const XML_FORMAT_LABEL: &str = "XML";
const XML_MAX_NODES: u32 = 250_000;
const COMMENT_OPEN: &str = "<!--";
const COMMENT_CLOSE: &str = "-->";
const CDATA_OPEN: &str = "<![CDATA[";
const CDATA_CLOSE: &str = "]]>";
const PI_OPEN: &str = "<?";
const PI_CLOSE: &str = "?>";
const DOCTYPE_DECL_PREFIX: &str = "<!doctype";
const FORBIDDEN_DTD_DECLS: &[&str] = &["<!entity", "<!element", "<!attlist", "<!notation"];

const TAG_ARTICLE: &str = "article";
const TAG_RFC: &str = "rfc";
const TAG_ARTICLE_TITLE: &str = "article-title";
const TAG_FRONT: &str = "front";
const TAG_BACK: &str = "back";
const TAG_ABSTRACT: &str = "abstract";
const TAG_TITLE: &str = "title";
const TAG_NAME: &str = "name";
const TAG_SUMMARY: &str = "summary";
const TAG_AUTHOR: &str = "author";
const TAG_APP_GROUP: &str = "app-group";
const TAG_APP: &str = "app";
const TAG_LABEL: &str = "label";
const TAG_CAPTION: &str = "caption";
const TAG_SECTION: &str = "section";
const TAG_OL: &str = "ol";
const TAG_TABLE: &str = "table";
const TAG_TR: &str = "tr";
const TAG_TEX_MATH: &str = "tex-math";
const TAG_ANNOTATION: &str = "annotation";
const TAG_CODE: &str = "code";
const TAG_DISP_FORMULA: &str = "disp-formula";
const ABSTRACT_HEADING_MARKDOWN: &str = "## Abstract\n\n";
const ABSTRACT_SUBHEADING_LEVEL: usize = 3;

/// Result of parsing a NISO JATS article XML payload from Europe PMC or PubMed Central.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JatsExtraction {
    /// Complete open-access JATS article containing a non-empty `<body>`.
    FullText {
        /// Article title from `<front>/<article-meta>/<title-group>/<article-title>`.
        title: Option<String>,
        /// Ordered author names from `<front>/<article-meta>`.
        authors: Vec<String>,
        /// Publication year from `<front>/<article-meta>/<pub-date>/<year>`.
        year: Option<u16>,
        /// Extracted abstract text.
        abstract_text: Option<String>,
        /// Normalized Markdown rendering of title, abstract, and `<body>`.
        markdown: String,
    },
    /// Bibliographic `<front>` metadata arrived without a `<body>` (restricted full text).
    BodyRestricted {
        /// Article title from `<article-title>`.
        title: Option<String>,
        /// Ordered author names.
        authors: Vec<String>,
        /// Publication year.
        year: Option<u16>,
        /// Extracted abstract text.
        abstract_text: Option<String>,
    },
}

/// Bibliographic metadata parsed from an `export.arxiv.org/api/query` Atom feed entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArxivAtomEntry {
    /// Paper title.
    pub title: Option<String>,
    /// Ordered author names.
    pub authors: Vec<String>,
    /// Publication year from `<published>`.
    pub year: Option<u16>,
    /// Abstract from `<summary>`.
    pub abstract_text: Option<String>,
    /// Optional journal DOI (`<arxiv:doi>`).
    pub doi: Option<String>,
}

/// Single-pass extracted XML payload dispatched by root element local name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XmlDocumentExtraction {
    /// NISO JATS `<article>` payload (full text or restricted front-matter metadata).
    Jats(JatsExtraction),
    /// IETF RFCXML v3 `<rfc>` document.
    Rfc(ExtractedText),
    /// Valid XML document whose root element is neither `<article>` nor `<rfc>`.
    UnsupportedRoot,
}

/// Rejects forbidden XML DTD markup declarations and internal DTD subsets.
///
/// Scans `xml` outside `<!-- ... -->` comments, `<? ... ?>` processing instructions, and
/// `<![CDATA[ ... ]]>` sections to block `<!ENTITY`, `<!ELEMENT`, `<!ATTLIST`, `<!NOTATION`, and
/// internal DTD subsets (`<!DOCTYPE ... [`).
///
/// External NISO JATS prologs such as `<!DOCTYPE article PUBLIC "-//NLM//DTD JATS ..." "...">`
/// without an internal `[` subset are allowed because `roxmltree` performs zero external I/O.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if a forbidden DTD/entity declaration or unterminated
/// `<!DOCTYPE` appears, or [`AdfontesError::Extraction`] on unterminated comments, PIs, or CDATA.
pub fn reject_unsafe_xml_dtd(xml: &str) -> Result<(), AdfontesError> {
    let trimmed_bom = xml.strip_prefix('\u{FEFF}').unwrap_or(xml);
    let bytes = trimmed_bom.as_bytes();
    let mut index: usize = 0;

    while let Some(rel_lt) = bytes
        .get(index..)
        .and_then(|slice| slice.iter().position(|&byte| byte == b'<'))
    {
        index += rel_lt;
        let rest = trimmed_bom.get(index..).unwrap_or_default();
        if rest.starts_with(COMMENT_OPEN) {
            index =
                advance_past_delimiter(rest, index, COMMENT_OPEN, COMMENT_CLOSE, "XML comment")?;
            continue;
        }
        if rest.starts_with(CDATA_OPEN) {
            index = advance_past_delimiter(rest, index, CDATA_OPEN, CDATA_CLOSE, "CDATA section")?;
            continue;
        }
        if rest.starts_with(PI_OPEN) {
            index = advance_past_delimiter(
                rest,
                index,
                PI_OPEN,
                PI_CLOSE,
                "XML processing instruction",
            )?;
            continue;
        }
        if FORBIDDEN_DTD_DECLS
            .iter()
            .any(|prefix| starts_with_ascii_ignore_case(rest, prefix))
        {
            return Err(AdfontesError::SafetyBlock {
                reason: "XML DTD markup/entity declarations are forbidden".to_owned(),
            });
        }
        if starts_with_ascii_ignore_case(rest, DOCTYPE_DECL_PREFIX) {
            index = validate_external_doctype(bytes, index)?;
            continue;
        }

        index += 1;
    }

    Ok(())
}

/// Parses XML once under DTD/node safety limits and dispatches `<article>` (`JATS`) or `<rfc>`.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] on forbidden DTD/entity payloads or node-limit
/// violations, or [`AdfontesError::Extraction`] on malformed XML.
pub fn extract_xml_document(xml: &str) -> Result<XmlDocumentExtraction, AdfontesError> {
    let document = parse_safe_xml(xml)?;
    let extracted = match document.root_element().tag_name().name() {
        TAG_ARTICLE => XmlDocumentExtraction::Jats(extract_jats_from_document(&document)),
        TAG_RFC => XmlDocumentExtraction::Rfc(extract_rfc_from_document(&document)),
        _ => XmlDocumentExtraction::UnsupportedRoot,
    };
    Ok(extracted)
}

/// Parses a NISO JATS XML document into normalized Markdown or [`JatsExtraction::BodyRestricted`].
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] on forbidden DTD/entity payloads or
/// [`AdfontesError::Extraction`] on malformed XML.
pub fn extract_jats_xml(xml: &str) -> Result<JatsExtraction, AdfontesError> {
    let document = parse_safe_xml(xml)?;
    Ok(extract_jats_from_document(&document))
}

/// Parses an IETF RFCXML v3 (`rfc<N>.xml`) specification into normalized Markdown.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] on forbidden DTD/entity payloads or
/// [`AdfontesError::Extraction`] on malformed XML.
pub fn extract_rfc_xml(xml: &str) -> Result<ExtractedText, AdfontesError> {
    let document = parse_safe_xml(xml)?;
    Ok(extract_rfc_from_document(&document))
}

/// Parses the first entry from an `export.arxiv.org/api/query` Atom XML feed.
///
/// Rejects `arXiv`'s silent `HTTP 200` API error entries cleanly without false positives on
/// real papers titled `"Error"`.
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] on forbidden DTD/entity payloads or
/// [`AdfontesError::Extraction`] if the feed is malformed or contains an `arXiv` API error entry.
pub fn parse_arxiv_atom_feed(xml: &str) -> Result<ArxivAtomEntry, AdfontesError> {
    let document = parse_safe_xml(xml)?;
    let root = document.root_element();
    let entry = root
        .children()
        .find(|node| node.has_tag_name("entry"))
        .ok_or_else(|| AdfontesError::Extraction {
            format: XML_FORMAT_LABEL,
            detail: "arXiv Atom feed contained no <entry> element".to_owned(),
        })?;

    let title = entry
        .children()
        .find(|node| node.has_tag_name(TAG_TITLE))
        .map(collect_inline_xml_text)
        .filter(|text| !text.is_empty());

    let entry_id = entry
        .children()
        .find(|node| node.has_tag_name("id"))
        .and_then(|node| node.text())
        .map_or_default(str::trim);

    let year = entry
        .children()
        .find(|node| node.has_tag_name("published"))
        .and_then(|node| node.text())
        .and_then(|text| text.trim().get(..4))
        .and_then(|digits| digits.parse::<u16>().ok());

    let is_api_error = title
        .as_deref()
        .is_some_and(|heading| heading.eq_ignore_ascii_case("error"))
        && (entry_id.contains("/api/errors") || year.is_none());

    if is_api_error {
        let summary = entry
            .children()
            .find(|node| node.has_tag_name(TAG_SUMMARY))
            .map_or_else(
                || "unknown arXiv API error".to_owned(),
                collect_inline_xml_text,
            );
        return Err(AdfontesError::Extraction {
            format: XML_FORMAT_LABEL,
            detail: format!("arXiv Atom API returned error entry: {summary}"),
        });
    }

    let authors = entry
        .children()
        .filter(|node| node.has_tag_name(TAG_AUTHOR))
        .filter_map(|author| {
            author
                .children()
                .find(|child| child.has_tag_name(TAG_NAME))
                .map(collect_inline_xml_text)
                .filter(|name| !name.is_empty())
        })
        .collect();

    let abstract_text = entry
        .children()
        .find(|node| node.has_tag_name(TAG_SUMMARY))
        .map(collect_inline_xml_text)
        .filter(|text| !text.is_empty());

    let doi = entry
        .children()
        .find(|node| node.has_tag_name("doi"))
        .map(collect_inline_xml_text)
        .filter(|text| !text.is_empty());

    Ok(ArxivAtomEntry {
        title,
        authors,
        year,
        abstract_text,
        doi,
    })
}

fn advance_past_delimiter(
    rest: &str,
    index: usize,
    open: &str,
    close: &str,
    label: &str,
) -> Result<usize, AdfontesError> {
    if let Some(close_rel) = rest.get(open.len()..).and_then(|slice| slice.find(close)) {
        return Ok(index + open.len() + close_rel + close.len());
    }
    Err(AdfontesError::Extraction {
        format: XML_FORMAT_LABEL,
        detail: format!("unterminated {label}"),
    })
}

fn validate_external_doctype(bytes: &[u8], start: usize) -> Result<usize, AdfontesError> {
    let mut cursor = start + DOCTYPE_DECL_PREFIX.len();
    let mut quote: Option<u8> = None;

    while let Some(&byte) = bytes.get(cursor) {
        match quote {
            Some(active_quote) => {
                if byte == active_quote {
                    quote = None;
                }
            }
            None => match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'[' => {
                    return Err(AdfontesError::SafetyBlock {
                        reason: "XML internal DTD subsets (`<!DOCTYPE ... [`) are forbidden"
                            .to_owned(),
                    });
                }
                b'>' => return Ok(cursor + 1),
                _ => {}
            },
        }
        cursor += 1;
    }

    Err(AdfontesError::SafetyBlock {
        reason: "unterminated XML `<!DOCTYPE` declaration is forbidden".to_owned(),
    })
}

const MAX_RFC_BLOCKQUOTE_DEPTH: usize = 8;

fn parse_safe_xml(xml: &str) -> Result<Document<'_>, AdfontesError> {
    reject_unsafe_xml_dtd(xml)?;
    let options = ParsingOptions {
        allow_dtd: true,
        nodes_limit: XML_MAX_NODES,
        entity_resolver: None,
    };
    Document::parse_with_options(xml, options).map_err(|parse_error| match parse_error {
        roxmltree::Error::NodesLimitReached => AdfontesError::SafetyBlock {
            reason: format!("XML document exceeded safety limit of {XML_MAX_NODES} nodes"),
        },
        roxmltree::Error::EntityReferenceLoop(_) => AdfontesError::SafetyBlock {
            reason: "XML entity reference expansion loop detected".to_owned(),
        },
        other => AdfontesError::Extraction {
            format: XML_FORMAT_LABEL,
            detail: format!("XML parse error: {other}"),
        },
    })
}

fn extract_jats_from_document(document: &Document<'_>) -> JatsExtraction {
    let root = document.root_element();
    let article = find_jats_article_root(root);
    let article_meta = find_jats_article_meta(article);

    let title = article_meta
        .and_then(extract_jats_title)
        .filter(|text| !text.is_empty());
    let authors = article_meta.map_or_default(collect_jats_authors);
    let year = article_meta.and_then(extract_jats_year);
    let abstract_text = article_meta
        .and_then(select_jats_abstract_node)
        .map(|node| render_jats_container(node, ABSTRACT_SUBHEADING_LEVEL))
        .filter(|text| !text.is_empty());

    let Some(body_node) = article.children().find(|node| node.has_tag_name("body")) else {
        return JatsExtraction::BodyRestricted {
            title,
            authors,
            year,
            abstract_text,
        };
    };

    let body_markdown = render_jats_container(body_node, 2);
    if body_markdown.is_empty() {
        return JatsExtraction::BodyRestricted {
            title,
            authors,
            year,
            abstract_text,
        };
    }

    let mut sections = Vec::new();
    if let Some(ref heading) = title {
        sections.push(format!("# {heading}"));
    }
    if let Some(ref abstract_body) = abstract_text {
        let mut block = String::from(ABSTRACT_HEADING_MARKDOWN);
        block.push_str(abstract_body);
        sections.push(block);
    }
    sections.push(body_markdown);
    if let Some(back_node) = article.children().find(|node| node.has_tag_name(TAG_BACK)) {
        let appendices = render_jats_appendices(back_node);
        if !appendices.is_empty() {
            sections.push(appendices);
        }
    }

    JatsExtraction::FullText {
        title,
        authors,
        year,
        abstract_text,
        markdown: normalize_markdown_spacing(&sections.join("\n\n")),
    }
}

fn extract_rfc_from_document(document: &Document<'_>) -> ExtractedText {
    let root = document.root_element();

    let front = root.children().find(|node| node.has_tag_name(TAG_FRONT));
    let title = front
        .and_then(|node| node.children().find(|child| child.has_tag_name(TAG_TITLE)))
        .map(collect_inline_xml_text)
        .filter(|text| !text.is_empty());

    let mut out = String::new();
    if let Some(ref heading) = title {
        push_markdown_heading(&mut out, 1, heading);
    }

    if let Some(abstract_node) = front.and_then(|node| {
        node.children()
            .find(|child| child.has_tag_name(TAG_ABSTRACT))
    }) {
        let text = render_rfc_container(abstract_node, ABSTRACT_SUBHEADING_LEVEL);
        if !text.is_empty() {
            out.push_str(ABSTRACT_HEADING_MARKDOWN);
            out.push_str(&text);
            out.push_str("\n\n");
        }
    }

    if let Some(middle) = root.children().find(|node| node.has_tag_name("middle")) {
        out.push_str(&render_rfc_container(middle, 2));
        out.push_str("\n\n");
    }
    if let Some(back) = root.children().find(|node| node.has_tag_name(TAG_BACK)) {
        for section in back
            .children()
            .filter(|node| node.has_tag_name(TAG_SECTION))
        {
            out.push_str(&render_rfc_single_section(section, 2));
            out.push_str("\n\n");
        }
    }

    ExtractedText {
        title,
        markdown: normalize_markdown_spacing(&out),
    }
}

fn find_jats_article_root<'a, 'input>(root: Node<'a, 'input>) -> Node<'a, 'input> {
    if root.has_tag_name(TAG_ARTICLE) {
        return root;
    }
    root.children()
        .find(|node| node.has_tag_name(TAG_ARTICLE))
        .unwrap_or(root)
}

fn find_jats_article_meta<'a, 'input>(article: Node<'a, 'input>) -> Option<Node<'a, 'input>> {
    let front = article
        .children()
        .find(|node| node.has_tag_name(TAG_FRONT))?;
    front
        .children()
        .find(|node| node.has_tag_name("article-meta"))
        .or(Some(front))
}

fn extract_jats_title(article_meta: Node<'_, '_>) -> Option<String> {
    if let Some(title_group) = article_meta
        .children()
        .find(|node| node.has_tag_name("title-group"))
        && let Some(article_title) = title_group
            .children()
            .find(|node| node.has_tag_name(TAG_ARTICLE_TITLE))
    {
        return Some(collect_inline_xml_text(article_title));
    }
    article_meta
        .children()
        .find(|node| node.has_tag_name(TAG_ARTICLE_TITLE))
        .map(collect_inline_xml_text)
}

fn collect_jats_authors(article_meta: Node<'_, '_>) -> Vec<String> {
    let mut authors = Vec::new();
    for contrib_group in article_meta
        .children()
        .filter(|node| node.has_tag_name("contrib-group"))
    {
        for contrib in contrib_group.children().filter(|node| {
            node.has_tag_name("contrib") && node.attribute("contrib-type") == Some(TAG_AUTHOR)
        }) {
            if let Some(name_node) = contrib
                .children()
                .find(|child| child.has_tag_name(TAG_NAME) || child.has_tag_name("string-name"))
            {
                let given = name_node
                    .children()
                    .find(|child| child.has_tag_name("given-names"))
                    .map_or_default(collect_inline_xml_text);
                let surname = name_node
                    .children()
                    .find(|child| child.has_tag_name("surname"))
                    .map_or_default(collect_inline_xml_text);
                let combined = if !given.is_empty() || !surname.is_empty() {
                    format!("{given} {surname}").trim().to_owned()
                } else {
                    collect_inline_xml_text(name_node)
                };
                if !combined.is_empty() {
                    authors.push(combined);
                }
            } else if let Some(collab) = contrib
                .children()
                .find(|child| child.has_tag_name("collab"))
            {
                let text = collect_inline_xml_text(collab);
                if !text.is_empty() {
                    authors.push(text);
                }
            }
        }
    }
    authors
}

fn extract_jats_year(article_meta: Node<'_, '_>) -> Option<u16> {
    article_meta
        .children()
        .filter(|node| node.has_tag_name("pub-date"))
        .find_map(|pub_date| {
            pub_date
                .children()
                .find(|child| child.has_tag_name("year"))
                .and_then(|year_node| year_node.text())
                .and_then(|digits| digits.trim().parse::<u16>().ok())
        })
}

fn select_jats_abstract_node<'a, 'input>(
    article_meta: Node<'a, 'input>,
) -> Option<Node<'a, 'input>> {
    let mut fallback = None;
    for candidate in article_meta
        .children()
        .filter(|node| node.has_tag_name(TAG_ABSTRACT))
    {
        let kind = candidate
            .attribute("abstract-type")
            .unwrap_or_default()
            .to_ascii_lowercase();
        if kind != "graphical" && kind != "toc" {
            return Some(candidate);
        }
        if fallback.is_none() {
            fallback = Some(candidate);
        }
    }
    fallback
}

fn render_jats_appendices(back: Node<'_, '_>) -> String {
    let mut out = String::new();
    let mut stack = Vec::new();
    for app in back
        .children()
        .filter(|node| node.has_tag_name(TAG_APP_GROUP) || node.has_tag_name(TAG_APP))
    {
        if app.has_tag_name(TAG_APP_GROUP) {
            for child in app.children().filter(|node| node.has_tag_name(TAG_APP)) {
                stack.push(JatsStep {
                    node: child,
                    heading_level: 2,
                });
            }
        } else {
            stack.push(JatsStep {
                node: app,
                heading_level: 2,
            });
        }
    }
    stack.reverse();
    while let Some(step) = stack.pop() {
        emit_jats_section_heading(step.node, step.heading_level, &mut out);
        let body = render_jats_container(step.node, step.heading_level + 1);
        if !body.is_empty() {
            out.push_str(&body);
            out.push_str("\n\n");
        }
    }
    normalize_markdown_spacing(&out)
}

fn render_jats_container(container: Node<'_, '_>, base_heading_level: usize) -> String {
    let mut out = String::new();
    let mut stack = Vec::new();
    push_jats_children_reversed(container, base_heading_level, &mut stack);

    while let Some(step) = stack.pop() {
        let node = step.node;
        match node.tag_name().name() {
            "sec" | TAG_APP => {
                emit_jats_section_heading(node, step.heading_level, &mut out);
                push_jats_children_reversed(node, step.heading_level.saturating_add(1), &mut stack);
            }
            "p" => {
                let text = collect_inline_xml_text(node);
                if !text.is_empty() {
                    out.push_str(&text);
                    out.push_str("\n\n");
                }
            }
            TAG_DISP_FORMULA => {
                let math = extract_jats_formula(node);
                if !math.is_empty() {
                    out.push_str("$$\n");
                    out.push_str(&math);
                    out.push_str("\n$$\n\n");
                }
            }
            "preformat" | TAG_CODE => {
                let code_text = collect_raw_text(node);
                push_fenced_code_block(&mut out, "", &code_text);
            }
            "list" => render_jats_list(node, &mut out),
            "table-wrap" => render_jats_table_wrap(node, &mut out),
            "fig" => render_jats_figure(node, &mut out),
            "boxed-text" | "disp-quote" => {
                push_jats_children_reversed(node, step.heading_level, &mut stack);
            }
            _ => {}
        }
    }

    normalize_markdown_spacing(&out)
}

struct JatsStep<'a, 'input> {
    node: Node<'a, 'input>,
    heading_level: usize,
}

fn push_jats_children_reversed<'a, 'input>(
    parent: Node<'a, 'input>,
    heading_level: usize,
    stack: &mut Vec<JatsStep<'a, 'input>>,
) {
    let mut children: Vec<Node<'a, 'input>> = parent.children().filter(Node::is_element).collect();
    children.reverse();
    for child in children {
        stack.push(JatsStep {
            node: child,
            heading_level,
        });
    }
}

fn emit_jats_section_heading(section: Node<'_, '_>, heading_level: usize, out: &mut String) {
    let label = section
        .children()
        .find(|child| child.has_tag_name(TAG_LABEL))
        .map_or_default(collect_inline_xml_text);
    let title = section
        .children()
        .find(|child| child.has_tag_name(TAG_TITLE))
        .map_or_default(collect_inline_xml_text);
    let heading = match (label.is_empty(), title.is_empty()) {
        (false, false) => format!("{label} {title}"),
        (true, false) => title,
        (false, true) => label,
        (true, true) => String::new(),
    };
    if !heading.is_empty() {
        push_markdown_heading(out, heading_level, &heading);
    }
}

fn render_jats_list(list: Node<'_, '_>, out: &mut String) {
    let ordered = list
        .attribute("list-type")
        .is_some_and(|kind| kind.eq_ignore_ascii_case("order"));
    for (idx, item) in list
        .children()
        .filter(|child| child.has_tag_name("list-item"))
        .enumerate()
    {
        let text = collect_inline_xml_text(item);
        if !text.is_empty() {
            if ordered {
                let _ = writeln!(out, "{}. {text}", idx + 1);
            } else {
                let _ = writeln!(out, "- {text}");
            }
        }
    }
    out.push('\n');
}

fn render_jats_table_wrap(table_wrap: Node<'_, '_>, out: &mut String) {
    let label = table_wrap
        .children()
        .find(|child| child.has_tag_name(TAG_LABEL))
        .map_or_default(collect_inline_xml_text);
    let caption = table_wrap
        .children()
        .find(|child| child.has_tag_name(TAG_CAPTION))
        .map_or_default(collect_inline_xml_text);
    let header_line = match (label.is_empty(), caption.is_empty()) {
        (false, false) => format!("{label}: {caption}"),
        (false, true) => label,
        (true, false) => caption,
        (true, true) => String::new(),
    };
    if !header_line.is_empty() {
        out.push_str("**");
        out.push_str(&header_line);
        out.push_str("**\n\n");
    }
    if let Some(table) = table_wrap
        .descendants()
        .find(|child| child.has_tag_name(TAG_TABLE))
    {
        let rendered = render_xml_table(table);
        if !rendered.is_empty() {
            out.push_str(&rendered);
            out.push_str("\n\n");
        }
    }
}

fn render_jats_figure(figure: Node<'_, '_>, out: &mut String) {
    let label = figure
        .children()
        .find(|child| child.has_tag_name(TAG_LABEL))
        .map_or_default(collect_inline_xml_text);
    let caption = figure
        .children()
        .find(|child| child.has_tag_name(TAG_CAPTION))
        .map_or_default(collect_inline_xml_text);
    if !label.is_empty() || !caption.is_empty() {
        out.push_str("> **");
        if label.is_empty() {
            out.push_str("Figure");
        } else {
            out.push_str(&label);
        }
        out.push_str(":** ");
        out.push_str(&caption);
        out.push_str("\n\n");
    }
}

fn render_rfc_single_section(section: Node<'_, '_>, heading_level: usize) -> String {
    let mut buffers = vec![String::new()];
    let mut stack = vec![RfcStep::Node {
        node: section,
        heading_level,
    }];
    drain_rfc_stack(&mut stack, &mut buffers);
    buffers.pop().unwrap_or_default()
}

fn render_rfc_container(parent: Node<'_, '_>, base_heading_level: usize) -> String {
    let mut buffers = vec![String::new()];
    let mut stack = Vec::new();
    push_rfc_children_reversed(parent, base_heading_level, &mut stack);
    drain_rfc_stack(&mut stack, &mut buffers);
    buffers.pop().unwrap_or_default()
}

enum RfcStep<'a, 'input> {
    Node {
        node: Node<'a, 'input>,
        heading_level: usize,
    },
    EndBlockquote,
}

fn drain_rfc_stack(stack: &mut Vec<RfcStep<'_, '_>>, buffers: &mut Vec<String>) {
    while let Some(step) = stack.pop() {
        match step {
            RfcStep::EndBlockquote => {
                if let Some(inner) = buffers.pop()
                    && let Some(parent_out) = buffers.last_mut()
                {
                    push_markdown_blockquote(parent_out, &inner);
                }
            }
            RfcStep::Node {
                node,
                heading_level,
            } => {
                let Some(out) = buffers.last_mut() else {
                    continue;
                };
                match node.tag_name().name() {
                    TAG_SECTION => {
                        let title = node
                            .children()
                            .find(|child| child.has_tag_name(TAG_NAME))
                            .map(collect_inline_xml_text)
                            .or_else(|| node.attribute(TAG_TITLE).map(sanitize_inline_text))
                            .unwrap_or_default();
                        if !title.is_empty() {
                            push_markdown_heading(out, heading_level, &title);
                        }
                        push_rfc_children_reversed(node, heading_level.saturating_add(1), stack);
                    }
                    "figure" => {
                        if let Some(caption) = node
                            .children()
                            .find(|child| child.has_tag_name(TAG_NAME))
                            .map(collect_inline_xml_text)
                            .filter(|text| !text.is_empty())
                        {
                            out.push_str("**");
                            out.push_str(&caption);
                            out.push_str("**\n\n");
                        }
                        push_rfc_children_reversed(node, heading_level, stack);
                    }
                    "t" => {
                        let text = collect_inline_xml_text(node);
                        if !text.is_empty() {
                            out.push_str(&text);
                            out.push_str("\n\n");
                        }
                    }
                    "sourcecode" | "artwork" => {
                        let lang = node.attribute("type").unwrap_or_default();
                        let code = collect_raw_text(node);
                        push_fenced_code_block(out, lang, &code);
                    }
                    "ul" | TAG_OL => render_rfc_list(node, out),
                    "dl" => render_rfc_definition_list(node, out),
                    TAG_TABLE => {
                        let rendered = render_xml_table(node);
                        if !rendered.is_empty() {
                            out.push_str(&rendered);
                            out.push_str("\n\n");
                        }
                    }
                    "blockquote" | "aside" => {
                        if buffers.len() <= MAX_RFC_BLOCKQUOTE_DEPTH {
                            buffers.push(String::new());
                            stack.push(RfcStep::EndBlockquote);
                        }
                        push_rfc_children_reversed(node, heading_level, stack);
                    }
                    _ => {}
                }
            }
        }
    }
}

fn push_rfc_children_reversed<'a, 'input>(
    parent: Node<'a, 'input>,
    heading_level: usize,
    stack: &mut Vec<RfcStep<'a, 'input>>,
) {
    let mut children: Vec<Node<'a, 'input>> = parent.children().filter(Node::is_element).collect();
    children.reverse();
    for child in children {
        stack.push(RfcStep::Node {
            node: child,
            heading_level,
        });
    }
}

fn render_rfc_list(list: Node<'_, '_>, out: &mut String) {
    for (item_idx, item) in list
        .children()
        .filter(|node| node.has_tag_name("li"))
        .enumerate()
    {
        let text = collect_inline_xml_text(item);
        if !text.is_empty() {
            if list.has_tag_name(TAG_OL) {
                let _ = writeln!(out, "{}. {text}", item_idx + 1);
            } else {
                let _ = writeln!(out, "- {text}");
            }
        }
    }
    out.push('\n');
}

fn render_rfc_definition_list(dl: Node<'_, '_>, out: &mut String) {
    let mut current_term = String::new();
    for item in dl.children().filter(Node::is_element) {
        if item.has_tag_name("dt") {
            current_term = collect_inline_xml_text(item);
        } else if item.has_tag_name("dd") {
            let definition = collect_inline_xml_text(item);
            if !current_term.is_empty() || !definition.is_empty() {
                out.push_str("- **");
                out.push_str(&current_term);
                out.push_str("**: ");
                out.push_str(&definition);
                out.push('\n');
            }
        }
    }
    out.push('\n');
}

fn render_xml_table(table: Node<'_, '_>) -> String {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut stack: Vec<Node<'_, '_>> = vec![table];

    while let Some(current) = stack.pop() {
        if current.has_tag_name(TAG_TR) {
            let cells: Vec<String> = current
                .children()
                .filter(|cell| cell.has_tag_name("th") || cell.has_tag_name("td"))
                .map(|cell| collect_inline_xml_text(cell).replace('|', "\\|"))
                .collect();
            if !cells.is_empty() {
                rows.push(cells);
            }
            continue;
        }
        let mut children: Vec<Node<'_, '_>> = current.children().filter(Node::is_element).collect();
        children.reverse();
        for child in children {
            let name = child.tag_name().name();
            if matches!(name, TAG_TR | "thead" | "tbody" | "tfoot") {
                stack.push(child);
            }
        }
    }

    render_gfm_table(&rows)
}

fn collect_inline_xml_text(node: Node<'_, '_>) -> String {
    let mut buffer = String::new();
    append_inline_xml(node, &mut buffer);
    sanitize_inline_text(&buffer)
}

fn append_inline_xml(root: Node<'_, '_>, out: &mut String) {
    let mut stack = Vec::new();
    let mut non_ws_emitted: usize = 0;
    push_inline_children_reversed(root, &mut stack);

    while let Some(step) = stack.pop() {
        match step {
            InlineStep::Text(text) => push_inline_str(out, text, &mut non_ws_emitted),
            InlineStep::EndXref {
                non_ws_at_start,
                target,
            } => {
                if non_ws_emitted == non_ws_at_start {
                    out.push('[');
                    out.push_str(target);
                    out.push(']');
                    non_ws_emitted = non_ws_emitted.saturating_add(1);
                }
            }
            InlineStep::Node(child) => {
                if child.is_text() {
                    if let Some(text) = child.text() {
                        push_inline_str(out, text, &mut non_ws_emitted);
                    }
                    continue;
                }
                if !child.is_element() {
                    continue;
                }
                match child.tag_name().name() {
                    "inline-formula" => {
                        let formula = extract_jats_formula(child);
                        if !formula.is_empty() {
                            out.push('$');
                            out.push_str(&formula);
                            out.push('$');
                            non_ws_emitted = non_ws_emitted.saturating_add(1);
                        }
                    }
                    TAG_DISP_FORMULA => {
                        let formula = extract_jats_formula(child);
                        if !formula.is_empty() {
                            out.push_str(" $$");
                            out.push_str(&formula);
                            out.push_str("$$ ");
                            non_ws_emitted = non_ws_emitted.saturating_add(1);
                        }
                    }
                    TAG_TEX_MATH => {
                        let formula = sanitize_inline_text(&collect_raw_text(child));
                        if !formula.is_empty() {
                            out.push('$');
                            out.push_str(&formula);
                            out.push('$');
                            non_ws_emitted = non_ws_emitted.saturating_add(1);
                        }
                    }
                    "tt" | "monospace" | TAG_CODE => {
                        let code = sanitize_inline_text(&collect_raw_text(child));
                        if !code.is_empty() {
                            push_inline_code(out, &code);
                            non_ws_emitted = non_ws_emitted.saturating_add(1);
                        }
                    }
                    "xref" => {
                        if let Some(target) = child
                            .attribute("target")
                            .map(str::trim)
                            .filter(|val| !val.is_empty())
                        {
                            stack.push(InlineStep::EndXref {
                                non_ws_at_start: non_ws_emitted,
                                target,
                            });
                        }
                        push_inline_children_reversed(child, &mut stack);
                    }
                    "sup" => {
                        out.push('^');
                        non_ws_emitted = non_ws_emitted.saturating_add(1);
                        push_inline_children_reversed(child, &mut stack);
                    }
                    "sub" => {
                        out.push('_');
                        non_ws_emitted = non_ws_emitted.saturating_add(1);
                        push_inline_children_reversed(child, &mut stack);
                    }
                    _ => push_inline_children_reversed(child, &mut stack),
                }
            }
        }
    }
}

fn push_inline_str(out: &mut String, text: &str, non_ws_emitted: &mut usize) {
    out.push_str(text);
    if text.chars().any(|character| !character.is_whitespace()) {
        *non_ws_emitted = non_ws_emitted.saturating_add(1);
    }
}

enum InlineStep<'a, 'input> {
    Node(Node<'a, 'input>),
    Text(&'a str),
    EndXref {
        non_ws_at_start: usize,
        target: &'a str,
    },
}

fn push_inline_children_reversed<'a, 'input>(
    parent: Node<'a, 'input>,
    stack: &mut Vec<InlineStep<'a, 'input>>,
) {
    let mut children: Vec<Node<'a, 'input>> = parent.children().collect();
    children.reverse();
    for child in children {
        if child.is_text() {
            if let Some(text) = child.text() {
                stack.push(InlineStep::Text(text));
            }
        } else if child.is_element() {
            stack.push(InlineStep::Node(child));
        }
    }
}

fn extract_jats_formula(formula: Node<'_, '_>) -> String {
    if let Some(tex_math) = formula
        .descendants()
        .find(|node| node.has_tag_name(TAG_TEX_MATH))
    {
        return sanitize_inline_text(&collect_raw_text(tex_math));
    }
    if let Some(tex_annotation) = formula.descendants().find(|node| {
        node.has_tag_name(TAG_ANNOTATION)
            && node
                .attribute("encoding")
                .is_some_and(|enc| enc.to_ascii_lowercase().contains("tex"))
    }) {
        return sanitize_inline_text(&collect_raw_text(tex_annotation));
    }
    if let Some(any_annotation) = formula
        .descendants()
        .find(|node| node.has_tag_name(TAG_ANNOTATION))
    {
        return sanitize_inline_text(&collect_raw_text(any_annotation));
    }
    sanitize_inline_text(&collect_raw_text(formula))
}

fn collect_raw_text(node: Node<'_, '_>) -> String {
    let mut text = String::new();
    for descendant in node.descendants() {
        if descendant.is_text()
            && let Some(chunk) = descendant.text()
        {
            text.push_str(chunk);
        }
    }
    text
}

fn starts_with_ascii_ignore_case(haystack: &str, prefix: &str) -> bool {
    haystack
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use rstest::rstest;

    use super::*;

    fn sample_jats_article() -> &'static str {
        indoc! {r#"
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
            <article>
              <front>
                <article-meta>
                  <title-group><article-title>Clinical Trial Synthesis</article-title></title-group>
                  <contrib-group>
                    <contrib contrib-type="author"><name><surname>Curie</surname><given-names>Marie</given-names></name></contrib>
                  </contrib-group>
                  <pub-date><year>2025</year></pub-date>
                  <abstract>
                    <sec><title>Background</title><p>Short summary.</p></sec>
                  </abstract>
                </article-meta>
              </front>
              <body>
                <sec>
                  <label>1</label>
                  <title>Methods</title>
                  <p>Energy satisfies <inline-formula><tex-math>E = mc^2</tex-math></inline-formula> and display <disp-formula><alternatives><tex-math>F = ma</tex-math><math><mi>glyphsoup</mi></math></alternatives></disp-formula>.</p>
                  <list list-type="order"><list-item><p>First step</p></list-item></list>
                  <table-wrap>
                    <label>Table 1</label>
                    <caption><p>Outcome metrics</p></caption>
                    <table>
                      <tr><th>Group</th><th>Score</th></tr>
                      <tr><td>A</td><td>10<table><tr><td>NESTED</td></tr></table></td></tr>
                    </table>
                  </table-wrap>
                </sec>
              </body>
              <back>
                <ref-list>
                  <ref><element-citation><article-title>Wrong Ref Title</article-title><year>1999</year></element-citation></ref>
                </ref-list>
              </back>
              <sub-article>
                <front-stub>
                  <contrib-group>
                    <contrib contrib-type="author"><name><surname>Reviewer</surname><given-names>Peer</given-names></name></contrib>
                  </contrib-group>
                </front-stub>
              </sub-article>
            </article>
        "#}
    }

    #[test]
    fn isolates_jats_frontmatter_metadata_from_references_and_subarticles() {
        let JatsExtraction::FullText {
            title,
            authors,
            year,
            ..
        } = extract_jats_xml(sample_jats_article()).unwrap()
        else {
            panic!("expected full text JATS");
        };
        assert_eq!(title.as_deref(), Some("Clinical Trial Synthesis"));
        assert_eq!(authors, vec!["Marie Curie"]);
        assert_eq!(year, Some(2025));
    }

    #[test]
    fn renders_jats_headings_math_lists_tables_and_isolates_nested_tables() {
        let JatsExtraction::FullText { markdown, .. } =
            extract_jats_xml(sample_jats_article()).unwrap()
        else {
            panic!("expected full text JATS");
        };
        insta::assert_snapshot!(markdown, @"
        # Clinical Trial Synthesis

        ## Abstract

        ### Background

        Short summary.

        ## 1 Methods

        Energy satisfies $E = mc^2$ and display $$F = ma$$ .

        1. First step

        **Table 1: Outcome metrics**

        | Group | Score |
        | --- | --- |
        | A | 10NESTED |
        ");
    }

    #[test]
    fn detects_bodyless_jats_frontmatter_as_restricted() {
        let jats = indoc! {r"
            <article>
              <front><article-meta><title-group><article-title>Restricted Study</article-title></title-group></article-meta></front>
            </article>
        "};
        assert!(matches!(
            extract_jats_xml(jats).unwrap(),
            JatsExtraction::BodyRestricted { .. }
        ));
    }

    #[rstest]
    #[case(indoc! {r#"
        <!-- safe comment mentioning <!ENTITY fake "x"> -->
        <!DOCTYPE article [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>
        <article><body><p>&xxe;</p></body></article>
    "#})]
    #[case(indoc! {r#"
        <?pi <!-- ?>
        <!DOCTYPE a [ <!ENTITY x "pwned"> ]>
        <a><!-- -->&x;</a>
    "#})]
    #[case(indoc! {r#"
        <?pi <![CDATA[ ?>
        <!DOCTYPE a [ <!ENTITY x "pwned"> ]>
        <a><![CDATA[]]> &x;</a>
    "#})]
    #[case(indoc! {r#"
        <?pi <!DOCTYPE " ?>
        <!DOCTYPE a [ <!ENTITY x 'pwned'><!-- " --> ]>
        <a>&x;</a>
    "#})]
    #[case(indoc! {r#"
        <!DOCTYPE a "unterminated
        <a>text</a>
    "#})]
    fn blocks_xxe_entity_internal_dtd_and_pi_smuggling(#[case] payload: &str) {
        assert!(matches!(
            extract_jats_xml(payload).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn handles_deeply_nested_xml_without_stack_overflow() {
        let open = "<italic>".repeat(4_000);
        let close = "</italic>".repeat(4_000);
        let xml = format!("<article><body><p>{open}DeepLeaf{close}</p></body></article>");

        let JatsExtraction::FullText { markdown, .. } = extract_jats_xml(&xml).unwrap() else {
            panic!("expected full text");
        };
        assert!(markdown.contains("DeepLeaf"));
    }

    #[test]
    fn parses_rfcxml_inline_code_and_empty_xrefs() {
        let rfc_xml = indoc! {r#"
            <rfc>
              <front><title>HTTP Semantics</title></front>
              <middle>
                <section>
                  <name>Overview</name>
                  <t>Request methods use <tt>GET</tt> as defined in <xref target="RFC9110"/>.</t>
                </section>
              </middle>
            </rfc>
        "#};
        let rfc = extract_rfc_xml(rfc_xml).unwrap();
        assert_eq!(rfc.title.as_deref(), Some("HTTP Semantics"));
        insta::assert_snapshot!(rfc.markdown, @"
        # HTTP Semantics

        ## Overview

        Request methods use `GET` as defined in [RFC9110].
        ");
    }

    #[test]
    fn parses_rfcxml_figure_artwork_and_sourcecode_blocks() {
        let rfc_xml = indoc! {r#"
            <rfc>
              <front><title>HTTP Semantics</title></front>
              <middle>
                <section>
                  <name>Grammar</name>
                  <figure>
                    <name>Packet Format</name>
                    <artwork>+-+---+-+</artwork>
                  </figure>
                  <figure>
                    <sourcecode type="abnf">field-name = token</sourcecode>
                  </figure>
                </section>
              </middle>
            </rfc>
        "#};
        let rfc = extract_rfc_xml(rfc_xml).unwrap();
        insta::assert_snapshot!(rfc.markdown, @"
        # HTTP Semantics

        ## Grammar

        **Packet Format**

        ```
        +-+---+-+
        ```

        ```abnf
        field-name = token
        ```
        ");
    }

    #[test]
    fn parses_arxiv_atom_happy_path_including_paper_titled_error() {
        let atom_ok = indoc! {r#"
            <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
              <entry>
                <id>http://arxiv.org/abs/2409.13740v1</id>
                <published>2024-09-20T17:00:00Z</published>
                <title>Error</title>
                <summary>Real paper titled Error.</summary>
                <author><name>Ada Lovelace</name></author>
                <arxiv:doi>10.1000/test</arxiv:doi>
              </entry>
            </feed>
        "#};
        let expected = ArxivAtomEntry {
            title: Some("Error".to_owned()),
            authors: vec!["Ada Lovelace".to_owned()],
            year: Some(2024),
            abstract_text: Some("Real paper titled Error.".to_owned()),
            doi: Some("10.1000/test".to_owned()),
        };
        assert_eq!(parse_arxiv_atom_feed(atom_ok).unwrap(), expected);
    }

    #[test]
    fn rejects_arxiv_atom_api_error_entry() {
        let atom_error = indoc! {r#"
            <feed xmlns="http://www.w3.org/2005/Atom">
              <entry>
                <id>http://arxiv.org/api/errors#incorrect_id_format</id>
                <title>Error</title>
                <summary>incorrect id format</summary>
              </entry>
            </feed>
        "#};
        assert!(matches!(
            parse_arxiv_atom_feed(atom_error).unwrap_err(),
            AdfontesError::Extraction { .. }
        ));
    }

    #[test]
    fn maps_xml_nodes_limit_exceeded_to_safety_block() {
        let xml = format!(
            "<article><body>{}</body></article>",
            "<p/>".repeat(usize::try_from(XML_MAX_NODES).unwrap() + 5)
        );
        assert!(matches!(
            extract_jats_xml(&xml).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn caps_deeply_nested_rfc_blockquotes_without_quadratic_growth() {
        let open = "<blockquote>".repeat(1_000);
        let close = "</blockquote>".repeat(1_000);
        let rfc_xml = format!(
            "<rfc><front><title>Quoted Spec</title></front><middle><section><name>Intro</name>{open}<t>Leaf</t>{close}</section></middle></rfc>"
        );
        let extracted = extract_rfc_xml(&rfc_xml).unwrap();
        assert!(extracted.markdown.contains("Leaf"));
        assert!(!extracted.markdown.contains("> > > > > > > > >"));
    }

    #[test]
    fn handles_deeply_nested_rfc_xrefs_in_linear_time() {
        let open = "<xref target=\"sec1\">".repeat(2_000);
        let close = "</xref>".repeat(2_000);
        let padding = " ".repeat(200_000);
        let rfc_xml = format!(
            "<rfc><front><title>Xref Spec</title></front><middle><section><name>Intro</name><t>{open}{padding}<xref target=\"inner\"/>{close} <xref target=\"sec2\"/></t></section></middle></rfc>"
        );
        let extracted = extract_rfc_xml(&rfc_xml).unwrap();
        assert!(extracted.markdown.contains("[inner]"));
        assert!(!extracted.markdown.contains("[sec1]"));
        assert!(extracted.markdown.contains("[sec2]"));
    }

    #[test]
    fn extract_xml_document_dispatches_article_rfc_and_other() {
        let rfc_xml = "<rfc><front><title>Spec</title></front><middle><section><name>Intro</name><t>Body.</t></section></middle></rfc>";
        let namespaced_jats = "\u{FEFF}<!-- αβγ — leading UTF-8 comment --><jats:article xmlns:jats=\"http://jats.nlm.nih.gov\"><jats:front><jats:article-meta><jats:title-group><jats:article-title>Study</jats:article-title></jats:title-group></jats:article-meta></jats:front></jats:article>";

        let jats = extract_xml_document(sample_jats_article()).unwrap();
        let ns_jats = extract_xml_document(namespaced_jats).unwrap();
        let rfc = extract_xml_document(rfc_xml).unwrap();
        let other = extract_xml_document("<other/>").unwrap();

        assert!(matches!(
            jats,
            XmlDocumentExtraction::Jats(JatsExtraction::FullText { .. })
        ));
        assert!(matches!(
            ns_jats,
            XmlDocumentExtraction::Jats(JatsExtraction::BodyRestricted { .. })
        ));
        assert!(matches!(rfc, XmlDocumentExtraction::Rfc(_)));
        assert_eq!(other, XmlDocumentExtraction::UnsupportedRoot);
    }

    #[test]
    fn extract_xml_document_blocks_internal_dtd() {
        let xxe = "<!DOCTYPE doc [<!ENTITY x SYSTEM \"file:///etc/passwd\">]><doc/>";
        assert!(matches!(
            extract_xml_document(xxe).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[rstest]
    #[case("<!-- unterminated UTF-8 — comment")]
    #[case("<?xml unterminated pi")]
    fn reject_unsafe_xml_dtd_reports_extraction_error_on_unterminated_delimiters(
        #[case] xml: &str,
    ) {
        assert!(matches!(
            reject_unsafe_xml_dtd(xml).unwrap_err(),
            AdfontesError::Extraction { .. }
        ));
    }

    #[test]
    fn reject_unsafe_xml_dtd_scans_multibyte_utf8_and_trailing_angle_bracket() {
        let clean = "\u{FEFF}<?xml version=\"1.0\"?><!-- αβγ --><doc>café &amp; naïve <";
        assert!(reject_unsafe_xml_dtd(clean).is_ok());
    }
}
