//! Math-preserving HTML5, JATS/RFCXML/Atom XML, IETF RFC, TeX, `CommonMark` TOC, and fragment extractors.
//!
//! All extractors operate strictly in memory over validated UTF-8 text or bounded archive bytes
//! without network access, isolating `cssparser`, `scraper`, `roxmltree`, and `pulldown_cmark`
//! behind pure domain outputs.

architecture_component!(Extract);

mod html;
mod rfc;
mod tex;
mod title;
mod xml;

pub use self::html::{clean_html_fragment, extract_html_to_markdown};
pub use self::rfc::clean_rfc_text;
pub use self::tex::extract_arxiv_source_archive;
pub use self::title::{
    extract_markdown_toc, extract_text_title, normalize_markdown_spacing, sanitize_inline_text,
};
pub use self::xml::{
    JatsExtraction, XmlDocumentExtraction, extract_jats_xml, extract_rfc_xml, extract_xml_document,
    parse_arxiv_atom_feed,
};
