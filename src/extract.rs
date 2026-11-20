//! Math-preserving HTML5, JATS/RFCXML/Atom XML, IETF RFC, TeX, and plain-text extractors.
//!
//! All extractors operate strictly in memory over validated UTF-8 text or bounded archive bytes
//! without network access, isolating `scraper` and `roxmltree` behind pure domain outputs.

architecture_component!(Extract);

mod html;
mod rfc;
mod tex;
mod title;
mod xml;

pub use self::html::extract_html_to_markdown;
pub use self::rfc::clean_rfc_text;
pub use self::tex::{extract_arxiv_source_archive, flatten_and_convert_tex};
pub use self::title::{
    ExtractedText, extract_markdown_toc, extract_text_title, normalize_markdown_spacing,
    sanitize_inline_text,
};
pub use self::xml::{
    ArxivAtomEntry, JatsExtraction, extract_jats_xml, extract_rfc_xml, parse_arxiv_atom_feed,
    reject_unsafe_xml_dtd,
};
