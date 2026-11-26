//! Iterative WHATWG HTML5-to-Markdown and inline-fragment extractors preserving mathematical notation and tables.
//!
//! Traverses `scraper::Html` iteratively using an explicit heap stack so arbitrarily deep DOM
//! trees never overflow the call stack. Preserves KaTeX, MathJax v2/v3, Distill, and LaTeXML
//! equations verbatim in full documents and decodes HTML entities cleanly inside snippets via
//! `clean_html_fragment` while stripping boilerplate, HTML comments, hidden controls, and
//! `cssparser`-validated invisible prompt-injection vectors.

use std::fmt::Write as _;

use cssparser::{Delimiter, ParseError, Parser as CssParser, Token, parse_important};
use cssparser_color::Color as CssColor;
use scraper::node::{Element, Node};
use scraper::{ElementRef, Html};

use super::title::{
    ExtractedText, normalize_markdown_spacing, push_fenced_code_block, push_inline_code,
    push_markdown_blockquote, render_gfm_table, sanitize_inline_text,
};

const TAG_BR: &str = "br";
const TAG_HORIZONTAL_RULE: &str = "hr";
const TAG_PRE: &str = "pre";
const TAG_LI: &str = "li";
const TAG_BLOCKQUOTE: &str = "blockquote";
const TAG_TITLE: &str = "title";
const TAG_MATH: &str = "math";
const TAG_SCRIPT: &str = "script";
const TAG_TR: &str = "tr";
const TAG_TD: &str = "td";
const TAG_TH: &str = "th";
const ATTR_ALTTEXT: &str = "alttext";
const ATTR_CLASS: &str = "class";
const ATTR_BLOCK: &str = "block";
const ATTR_DISPLAY: &str = "display";
const ATTR_TYPE: &str = "type";
const CSS_PROP_DISPLAY: &str = "display";
const CSS_PROP_CLIP: &str = "clip";
const VALUE_TRUE: &str = "true";
const ATTR_HIDDEN: &str = "hidden";
const KW_AUTO: &str = "auto";
const KW_CLIP: &str = "clip";
const KW_HIDDEN: &str = "hidden";
const KW_NONE: &str = "none";
const CSS_RECT_ARG_COUNT: usize = 4;
const CSS_INSET_MAX_ARGS: usize = 4;
const CSS_SCALE3D_ARG_COUNT: usize = 3;
const CSS_GLOBAL_KEYWORDS: &[&str] = &["initial", "inherit", "unset", "revert", "revert-layer"];
const FONT_SIZE_KEYWORDS: &[&str] = &[
    "xx-small",
    "x-small",
    "small",
    "medium",
    "large",
    "x-large",
    "xx-large",
    "xxx-large",
    "smaller",
    "larger",
    "math",
];
const BOX_DIMENSION_KEYWORDS: &[&str] = &[
    "auto",
    "none",
    "min-content",
    "max-content",
    "fit-content",
    "stretch",
];
const GEOMETRY_BOX_KEYWORDS: &[&str] = &[
    "border-box",
    "padding-box",
    "content-box",
    "margin-box",
    "fill-box",
    "stroke-box",
    "view-box",
];
const TRANSFORM_OTHER_FUNCTIONS: &[&str] = &[
    "matrix",
    "matrix3d",
    "translate",
    "translatex",
    "translatey",
    "translatez",
    "translate3d",
    "rotate",
    "rotatex",
    "rotatey",
    "rotatez",
    "rotate3d",
    "skew",
    "skewx",
    "skewy",
    "perspective",
];
const CLIP_OTHER_SHAPE_FUNCTIONS: &[&str] = &[
    "circle", "ellipse", "polygon", "path", "xywh", "rect", "shape",
];
const CLASS_KATEX_DISPLAY: &str = "katex-display";
const MAX_SCORE_CHARS: usize = 50_000;
const MIN_BODY_FALLBACK_CHARS: usize = 200;
const ARTICLE_COVERAGE_DENOMINATOR: usize = 5;
const ARTICLE_COVERAGE_NUMERATOR: usize = 2;
const MAX_BLOCKQUOTE_DEPTH: usize = 8;

const SKIPPED_HTML_TAGS: &[&str] = &[
    "head", "style", "noscript", "svg", "canvas", "template", "iframe", "nav", "footer", "aside",
    "form", "button", "select", "textarea", "input",
];

const BLOCK_NEWLINE_TAGS: &[&str] = &[
    "p",
    "div",
    "section",
    "article",
    "main",
    "header",
    "figure",
    "figcaption",
    "dl",
    "dt",
    "dd",
];

/// Extracts clean, math-preserving Markdown and document title from raw HTML5.
#[must_use]
pub fn extract_html_to_markdown(html: &str) -> ExtractedText {
    let document = Html::parse_document(html);
    let title = extract_html_title(&document);
    let root_elem = select_content_root(&document);

    let mut state = HtmlRenderState::new();
    state.walk_element(root_elem, true);

    ExtractedText {
        title,
        markdown: normalize_markdown_spacing(&state.into_output()),
    }
}

/// Sanitizes an HTML or JATS inline markup snippet into clean single-line text.
///
/// Decodes HTML entities once via `html5ever`, skips `<script>`, `<style>`, `<noscript>`,
/// `<template>`, and hidden elements, preserves spaces at block/break boundaries, and collapses
/// whitespace without stripping decoded `<` or `>` math inequalities such as `"p < 0.05"`.
///
/// Parsed under WHATWG HTML5 fragment rules, so self-closing `/>` syntax on non-void tags such as
/// `<aside/>` or `<span style="display:none"/>` opens an HTML element rather than acting as an XML
/// empty element.
#[must_use]
pub fn clean_html_fragment(fragment: &str) -> String {
    let parsed = Html::parse_fragment(fragment);
    let mut output = String::new();
    let mut stack: Vec<WalkStep<'_>> = Vec::new();
    for child in parsed.tree.root().children().rev() {
        if let Some(root_elem) = ElementRef::wrap(child) {
            stack.push(WalkStep::EnterElement(root_elem));
        } else if let Node::Text(chunk) = child.value() {
            stack.push(WalkStep::Text(chunk.text.as_ref()));
        }
    }

    while let Some(step) = stack.pop() {
        match step {
            WalkStep::Text(text) => output.push_str(text),
            WalkStep::Exit(_) => output.push(' '),
            WalkStep::EnterElement(elem_ref) => {
                let element = elem_ref.value();
                let tag = element.name();
                if tag == TAG_SCRIPT || should_skip_element(tag, element) {
                    continue;
                }
                if is_fragment_break_tag(tag) {
                    output.push(' ');
                    stack.push(WalkStep::Exit(CloseAction::None));
                }
                push_children_reversed(elem_ref, &mut stack);
            }
        }
    }

    sanitize_inline_text(&output)
}

struct HtmlRenderState {
    buffers: Vec<String>,
    list_stack: Vec<ListContext>,
    in_table_cell: bool,
    in_unannotated_katex_depth: usize,
}

impl HtmlRenderState {
    fn new() -> Self {
        Self {
            buffers: vec![String::new()],
            list_stack: Vec::new(),
            in_table_cell: false,
            in_unannotated_katex_depth: 0,
        }
    }

    fn new_table_cell() -> Self {
        Self {
            buffers: vec![String::new()],
            list_stack: Vec::new(),
            in_table_cell: true,
            in_unannotated_katex_depth: 0,
        }
    }

    fn into_output(mut self) -> String {
        self.buffers.pop().unwrap_or_default()
    }

    fn walk_element<'a>(&mut self, root: ElementRef<'a>, include_self: bool) {
        let mut stack: Vec<WalkStep<'a>> = Vec::new();
        if include_self {
            stack.push(WalkStep::EnterElement(root));
        } else {
            push_children_reversed(root, &mut stack);
        }

        while let Some(step) = stack.pop() {
            match step {
                WalkStep::Text(text) => self.push_text(text),
                WalkStep::Exit(action) => self.apply_close(action),
                WalkStep::EnterElement(elem_ref) => {
                    if let Some(close_action) = self.enter_element(elem_ref) {
                        stack.push(WalkStep::Exit(close_action));
                        push_children_reversed(elem_ref, &mut stack);
                    }
                }
            }
        }
    }

    fn enter_element(&mut self, elem_ref: ElementRef<'_>) -> Option<CloseAction> {
        let element = elem_ref.value();
        let tag = element.name();

        if is_explicitly_hidden(element) {
            return None;
        }

        // MathJax v2 stores raw TeX inside `<script type="math/tex...">`.
        if tag == TAG_SCRIPT {
            if is_math_tex_script(element)
                && let Some(script_type) = element.attr(ATTR_TYPE)
            {
                let tex = sanitize_inline_text(&collect_element_raw_text(elem_ref));
                let is_display = script_type.to_ascii_lowercase().contains("mode=display");
                self.push_math(&tex, is_display);
            }
            return None;
        }

        if should_skip_element(tag, element) {
            return None;
        }

        if has_class(element, CLASS_KATEX_DISPLAY) || has_class(element, "katex") {
            return self.enter_katex_element(elem_ref, element);
        }

        if self.try_handle_special_element(tag, elem_ref, element) {
            return None;
        }

        if let Some(level) = parse_html_heading_level(tag) {
            self.ensure_blank_line();
            let out = self.current_output();
            for _ in 0..level {
                out.push('#');
            }
            out.push(' ');
            return Some(CloseAction::BlankLine);
        }

        match tag {
            TAG_BR => {
                self.current_output().push('\n');
                None
            }
            TAG_HORIZONTAL_RULE => {
                self.ensure_blank_line();
                self.current_output().push_str("---\n\n");
                None
            }
            TAG_PRE => {
                self.ensure_blank_line();
                let lang = detect_code_block_language(elem_ref).unwrap_or_default();
                let body = collect_visible_raw_text(elem_ref);
                push_fenced_code_block(self.current_output(), &lang, &body);
                None
            }
            "code" => {
                let inline = collect_visible_inline_text(elem_ref);
                push_inline_code(self.current_output(), &inline);
                None
            }
            "strong" | "b" => {
                self.current_output().push_str("**");
                Some(CloseAction::Bold)
            }
            "em" | "i" => {
                self.current_output().push('*');
                Some(CloseAction::Italic)
            }
            "a" => {
                let href = element.attr("href").and_then(sanitize_safe_markdown_href);
                if href.is_some() {
                    self.current_output().push('[');
                }
                Some(CloseAction::Link(href))
            }
            "ul" => Some(self.enter_list(false)),
            "ol" => Some(self.enter_list(true)),
            TAG_LI => Some(self.enter_list_item()),
            TAG_BLOCKQUOTE => {
                self.ensure_blank_line();
                if self.buffers.len() <= MAX_BLOCKQUOTE_DEPTH {
                    self.buffers.push(String::new());
                    Some(CloseAction::Blockquote)
                } else {
                    Some(CloseAction::BlankLine)
                }
            }
            other if BLOCK_NEWLINE_TAGS.contains(&other) => {
                self.ensure_blank_line();
                Some(CloseAction::BlankLine)
            }
            _ => Some(CloseAction::None),
        }
    }

    fn enter_list(&mut self, ordered: bool) -> CloseAction {
        self.ensure_newline();
        self.list_stack.push(ListContext {
            ordered,
            next_index: 1,
        });
        CloseAction::List
    }

    fn enter_katex_element(
        &mut self,
        elem_ref: ElementRef<'_>,
        element: &Element,
    ) -> Option<CloseAction> {
        if self.in_unannotated_katex_depth == 0 {
            let is_display = has_class(element, CLASS_KATEX_DISPLAY);
            if let Some(tex) = find_tex_annotation(elem_ref) {
                self.push_math(&tex, is_display);
                return None;
            }
            self.in_unannotated_katex_depth = 1;
            return Some(CloseAction::UnannotatedKatex);
        }
        Some(CloseAction::None)
    }

    fn enter_list_item(&mut self) -> CloseAction {
        self.ensure_newline();
        let indent_level = self.list_stack.len().saturating_sub(1);
        let prefix = if let Some(list_context) = self.list_stack.last_mut() {
            if list_context.ordered {
                let idx = list_context.next_index;
                list_context.next_index += 1;
                format!("{idx}. ")
            } else {
                "- ".to_owned()
            }
        } else {
            "- ".to_owned()
        };
        let out = self.current_output();
        for _ in 0..indent_level {
            out.push_str("  ");
        }
        out.push_str(&prefix);
        CloseAction::Newline
    }

    fn apply_close(&mut self, action: CloseAction) {
        match action {
            CloseAction::None | CloseAction::Link(None) => {}
            CloseAction::BlankLine => self.ensure_blank_line(),
            CloseAction::Newline => self.ensure_newline(),
            CloseAction::Bold => self.current_output().push_str("**"),
            CloseAction::Italic => self.current_output().push('*'),
            CloseAction::Link(Some(href)) => {
                let out = self.current_output();
                out.push_str("](");
                out.push_str(&href);
                out.push(')');
            }
            CloseAction::List => {
                let _ = self.list_stack.pop();
                self.ensure_blank_line();
            }
            CloseAction::UnannotatedKatex => {
                self.in_unannotated_katex_depth = self.in_unannotated_katex_depth.saturating_sub(1);
            }
            CloseAction::Blockquote => {
                let inner = self.buffers.pop().unwrap_or_default();
                push_markdown_blockquote(self.current_output(), &inner);
            }
        }
    }

    fn try_handle_special_element(
        &mut self,
        tag: &str,
        elem_ref: ElementRef<'_>,
        element: &Element,
    ) -> bool {
        if tag == "mjx-container" {
            let is_display = element
                .attr(ATTR_DISPLAY)
                .is_some_and(|val| val.eq_ignore_ascii_case(VALUE_TRUE));
            let formula = find_tex_annotation(elem_ref)
                .or_else(|| find_math_alttext(elem_ref))
                .or_else(|| find_assistive_mathml_text(elem_ref));
            if let Some(tex) = formula {
                self.push_math(&tex, is_display);
            }
            return true;
        }

        if tag == TAG_MATH {
            let is_display = element
                .attr(ATTR_DISPLAY)
                .is_some_and(|val| val.eq_ignore_ascii_case(ATTR_BLOCK));
            let formula = element
                .attr(ATTR_ALTTEXT)
                .map(sanitize_inline_text)
                .filter(|val| !val.is_empty())
                .or_else(|| find_tex_annotation(elem_ref))
                .or_else(|| {
                    let fallback = sanitize_inline_text(&collect_visible_raw_text(elem_ref));
                    (!fallback.is_empty()).then_some(fallback)
                });
            if let Some(tex) = formula {
                self.push_math(&tex, is_display);
            }
            return true;
        }

        if tag == "d-math" {
            let is_display = element.attr(ATTR_BLOCK).is_some();
            let tex = sanitize_inline_text(&collect_visible_raw_text(elem_ref));
            self.push_math(&tex, is_display);
            return true;
        }

        if tag == "d-code" {
            let code = collect_visible_raw_text(elem_ref);
            if element.attr(ATTR_BLOCK).is_some() {
                let lang = element.attr("language").unwrap_or_default();
                self.ensure_blank_line();
                push_fenced_code_block(self.current_output(), lang, &code);
            } else {
                let inline = sanitize_inline_text(&code);
                push_inline_code(self.current_output(), &inline);
            }
            return true;
        }

        if tag == "table" {
            if self.in_table_cell {
                // GFM pipe tables cannot nest tables; flatten nested tables on the iterative
                // heap stack so arbitrarily deep `<table><tr><td>` markup never recurses.
                return false;
            }
            if has_class(element, "ltx_eqn_table")
                || has_class(element, "ltx_equation")
                || has_class(element, "ltx_equationgroup")
            {
                self.render_latexml_equation_table(elem_ref);
                return true;
            }
            let rendered = render_html_gfm_table(elem_ref);
            if !rendered.is_empty() {
                self.ensure_blank_line();
                self.current_output().push_str(&rendered);
                self.ensure_blank_line();
            }
            return true;
        }

        false
    }

    fn render_latexml_equation_table(&mut self, table: ElementRef<'_>) {
        let mut emitted = false;
        let rows = collect_direct_tr_elements(table);
        for tr in rows {
            let mut row_formulas = Vec::new();
            let mut eqno: Option<String> = None;

            for cell in tr.children().filter_map(ElementRef::wrap) {
                let cell_tag = cell.value().name();
                if should_skip_element(cell_tag, cell.value()) {
                    continue;
                }
                if has_class(cell.value(), "ltx_eqn_eqno") {
                    let label = collect_visible_inline_text(cell);
                    if !label.is_empty() {
                        eqno = Some(label);
                    }
                    continue;
                }
                collect_visible_math_formulas(cell, &mut row_formulas);
            }

            if !row_formulas.is_empty() {
                let mut combined = row_formulas.join(" ");
                if let Some(label) = eqno {
                    let _ = write!(combined, " \\quad {label}");
                }
                self.push_math(&combined, true);
                emitted = true;
            }
        }
        if !emitted {
            let fallback = collect_visible_inline_text(table);
            if !fallback.is_empty() {
                self.push_math(&fallback, true);
            }
        }
    }

    fn push_math(&mut self, tex: &str, is_display: bool) {
        let trimmed = tex.trim();
        if trimmed.is_empty() {
            return;
        }
        if is_display {
            self.ensure_blank_line();
            let out = self.current_output();
            out.push_str("$$\n");
            out.push_str(trimmed);
            out.push_str("\n$$\n\n");
        } else {
            let out = self.current_output();
            out.push('$');
            out.push_str(trimmed);
            out.push('$');
        }
    }

    fn push_text(&mut self, raw: &str) {
        let cleaned = collapse_inline_whitespace(raw);
        if cleaned.is_empty() {
            return;
        }
        let out = self.current_output();
        if cleaned == " " {
            let needs_space = !out.is_empty() && !out.ends_with([' ', '\n', '(', '[']);
            if needs_space {
                out.push(' ');
            }
            return;
        }
        if cleaned.starts_with(' ') && (out.is_empty() || out.ends_with([' ', '\n'])) {
            out.push_str(cleaned.trim_start());
        } else {
            out.push_str(&cleaned);
        }
    }

    fn ensure_newline(&mut self) {
        let out = self.current_output();
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
    }

    fn ensure_blank_line(&mut self) {
        let out = self.current_output();
        if out.is_empty() {
            return;
        }
        if !out.ends_with('\n') {
            out.push_str("\n\n");
        } else if !out.ends_with("\n\n") {
            out.push('\n');
        }
    }

    fn current_output(&mut self) -> &mut String {
        if self.buffers.is_empty() {
            self.buffers.push(String::new());
        }
        let last_index = self.buffers.len() - 1;
        &mut self.buffers[last_index]
    }
}

fn select_content_root(document: &Html) -> ElementRef<'_> {
    let root = document.root_element();
    let candidates = score_visible_root_candidates(root);

    if let Some(ltx_article) = candidates.best_ltx {
        return ltx_article.element;
    }

    if let Some(article) = candidates.best_article {
        if let Some(main) = candidates.best_main
            && article
                .char_count
                .saturating_mul(ARTICLE_COVERAGE_DENOMINATOR)
                < main.char_count.saturating_mul(ARTICLE_COVERAGE_NUMERATOR)
        {
            return main.element;
        }
        if candidates.best_main.is_none()
            && let Some(body) = candidates.visible_body
            && body.char_count >= MIN_BODY_FALLBACK_CHARS
            && article
                .char_count
                .saturating_mul(ARTICLE_COVERAGE_DENOMINATOR)
                < body.char_count.saturating_mul(ARTICLE_COVERAGE_NUMERATOR)
        {
            return body.element;
        }
        return article.element;
    }

    if let Some(main) = candidates.best_main {
        return main.element;
    }
    if let Some(body) = candidates.visible_body {
        return body.element;
    }
    root
}

struct RootCandidates<'a> {
    best_ltx: Option<VisibleCandidate<'a>>,
    best_article: Option<VisibleCandidate<'a>>,
    best_main: Option<VisibleCandidate<'a>>,
    visible_body: Option<VisibleCandidate<'a>>,
}

#[derive(Clone, Copy)]
struct VisibleCandidate<'a> {
    element: ElementRef<'a>,
    char_count: usize,
    pre_order: usize,
}

/// Scores all visible `ltx_document`, `<article>`, `<main>`, and `<body>` candidates in a single
/// $O(N)$ iterative DFS pass, pruning hidden subtrees on entry without ancestor walks.
fn score_visible_root_candidates(root: ElementRef<'_>) -> RootCandidates<'_> {
    let mut candidates = RootCandidates {
        best_ltx: None,
        best_article: None,
        best_main: None,
        visible_body: None,
    };
    let mut stack = vec![ScoreStep::Enter(root)];
    let mut count_stack: Vec<usize> = vec![0];
    let mut next_pre_order: usize = 0;

    while let Some(step) = stack.pop() {
        match step {
            ScoreStep::Text(chunk) => {
                if let Some(current_count) = count_stack.last_mut()
                    && *current_count < MAX_SCORE_CHARS
                {
                    let added = chunk
                        .chars()
                        .filter(|character| !character.is_whitespace())
                        .count();
                    *current_count = current_count.saturating_add(added).min(MAX_SCORE_CHARS);
                }
            }
            ScoreStep::Enter(elem) => {
                let tag = elem.value().name();
                if (tag == TAG_SCRIPT && !is_math_tex_script(elem.value()))
                    || should_skip_element(tag, elem.value())
                {
                    continue;
                }
                let pre_order = next_pre_order;
                next_pre_order += 1;
                count_stack.push(0);
                stack.push(ScoreStep::Exit {
                    element: elem,
                    pre_order,
                });
                let mut child_steps: Vec<ScoreStep<'_>> = elem
                    .children()
                    .filter_map(|child| match child.value() {
                        Node::Text(text) => Some(ScoreStep::Text(text.text.as_ref())),
                        Node::Element(_) => ElementRef::wrap(child).map(ScoreStep::Enter),
                        _ => None,
                    })
                    .collect();
                child_steps.reverse();
                stack.extend(child_steps);
            }
            ScoreStep::Exit { element, pre_order } => {
                let char_count = count_stack.pop().unwrap_or(0);
                if let Some(parent_count) = count_stack.last_mut() {
                    *parent_count = parent_count.saturating_add(char_count).min(MAX_SCORE_CHARS);
                }
                let candidate = VisibleCandidate {
                    element,
                    char_count,
                    pre_order,
                };
                let name = element.value().name();
                if name == "article" {
                    if has_class(element.value(), "ltx_document") {
                        update_best_candidate(&mut candidates.best_ltx, candidate);
                    }
                    update_best_candidate(&mut candidates.best_article, candidate);
                } else if name == "main" {
                    update_best_candidate(&mut candidates.best_main, candidate);
                } else if name == "body" {
                    update_best_candidate(&mut candidates.visible_body, candidate);
                }
            }
        }
    }

    candidates
}

enum ScoreStep<'a> {
    Enter(ElementRef<'a>),
    Text(&'a str),
    Exit {
        element: ElementRef<'a>,
        pre_order: usize,
    },
}

fn update_best_candidate<'a>(
    slot: &mut Option<VisibleCandidate<'a>>,
    candidate: VisibleCandidate<'a>,
) {
    let better = slot.as_ref().is_none_or(|current| {
        candidate.char_count > current.char_count
            || (candidate.char_count == current.char_count
                && candidate.pre_order < current.pre_order)
    });
    if better {
        *slot = Some(candidate);
    }
}

fn extract_html_title(document: &Html) -> Option<String> {
    let root = document.root_element();
    for descendant in root.descendants() {
        let Some(elem) = ElementRef::wrap(descendant) else {
            continue;
        };
        if elem.value().name() == "meta" {
            let prop = elem
                .value()
                .attr("property")
                .or_else(|| elem.value().attr("name"))
                .unwrap_or_default();
            if (prop.eq_ignore_ascii_case("citation_title")
                || prop.eq_ignore_ascii_case("og:title"))
                && let Some(content) = elem.value().attr("content")
            {
                let cleaned = sanitize_inline_text(content);
                if !cleaned.is_empty() {
                    return Some(cleaned);
                }
            }
        }
    }

    if let Some(h1_title) = find_first_visible_h1(root) {
        return Some(h1_title);
    }

    for node in root.descendants() {
        let Some(elem) = ElementRef::wrap(node) else {
            continue;
        };
        if elem.value().name() == TAG_TITLE {
            let cleaned = sanitize_inline_text(&collect_element_raw_text(elem));
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }
    None
}

fn find_first_visible_h1(root: ElementRef<'_>) -> Option<String> {
    let mut stack = vec![root];
    while let Some(elem) = stack.pop() {
        let tag = elem.value().name();
        if tag == TAG_SCRIPT || should_skip_element(tag, elem.value()) {
            continue;
        }
        if tag == "h1" {
            let cleaned = collect_visible_inline_text(elem);
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
            continue;
        }
        let mut children: Vec<ElementRef<'_>> =
            elem.children().filter_map(ElementRef::wrap).collect();
        children.reverse();
        stack.extend(children);
    }
    None
}

fn find_math_alttext(elem: ElementRef<'_>) -> Option<String> {
    for descendant in elem.descendants() {
        let Some(child) = ElementRef::wrap(descendant) else {
            continue;
        };
        if child.value().name() == TAG_MATH
            && let Some(alt) = child.value().attr(ATTR_ALTTEXT)
        {
            let cleaned = sanitize_inline_text(alt);
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }
    None
}

fn find_assistive_mathml_text(elem: ElementRef<'_>) -> Option<String> {
    for descendant in elem.descendants() {
        let Some(child) = ElementRef::wrap(descendant) else {
            continue;
        };
        let name = child.value().name();
        if name == "mjx-assistive-mml" || name == TAG_MATH {
            let cleaned = sanitize_inline_text(&collect_visible_raw_text(child));
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
    }
    None
}

fn collect_visible_math_formulas(root: ElementRef<'_>, formulas: &mut Vec<String>) {
    let mut stack: Vec<ElementRef<'_>> = vec![root];
    while let Some(elem) = stack.pop() {
        let tag = elem.value().name();
        if should_skip_element(tag, elem.value()) {
            continue;
        }
        if tag == TAG_MATH {
            if let Some(tex) = elem
                .value()
                .attr(ATTR_ALTTEXT)
                .map(sanitize_inline_text)
                .filter(|val| !val.is_empty())
                .or_else(|| find_tex_annotation(elem))
            {
                formulas.push(tex);
            }
            continue;
        }
        let mut children: Vec<ElementRef<'_>> =
            elem.children().filter_map(ElementRef::wrap).collect();
        children.reverse();
        stack.extend(children);
    }
}

fn find_tex_annotation(elem: ElementRef<'_>) -> Option<String> {
    for descendant in elem.descendants() {
        let Some(child) = ElementRef::wrap(descendant) else {
            continue;
        };
        if child.value().name() != "annotation" {
            continue;
        }
        let encoding = child
            .value()
            .attr("encoding")
            .unwrap_or_default()
            .to_ascii_lowercase();
        if encoding.contains("tex") {
            let tex = sanitize_inline_text(&collect_element_raw_text(child));
            if !tex.is_empty() {
                return Some(tex);
            }
        }
    }
    None
}

fn detect_code_block_language(pre: ElementRef<'_>) -> Option<String> {
    for descendant in pre.descendants() {
        let Some(elem) = ElementRef::wrap(descendant) else {
            continue;
        };
        if let Some(classes) = elem.value().attr(ATTR_CLASS) {
            for cls in classes.split_whitespace() {
                if let Some(lang) = cls
                    .strip_prefix("language-")
                    .or_else(|| cls.strip_prefix("lang-"))
                {
                    let clean = super::title::sanitize_code_language(lang);
                    if !clean.is_empty() {
                        return Some(clean);
                    }
                }
            }
        }
    }
    None
}

fn render_html_gfm_table(table: ElementRef<'_>) -> String {
    let mut rows: Vec<Vec<String>> = Vec::new();
    for tr in collect_direct_tr_elements(table) {
        let cells: Vec<String> = tr
            .children()
            .filter_map(ElementRef::wrap)
            .filter(|cell| {
                let name = cell.value().name();
                (name == TAG_TH || name == TAG_TD) && !should_skip_element(name, cell.value())
            })
            .map(|cell| render_table_cell_text(cell).replace('|', "\\|"))
            .collect();
        if !cells.is_empty() {
            rows.push(cells);
        }
    }
    render_gfm_table(&rows)
}

fn collect_direct_tr_elements(table: ElementRef<'_>) -> Vec<ElementRef<'_>> {
    let mut result = Vec::new();
    let mut stack: Vec<ElementRef<'_>> = vec![table];
    while let Some(current) = stack.pop() {
        let current_tag = current.value().name();
        if current != table && should_skip_element(current_tag, current.value()) {
            continue;
        }
        if current_tag == TAG_TR {
            result.push(current);
            continue;
        }
        let mut children: Vec<ElementRef<'_>> =
            current.children().filter_map(ElementRef::wrap).collect();
        children.reverse();
        for child in children {
            if matches!(child.value().name(), TAG_TR | "thead" | "tbody" | "tfoot") {
                stack.push(child);
            }
        }
    }
    result
}

fn render_table_cell_text(cell: ElementRef<'_>) -> String {
    let mut state = HtmlRenderState::new_table_cell();
    state.walk_element(cell, false);
    sanitize_inline_text(&state.into_output())
}

fn collect_element_raw_text(elem: ElementRef<'_>) -> String {
    let mut text = String::new();
    for node in elem.descendants() {
        if let Node::Text(chunk) = node.value() {
            text.push_str(chunk.text.as_ref());
        }
    }
    text
}

fn collect_visible_inline_text(elem: ElementRef<'_>) -> String {
    sanitize_inline_text(&collect_visible_raw_text(elem))
}

fn collect_visible_raw_text(elem: ElementRef<'_>) -> String {
    let mut out = String::new();
    let mut stack: Vec<WalkStep<'_>> = Vec::new();
    push_children_reversed(elem, &mut stack);
    while let Some(step) = stack.pop() {
        match step {
            WalkStep::Text(chunk) => out.push_str(chunk),
            WalkStep::EnterElement(child_elem) => {
                let tag = child_elem.value().name();
                if tag != TAG_SCRIPT && !should_skip_element(tag, child_elem.value()) {
                    push_children_reversed(child_elem, &mut stack);
                }
            }
            WalkStep::Exit(_) => {}
        }
    }
    out
}

fn should_skip_element(tag_lower: &str, element: &Element) -> bool {
    if SKIPPED_HTML_TAGS.contains(&tag_lower) {
        return true;
    }
    if is_explicitly_hidden(element) || has_aria_hidden_true(element) {
        return true;
    }
    has_class(element, "katex-html")
        || has_class(element, "ltx_page_navbar")
        || has_class(element, "ltx_page_footer")
}

fn is_math_tex_script(element: &Element) -> bool {
    !is_explicitly_hidden(element)
        && !has_aria_hidden_true(element)
        && element
            .attr(ATTR_TYPE)
            .is_some_and(|script_type| script_type.to_ascii_lowercase().contains("math/tex"))
}

fn is_explicitly_hidden(element: &Element) -> bool {
    element.attr(ATTR_HIDDEN).is_some() || element.attr("style").is_some_and(is_hidden_inline_style)
}

fn has_aria_hidden_true(element: &Element) -> bool {
    element
        .attr("aria-hidden")
        .is_some_and(|val| val.trim().eq_ignore_ascii_case(VALUE_TRUE))
}

fn has_class(element: &Element, target: &str) -> bool {
    element
        .attr(ATTR_CLASS)
        .is_some_and(|classes| classes.split_whitespace().any(|cls| cls == target))
}

fn is_fragment_break_tag(tag_lower: &str) -> bool {
    let local = tag_lower
        .rsplit_once(':')
        .map_or(tag_lower, |(_, suffix)| suffix);
    parse_html_heading_level(local).is_some()
        || BLOCK_NEWLINE_TAGS.contains(&local)
        || matches!(
            local,
            TAG_BR
                | TAG_HORIZONTAL_RULE
                | TAG_PRE
                | TAG_LI
                | TAG_TR
                | TAG_TD
                | TAG_TH
                | TAG_BLOCKQUOTE
                | TAG_TITLE
                | "sec"
                | "list-item"
                | "disp-quote"
        )
}

fn parse_html_heading_level(tag: &str) -> Option<usize> {
    let [b'h', level @ b'1'..=b'6'] = tag.as_bytes() else {
        return None;
    };
    Some(usize::from(*level - b'0'))
}

const BIT_DISPLAY: u16 = 1 << 0;
const BIT_VISIBILITY: u16 = 1 << 1;
const BIT_CONTENT_VISIBILITY: u16 = 1 << 2;
const BIT_OPACITY: u16 = 1 << 3;
const BIT_FONT_SIZE: u16 = 1 << 4;
const BIT_COLOR: u16 = 1 << 5;
const BIT_TRANSFORM: u16 = 1 << 6;
const BIT_CLIP_PATH: u16 = 1 << 7;
const BIT_POSITION: u16 = 1 << 8;
const BIT_CLIP: u16 = 1 << 9;
const BIT_WIDTH: u16 = 1 << 10;
const BIT_MAX_WIDTH: u16 = 1 << 11;
const BIT_HEIGHT: u16 = 1 << 12;
const BIT_MAX_HEIGHT: u16 = 1 << 13;
const BIT_OVERFLOW_X: u16 = 1 << 14;
const BIT_OVERFLOW_Y: u16 = 1 << 15;
const DIRECT_HIDE_BITS: u16 = BIT_DISPLAY
    | BIT_VISIBILITY
    | BIT_CONTENT_VISIBILITY
    | BIT_OPACITY
    | BIT_FONT_SIZE
    | BIT_COLOR
    | BIT_TRANSFORM
    | BIT_CLIP_PATH;

fn is_hidden_inline_style(style: &str) -> bool {
    let mut input = CssParser::new(style);
    let mut cascade = InlineStyleCascade::default();
    while !input.is_exhausted() {
        let _ =
            input.parse_until_after(Delimiter::Semicolon, |decl| cascade.parse_declaration(decl));
    }
    cascade.is_hidden()
}

#[derive(Default)]
struct InlineStyleCascade {
    hidden: u16,
    important: u16,
}

impl InlineStyleCascade {
    fn parse_declaration(&mut self, decl: &mut CssParser<'_>) -> Result<(), ParseError<()>> {
        let name = decl.expect_ident()?.to_ascii_lowercase();
        decl.expect_colon()?;
        let update = decl.parse_until_before(Delimiter::Bang, |value| {
            let update = parse_css_update(&name, value)?;
            value.expect_exhausted()?;
            Ok(update)
        })?;
        let important = if decl.is_exhausted() {
            false
        } else {
            parse_important(decl)?;
            decl.expect_exhausted()?;
            true
        };
        match update {
            CssUpdate::Single { bit, hidden } => self.set(bit, hidden, important),
            CssUpdate::Overflow { x_hidden, y_hidden } => {
                self.set(BIT_OVERFLOW_X, x_hidden, important);
                self.set(BIT_OVERFLOW_Y, y_hidden, important);
            }
        }
        Ok(())
    }

    const fn set(&mut self, bit: u16, is_hidden: bool, is_important: bool) {
        if is_important || (self.important & bit) == 0 {
            if is_hidden {
                self.hidden |= bit;
            } else {
                self.hidden &= !bit;
            }
            if is_important {
                self.important |= bit;
            }
        }
    }

    const fn is_hidden(&self) -> bool {
        let clipped = (self.hidden & (BIT_CLIP | BIT_POSITION)) == (BIT_CLIP | BIT_POSITION);
        let zero_box = (self.hidden & (BIT_WIDTH | BIT_MAX_WIDTH)) != 0
            && (self.hidden & (BIT_HEIGHT | BIT_MAX_HEIGHT)) != 0
            && (self.hidden & (BIT_OVERFLOW_X | BIT_OVERFLOW_Y)) != 0;
        (self.hidden & DIRECT_HIDE_BITS) != 0 || clipped || zero_box
    }
}

enum CssUpdate {
    Single { bit: u16, hidden: bool },
    Overflow { x_hidden: bool, y_hidden: bool },
}

fn parse_css_update(
    property_name: &str,
    input: &mut CssParser<'_>,
) -> Result<CssUpdate, ParseError<()>> {
    let (bit, hidden) = match property_name {
        CSS_PROP_DISPLAY => (
            BIT_DISPLAY,
            input.expect_ident()?.eq_ignore_ascii_case(KW_NONE),
        ),
        "visibility" => {
            let ident = input.expect_ident()?;
            (
                BIT_VISIBILITY,
                ident.eq_ignore_ascii_case(KW_HIDDEN) || ident.eq_ignore_ascii_case("collapse"),
            )
        }
        "content-visibility" => (
            BIT_CONTENT_VISIBILITY,
            input.expect_ident()?.eq_ignore_ascii_case(KW_HIDDEN),
        ),
        "opacity" => (
            BIT_OPACITY,
            match input.next()? {
                Token::Number { value, .. }
                | Token::Percentage {
                    unit_value: value, ..
                } => value.is_finite() && *value <= 0.0,
                _ => return Err(ParseError::unexpected_token()),
            },
        ),
        "font-size" => (
            BIT_FONT_SIZE,
            parse_box_dimension(input, FONT_SIZE_KEYWORDS)?,
        ),
        "width" => (
            BIT_WIDTH,
            parse_box_dimension(input, BOX_DIMENSION_KEYWORDS)?,
        ),
        "max-width" => (
            BIT_MAX_WIDTH,
            parse_box_dimension(input, BOX_DIMENSION_KEYWORDS)?,
        ),
        "height" => (
            BIT_HEIGHT,
            parse_box_dimension(input, BOX_DIMENSION_KEYWORDS)?,
        ),
        "max-height" => (
            BIT_MAX_HEIGHT,
            parse_box_dimension(input, BOX_DIMENSION_KEYWORDS)?,
        ),
        "color" => (BIT_COLOR, is_transparent_css_color(input)?),
        "transform" => (BIT_TRANSFORM, parse_transform_hidden(input)?),
        "clip-path" => (BIT_CLIP_PATH, parse_hidden_clip_path(input)?),
        "position" => (BIT_POSITION, parse_position_hidden(input)?),
        CSS_PROP_CLIP => (BIT_CLIP, parse_zero_clip_rect(input)?),
        "overflow" => {
            let first = parse_overflow_keyword(input)?;
            let second = if input.is_exhausted() {
                first
            } else {
                parse_overflow_keyword(input)?
            };
            return Ok(CssUpdate::Overflow {
                x_hidden: first,
                y_hidden: second,
            });
        }
        "overflow-x" => (BIT_OVERFLOW_X, parse_overflow_keyword(input)?),
        "overflow-y" => (BIT_OVERFLOW_Y, parse_overflow_keyword(input)?),
        _ => return Err(ParseError::unexpected_token()),
    };
    Ok(CssUpdate::Single { bit, hidden })
}

fn parse_box_dimension(
    input: &mut CssParser<'_>,
    allowed_keywords: &[&str],
) -> Result<bool, ParseError<()>> {
    match input.next()? {
        Token::Ident(ident)
            if CSS_GLOBAL_KEYWORDS
                .iter()
                .chain(allowed_keywords.iter())
                .any(|keyword| ident.eq_ignore_ascii_case(keyword)) =>
        {
            Ok(false)
        }
        Token::Number { value, .. } if *value == 0.0 => Ok(true),
        Token::Dimension { value, .. } if value.is_finite() && *value >= 0.0 => Ok(*value == 0.0),
        Token::Percentage { unit_value, .. } if unit_value.is_finite() && *unit_value >= 0.0 => {
            Ok(*unit_value == 0.0)
        }
        _ => Err(ParseError::unexpected_token()),
    }
}

fn is_transparent_css_color(input: &mut CssParser<'_>) -> Result<bool, ParseError<()>> {
    let alpha = match CssColor::parse(input)? {
        CssColor::CurrentColor => return Ok(false),
        CssColor::Rgba(color) => color.alpha,
        CssColor::Hsl(color) => color.alpha.unwrap_or(0.0),
        CssColor::Hwb(color) => color.alpha.unwrap_or(0.0),
        CssColor::Lab(color) => color.alpha.unwrap_or(0.0),
        CssColor::Lch(color) => color.alpha.unwrap_or(0.0),
        CssColor::Oklab(color) => color.alpha.unwrap_or(0.0),
        CssColor::Oklch(color) => color.alpha.unwrap_or(0.0),
        CssColor::ColorFunction(color) => color.alpha.unwrap_or(0.0),
    };
    Ok(alpha <= 0.0)
}

fn parse_transform_hidden(input: &mut CssParser<'_>) -> Result<bool, ParseError<()>> {
    let mut saw_none = false;
    let mut func_count = 0usize;
    let mut zero_scale = false;
    while !input.is_exhausted() {
        match input.next()? {
            Token::Ident(ident)
                if ident.eq_ignore_ascii_case(KW_NONE) && !saw_none && func_count == 0 =>
            {
                saw_none = true;
            }
            Token::Function(name) if !saw_none => {
                func_count += 1;
                if name.eq_ignore_ascii_case("scale") {
                    zero_scale |=
                        input.parse_nested_block(|nested| parse_scale_args(nested, 1, 2))?;
                } else if name.eq_ignore_ascii_case("scalex") || name.eq_ignore_ascii_case("scaley")
                {
                    zero_scale |=
                        input.parse_nested_block(|nested| parse_scale_args(nested, 1, 1))?;
                } else if name.eq_ignore_ascii_case("scale3d") {
                    zero_scale |= input.parse_nested_block(|nested| {
                        parse_scale_args(nested, CSS_SCALE3D_ARG_COUNT, CSS_SCALE3D_ARG_COUNT)
                    })?;
                } else if TRANSFORM_OTHER_FUNCTIONS
                    .iter()
                    .any(|func| name.eq_ignore_ascii_case(func))
                {
                    input.parse_nested_block(consume_non_empty_valid_tokens)?;
                } else {
                    return Err(ParseError::unexpected_token());
                }
            }
            _ => return Err(ParseError::unexpected_token()),
        }
    }
    if saw_none || func_count > 0 {
        Ok(zero_scale)
    } else {
        Err(ParseError::unexpected_token())
    }
}

fn parse_scale_args(
    nested: &mut CssParser<'_>,
    min_args: usize,
    max_args: usize,
) -> Result<bool, ParseError<()>> {
    let mut count = 0usize;
    let mut zero_xy = false;
    loop {
        if count > 0 {
            if nested.is_exhausted() {
                break;
            }
            nested.expect_comma()?;
        }
        if count >= max_args {
            return Err(ParseError::unexpected_token());
        }
        let value = match nested.next()? {
            Token::Number { value, .. }
            | Token::Percentage {
                unit_value: value, ..
            } if value.is_finite() => *value,
            _ => return Err(ParseError::unexpected_token()),
        };
        if count < 2 && value == 0.0 {
            zero_xy = true;
        }
        count += 1;
        if nested.is_exhausted() {
            break;
        }
    }
    if count < min_args {
        return Err(ParseError::unexpected_token());
    }
    Ok(zero_xy)
}

fn parse_hidden_clip_path(input: &mut CssParser<'_>) -> Result<bool, ParseError<()>> {
    let hidden = match input.next()? {
        Token::Ident(ident)
            if CSS_GLOBAL_KEYWORDS
                .iter()
                .chain(GEOMETRY_BOX_KEYWORDS.iter())
                .copied()
                .chain(std::iter::once(KW_NONE))
                .any(|keyword| ident.eq_ignore_ascii_case(keyword)) =>
        {
            return Ok(false);
        }
        Token::Function(name) if name.eq_ignore_ascii_case("inset") => {
            input.parse_nested_block(parse_inset_collapsed)?
        }
        Token::Function(name)
            if CLIP_OTHER_SHAPE_FUNCTIONS
                .iter()
                .any(|func| name.eq_ignore_ascii_case(func)) =>
        {
            input.parse_nested_block(consume_non_empty_valid_tokens)?;
            false
        }
        _ => return Err(ParseError::unexpected_token()),
    };
    if !input.is_exhausted() {
        let box_ident = input.expect_ident()?;
        if !GEOMETRY_BOX_KEYWORDS
            .iter()
            .any(|keyword| box_ident.eq_ignore_ascii_case(keyword))
        {
            return Err(ParseError::unexpected_token());
        }
    }
    Ok(hidden)
}

fn parse_inset_collapsed(nested: &mut CssParser<'_>) -> Result<bool, ParseError<()>> {
    let mut edges: [Option<f32>; CSS_INSET_MAX_ARGS] = [None; CSS_INSET_MAX_ARGS];
    let mut count = 0usize;
    while let Ok(token) = nested.next() {
        if let Token::Ident(ident) = token
            && ident.eq_ignore_ascii_case("round")
        {
            consume_non_empty_valid_tokens(nested)?;
            break;
        }
        if count >= CSS_INSET_MAX_ARGS {
            return Err(ParseError::unexpected_token());
        }
        edges[count] = match token {
            Token::Percentage { unit_value, .. } if unit_value.is_finite() => Some(*unit_value),
            Token::Number { value, .. } if *value == 0.0 => Some(0.0),
            Token::Dimension { value, .. } if value.is_finite() => (*value == 0.0).then_some(0.0),
            Token::Function(_) => None,
            _ => return Err(ParseError::unexpected_token()),
        };
        count += 1;
    }

    let (top, right, bottom, left) = match edges.get(..count) {
        Some(&[all]) => (all, all, all, all),
        Some(&[vertical, horizontal]) => (vertical, horizontal, vertical, horizontal),
        Some(&[top, horizontal, bottom]) => (top, horizontal, bottom, horizontal),
        Some(&[top, right, bottom, left]) => (top, right, bottom, left),
        _ => return Err(ParseError::unexpected_token()),
    };
    let collapses = |first: Option<f32>, second: Option<f32>| {
        first.is_some_and(|first_offset| {
            second.is_some_and(|second_offset| (first_offset + second_offset) >= 1.0)
        })
    };
    Ok(collapses(top, bottom) || collapses(left, right))
}

fn parse_position_hidden(input: &mut CssParser<'_>) -> Result<bool, ParseError<()>> {
    let ident = input.expect_ident()?;
    if ident.eq_ignore_ascii_case("absolute") || ident.eq_ignore_ascii_case("fixed") {
        Ok(true)
    } else if ident.eq_ignore_ascii_case("static")
        || ident.eq_ignore_ascii_case("relative")
        || ident.eq_ignore_ascii_case("sticky")
    {
        Ok(false)
    } else {
        Err(ParseError::unexpected_token())
    }
}

fn parse_zero_clip_rect(input: &mut CssParser<'_>) -> Result<bool, ParseError<()>> {
    match input.next()? {
        Token::Ident(ident) if ident.eq_ignore_ascii_case(KW_AUTO) => return Ok(false),
        Token::Function(name) if name.eq_ignore_ascii_case("rect") => {}
        _ => return Err(ParseError::unexpected_token()),
    }
    input.parse_nested_block(|nested| {
        let mut collapsed = [false; CSS_RECT_ARG_COUNT];
        let mut uses_commas = None;
        for (index, slot) in collapsed.iter_mut().enumerate() {
            if index > 0 {
                let saw_comma = nested.try_parse(CssParser::expect_comma).is_ok();
                match uses_commas {
                    None => uses_commas = Some(saw_comma),
                    Some(expected) if expected != saw_comma => {
                        return Err(ParseError::unexpected_token());
                    }
                    Some(_) => {}
                }
            }
            *slot = parse_clip_rect_arg(nested, index)?;
        }
        nested.expect_exhausted()?;
        let [top, right, bottom, left] = collapsed;
        Ok((top && bottom) || (right && left))
    })
}

fn parse_clip_rect_arg(nested: &mut CssParser<'_>, index: usize) -> Result<bool, ParseError<()>> {
    match nested.next()? {
        Token::Ident(ident) if ident.eq_ignore_ascii_case(KW_AUTO) => {
            Ok(index == 0 || index + 1 == CSS_RECT_ARG_COUNT)
        }
        Token::Number { value, .. } if *value == 0.0 => Ok(true),
        Token::Dimension { value, .. } if value.is_finite() => Ok(*value == 0.0),
        _ => Err(ParseError::unexpected_token()),
    }
}

fn parse_overflow_keyword(input: &mut CssParser<'_>) -> Result<bool, ParseError<()>> {
    let ident = input.expect_ident()?;
    if ident.eq_ignore_ascii_case(KW_HIDDEN) || ident.eq_ignore_ascii_case(KW_CLIP) {
        Ok(true)
    } else if ident.eq_ignore_ascii_case("visible")
        || ident.eq_ignore_ascii_case("scroll")
        || ident.eq_ignore_ascii_case(KW_AUTO)
    {
        Ok(false)
    } else {
        Err(ParseError::unexpected_token())
    }
}

fn consume_non_empty_valid_tokens(nested: &mut CssParser<'_>) -> Result<(), ParseError<()>> {
    let mut saw_token = false;
    while let Ok(token) = nested.next() {
        if matches!(token, Token::BadUrl(_) | Token::BadString(_)) {
            return Err(ParseError::unexpected_token());
        }
        saw_token = true;
    }
    if saw_token {
        Ok(())
    } else {
        Err(ParseError::unexpected_token())
    }
}

enum WalkStep<'a> {
    EnterElement(ElementRef<'a>),
    Exit(CloseAction),
    Text(&'a str),
}

enum CloseAction {
    None,
    BlankLine,
    Newline,
    Bold,
    Italic,
    Link(Option<String>),
    List,
    Blockquote,
    UnannotatedKatex,
}

#[derive(Clone, Copy)]
struct ListContext {
    ordered: bool,
    next_index: usize,
}

fn push_children_reversed<'a>(elem: ElementRef<'a>, stack: &mut Vec<WalkStep<'a>>) {
    let mut children: Vec<WalkStep<'a>> = Vec::new();
    for child in elem.children() {
        match child.value() {
            Node::Text(chunk) => children.push(WalkStep::Text(chunk.text.as_ref())),
            Node::Element(_) => {
                if let Some(child_elem) = ElementRef::wrap(child) {
                    children.push(WalkStep::EnterElement(child_elem));
                }
            }
            _ => {}
        }
    }
    while let Some(step) = children.pop() {
        stack.push(step);
    }
}

fn collapse_inline_whitespace(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut prev_space = false;
    for character in input.chars() {
        if character.is_whitespace() {
            if !prev_space {
                output.push(' ');
                prev_space = true;
            }
        } else {
            prev_space = false;
            output.push(character);
        }
    }
    output
}

fn sanitize_safe_markdown_href(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.bytes().any(|byte| byte <= 0x1F || byte == 0x7F) {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    let allowed = lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("mailto:")
        || lower.starts_with('#')
        || is_scheme_less_relative_href(trimmed);
    if !allowed {
        return None;
    }

    let mut encoded = String::with_capacity(trimmed.len());
    for character in trimmed.chars() {
        match character {
            ' ' => encoded.push_str("%20"),
            '(' => encoded.push_str("%28"),
            ')' => encoded.push_str("%29"),
            '<' => encoded.push_str("%3C"),
            '>' => encoded.push_str("%3E"),
            '\\' => encoded.push_str("%5C"),
            other => encoded.push(other),
        }
    }
    Some(encoded)
}

fn is_scheme_less_relative_href(href: &str) -> bool {
    if href.starts_with("//") {
        return false;
    }
    let before_delim = href.split(['/', '?', '#']).next().unwrap_or(href);
    !before_delim.contains(':')
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use rstest::rstest;

    use super::*;

    fn build_nested_divs(depth: usize, leaf: &str) -> String {
        let mut html = String::from("<article>");
        for _ in 0..depth {
            html.push_str("<div>");
        }
        html.push_str(leaf);
        for _ in 0..depth {
            html.push_str("</div>");
        }
        html.push_str("</article>");
        html
    }

    #[test]
    fn preserves_katex_and_mathjax_v2_and_v3_without_glyph_duplication() {
        let html = indoc! {r#"
            <article>
              <p>KaTeX: <span class="katex"><span class="katex-mathml"><math><annotation encoding="application/x-tex">\alpha + \beta</annotation></math></span><span class="katex-html" aria-hidden="true">α+β</span></span></p>
              <p>MathJax2: <script type="math/tex">f(x) = x^2</script></p>
              <mjx-container display="true"><mjx-math>glyphsoup</mjx-math><mjx-assistive-mml><math><annotation encoding="application/x-tex">\sum_{i=1}^n x_i</annotation></math></mjx-assistive-mml></mjx-container>
            </article>
        "#};
        let extracted = extract_html_to_markdown(html);
        insta::assert_snapshot!(extracted.markdown, @r"
        KaTeX: $\alpha + \beta$

        MathJax2: $f(x) = x^2$

        $$
        \sum_{i=1}^n x_i
        $$
        ");
    }

    #[test]
    fn preserves_latexml_equation_tables_with_numbers_and_distill_math() {
        let html = indoc! {r#"
            <article>
              <table class="ltx_eqn_table">
                <tr>
                  <td><math alttext="\mathrm{SE} = \sqrt{\mathrm{Var}/n}">SE</math></td>
                  <td class="ltx_eqn_eqno">(4)</td>
                </tr>
              </table>
              <d-math block>\int_0^1 t\,dt</d-math>
            </article>
        "#};
        let extracted = extract_html_to_markdown(html);
        insta::assert_snapshot!(extracted.markdown, @r"
        $$
        \mathrm{SE} = \sqrt{\mathrm{Var}/n} \quad (4)
        $$

        $$
        \int_0^1 t\,dt
        $$
        ");
    }

    #[rstest]
    #[case("display: none !important")]
    #[case("display: /*comment*/ none !important")]
    #[case("color: red; visibility: hidden !IMPORTANT;")]
    #[case("content-visibility: hidden !important")]
    #[case("opacity: 0% !important")]
    #[case("font-size: 0in !important")]
    #[case("color: transparent")]
    #[case("color: #0000")]
    #[case("color: #ffffff00")]
    #[case("transform: scale(0)")]
    #[case("width: 0; height: 0; overflow: hidden")]
    #[case("width: 0; height: 0; overflow-x: hidden")]
    #[case(r"\64 isplay: none")]
    #[case("font-family: '/*'; bad: [closed] !bogus; display: none")]
    #[case(r"123bad: x; display: \6e one")]
    #[case("display: none !important; display: block")]
    #[case("color: rgb(0 0 0 / 0%)")]
    #[case("color: rgb(0 0 0 / none)")]
    #[case("color: rgb(none none none / 0%)")]
    #[case("color: oklch(0 0 0 / 0)")]
    #[case("color: color(srgb 0 0 0 / 0)")]
    #[case("clip-path: inset(50%)")]
    #[case("clip-path: inset(60% 0 40% 0)")]
    #[case("position: absolute; clip: rect(auto, 0, 0, auto)")]
    #[case("position: absolute; clip: rect(0, 0, auto, 0)")]
    #[case("position: absolute; clip: rect(0, 0, 0, 0); clip: rect(10%, 10%, 10%, 10%)")]
    #[case("clip-path: inset(50%) border-box")]
    #[case("clip-path: inset(50%); clip-path: inset(10)")]
    #[case("font-size: 0; font-size: -10px")]
    #[case("font-size: 0; font-size: 'bogus'")]
    #[case("width: 0; height: 0; overflow: hidden; width: -1px")]
    #[case("transform: translateX(10px) scale(0)")]
    #[case("transform: scale(1, 0)")]
    #[case("transform: scale(0); transform: scale(1, invalid)")]
    fn strips_hidden_css_across_blocks_special_elements_and_tables(#[case] style: &str) {
        let html = format!(
            concat!(
                "<div style=\"{style}\"><article><h1>HIDDEN TITLE</h1><p>HIDDEN ROOT</p></article></div>",
                "<main>",
                "<h1>Visible Title</h1>",
                "<p>Visible</p>",
                "<div style=\"{style}\">INJECT_DIV</div>",
                "<table style=\"{style}\"><tr><td>INJECT_TABLE</td></tr></table>",
                "<table><tr><th>K</th></tr><tr style=\"{style}\"><td>INJECT_ROW</td></tr><tr><td>OK</td></tr></table>",
                "<math style=\"{style}\">INJECT_MATH</math>",
                "<d-math style=\"{style}\">INJECT_DMATH</d-math>",
                "<d-math>x + <span style=\"{style}\">INJECT_NESTED_DMATH</span></d-math>",
                "<!-- secret -->",
                "</main>"
            ),
            style = style
        );
        let extracted = extract_html_to_markdown(&html);
        let leaked: Vec<&str> = [
            "HIDDEN TITLE",
            "HIDDEN ROOT",
            "INJECT_DIV",
            "INJECT_TABLE",
            "INJECT_ROW",
            "INJECT_MATH",
            "INJECT_DMATH",
            "INJECT_NESTED_DMATH",
            "secret",
        ]
        .into_iter()
        .filter(|needle| extracted.markdown.contains(needle))
        .collect();
        assert_eq!(extracted.title.as_deref(), Some("Visible Title"));
        assert!(extracted.markdown.contains("Visible"));
        assert!(extracted.markdown.contains("| OK |"));
        assert!(leaked.is_empty(), "leaked {leaked:?} for style {style}");
    }

    #[rstest]
    #[case("color: rgb(255 0 0 / 50%)")]
    #[case("color: #g000")]
    #[case("opacity: 0.5 !important")]
    #[case("transform: scale(1.2)")]
    #[case("transform: scale(-1)")]
    #[case("transform: scaleX(-1)")]
    #[case("transform: scale3d(1, 1, 0)")]
    #[case("font-size: -10px")]
    #[case("width: -1px; height: -1px; overflow: hidden")]
    #[case("clip-path: inset(0 round 100%)")]
    #[case("clip: rect(0, 0, 0, 0)")]
    #[case("clip: rect(-20px, 0, 0, -20px)")]
    #[case("clip: rect(0, auto, auto, 0)")]
    #[case("display: none; display: block")]
    #[case("width: 0; width: 100px; height: 0; overflow: hidden")]
    #[case("width: 0; height: 0; overflow: hidden; overflow: visible")]
    #[case("display: none bogus")]
    #[case("transform: scale(0) !important !important")]
    #[case("overflow: hidden; overflow: visible bogus")]
    fn preserves_visible_css_declarations(#[case] style: &str) {
        assert!(!is_hidden_inline_style(style));
    }

    #[test]
    fn selects_main_over_small_sidebar_teaser_article() {
        let html = indoc! {r"
            <body>
              <header><article><p>Promo card</p></article></header>
              <main>
                <h1>Full Research Paper</h1>
                <p>This is the actual body of the engineering paper with full derivations and results.</p>
              </main>
            </body>
        "};
        let extracted = extract_html_to_markdown(html);
        insta::assert_snapshot!(extracted.markdown, @"
        # Full Research Paper

        This is the actual body of the engineering paper with full derivations and results.
        ");
    }

    #[test]
    fn preserves_multirow_gfm_table_order_and_headerless_tables() {
        let html = indoc! {r"
            <article>
              <table>
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                  <tr><td>Row 1</td><td>10</td></tr>
                  <tr><td>Row 2</td><td>20</td></tr>
                  <tr><td>A | B</td><td>30</td></tr>
                </tbody>
              </table>
              <table>
                <tr><td>First</td><td>1</td></tr>
                <tr><td>Second</td><td>2</td></tr>
              </table>
            </article>
        "};
        let extracted = extract_html_to_markdown(html);
        insta::assert_snapshot!(extracted.markdown, @r"
        | Metric | Value |
        | --- | --- |
        | Row 1 | 10 |
        | Row 2 | 20 |
        | A \| B | 30 |

        | First | 1 |
        | --- | --- |
        | Second | 2 |
        ");
    }

    #[test]
    fn renders_multiline_blockquotes_and_collision_safe_code() {
        let html = indoc! {r#"
            <article>
              <blockquote><p>Line one.</p><p>Line two.</p></blockquote>
              <p>Compare <code>alpha</code> <code>beta</code> and <code>`nested`</code>.</p>
              <pre><code class="language-rust">let fence = "```";</code></pre>
            </article>
        "#};
        let extracted = extract_html_to_markdown(html);
        insta::assert_snapshot!(extracted.markdown, @r#"
        > Line one.
        >
        > Line two.

        Compare `alpha` `beta` and `` `nested` ``.

        ````rust
        let fence = "```";
        ````
        "#);
    }

    #[test]
    fn encodes_markdown_links_and_blocks_unsafe_href_schemes() {
        let html = indoc! {r#"
            <article>
              <p>
                <a href="https://example.com/a) [evil](https://x.test">Safe</a>
                <a href="https://example.com/p\">Slash</a>
                <a href="file:///etc/passwd">NoFile</a>
                <a href="java&#x09;script:alert(1)">NoJs</a>
              </p>
            </article>
        "#};
        let extracted = extract_html_to_markdown(html);
        assert!(
            extracted
                .markdown
                .contains("[Safe](https://example.com/a%29%20[evil]%28https://x.test)")
        );
        assert!(
            extracted
                .markdown
                .contains("[Slash](https://example.com/p%5C)")
        );
        assert!(!extracted.markdown.contains("file:///etc/passwd"));
        assert!(!extracted.markdown.contains("alert(1)"));
    }

    #[test]
    fn handles_deeply_nested_dom_without_stack_overflow() {
        let html = build_nested_divs(800, "<p>Deep Leaf Content</p>");
        let extracted = extract_html_to_markdown(&html);
        assert_eq!(extracted.markdown, "Deep Leaf Content");
    }

    #[test]
    fn handles_deeply_nested_tables_without_stack_overflow() {
        let open = "<table><tr><td>".repeat(3_000);
        let close = "</td></tr></table>".repeat(3_000);
        let html = format!("<article>{open}TableLeaf{close}</article>");
        let extracted = extract_html_to_markdown(&html);
        assert!(extracted.markdown.contains("TableLeaf"));
    }

    #[test]
    fn caps_deeply_nested_html_blockquotes_without_quadratic_growth() {
        let open = "<blockquote>".repeat(1_000);
        let close = "</blockquote>".repeat(1_000);
        let html = format!("<article>{open}<p>Quoted</p>{close}</article>");
        let extracted = extract_html_to_markdown(&html);
        assert!(extracted.markdown.contains("Quoted"));
        assert!(!extracted.markdown.contains("> > > > > > > > >"));
    }

    #[test]
    fn scores_deeply_nested_articles_in_linear_time() {
        let open = "<article>".repeat(3_000);
        let close = "</article>".repeat(3_000);
        let html = format!("<body>{open}<p>ArticleLeaf</p>{close}</body>");
        let extracted = extract_html_to_markdown(&html);
        assert!(extracted.markdown.contains("ArticleLeaf"));
    }

    #[test]
    fn handles_nested_unannotated_katex_empty_h1_and_script_inside_dmath() {
        let katex = format!(
            "{}Leaf{}",
            "<span class=\"katex\">".repeat(2_000),
            "</span>".repeat(2_000)
        );
        let empty_h1s = format!(
            "{}{}",
            "<h1><table><tr><td>".repeat(1_500),
            "</td></tr></table></h1>".repeat(1_500)
        );
        let html = format!(
            "<article>{empty_h1s}{katex}<d-math>x + <script>INJECT_SCRIPT</script>1</d-math></article>"
        );
        let extracted = extract_html_to_markdown(&html);
        assert!(extracted.markdown.contains("Leaf"));
        assert!(!extracted.markdown.contains("INJECT_SCRIPT"));
    }

    #[rstest]
    #[case(
        "<jats:p>Significance at <i>p</i> &lt; 0.05 and <b>n</b> &gt; 10.</jats:p>",
        "Significance at p < 0.05 and n > 10."
    )]
    #[case("<p>First.</p><p>Second.</p>", "First. Second.")]
    #[case(
        "<script>secret()</script><style>.x{}</style><noscript>off</noscript><template>tpl</template><span hidden>hidden</span><span style=\"display:none\">prompt</span>Visible",
        "Visible"
    )]
    #[case(
        "<jats:list-item>Alpha</jats:list-item><jats:list-item>p < 0.05</jats:list-item>",
        "Alpha p < 0.05"
    )]
    #[case("", "")]
    fn clean_html_fragment_preserves_decoded_inequalities_breaks_and_hiding(
        #[case] fragment: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(clean_html_fragment(fragment), expected);
    }
}
