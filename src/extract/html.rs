//! Iterative WHATWG HTML5-to-Markdown extractor preserving mathematical notation and tables.
//!
//! Traverses `scraper::Html` iteratively using an explicit heap stack so arbitrarily deep DOM
//! trees never overflow the call stack. Preserves KaTeX, MathJax v2/v3, Distill, and LaTeXML
//! equations verbatim while stripping boilerplate, HTML comments, hidden controls, and
//! CSS-comment-stripped, `!important`-normalized invisible prompt-injection vectors.

use std::fmt::Write as _;

use scraper::node::{Element, Node};
use scraper::{ElementRef, Html};

use super::title::{
    ExtractedText, normalize_markdown_spacing, push_fenced_code_block, push_inline_code,
    render_gfm_table, sanitize_inline_text,
};

const TAG_H1: &str = "h1";
const TAG_OL: &str = "ol";
const TAG_ARTICLE: &str = "article";
const TAG_MATH: &str = "math";
const TAG_SCRIPT: &str = "script";
const TAG_TABLE: &str = "table";
const TAG_TR: &str = "tr";
const ATTR_ALTTEXT: &str = "alttext";
const ATTR_CLASS: &str = "class";
const ATTR_DISPLAY: &str = "display";
const ATTR_BLOCK: &str = "block";
const ATTR_HIDDEN: &str = "hidden";
const ATTR_TYPE: &str = "type";
const CSS_CLIP: &str = "clip";
const VALUE_TRUE: &str = "true";
const CLASS_KATEX_DISPLAY: &str = "katex-display";
const MAX_SCORE_CHARS: usize = 50_000;
const MIN_BODY_FALLBACK_CHARS: usize = 200;
const ARTICLE_COVERAGE_DENOMINATOR: usize = 5;
const ARTICLE_COVERAGE_NUMERATOR: usize = 2;
const CSS_RECT_ARG_COUNT: usize = 4;
const CSS_HEX_RGBA_SHORT_LEN: usize = 4;
const CSS_HEX_RGBA_LONG_LEN: usize = 8;
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

/// Returns `true` when an inline CSS `style` attribute visually hides content (stripping CSS
/// comments and `!important`), protecting agents from invisible prompt-injection payloads.
#[must_use]
pub fn is_hidden_inline_style(style: &str) -> bool {
    let stripped = strip_css_comments(style);
    let mut zero_width = false;
    let mut zero_height = false;
    let mut overflow_hidden = false;

    for declaration in stripped.split(';') {
        let Some((raw_prop, raw_val)) = declaration.split_once(':') else {
            continue;
        };
        let prop = raw_prop.trim().to_ascii_lowercase();
        let val = strip_css_important(raw_val).to_ascii_lowercase();
        match prop.as_str() {
            ATTR_DISPLAY if val == "none" => return true,
            "visibility" if matches!(val.as_str(), ATTR_HIDDEN | "collapse") => return true,
            "content-visibility" if val == ATTR_HIDDEN => return true,
            "opacity" if is_zero_css_opacity(&val) => return true,
            "font-size" if is_zero_css_length(&val) => return true,
            "color" if is_transparent_css_color(&val) => return true,
            "transform" if has_zero_scale_transform(&val) => return true,
            "clip-path" if is_hidden_clip_path(&val) => return true,
            CSS_CLIP if is_zero_clip_rect(&val) => return true,
            "width" | "max-width" if is_zero_css_length(&val) => zero_width = true,
            "height" | "max-height" if is_zero_css_length(&val) => zero_height = true,
            "overflow" | "overflow-x" | "overflow-y"
                if matches!(val.as_str(), ATTR_HIDDEN | CSS_CLIP) =>
            {
                overflow_hidden = true;
            }
            _ => {}
        }
    }

    zero_width && zero_height && overflow_hidden
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
        let tag = element.name().to_ascii_lowercase();

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

        if should_skip_element(&tag, element) {
            return None;
        }

        if has_class(element, CLASS_KATEX_DISPLAY) || has_class(element, "katex") {
            return self.enter_katex_element(elem_ref, element);
        }

        if self.try_handle_special_element(&tag, elem_ref, element) {
            return None;
        }

        match tag.as_str() {
            TAG_H1 | "h2" | "h3" | "h4" | "h5" | "h6" => {
                let level = tag
                    .as_bytes()
                    .get(1)
                    .map_or(2, |digit| usize::from(digit.saturating_sub(b'0')));
                self.ensure_blank_line();
                let out = self.current_output();
                for _ in 0..level {
                    out.push('#');
                }
                out.push(' ');
                Some(CloseAction::BlankLine)
            }
            "br" => {
                self.current_output().push('\n');
                None
            }
            "hr" => {
                self.ensure_blank_line();
                self.current_output().push_str("---\n\n");
                None
            }
            "pre" => {
                self.ensure_blank_line();
                let lang = detect_code_block_language(elem_ref).unwrap_or_default();
                let body = collect_visible_raw_text(elem_ref);
                push_fenced_code_block(self.current_output(), &lang, &body);
                None
            }
            "code" => {
                let inline = sanitize_inline_text(&collect_visible_text(elem_ref));
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
            "ul" | TAG_OL => {
                self.ensure_newline();
                self.list_stack.push(ListContext {
                    ordered: tag == TAG_OL,
                    next_index: 1,
                });
                Some(CloseAction::List)
            }
            "li" => Some(self.enter_list_item()),
            "blockquote" => {
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
                let normalized = normalize_markdown_spacing(&inner);
                if !normalized.is_empty() {
                    let out = self.current_output();
                    for line in normalized.lines() {
                        if line.is_empty() {
                            out.push_str(">\n");
                        } else {
                            let _ = writeln!(out, "> {line}");
                        }
                    }
                    out.push('\n');
                }
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

        if tag == TAG_TABLE {
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
                let cell_tag = cell.value().name().to_ascii_lowercase();
                if should_skip_element(&cell_tag, cell.value()) {
                    continue;
                }
                if has_class(cell.value(), "ltx_eqn_eqno") {
                    let label = sanitize_inline_text(&collect_visible_text(cell));
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
            let fallback = sanitize_inline_text(&collect_visible_text(table));
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
                let tag = elem.value().name().to_ascii_lowercase();
                if (tag == TAG_SCRIPT && !is_math_tex_script(elem.value()))
                    || should_skip_element(&tag, elem.value())
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
                if name.eq_ignore_ascii_case(TAG_ARTICLE) {
                    if has_class(element.value(), "ltx_document") {
                        update_best_candidate(&mut candidates.best_ltx, candidate);
                    }
                    update_best_candidate(&mut candidates.best_article, candidate);
                } else if name.eq_ignore_ascii_case("main") {
                    update_best_candidate(&mut candidates.best_main, candidate);
                } else if name.eq_ignore_ascii_case("body") {
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
        if elem.value().name().eq_ignore_ascii_case("meta") {
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
        if elem.value().name().eq_ignore_ascii_case("title") {
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
        let tag = elem.value().name().to_ascii_lowercase();
        if tag == TAG_SCRIPT || should_skip_element(&tag, elem.value()) {
            continue;
        }
        if tag == TAG_H1 {
            let cleaned = sanitize_inline_text(&collect_visible_text(elem));
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

fn strip_css_comments(style: &str) -> String {
    let mut out = String::with_capacity(style.len());
    let mut cursor: usize = 0;
    while let Some(rest) = style.get(cursor..) {
        let Some(start_rel) = rest.find("/*") else {
            out.push_str(rest);
            break;
        };
        if let Some(before) = rest.get(..start_rel) {
            out.push_str(before);
        }
        let after_open = start_rel + 2;
        if let Some(end_rel) = rest.get(after_open..).and_then(|slice| slice.find("*/")) {
            out.push(' ');
            cursor += after_open + end_rel + 2;
        } else {
            break;
        }
    }
    out
}

fn strip_css_important(raw_value: &str) -> &str {
    let trimmed = raw_value.trim();
    if let Some(bang_index) = trimmed.rfind('!') {
        let after_bang = trimmed.get(bang_index + 1..).unwrap_or_default().trim();
        if after_bang.eq_ignore_ascii_case("important") {
            return trimmed.get(..bang_index).unwrap_or_default().trim();
        }
    }
    trimmed
}

fn is_transparent_css_color(value: &str) -> bool {
    if value == "transparent" {
        return true;
    }
    if let Some(hex) = value.strip_prefix('#')
        && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
        && ((hex.len() == CSS_HEX_RGBA_SHORT_LEN && hex.ends_with('0'))
            || (hex.len() == CSS_HEX_RGBA_LONG_LEN && hex.ends_with("00")))
    {
        return true;
    }
    for prefix in ["rgba(", "hsla(", "rgb(", "hsl("] {
        if let Some(rest) = value.strip_prefix(prefix)
            && let Some(inner) = rest.strip_suffix(')')
        {
            let alpha_candidate = inner
                .rsplit_once('/')
                .map(|(_, after)| after.trim())
                .or_else(|| {
                    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
                    (parts.len() == CSS_RECT_ARG_COUNT).then(|| parts[3])
                });
            if alpha_candidate.is_some_and(is_zero_css_opacity) {
                return true;
            }
        }
    }
    false
}

fn has_zero_scale_transform(value: &str) -> bool {
    for func in ["scale(", "scalex(", "scaley(", "scale3d("] {
        if let Some(pos) = value.find(func)
            && let Some(rest) = value.get(pos + func.len()..)
            && let Some(end) = rest.find(')')
            && let Some(args) = rest.get(..end)
            && args.split(',').any(|arg| is_zero_css_number(arg.trim()))
        {
            return true;
        }
    }
    false
}

fn is_hidden_clip_path(value: &str) -> bool {
    if let Some(rest) = value.strip_prefix("inset(")
        && let Some(inner) = rest.strip_suffix(')')
    {
        return inner.split_whitespace().any(|tok| {
            tok.strip_suffix('%')
                .and_then(|percent| percent.parse::<f64>().ok())
                .is_some_and(|pct| pct >= 100.0)
        });
    }
    false
}

fn is_zero_clip_rect(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("rect(") else {
        return false;
    };
    let Some(inner) = rest.strip_suffix(')') else {
        return false;
    };
    let tokens: Vec<&str> = inner
        .split([',', ' ', '\t'])
        .map(str::trim)
        .filter(|tok| !tok.is_empty())
        .collect();
    tokens.len() == CSS_RECT_ARG_COUNT && tokens.iter().all(|tok| is_zero_css_length(tok))
}

fn is_zero_css_opacity(value: &str) -> bool {
    let numeric = value.strip_suffix('%').unwrap_or(value).trim();
    is_zero_css_number(numeric)
}

fn is_zero_css_length(value: &str) -> bool {
    let trimmed = value.trim();
    let Some(split_pos) = split_css_number_prefix(trimmed) else {
        return false;
    };
    let (number_part, unit_part) = trimmed.split_at(split_pos);
    let unit_ok = unit_part.is_empty()
        || unit_part == "%"
        || unit_part.bytes().all(|byte| byte.is_ascii_alphabetic());
    unit_ok && is_zero_css_number(number_part)
}

fn split_css_number_prefix(trimmed: &str) -> Option<usize> {
    let bytes = trimmed.as_bytes();
    let mut index: usize = 0;
    if matches!(bytes.first(), Some(b'+' | b'-')) {
        index += 1;
    }
    let mut saw_digit = false;
    let mut saw_dot = false;
    while let Some(&byte) = bytes.get(index) {
        if byte.is_ascii_digit() {
            saw_digit = true;
            index += 1;
        } else if byte == b'.' && !saw_dot {
            saw_dot = true;
            index += 1;
        } else {
            break;
        }
    }
    saw_digit.then_some(index)
}

fn is_zero_css_number(value: &str) -> bool {
    value
        .parse::<f64>()
        .is_ok_and(|number| number.is_finite() && number <= 0.0)
}

fn find_math_alttext(elem: ElementRef<'_>) -> Option<String> {
    for descendant in elem.descendants() {
        let Some(child) = ElementRef::wrap(descendant) else {
            continue;
        };
        if child.value().name().eq_ignore_ascii_case(TAG_MATH)
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
        if child
            .value()
            .name()
            .eq_ignore_ascii_case("mjx-assistive-mml")
            || child.value().name().eq_ignore_ascii_case(TAG_MATH)
        {
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
        let tag = elem.value().name().to_ascii_lowercase();
        if should_skip_element(&tag, elem.value()) {
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
        if !child.value().name().eq_ignore_ascii_case("annotation") {
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
                let name = cell.value().name().to_ascii_lowercase();
                (name == "th" || name == "td") && !should_skip_element(&name, cell.value())
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
        let current_tag = current.value().name().to_ascii_lowercase();
        if current != table && should_skip_element(&current_tag, current.value()) {
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
            let tag = child.value().name().to_ascii_lowercase();
            if matches!(tag.as_str(), TAG_TR | "thead" | "tbody" | "tfoot") {
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

fn collect_visible_text(elem: ElementRef<'_>) -> String {
    collect_visible_raw_text(elem)
}

fn collect_visible_raw_text(elem: ElementRef<'_>) -> String {
    let mut out = String::new();
    let mut stack: Vec<WalkStep<'_>> = Vec::new();
    push_children_reversed(elem, &mut stack);
    while let Some(step) = stack.pop() {
        match step {
            WalkStep::Text(chunk) => out.push_str(chunk),
            WalkStep::EnterElement(child_elem) => {
                let tag = child_elem.value().name().to_ascii_lowercase();
                if tag != TAG_SCRIPT && !should_skip_element(&tag, child_elem.value()) {
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
        assert!(extracted.markdown.contains(r"$\alpha + \beta$"));
        assert!(!extracted.markdown.contains("α+β"));
        assert!(extracted.markdown.contains("$f(x) = x^2$"));
        assert!(!extracted.markdown.contains("glyphsoup"));
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
        assert!(
            extracted
                .markdown
                .contains(r"\mathrm{SE} = \sqrt{\mathrm{Var}/n} \quad (4)")
        );
        assert!(extracted.markdown.contains(r"\int_0^1 t\,dt"));
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
        assert!(extracted.markdown.contains("Full Research Paper"));
        assert!(
            extracted
                .markdown
                .contains("actual body of the engineering paper")
        );
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
        let row1 = extracted.markdown.find("| Row 1 | 10 |").unwrap();
        let row2 = extracted.markdown.find("| Row 2 | 20 |").unwrap();
        let row3 = extracted.markdown.find(r"| A \| B | 30 |").unwrap();
        let first = extracted.markdown.find("| First | 1 |").unwrap();
        let second = extracted.markdown.find("| Second | 2 |").unwrap();

        assert!(row1 < row2);
        assert!(row2 < row3);
        assert!(first < second);
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
        let expected_quote = indoc! {r"
            > Line one.
            >
            > Line two.
        "};
        let expected_code = indoc! {r#"
            ````rust
            let fence = "```";
            ````
        "#};
        let extracted = extract_html_to_markdown(html);
        assert!(extracted.markdown.contains(expected_quote.trim_end()));
        assert!(extracted.markdown.contains("`alpha` `beta`"));
        assert!(extracted.markdown.contains("`` `nested` ``"));
        assert!(extracted.markdown.contains(expected_code.trim_end()));
    }

    #[test]
    fn encodes_markdown_links_and_blocks_unsafe_href_schemes() {
        let html = indoc! {r#"
            <article>
              <p>
                <a href="https://example.com/a) [evil](https://x.test">Safe</a>
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
}
