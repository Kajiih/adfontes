# `adfontes` Roadmap

Tracked follow-up directions and enhancements to evaluate after `v0.2.0`. Items here represent scoped design areas rather than fixed commitments.

## 1. First-Class Text Path for PDFs

- **Staged extracted text alongside PDFs**: Persist the per-page text already extracted during PDF indexing (currently up to `250` pages) into a companion Markdown file (`<slug>.extracted.md` with `## Page N` anchors and line-range mapping in `<slug>.pages.md`), surfacing its path in the compact manifest alongside `STAGED_PDF`.
- **On-demand page-range text extraction**: Emit companion extracted text when slicing PDFs (or expose a text-extraction mode on `slice_pdf`) so text-only readers can inspect arbitrary page spans — including pages beyond the initial `250`-page indexing cap on book-length PDFs — while multimodal readers retain lossless vector PDF slices for equations, figures, tables, and scanned pages.
- **Reader-aware navigation directives**: Update manifest `NAVIGATION:` lines and the agent skill (Section 3) to present both the visual PDF path and the extracted text path explicitly, flagging low-character/scanned pages that require visual inspection.

## 2. Large-PDF Support & Structured Size Limits

- **Aligned, configurable download ceilings**: Raise the default HTTP size ceiling for PDFs to match the `100 MiB` local PDF limit (while keeping tighter defaults for HTML/XML/JSON) and expose bounded environment overrides (`ADFONTES_MAX_RESPONSE_BYTES`, `ADFONTES_TIMEOUT_SECS`) for book-length sources.
- **Outline page numbers & graceful long-book indexing**: Resolve PDF `/Outlines` destinations (`/Dest` and `/A`) to 1-indexed page numbers so agents can jump directly to chapters in multi-hundred-page books, and stop full-book text indexing cleanly once the `250`-page or cumulative `32 MiB` indexing-text budget is reached (without aborting staging), while keeping per-stream decompression-bomb checks fail-closed.
- **Structured oversized-payload outcome**: Return a compact manifest (`STATUS: METADATA_ONLY` with an explicit size-limit reason, or `STATUS: TOO_LARGE`) reporting actual size, active limit, and next steps instead of a bare tool error when a direct target exceeds the configured download ceiling.

## 3. Agent Skill Collocation

- **In-repo `skills/adfontes/SKILL.md`**: Ship a self-contained [`agentskills.io`](https://agentskills.io/specification) skill directly inside this repository so CLI flags, stdio MCP schemas (`fetch`, `slice_pdf`), and adaptive PDF-slicing workflows stay version-locked to releases and directly consumable by coding-agent skill installers.

## 4. Claim & Citation Verification

- **Lightweight post-hoc verification against staged full text**: Evaluate a local verification subcommand or companion workflow (borrowing post-hoc citation-grounding patterns from [PaperQA2](https://arxiv.org/abs/2409.13740) and [OpenScholar](https://doi.org/10.1038/s41586-025-10072-4) without requiring a hosted index or vector database) that checks quoted claims, equations, and bibliographic metadata against staged `.md` and `.pdf` artifacts.

## 5. Source Coverage & Extractors

- **Additional specification shorthands**: Evaluate first-class identifier parsers and canonical waterfalls for Python PEPs (`PEP 695`), Rust RFCs, and W3C technical reports alongside `RFC <N>`.
- **TeX source flattening fidelity**: Improve custom macro expansion (`\newcommand` / `\DeclareMathOperator`) and `.bbl` / bibliography inclusion when flattening arXiv TeX tarballs that separate notation macros from theorem statements.
- **Keyless bibliographic discovery (evaluate)**: Assess whether a read-only `search` subcommand over OpenAlex, Crossref, and arXiv APIs (returning compact candidate lists with open-access indicators ready for `fetch`) belongs in `adfontes` or should stay outside its primary-source reader scope.

## 6. Distribution & Maintenance

- **Signed macOS standalone binaries**: Sign and notarize standalone `aarch64-apple-darwin` and `x86_64-apple-darwin` GitHub Release binaries with an Apple Developer ID certificate so Gatekeeper hosts can run archives without clearing quarantine attributes (PyPI `cp310-abi3` wheels already run cleanly via `dlopen`).
- **Remove `from_env` helper suppressions**: Drop the two `// omni:ignore [environment-variable-in-function]` comments inside `first_non_empty_env` ([src/transport.rs](src/transport.rs)) once `omnicoach` propagates `from_env` configuration-boundary ownership across intra-file `ScopeGraph` call edges.

## References

- [Agent Skills Specification (`agentskills.io`)](https://agentskills.io/specification) — Progressive-disclosure `<skill>/SKILL.md` packaging convention for in-repo agent skills.
- [PaperQA2 (Skarlinski et al., 2024)](https://arxiv.org/abs/2409.13740) — Full-text staging, contextual evidence gathering, and citation verification workflows.
- [OpenScholar (Asai et al., Nature 2026)](https://doi.org/10.1038/s41586-025-10072-4) — Post-hoc citation attribution and claim verification against retrieved full-text passages.
