# `adfontes` Roadmap

Tracked follow-up directions and enhancements to evaluate after `v0.1.0`. Items here represent scoped design areas rather than fixed commitments.

## 1. Agent Skill Collocation

- **In-repo `skills/adfontes/SKILL.md`**: Ship a self-contained [`agentskills.io`](https://agentskills.io/specification) skill directly inside this repository so CLI flags, stdio MCP schemas (`fetch`, `slice_pdf`), and adaptive PDF-slicing workflows stay version-locked to releases and directly consumable by coding-agent skill installers.

## 2. Claim & Citation Verification

- **Lightweight post-hoc verification against staged full text**: Evaluate a local verification subcommand or companion workflow (borrowing post-hoc citation-grounding patterns from [PaperQA2](https://arxiv.org/abs/2409.13740) and [OpenScholar](https://doi.org/10.1038/s41586-025-10072-4) without requiring a hosted index or vector database) that checks quoted claims, equations, and bibliographic metadata against staged `.md` and `.pdf` artifacts.

## 3. Source Coverage & Extractors

- **Additional specification shorthands**: Evaluate first-class identifier parsers and canonical waterfalls for Python PEPs (`PEP 695`), Rust RFCs, and W3C technical reports alongside `RFC <N>`.
- **TeX source flattening fidelity**: Improve custom macro expansion (`\newcommand` / `\DeclareMathOperator`) and `.bbl` / bibliography inclusion when flattening arXiv TeX tarballs that separate notation macros from theorem statements.
- **Keyless bibliographic discovery (evaluate)**: Assess whether a read-only `search` subcommand over OpenAlex, Crossref, and arXiv APIs (returning compact candidate lists with open-access indicators ready for `fetch`) belongs in `adfontes` or should stay outside its primary-source reader scope.

## 4. Distribution & Maintenance

- **Signed macOS standalone binaries**: Sign and notarize standalone `aarch64-apple-darwin` and `x86_64-apple-darwin` GitHub Release binaries with an Apple Developer ID certificate so Gatekeeper hosts can run archives without clearing quarantine attributes (PyPI `cp310-abi3` wheels already run cleanly via `dlopen`).
- **Remove `from_env` helper suppressions**: Drop the two `// omni:ignore [environment-variable-in-function]` comments inside `first_non_empty_env` ([src/transport.rs](src/transport.rs)) once `omnicoach` propagates `from_env` configuration-boundary ownership across intra-file `ScopeGraph` call edges.

## References

- [Agent Skills Specification (`agentskills.io`)](https://agentskills.io/specification) — Progressive-disclosure `<skill>/SKILL.md` packaging convention for in-repo agent skills.
- [PaperQA2 (Skarlinski et al., 2024)](https://arxiv.org/abs/2409.13740) — Full-text staging, contextual evidence gathering, and citation verification workflows.
- [OpenScholar (Asai et al., Nature 2026)](https://doi.org/10.1038/s41586-025-10072-4) — Post-hoc citation attribution and claim verification against retrieved full-text passages.
