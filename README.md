# `adfontes`

Full-text academic and technical literature reader CLI and stdio Model Context Protocol (**MCP**) server for coding agents.

Given a paper identifier (`arXiv`, `DOI`, `PMCID`, `PMID`), specification (`IETF RFC`), or URL, `adfontes` fetches open-access full text, stages it locally, and prints a compact provenance header with a line- or page-indexed Table of Contents. Paywalled works return bibliographic metadata and an abstract without fabricating full text.

---

## Installation

### Python (`uv`)

```bash
uv tool install --managed-python adfontes
```

### Standalone Binary

Install with [cargo-binstall](https://github.com/cargo-bins/cargo-binstall), or compile from [crates.io](https://crates.io/crates/adfontes):

```bash
cargo binstall adfontes          # prebuilt GitHub Release binary
cargo install adfontes --locked  # compile from source
```

---

## Quickstart

Fetch a paper, specification, or web article:

```bash
adfontes fetch 2409.13740                          # arXiv paper
adfontes fetch 10.1038/s41586-025-10072-4          # DOI
adfontes fetch PMC11456789                         # PubMed Central (or PMID:38123456)
adfontes fetch "RFC 9110"                          # IETF RFC
adfontes fetch https://example.org/article         # web page, Markdown, GitHub blob, or PDF
adfontes fetch 2409.13740 -o /tmp/lit              # custom staging directory
```

Extract a lossless `1`–`8`-page vector slice from a staged PDF:

```bash
adfontes slice-pdf /tmp/adfontes/paper.pdf -p 8-10
```

Start the stdio MCP server:

```bash
adfontes mcp
```

Register `adfontes` in `mcp_config.json` to expose the `fetch` and `slice_pdf` tools:

```json
{
  "mcpServers": {
    "adfontes": {
      "command": "/absolute/path/to/adfontes",
      "args": ["mcp"]
    }
  }
}
```

---

## Supported Sources

`adfontes` resolves each target through a keyless open-access waterfall:

| Source | Examples | Resolution Order |
| :--- | :--- | :--- |
| **arXiv** | `2409.13740`, `arXiv:2409.13740v1` | `arxiv.org/html` → `ar5iv` → TeX source → PDF |
| **DOI** | `10.1038/...`, `https://doi.org/...` | Crossref/OpenAlex/Unpaywall → PMCID/arXiv → open-access PDF/HTML → metadata only |
| **PMCID / PMID** | `PMC11456789`, `PMID:38123456` | Europe PMC JATS XML → DOI/arXiv fallback → open-access PDF → metadata only |
| **IETF RFC** | `RFC 9110`, `rfc7231` | Datatracker status notes → RFCXML v3 → plain text |
| **Web / GitHub** | `https://...`, `github.com/.../blob/...` | GitHub raw rewrite → `text/markdown` → HTML math/table extractor, text, or PDF |

---

## Configuration

Configure staging confinement, outbound host rules, and polite-pool metadata access through environment variables:

| Variable | Purpose |
| :--- | :--- |
| `ADFONTES_STAGING_DIR` | Default staging directory. |
| `ADFONTES_EXTRA_ALLOWED_ROOTS` | Additional allowed staging roots (`:` on Unix, `;` on Windows). |
| `ADFONTES_FORBIDDEN_PATH_PREFIXES` | Additional absolute path prefixes blocked from staging. |
| `ADFONTES_BLOCKED_HOST_SUFFIXES` | Comma-separated domain suffixes blocked from outbound HTTP(S). |
| `ADFONTES_CONTACT_EMAIL` | Contact email enabling Unpaywall and OpenAlex polite-pool queries. |
| `ADFONTES_DISABLE_PDF_SLICE` | Truthy value (`1`, `true`, `yes`, `on`) disables local PDF slicing. |
| `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY` | Standard HTTP(S) proxy settings (SOCKS proxies are rejected). |

---

## License

MIT — see [LICENSE](LICENSE).
