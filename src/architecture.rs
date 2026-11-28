//! Architectural component catalog and Directed Acyclic Graph (DAG) contract.
//!
//! Defines [`ArchitectureComponent`] and compile-time topological acyclicity assertions across
//! the declared catalog. `tests/architecture_conformance.rs` verifies root declarations,
//! top-level module parity, the narrowed `pub` surface of `src/lib.rs`, and the
//! `crates/adfontes-py` -> `adfontes::cli` boundary, while intra-crate component edges and
//! external crate ownership document the structural contract checked via `omnicoach graph`.

architecture_component!(Architecture);

use strum::{AsRefStr, Display, EnumString, VariantArray};

macro_rules! define_submodules {
    ($(
        $sub:ident => [ $( $sub_dep:ident ),* $(,)? ] $( (modules: { $($nested:tt)* }) )?
    ),* $(,)?) => {
        const _: () = {
            struct _SubmoduleOrder {
                $( $sub: usize, )*
            }
            let mut _index = 0usize;
            let _order = _SubmoduleOrder {
                $(
                    $sub: {
                        let current = _index;
                        _index += 1;
                        current
                    },
                )*
            };
            let _ = ( $( _order.$sub, )* );
            $(
                $(
                    assert!(
                        _order.$sub_dep < _order.$sub,
                        concat!(
                            "submodule `",
                            stringify!($sub),
                            "` cannot depend on sibling `",
                            stringify!($sub_dep),
                            "`: dependencies must be declared before their dependents in `modules: {{ ... }}` to guarantee acyclicity",
                        ),
                    );
                )*
                define_submodules!($( $($nested)* )?);
            )*
        };
    };
}

/// Defines `pub enum ArchitectureComponent` and compile-time `const` assertions requiring every
/// component and submodule dependency to be declared before its dependent.
macro_rules! define_architecture {
    ($(
        $(#[$meta:meta])*
        $node:ident => [ $( $dep:ident ),* $(,)? ]
            $( (owns_external: [ $( $ext:literal ),* $(,)? ]) )?
            $( (modules: { $($mods:tt)* }) )?
    ),* $(,)?) => {
        /// Architectural components representing bounded functional units across `adfontes`.
        #[derive(
            Debug,
            Clone,
            Copy,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            Display,
            EnumString,
            AsRefStr,
            VariantArray,
        )]
        pub enum ArchitectureComponent {
            $(
                $(#[$meta])*
                $node,
            )*
        }

        const _: () = {
            $(
                let _ = ( $( $( $ext, )* )? );
                $(
                    assert!(
                        (ArchitectureComponent::$dep as usize) < (ArchitectureComponent::$node as usize),
                        concat!(
                            "ArchitectureComponent::",
                            stringify!($node),
                            " cannot depend on ArchitectureComponent::",
                            stringify!($dep),
                            ": dependencies must be declared before their dependents in define_architecture! to guarantee acyclicity",
                        ),
                    );
                )*
                define_submodules!($( $($mods)* )?);
            )*
        };
    };
}

define_architecture! {
    /// Architectural component catalog and compile-time DAG assertions (`architecture`).
    Architecture => [] (owns_external: ["strum"]),
    /// Validated scholarly identifiers, URLs, page spans, and document domain models (`model`).
    Model        => [] (owns_external: ["thiserror", "url", "percent_encoding"]),
    /// Bounded pure-Rust gzip/deflate decompression and `tar` unpacking (`archive`).
    Archive      => [Model] (owns_external: ["flate2", "tar"]),
    /// SSRF-pinned async HTTP(S) client and charset decoder (`transport`).
    Transport    => [Archive, Model] (owns_external: ["reqwest", "rustls", "rustls_platform_verifier", "ipnet", "mime", "encoding_rs"]),
    /// Pure-Rust PDF page indexing and lossless vector page slicing (`pdf`).
    Pdf          => [Model] (owns_external: ["lopdf"]),
    /// Math-preserving HTML5, JATS/RFCXML/Atom XML, RFC, TeX, and title extractors (`extract`).
    Extract      => [Archive, Model] (owns_external: ["cssparser", "cssparser_color", "scraper", "roxmltree", "pulldown_cmark"]) (modules: {
        title => [],
        html  => [title],
        xml   => [title],
        rfc   => [title],
        tex   => [title],
    }),
    /// Keyless scholarly retrieval waterfalls and format dispatch (`resolve`).
    Resolve      => [Extract, Pdf, Transport, Model] (modules: {
        providers => [],
        content   => [],
    }),
    /// Confined filesystem staging, TOC/page indexing, PDF slicing, and compact manifests (`stage`).
    Stage        => [Extract, Pdf, Model] (owns_external: ["sha2", "tempfile", "rustix", "dunce"]) (modules: {
        manifest => [],
        sandbox  => [],
    }),
    /// Newline-delimited JSON-RPC 2.0 stdio MCP server exposing `fetch` and `slice_pdf` (`mcp`).
    Mcp          => [Stage, Model] (owns_external: ["serde"]),
    /// Command-line interface parsing, startup env binding, and EPIPE-safe dispatch (`cli`).
    Cli          => [Mcp, Stage, Resolve, Transport, Model] (owns_external: ["clap"]),
    /// Thin binary executable entrypoint (`src/bin/adfontes.rs`).
    Bin          => [Cli],
}

impl ArchitectureComponent {
    /// Returns every declared [`ArchitectureComponent`] in topological order.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        Self::VARIANTS
    }
}
