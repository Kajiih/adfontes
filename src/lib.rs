//! Full-text academic and technical literature reader library.
//!
//! Provides scholarly identifier parsing, SSRF-validated HTTP transport, math-preserving
//! Markdown extraction, lossless PDF page slicing, confined staging, and stdio MCP serving.

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

/// Declares the [`crate::architecture::ArchitectureComponent`] that the enclosing root module belongs to.
#[doc(hidden)]
#[macro_export]
macro_rules! architecture_component {
    ($component:ident) => {
        const _ARCHITECTURE_COMPONENT: $crate::architecture::ArchitectureComponent =
            $crate::architecture::ArchitectureComponent::$component;
    };
}

#[doc(hidden)]
pub mod architecture;
pub(crate) mod archive;
pub mod cli;
pub(crate) mod extract;
pub(crate) mod mcp;
pub(crate) mod model;
pub(crate) mod pdf;
pub(crate) mod resolve;
pub(crate) mod stage;
pub(crate) mod transport;
