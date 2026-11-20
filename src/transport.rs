//! SSRF-pinned synchronous `GET`/`HEAD` HTTP(S) transport and WHATWG charset decoding.
//!
//! Encapsulates `ureq` (`rustls` + OS platform verifier) and `encoding_rs` behind hop-by-hop
//! redirect validation, DNS/IP pinning against rebinding attacks, strict scheme-specific HTTP(S)
//! proxy selection, end-to-end request deadlines, bounded body streaming, and safe `gzip`/`deflate`
//! content decoding.

architecture_component!(Transport);

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use encoding_rs::Encoding;
use ureq::Agent;
use ureq::config::Config;
use ureq::http::Uri;
use ureq::tls::{RootCerts, TlsConfig};
use ureq::unversioned::resolver::{ResolvedSocketAddrs, Resolver};
use ureq::unversioned::transport::{DefaultConnector, NextTimeout};

use crate::archive::{is_gzip_magic, safe_gunzip_bytes, safe_inflate_bytes};
use crate::model::{AdfontesError, HttpUrl, UrlHost};

/// Default end-to-end HTTP request timeout (`20 s`, shared across redirect hops).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// Default maximum HTTP redirect hops followed per request.
pub const DEFAULT_MAX_REDIRECTS: usize = 6;

/// Default maximum response body size (`25 MiB`, covering large open-access academic PDFs).
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 25 * 1024 * 1024;

/// Default descriptive `User-Agent` sent on outbound scholarly API requests.
pub const DEFAULT_USER_AGENT: &str = "adfontes/0.1.0 (+https://github.com/Kajiih/adfontes)";

const MAX_PINNED_SOCKET_ADDRS: usize = 16;
const BODY_READ_CHUNK_SIZE: usize = 16 * 1024;
const MAX_CONTENT_ENCODING_LAYERS: usize = 2;
const LOCALHOST_NAME: &str = "localhost";
const SCHEME_HTTP: &str = "http";
const SCHEME_HTTPS: &str = "https";
const HEADER_CONTENT_ENCODING: &str = "content-encoding";
const ENCODING_GZIP: &str = "gzip";
const ENCODING_X_GZIP: &str = "x-gzip";
const ASCII_PRINTABLE_FIRST: u8 = 0x20;
const ASCII_PRINTABLE_LAST: u8 = 0x7E;
const MAX_IPV4_PREFIX_BITS: u8 = 32;
const MAX_IPV6_PREFIX_BITS: u8 = 128;

const HTTP_STATUS_MOVED_PERMANENTLY: u16 = 301;
const HTTP_STATUS_FOUND: u16 = 302;
const HTTP_STATUS_SEE_OTHER: u16 = 303;
const HTTP_STATUS_TEMPORARY_REDIRECT: u16 = 307;
const HTTP_STATUS_PERMANENT_REDIRECT: u16 = 308;

const IPV4_CGNAT_FIRST: u8 = 100;
const IPV4_CGNAT_MASK: u8 = 0xC0;
const IPV4_CGNAT_SECOND: u8 = 64;
const IPV4_IETF_PROTOCOL_FIRST: u8 = 192;
const IPV4_SIX_TO_FOUR_RELAY_SECOND: u8 = 88;
const IPV4_SIX_TO_FOUR_RELAY_THIRD: u8 = 99;
const IPV4_BENCHMARK_FIRST: u8 = 198;
const IPV4_BENCHMARK_MASK: u8 = 0xFE;
const IPV4_BENCHMARK_SECOND: u8 = 18;
const IPV4_CLASS_E_START: u8 = 240;

const IPV6_GLOBAL_UNICAST_MASK: u16 = 0xE000;
const IPV6_GLOBAL_UNICAST_PREFIX: u16 = 0x2000;
const IPV6_IETF_SPECIAL_HEAD: u16 = 0x2001;
const IPV6_IETF_PROTOCOL_MASK: u16 = 0xFE00;
const IPV6_DOCUMENTATION_SECOND: u16 = 0x0DB8;
const IPV6_RFC9637_DOC_HEAD: u16 = 0x3FFF;
const IPV6_RFC9637_DOC_SECOND_MASK: u16 = 0xF000;
const IPV6_SIX_TO_FOUR_HEAD: u16 = 0x2002;
const IPV6_NAT64_HEAD: u16 = 0x0064;
const IPV6_NAT64_SECOND: u16 = 0xFF9B;

const FORBIDDEN_EXACT_HOSTS: &[&str] = &[
    LOCALHOST_NAME,
    "ip6-localhost",
    "ip6-loopback",
    "localtest.me",
    "lvh.me",
    "metadata",
    "nip.io",
    "sslip.io",
    "xip.io",
    "vcap.me",
    "traefik.me",
    "local.gd",
    "rbndr.us",
];

const FORBIDDEN_HOST_SUFFIXES: &[&str] = &[
    ".localhost",
    ".local",
    ".internal",
    ".localdomain",
    ".home.arpa",
    ".corp",
    ".intranet",
    ".lan",
    ".home",
    ".localtest.me",
    ".lvh.me",
    ".nip.io",
    ".sslip.io",
    ".xip.io",
    ".vcap.me",
    ".traefik.me",
    ".local.gd",
    ".rbndr.us",
];

const RAW_ARCHIVE_MIME_TYPES: &[&str] = &[
    "application/gzip",
    "application/x-gzip",
    "application/tar",
    "application/x-tar",
    "application/x-eprint-tar",
    "application/x-eprint",
];

const SENSITIVE_REDIRECT_HEADERS: &[&str] = &["authorization", "proxy-authorization", "cookie"];

/// Scheme-specific HTTP(S) proxy configuration parsed once at process startup.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProxyConfig {
    /// Proxy URL for `http://` targets (`HTTP_PROXY` / `http_proxy`).
    pub http_proxy: Option<String>,
    /// Proxy URL for `https://` targets (`HTTPS_PROXY` / `https_proxy`).
    pub https_proxy: Option<String>,
    /// Fallback proxy URL for either scheme (`ALL_PROXY` / `all_proxy`).
    pub all_proxy: Option<String>,
    /// Comma-separated host/suffix/CIDR exclusions (`NO_PROXY` / `no_proxy`).
    pub no_proxy: Vec<String>,
    /// Permits proxying when local DNS cannot resolve the destination (disabled by default so
    /// split-horizon/internal names fail closed).
    pub allow_proxy_remote_dns: bool,
}

impl ProxyConfig {
    /// Returns an empty proxy configuration that always connects directly.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            http_proxy: None,
            https_proxy: None,
            all_proxy: None,
            no_proxy: Vec::new(),
            allow_proxy_remote_dns: false,
        }
    }

    /// Reads scheme-specific proxy settings from standard environment variables once at startup.
    #[must_use]
    pub fn from_env() -> Self {
        let plain_http = first_non_empty_env("http_proxy", "HTTP_PROXY");
        let secure_https = first_non_empty_env("https_proxy", "HTTPS_PROXY");
        let all_proxy = first_non_empty_env("all_proxy", "ALL_PROXY");
        let no_proxy_raw = first_non_empty_env("no_proxy", "NO_PROXY").unwrap_or_default();
        let no_proxy = no_proxy_raw
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(str::to_ascii_lowercase)
            .collect();
        Self {
            http_proxy: plain_http,
            https_proxy: secure_https,
            all_proxy,
            no_proxy,
            allow_proxy_remote_dns: false,
        }
    }

    /// Selects and validates the HTTP(S) proxy URL applicable to `url` by scheme and `NO_PROXY`.
    ///
    /// Rejects unsupported proxy schemes such as `socks5://` cleanly instead of passing them to
    /// `ureq` without `socks-proxy` enabled.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::Transport`] if the matched proxy URL is malformed or uses a
    /// non-HTTP(S) scheme (such as `socks4://` / `socks5://`).
    pub fn resolve_proxy_for_url(
        &self,
        url: &HttpUrl,
        bypass_loopback: bool,
    ) -> Result<Option<String>, AdfontesError> {
        if bypass_loopback && is_loopback_url(url) {
            return Ok(None);
        }
        if matches_no_proxy(url, &self.no_proxy) {
            return Ok(None);
        }
        let selected = match url.scheme() {
            SCHEME_HTTPS => self.https_proxy.as_deref().or(self.all_proxy.as_deref()),
            SCHEME_HTTP => self.http_proxy.as_deref().or(self.all_proxy.as_deref()),
            _ => None,
        };
        let Some(raw_proxy) = selected.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        validate_http_proxy_url(raw_proxy)?;
        Ok(Some(raw_proxy.to_owned()))
    }
}

/// Runtime configuration for [`HttpClient`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportConfig {
    /// End-to-end deadline shared across DNS resolution and all redirect hops.
    pub timeout: Duration,
    /// Maximum number of `3xx` redirects to follow.
    pub max_redirects: usize,
    /// Maximum wire and decompressed response body bytes permitted.
    pub max_response_bytes: usize,
    /// `User-Agent` header value (must be visible ASCII without CRLF).
    pub user_agent: String,
    /// Boundary-parsed proxy configuration.
    pub proxy: ProxyConfig,
    /// Additional operator-configured blocked host suffixes (`ADFONTES_BLOCKED_HOST_SUFFIXES`),
    /// each matching both the exact domain and any subdomain.
    pub blocked_host_suffixes: Vec<String>,
    /// Exact loopback socket addresses (`127.0.0.1:<port>` or `[::1]:<port>`) permitted during
    /// hermetic local fixture tests (empty in production).
    pub allowed_loopback_addrs: BTreeSet<SocketAddr>,
}

impl TransportConfig {
    /// Reads proxy settings and optional `ADFONTES_BLOCKED_HOST_SUFFIXES` once at startup.
    #[must_use]
    pub fn from_env() -> Self {
        let blocked_host_suffixes = std::env::var("ADFONTES_BLOCKED_HOST_SUFFIXES")
            .ok()
            .map_or_default(|raw| parse_blocked_host_suffixes(&raw));
        Self {
            proxy: ProxyConfig::from_env(),
            blocked_host_suffixes,
            ..Self::default()
        }
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            max_redirects: DEFAULT_MAX_REDIRECTS,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            user_agent: DEFAULT_USER_AGENT.to_owned(),
            proxy: ProxyConfig::none(),
            blocked_host_suffixes: Vec::new(),
            allowed_loopback_addrs: BTreeSet::new(),
        }
    }
}

/// Completed HTTP response with final post-redirect URL, headers, and bounded body bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// HTTP status code (`200`, `404`, `429`, etc.).
    pub status: u16,
    /// Intermediate validated redirect hop URLs visited between the initial URL and [`Self::final_url`].
    pub redirect_chain: Vec<HttpUrl>,
    /// Final validated URL after following all redirects.
    pub final_url: HttpUrl,
    /// Raw `Content-Type` header value, if sent.
    pub content_type: Option<String>,
    /// Lowercased response headers (multi-value headers combined per RFC 9110 §5.3).
    pub headers: BTreeMap<String, String>,
    /// Bounded (and, when appropriate, content-decoded) response body bytes.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Returns the lowercased media type without `; charset=...` parameters.
    #[must_use]
    pub fn mime_type(&self) -> String {
        self.content_type
            .as_deref()
            .and_then(|value| value.split(';').next())
            .map_or_default(|media| media.trim().to_ascii_lowercase())
    }

    /// Decodes [`Self::body`] to UTF-8 using BOM detection, HTTP `charset=` label, or UTF-8 fallback.
    #[must_use]
    pub fn decode_text(&self) -> String {
        if let Some((bom_encoding, bom_len)) = Encoding::for_bom(&self.body) {
            let slice = self.body.get(bom_len..).unwrap_or_default();
            let (decoded, _) = bom_encoding.decode_without_bom_handling(slice);
            return decoded.into_owned();
        }
        let encoding = self
            .content_type
            .as_deref()
            .and_then(extract_charset_label)
            .and_then(|label| Encoding::for_label(label.as_bytes()))
            .unwrap_or(encoding_rs::UTF_8);
        let (decoded, _, _) = encoding.decode(&self.body);
        decoded.into_owned()
    }
}

/// Synchronous SSRF-pinned HTTP(S) client supporting only read-only `GET` and `HEAD` methods.
#[derive(Debug, Clone)]
pub struct HttpClient {
    config: TransportConfig,
    tls_config: TlsConfig,
}

impl HttpClient {
    /// Constructs a new [`HttpClient`] caching the OS `PlatformVerifier` TLS configuration.
    #[must_use]
    pub fn new(config: TransportConfig) -> Self {
        let tls_config = TlsConfig::builder()
            .root_certs(RootCerts::PlatformVerifier)
            .build();
        Self { config, tls_config }
    }

    /// Returns a reference to the active [`TransportConfig`].
    #[must_use]
    pub const fn config(&self) -> &TransportConfig {
        &self.config
    }

    /// Executes a validated `GET` request with optional `Accept` header and extra headers.
    ///
    /// Sensitive headers (`Authorization`, `Cookie`, `Proxy-Authorization`) in `extra_headers`
    /// are automatically stripped across cross-origin redirects or `https` → `http` downgrades.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::SafetyBlock`] if any hop targets a forbidden host/IP, exceeds the
    /// redirect cap, or exceeds `max_response_bytes`; returns [`AdfontesError::Transport`] on
    /// network, DNS, header, proxy, or TLS failures.
    pub fn get(
        &self,
        url: &HttpUrl,
        accept: Option<&str>,
        extra_headers: &[(&str, &str)],
    ) -> Result<HttpResponse, AdfontesError> {
        self.execute(HttpMethod::Get, url, accept, extra_headers)
    }

    /// Executes a validated `HEAD` request without reading a response body.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::SafetyBlock`] or [`AdfontesError::Transport`] on validation or
    /// network failures.
    pub fn head(&self, url: &HttpUrl) -> Result<HttpResponse, AdfontesError> {
        self.execute(HttpMethod::Head, url, None, &[])
    }

    /// Validates `url` against static SSRF and configured blocked-host rules without performing
    /// DNS resolution or network I/O.
    ///
    /// # Errors
    ///
    /// Returns [`AdfontesError::SafetyBlock`] if `url` violates built-in or configured host policy.
    pub fn validate_url_policy(&self, url: &HttpUrl) -> Result<(), AdfontesError> {
        let allow_loopback = self.is_loopback_target_allowed(url);
        validate_hostname_policy_with_suffixes(
            url,
            allow_loopback,
            &self.config.blocked_host_suffixes,
        )
    }

    fn execute(
        &self,
        method: HttpMethod,
        initial_url: &HttpUrl,
        accept: Option<&str>,
        extra_headers: &[(&str, &str)],
    ) -> Result<HttpResponse, AdfontesError> {
        validate_header_value("User-Agent", &self.config.user_agent, initial_url.as_str())?;
        validate_extra_headers(extra_headers, initial_url.as_str())?;

        let deadline = Instant::now()
            .checked_add(self.config.timeout)
            .unwrap_or_else(Instant::now);
        let initial_origin = url_origin(initial_url);
        let mut current_url = initial_url.clone();
        let mut redirect_chain = Vec::new();

        for hop in 0..=self.config.max_redirects {
            let remaining = remaining_timeout(deadline, current_url.as_str())?;
            let allow_loopback = self.is_loopback_target_allowed(&current_url);
            let proxy_url = self
                .config
                .proxy
                .resolve_proxy_for_url(&current_url, allow_loopback)?;
            let validated_addrs = validate_and_resolve_hop(
                &current_url,
                &self.config,
                allow_loopback,
                proxy_url.is_some(),
                remaining,
            )?;

            let hop_timeout = remaining_timeout(deadline, current_url.as_str())?;
            let agent = build_hop_agent(
                &self.config,
                &self.tls_config,
                validated_addrs,
                proxy_url.as_deref(),
                hop_timeout,
            )?;
            let same_origin = url_origin(&current_url) == initial_origin;
            let hop_headers = filter_hop_headers(extra_headers, same_origin);
            let response = send_single_hop(&agent, method, &current_url, accept, &hop_headers)?;
            let status = response.status().as_u16();
            let raw_headers = response.headers();

            if is_redirect_status(status) {
                let location = extract_single_location_header(raw_headers, current_url.as_str())?;
                if hop >= self.config.max_redirects {
                    return Err(AdfontesError::SafetyBlock {
                        reason: format!(
                            "exceeded maximum of {} HTTP redirects starting from {initial_url}",
                            self.config.max_redirects
                        ),
                    });
                }
                if hop > 0 {
                    redirect_chain.push(current_url.clone());
                }
                current_url = current_url.join_redirect(&location)?;
                continue;
            }

            let content_encodings =
                extract_content_encoding_tokens(raw_headers, current_url.as_str())?;
            let mut headers = collect_response_headers(raw_headers);
            let content_type = headers.get("content-type").cloned();
            let raw_body = match method {
                HttpMethod::Head => Vec::new(),
                HttpMethod::Get => read_bounded_http_body(
                    response.into_body(),
                    self.config.max_response_bytes,
                    current_url.as_str(),
                )?,
            };
            let decoded = decode_content_encoding(
                raw_body,
                &content_encodings,
                content_type.as_deref(),
                self.config.max_response_bytes,
            )?;
            if decoded.was_decoded {
                headers.remove(HEADER_CONTENT_ENCODING);
                headers.remove("content-length");
            }

            return Ok(HttpResponse {
                status,
                redirect_chain,
                final_url: current_url,
                content_type,
                headers,
                body: decoded.bytes,
            });
        }

        Err(AdfontesError::SafetyBlock {
            reason: format!(
                "exceeded maximum of {} HTTP redirects starting from {initial_url}",
                self.config.max_redirects
            ),
        })
    }

    fn is_loopback_target_allowed(&self, url: &HttpUrl) -> bool {
        if self.config.allowed_loopback_addrs.is_empty() {
            return false;
        }
        let port = url.port_or_default();
        match url.host() {
            UrlHost::Ip(ip) => {
                ip.is_loopback()
                    && self
                        .config
                        .allowed_loopback_addrs
                        .contains(&SocketAddr::new(ip, port))
            }
            UrlHost::Domain(domain) => {
                domain.eq_ignore_ascii_case(LOCALHOST_NAME)
                    && self
                        .config
                        .allowed_loopback_addrs
                        .iter()
                        .any(|addr| addr.port() == port && addr.ip().is_loopback())
            }
        }
    }
}

/// Returns `true` if `address` is non-global under an IANA Global Unicast allowlist.
///
/// IPv4 blocks RFC 1918, loopback, link-local, `0.0.0.0/8`, CGNAT `100.64.0.0/10`, IETF
/// `192.0.0.0/24`, deprecated 6to4 relay `192.88.99.0/24`, benchmarking `198.18.0.0/15`,
/// documentation, multicast, broadcast, and Class E `240.0.0.0/4`.
/// IPv6 permits only IANA Global Unicast (`2000::/3`, excluding documentation/benchmark/ORCHID
/// sub-blocks and checking embedded IPv4 in 6to4/Teredo) and NAT64 WKP `64:ff9b::/96` when the
/// embedded IPv4 address is global.
#[must_use]
pub const fn is_forbidden_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(ipv4) => is_forbidden_ipv4(ipv4),
        IpAddr::V6(ipv6) => is_forbidden_ipv6(ipv6),
    }
}

/// Validates a URL's hostname against static SSRF blocklists without performing DNS lookups.
///
/// Rejects single-label domains (which trigger `/etc/resolv.conf` search-domain expansion),
/// local/internal suffixes, metadata services, and wildcard DNS rebinding domains (`nip.io`,
/// `sslip.io`, `xip.io`, `vcap.me`, `traefik.me`, `local.gd`, `rbndr.us`).
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the host violates SSRF policy.
pub fn validate_hostname_policy(url: &HttpUrl, allow_loopback: bool) -> Result<(), AdfontesError> {
    validate_hostname_policy_with_suffixes(url, allow_loopback, &[])
}

/// Validates a URL's hostname against built-in SSRF blocklists plus additional operator-configured
/// host suffixes (each matching both the exact domain and any subdomain).
///
/// # Errors
///
/// Returns [`AdfontesError::SafetyBlock`] if the host violates SSRF policy.
pub fn validate_hostname_policy_with_suffixes(
    url: &HttpUrl,
    allow_loopback: bool,
    extra_blocked_suffixes: &[String],
) -> Result<(), AdfontesError> {
    let mut validated_suffixes = Vec::with_capacity(extra_blocked_suffixes.len());
    for raw in extra_blocked_suffixes {
        validated_suffixes.push(normalize_and_validate_blocked_suffix(raw)?);
    }

    match url.host() {
        UrlHost::Ip(address) => {
            if address.is_loopback() && allow_loopback {
                return Ok(());
            }
            if is_forbidden_ip(address) {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!("direct access to non-global IP `{address}` is blocked"),
                });
            }
        }
        UrlHost::Domain(domain) => {
            let lower = domain.to_ascii_lowercase();
            if lower.is_empty() {
                return Err(AdfontesError::SafetyBlock {
                    reason: "URL host is empty".to_owned(),
                });
            }
            if allow_loopback && lower == LOCALHOST_NAME {
                return Ok(());
            }
            let matches_extra = validated_suffixes
                .iter()
                .any(|suffix| matches_validated_host_suffix(&lower, suffix));
            if FORBIDDEN_EXACT_HOSTS.contains(&lower.as_str())
                || FORBIDDEN_HOST_SUFFIXES
                    .iter()
                    .any(|suffix| lower.ends_with(suffix))
                || matches_extra
            {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!("host `{domain}` is on the SSRF blocklist"),
                });
            }
            if !lower.contains('.') {
                return Err(AdfontesError::SafetyBlock {
                    reason: format!("single-label hostname `{domain}` is blocked"),
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum HttpMethod {
    Get,
    Head,
}

#[derive(Debug, PartialEq, Eq)]
struct UrlOrigin {
    scheme: String,
    host: String,
    port: u16,
}

struct DecodedBody {
    bytes: Vec<u8>,
    was_decoded: bool,
}

fn url_origin(url: &HttpUrl) -> UrlOrigin {
    let host = match url.host() {
        UrlHost::Domain(domain) => domain.to_ascii_lowercase(),
        UrlHost::Ip(ip) => ip.to_string(),
    };
    UrlOrigin {
        scheme: url.scheme().to_owned(),
        host,
        port: url.port_or_default(),
    }
}

fn remaining_timeout(deadline: Instant, url: &str) -> Result<Duration, AdfontesError> {
    let now = Instant::now();
    if now >= deadline {
        return Err(AdfontesError::Transport {
            url: url.to_owned(),
            detail: "HTTP request exceeded end-to-end timeout deadline".to_owned(),
        });
    }
    Ok(deadline.saturating_duration_since(now))
}

fn validate_extra_headers(headers: &[(&str, &str)], url: &str) -> Result<(), AdfontesError> {
    for &(name, value) in headers {
        let valid_name = !name.is_empty()
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
        if !valid_name || name.eq_ignore_ascii_case("host") {
            return Err(AdfontesError::Transport {
                url: url.to_owned(),
                detail: format!("disallowed HTTP header name `{name}`"),
            });
        }
        validate_header_value(name, value, url)?;
    }
    Ok(())
}

fn validate_header_value(name: &str, value: &str, url: &str) -> Result<(), AdfontesError> {
    let valid = !value.is_empty()
        && value.bytes().all(|byte| {
            byte == b'\t' || (ASCII_PRINTABLE_FIRST..=ASCII_PRINTABLE_LAST).contains(&byte)
        });
    if !valid {
        return Err(AdfontesError::Transport {
            url: url.to_owned(),
            detail: format!("invalid HTTP header value for `{name}`"),
        });
    }
    Ok(())
}

fn filter_hop_headers<'a>(
    extra_headers: &'a [(&'a str, &'a str)],
    same_origin: bool,
) -> Vec<(&'a str, &'a str)> {
    if same_origin {
        return extra_headers.to_vec();
    }
    extra_headers
        .iter()
        .copied()
        .filter(|&(name, _)| {
            !SENSITIVE_REDIRECT_HEADERS
                .iter()
                .any(|sensitive| name.eq_ignore_ascii_case(sensitive))
        })
        .collect()
}

fn validate_and_resolve_hop(
    url: &HttpUrl,
    config: &TransportConfig,
    allow_loopback: bool,
    using_proxy: bool,
    timeout: Duration,
) -> Result<Vec<SocketAddr>, AdfontesError> {
    validate_hostname_policy_with_suffixes(url, allow_loopback, &config.blocked_host_suffixes)?;
    let port = url.port_or_default();

    let all_addrs: Vec<SocketAddr> = match url.host() {
        UrlHost::Ip(address) => vec![SocketAddr::new(address, port)],
        UrlHost::Domain(domain) => match resolve_domain_with_timeout(domain, port, timeout) {
            Ok(addresses) => addresses,
            Err(dns_error) => {
                if using_proxy && config.proxy.allow_proxy_remote_dns {
                    return Ok(Vec::new());
                }
                return Err(AdfontesError::Transport {
                    url: url.to_string(),
                    detail: dns_error,
                });
            }
        },
    };

    if all_addrs.is_empty() {
        return Err(AdfontesError::Transport {
            url: url.to_string(),
            detail: format!("DNS returned no addresses for `{}`", url.host_str()),
        });
    }

    for socket_addr in &all_addrs {
        let ip = socket_addr.ip();
        if allow_loopback && ip.is_loopback() && config.allowed_loopback_addrs.contains(socket_addr)
        {
            continue;
        }
        if is_forbidden_ip(ip) {
            return Err(AdfontesError::SafetyBlock {
                reason: format!("host `{}` resolved to non-global IP `{ip}`", url.host_str()),
            });
        }
    }

    Ok(all_addrs
        .into_iter()
        .take(MAX_PINNED_SOCKET_ADDRS)
        .collect())
}

const MAX_CONCURRENT_DNS_LOOKUPS: usize = 16;
static ACTIVE_DNS_LOOKUPS: AtomicUsize = AtomicUsize::new(0);

fn resolve_domain_with_timeout(
    domain: &str,
    port: u16,
    timeout: Duration,
) -> Result<Vec<SocketAddr>, String> {
    let slot_guard = DnsLookupSlot::try_acquire().ok_or_else(|| {
        format!("DNS resolver busy ({MAX_CONCURRENT_DNS_LOOKUPS} concurrent lookups active) for `{domain}`")
    })?;
    let target = (domain.to_owned(), port);
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _guard = slot_guard;
        let lookup = target
            .to_socket_addrs()
            .map(Iterator::collect::<Vec<SocketAddr>>);
        let _ = sender.send(lookup);
    });
    match receiver.recv_timeout(timeout) {
        Ok(Ok(addresses)) => Ok(addresses),
        Ok(Err(io_error)) => Err(format!("DNS resolution failed for `{domain}`: {io_error}")),
        Err(RecvTimeoutError::Timeout) => Err(format!("DNS resolution timed out for `{domain}`")),
        Err(RecvTimeoutError::Disconnected) => {
            Err(format!("DNS resolver worker terminated for `{domain}`"))
        }
    }
}

struct DnsLookupSlot;

impl DnsLookupSlot {
    fn try_acquire() -> Option<Self> {
        let mut current = ACTIVE_DNS_LOOKUPS.load(Ordering::Acquire);
        while current < MAX_CONCURRENT_DNS_LOOKUPS {
            match ACTIVE_DNS_LOOKUPS.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(Self),
                Err(observed) => current = observed,
            }
        }
        None
    }
}

impl Drop for DnsLookupSlot {
    fn drop(&mut self) {
        ACTIVE_DNS_LOOKUPS.fetch_sub(1, Ordering::AcqRel);
    }
}

fn build_hop_agent(
    transport_config: &TransportConfig,
    tls_config: &TlsConfig,
    validated_addrs: Vec<SocketAddr>,
    proxy_url: Option<&str>,
    hop_timeout: Duration,
) -> Result<Agent, AdfontesError> {
    let mut builder = Config::builder()
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_global(Some(hop_timeout))
        .timeout_connect(Some(hop_timeout))
        .user_agent(&transport_config.user_agent)
        .tls_config(tls_config.clone());

    if let Some(proxy_raw) = proxy_url {
        let proxy =
            ureq::Proxy::new(proxy_raw).map_err(|proxy_error| AdfontesError::Transport {
                url: proxy_raw.to_owned(),
                detail: format!("invalid proxy URL: {proxy_error}"),
            })?;
        builder = builder.proxy(Some(proxy));
        return Ok(Agent::new_with_config(builder.build()));
    }

    builder = builder.proxy(None);
    let resolver = PinnedResolver::new(validated_addrs);
    Ok(Agent::with_parts(
        builder.build(),
        DefaultConnector::default(),
        resolver,
    ))
}

#[derive(Debug)]
struct PinnedResolver {
    addresses: Vec<SocketAddr>,
}

impl PinnedResolver {
    const fn new(addresses: Vec<SocketAddr>) -> Self {
        Self { addresses }
    }
}

impl Resolver for PinnedResolver {
    fn resolve(
        &self,
        _uri: &Uri,
        _config: &Config,
        _timeout: NextTimeout,
    ) -> Result<ResolvedSocketAddrs, ureq::Error> {
        let mut resolved = self.empty();
        for &address in self.addresses.iter().take(MAX_PINNED_SOCKET_ADDRS) {
            resolved.push(address);
        }
        if resolved.is_empty() {
            return Err(ureq::Error::HostNotFound);
        }
        Ok(resolved)
    }
}

fn send_single_hop(
    agent: &Agent,
    method: HttpMethod,
    url: &HttpUrl,
    accept: Option<&str>,
    extra_headers: &[(&str, &str)],
) -> Result<ureq::http::Response<ureq::Body>, AdfontesError> {
    match method {
        HttpMethod::Get => {
            let mut request = agent
                .get(url.as_str())
                .header("Accept-Encoding", "gzip, deflate");
            if let Some(accept_value) = accept {
                request = request.header("Accept", accept_value);
            }
            for &(name, value) in extra_headers {
                request = request.header(name, value);
            }
            request
                .call()
                .map_err(|http_error| AdfontesError::Transport {
                    url: url.to_string(),
                    detail: http_error.to_string(),
                })
        }
        HttpMethod::Head => {
            let mut request = agent.head(url.as_str());
            for &(name, value) in extra_headers {
                request = request.header(name, value);
            }
            request
                .call()
                .map_err(|http_error| AdfontesError::Transport {
                    url: url.to_string(),
                    detail: http_error.to_string(),
                })
        }
    }
}

fn extract_single_location_header(
    headers: &ureq::http::HeaderMap,
    url: &str,
) -> Result<String, AdfontesError> {
    let mut locations = Vec::new();
    for raw_value in headers.get_all("location") {
        let text = raw_value.to_str().map_err(|_| AdfontesError::Transport {
            url: url.to_owned(),
            detail: "HTTP redirect Location header contains non-ASCII bytes".to_owned(),
        })?;
        locations.push(text.to_owned());
    }
    match locations.as_slice() {
        [] => Err(AdfontesError::Transport {
            url: url.to_owned(),
            detail: "HTTP redirect response missing Location header".to_owned(),
        }),
        [single] => Ok(single.clone()),
        _ => Err(AdfontesError::SafetyBlock {
            reason: "HTTP redirect response contained multiple Location headers".to_owned(),
        }),
    }
}

fn extract_content_encoding_tokens(
    headers: &ureq::http::HeaderMap,
    url: &str,
) -> Result<Vec<String>, AdfontesError> {
    let mut tokens = Vec::new();
    for raw_value in headers.get_all(HEADER_CONTENT_ENCODING) {
        let text = raw_value.to_str().map_err(|_| AdfontesError::Transport {
            url: url.to_owned(),
            detail: "HTTP Content-Encoding header contains non-ASCII bytes".to_owned(),
        })?;
        for part in text.split(',') {
            let normalized = part.trim().to_ascii_lowercase();
            if !normalized.is_empty() && normalized != "identity" {
                tokens.push(normalized);
            }
        }
    }
    Ok(tokens)
}

fn collect_response_headers(headers: &ureq::http::HeaderMap) -> BTreeMap<String, String> {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in headers {
        if let Ok(text) = value.to_str() {
            let key = name.as_str().to_ascii_lowercase();
            let trimmed = text.trim();
            map.entry(key)
                .and_modify(|existing| {
                    existing.push_str(", ");
                    existing.push_str(trimmed);
                })
                .or_insert_with(|| trimmed.to_owned());
        }
    }
    map
}

fn read_bounded_http_body(
    body: ureq::Body,
    max_bytes: usize,
    url: &str,
) -> Result<Vec<u8>, AdfontesError> {
    if let Some(declared_len) = body.content_length()
        && declared_len > max_bytes as u64
    {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("HTTP Content-Length {declared_len} exceeds {max_bytes}-byte limit"),
        });
    }

    let reader_limit = (max_bytes as u64).saturating_add(1);
    let mut reader = body.into_with_config().limit(reader_limit).reader();
    let mut output = Vec::new();
    let mut chunk = [0u8; BODY_READ_CHUNK_SIZE];
    while output.len() < max_bytes {
        let remaining = max_bytes - output.len();
        let to_read = remaining.min(BODY_READ_CHUNK_SIZE);
        let Some(target) = chunk.get_mut(..to_read) else {
            break;
        };
        let bytes_read = reader
            .read(target)
            .map_err(|io_error| AdfontesError::Transport {
                url: url.to_owned(),
                detail: format!("failed reading HTTP body: {io_error}"),
            })?;
        if bytes_read == 0 {
            return Ok(output);
        }
        if let Some(slice) = target.get(..bytes_read) {
            output.extend_from_slice(slice);
        }
    }

    let mut overflow_probe = [0u8; 1];
    let probe_read =
        reader
            .read(&mut overflow_probe)
            .map_err(|io_error| AdfontesError::Transport {
                url: url.to_owned(),
                detail: format!("failed reading HTTP body: {io_error}"),
            })?;
    if probe_read > 0 {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("HTTP response body exceeded {max_bytes}-byte limit"),
        });
    }
    Ok(output)
}

fn decode_content_encoding(
    raw_body: Vec<u8>,
    encodings: &[String],
    content_type: Option<&str>,
    max_bytes: usize,
) -> Result<DecodedBody, AdfontesError> {
    if raw_body.is_empty() || encodings.is_empty() {
        return Ok(DecodedBody {
            bytes: raw_body,
            was_decoded: false,
        });
    }

    if encodings.len() > MAX_CONTENT_ENCODING_LAYERS {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "HTTP Content-Encoding stacked {} layers (max {MAX_CONTENT_ENCODING_LAYERS})",
                encodings.len()
            ),
        });
    }

    let mut seen_layers = BTreeSet::new();
    for encoding in encodings {
        let canonical = if encoding == ENCODING_X_GZIP {
            ENCODING_GZIP
        } else {
            encoding.as_str()
        };
        if !seen_layers.insert(canonical) {
            return Err(AdfontesError::SafetyBlock {
                reason: format!("duplicate HTTP Content-Encoding layer `{encoding}` is blocked"),
            });
        }
    }

    let media_type = content_type
        .and_then(|value| value.split(';').next())
        .map_or_default(|media| media.trim().to_ascii_lowercase());
    let is_single_gzip =
        matches!(encodings, [single] if single == ENCODING_GZIP || single == ENCODING_X_GZIP);
    if is_single_gzip
        && RAW_ARCHIVE_MIME_TYPES.contains(&media_type.as_str())
        && is_gzip_magic(&raw_body)
    {
        return Ok(DecodedBody {
            bytes: raw_body,
            was_decoded: false,
        });
    }

    let mut current = raw_body;
    for encoding in encodings.iter().rev() {
        current = match encoding.as_str() {
            ENCODING_GZIP | ENCODING_X_GZIP => safe_gunzip_bytes(&current, max_bytes)?,
            "deflate" => safe_inflate_bytes(&current, max_bytes)?,
            other => {
                return Err(AdfontesError::Extraction {
                    format: "HTTP",
                    detail: format!("unsupported Content-Encoding `{other}`"),
                });
            }
        };
    }
    Ok(DecodedBody {
        bytes: current,
        was_decoded: true,
    })
}

const fn is_redirect_status(status: u16) -> bool {
    matches!(
        status,
        HTTP_STATUS_MOVED_PERMANENTLY
            | HTTP_STATUS_FOUND
            | HTTP_STATUS_SEE_OTHER
            | HTTP_STATUS_TEMPORARY_REDIRECT
            | HTTP_STATUS_PERMANENT_REDIRECT
    )
}

const fn is_forbidden_ipv6(ipv6: Ipv6Addr) -> bool {
    let [seg0, seg1, seg2, seg3, seg4, seg5, seg6, seg7] = ipv6.segments();

    // NAT64 well-known prefix (`64:ff9b::/96`) is permitted only when its embedded IPv4 is global.
    if seg0 == IPV6_NAT64_HEAD
        && seg1 == IPV6_NAT64_SECOND
        && seg2 == 0
        && seg3 == 0
        && seg4 == 0
        && seg5 == 0
    {
        return is_forbidden_ipv4(ipv4_from_segments(seg6, seg7));
    }

    // Reject everything outside IANA Global Unicast (`2000::/3`).
    if (seg0 & IPV6_GLOBAL_UNICAST_MASK) != IPV6_GLOBAL_UNICAST_PREFIX {
        return true;
    }

    // RFC 3849 documentation (`2001:db8::/32`) and RFC 9637 documentation (`3fff::/20`).
    let is_rfc3849_doc = seg0 == IPV6_IETF_SPECIAL_HEAD && seg1 == IPV6_DOCUMENTATION_SECOND;
    let is_rfc9637_doc =
        seg0 == IPV6_RFC9637_DOC_HEAD && (seg1 & IPV6_RFC9637_DOC_SECOND_MASK) == 0;
    if is_rfc3849_doc || is_rfc9637_doc {
        return true;
    }

    // Teredo (`2001:0000::/32`) embeds server IPv4 in segments 2..4 and inverted client IPv4 in 6..8.
    if seg0 == IPV6_IETF_SPECIAL_HEAD && seg1 == 0 {
        let server_ip = ipv4_from_segments(seg2, seg3);
        let client_ip = ipv4_from_segments(!seg6, !seg7);
        return is_forbidden_ipv4(server_ip) || is_forbidden_ipv4(client_ip);
    }

    // Non-global IETF protocol assignments (`2001::/23`, covering `2001:2::/48`, ORCHID `2001:10::/28`, etc.).
    if seg0 == IPV6_IETF_SPECIAL_HEAD && (seg1 & IPV6_IETF_PROTOCOL_MASK) == 0 {
        return true;
    }

    // 6to4 (`2002::/16`) embeds an IPv4 address in segments 1 and 2.
    if seg0 == IPV6_SIX_TO_FOUR_HEAD {
        return is_forbidden_ipv4(ipv4_from_segments(seg1, seg2));
    }

    false
}

const fn is_forbidden_ipv4(ipv4: Ipv4Addr) -> bool {
    let [first, second, third, _fourth] = ipv4.octets();
    let is_this_network = first == 0;
    let is_cgnat = first == IPV4_CGNAT_FIRST && (second & IPV4_CGNAT_MASK) == IPV4_CGNAT_SECOND;
    let is_ietf_protocol = first == IPV4_IETF_PROTOCOL_FIRST && second == 0 && third == 0;
    let is_six_to_four_relay = first == IPV4_IETF_PROTOCOL_FIRST
        && second == IPV4_SIX_TO_FOUR_RELAY_SECOND
        && third == IPV4_SIX_TO_FOUR_RELAY_THIRD;
    let is_benchmarking =
        first == IPV4_BENCHMARK_FIRST && (second & IPV4_BENCHMARK_MASK) == IPV4_BENCHMARK_SECOND;
    let is_reserved_class_e = first >= IPV4_CLASS_E_START;

    ipv4.is_private()
        || ipv4.is_loopback()
        || ipv4.is_link_local()
        || ipv4.is_broadcast()
        || ipv4.is_documentation()
        || ipv4.is_multicast()
        || ipv4.is_unspecified()
        || is_this_network
        || is_cgnat
        || is_ietf_protocol
        || is_six_to_four_relay
        || is_benchmarking
        || is_reserved_class_e
}

const fn ipv4_from_segments(high: u16, low: u16) -> Ipv4Addr {
    let [hi0, hi1] = high.to_be_bytes();
    let [lo0, lo1] = low.to_be_bytes();
    Ipv4Addr::new(hi0, hi1, lo0, lo1)
}

fn validate_http_proxy_url(raw_proxy: &str) -> Result<(), AdfontesError> {
    let Some((scheme, rest)) = raw_proxy.split_once("://") else {
        return Err(AdfontesError::Transport {
            url: raw_proxy.to_owned(),
            detail: "proxy URL must begin with `http://` or `https://`".to_owned(),
        });
    };
    if !scheme.eq_ignore_ascii_case(SCHEME_HTTP) && !scheme.eq_ignore_ascii_case(SCHEME_HTTPS) {
        return Err(AdfontesError::Transport {
            url: raw_proxy.to_owned(),
            detail: format!(
                "unsupported proxy scheme `{scheme}` (only http and https proxies are supported)"
            ),
        });
    }
    if rest.is_empty() || raw_proxy.chars().any(char::is_control) {
        return Err(AdfontesError::Transport {
            url: raw_proxy.to_owned(),
            detail: "proxy URL is empty or contains control characters".to_owned(),
        });
    }
    Ok(())
}

fn parse_blocked_host_suffixes(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(str::to_owned)
        .collect()
}

fn normalize_and_validate_blocked_suffix(raw: &str) -> Result<String, AdfontesError> {
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix("*.")
        .unwrap_or(trimmed)
        .trim_matches('.')
        .to_ascii_lowercase();
    let valid_labels = !stripped.is_empty()
        && stripped.split('.').all(|label| {
            !label.is_empty()
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        });
    if !valid_labels {
        return Err(AdfontesError::SafetyBlock {
            reason: format!(
                "invalid ADFONTES_BLOCKED_HOST_SUFFIXES entry `{raw}` (expected DNS host suffix)"
            ),
        });
    }
    Ok(stripped)
}

fn matches_validated_host_suffix(lower_domain: &str, suffix: &str) -> bool {
    if lower_domain == suffix {
        return true;
    }
    lower_domain.len() > suffix.len()
        && lower_domain.ends_with(suffix)
        && lower_domain
            .as_bytes()
            .get(lower_domain.len() - suffix.len() - 1)
            == Some(&b'.')
}

fn first_non_empty_env(primary: &str, fallback: &str) -> Option<String> {
    // omni:ignore [environment-variable-in-function] -- helper invoked exclusively from ProxyConfig::from_env
    std::env::var(primary)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            // omni:ignore [environment-variable-in-function] -- helper invoked exclusively from ProxyConfig::from_env
            std::env::var(fallback)
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
}

fn is_loopback_url(url: &HttpUrl) -> bool {
    match url.host() {
        UrlHost::Ip(ip) => ip.is_loopback(),
        UrlHost::Domain(domain) => domain.eq_ignore_ascii_case(LOCALHOST_NAME),
    }
}

fn matches_no_proxy(url: &HttpUrl, no_proxy: &[String]) -> bool {
    let port = url.port_or_default();
    for raw_rule in no_proxy {
        let Some(entry) = parse_no_proxy_entry(raw_rule) else {
            continue;
        };
        if entry.matches(url.host(), port) {
            return true;
        }
    }
    false
}

enum NoProxyEntry {
    MatchAll,
    Ip { address: IpAddr, port: Option<u16> },
    Cidr { network: IpAddr, prefix_len: u8 },
    Domain { suffix: String, port: Option<u16> },
}

impl NoProxyEntry {
    fn matches(&self, target_host: UrlHost<'_>, target_port: u16) -> bool {
        match self {
            Self::MatchAll => true,
            Self::Ip { address, port } => {
                port.is_none_or(|required| required == target_port)
                    && target_host == UrlHost::Ip(*address)
            }
            Self::Cidr {
                network,
                prefix_len,
            } => match target_host {
                UrlHost::Ip(target_ip) => cidr_contains(*network, *prefix_len, target_ip),
                UrlHost::Domain(_) => false,
            },
            Self::Domain { suffix, port } => {
                if !port.is_none_or(|required| required == target_port) {
                    return false;
                }
                let UrlHost::Domain(domain) = target_host else {
                    return false;
                };
                let lower_domain = domain.to_ascii_lowercase();
                lower_domain == *suffix
                    || lower_domain
                        .strip_suffix(suffix.as_str())
                        .is_some_and(|prefix| prefix.ends_with('.'))
            }
        }
    }
}

fn parse_no_proxy_entry(raw_rule: &str) -> Option<NoProxyEntry> {
    let trimmed = raw_rule.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == "*" {
        return Some(NoProxyEntry::MatchAll);
    }

    if let Some((ip_part, prefix_part)) = trimmed.split_once('/') {
        let network: IpAddr = ip_part.trim().parse().ok()?;
        let prefix_len: u8 = prefix_part.trim().parse().ok()?;
        let max_prefix = match network {
            IpAddr::V4(_) => MAX_IPV4_PREFIX_BITS,
            IpAddr::V6(_) => MAX_IPV6_PREFIX_BITS,
        };
        if prefix_len <= max_prefix {
            return Some(NoProxyEntry::Cidr {
                network,
                prefix_len,
            });
        }
        return None;
    }

    if let Some(rest) = trimmed.strip_prefix('[') {
        let (ipv6_text, tail) = rest.split_once(']')?;
        let address = IpAddr::V6(ipv6_text.parse::<Ipv6Addr>().ok()?);
        let port = if tail.is_empty() {
            None
        } else {
            let digits = tail.strip_prefix(':')?;
            Some(digits.parse::<u16>().ok()?)
        };
        return Some(NoProxyEntry::Ip { address, port });
    }

    if let Ok(address) = trimmed.parse::<IpAddr>() {
        return Some(NoProxyEntry::Ip {
            address,
            port: None,
        });
    }

    let (host_part, port) = if let Some((left, right)) = trimmed.rsplit_once(':')
        && !left.contains(':')
    {
        (left, Some(right.parse::<u16>().ok()?))
    } else {
        (trimmed, None)
    };

    if let Ok(address) = host_part.parse::<IpAddr>() {
        return Some(NoProxyEntry::Ip { address, port });
    }

    let normalized = host_part
        .strip_prefix("*.")
        .unwrap_or(host_part)
        .trim_start_matches('.')
        .trim_end_matches('.')
        .to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    Some(NoProxyEntry::Domain {
        suffix: normalized,
        port,
    })
}

fn cidr_contains(network: IpAddr, prefix_len: u8, target: IpAddr) -> bool {
    match (network, target) {
        (IpAddr::V4(net_v4), IpAddr::V4(target_v4)) => {
            if prefix_len == 0 {
                return true;
            }
            let mask = u32::MAX << (MAX_IPV4_PREFIX_BITS - prefix_len);
            (u32::from(net_v4) & mask) == (u32::from(target_v4) & mask)
        }
        (IpAddr::V6(net_v6), IpAddr::V6(target_v6)) => {
            if prefix_len == 0 {
                return true;
            }
            let mask = u128::MAX << (MAX_IPV6_PREFIX_BITS - prefix_len);
            (u128::from(net_v6) & mask) == (u128::from(target_v6) & mask)
        }
        _ => false,
    }
}

fn extract_charset_label(content_type: &str) -> Option<&str> {
    for parameter in content_type.split(';').skip(1) {
        let Some((key, value)) = parameter.split_once('=') else {
            continue;
        };
        if key.trim().eq_ignore_ascii_case("charset") {
            let trimmed = value.trim().trim_matches('"').trim_matches('\'');
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::thread;

    use rstest::rstest;

    use super::*;
    use crate::archive::{test_gzip_bytes, test_zlib_bytes};

    struct TestResponse {
        status_line: &'static str,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    struct CapturedRequest {
        path: String,
        headers: BTreeMap<String, String>,
    }

    fn spawn_loopback_server<F>(
        max_requests: usize,
        handler: F,
    ) -> (SocketAddr, thread::JoinHandle<()>)
    where
        F: Fn(&CapturedRequest) -> TestResponse + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let bound_addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            for _ in 0..max_requests {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let request = {
                    let mut reader = BufReader::new(&stream);
                    let mut line = String::new();
                    let _ = reader.read_line(&mut line);
                    let path = line.split_whitespace().nth(1).unwrap_or("/").to_owned();
                    let mut headers = BTreeMap::new();
                    let mut header_line = String::new();
                    while reader.read_line(&mut header_line).unwrap_or(0) > 0 {
                        if header_line == "\r\n" || header_line == "\n" {
                            break;
                        }
                        if let Some((key, value)) = header_line.split_once(':') {
                            headers
                                .insert(key.trim().to_ascii_lowercase(), value.trim().to_owned());
                        }
                        header_line.clear();
                    }
                    CapturedRequest { path, headers }
                };
                let response = handler(&request);
                let mut raw = String::new();
                raw.push_str("HTTP/1.1 ");
                raw.push_str(response.status_line);
                raw.push_str("\r\nContent-Length: ");
                raw.push_str(&response.body.len().to_string());
                raw.push_str("\r\nConnection: close\r\n");
                for (name, value) in &response.headers {
                    raw.push_str(name);
                    raw.push_str(": ");
                    raw.push_str(value);
                    raw.push_str("\r\n");
                }
                raw.push_str("\r\n");
                let _ = stream.write_all(raw.as_bytes());
                let _ = stream.write_all(&response.body);
                let _ = stream.flush();
            }
        });
        (bound_addr, handle)
    }

    fn loopback_client(allowed_addrs: &[SocketAddr], max_response_bytes: usize) -> HttpClient {
        let mut addrs = BTreeSet::new();
        for &addr in allowed_addrs {
            addrs.insert(addr);
        }
        HttpClient::new(TransportConfig {
            timeout: Duration::from_secs(5),
            max_response_bytes,
            allowed_loopback_addrs: addrs,
            ..TransportConfig::default()
        })
    }

    #[rstest]
    #[case("127.0.0.1", true)]
    #[case("10.0.0.1", true)]
    #[case("169.254.169.254", true)]
    #[case("100.64.0.1", true)]
    #[case("192.0.0.1", true)]
    #[case("192.88.99.1", true)]
    #[case("198.18.0.1", true)]
    #[case("240.0.0.1", true)]
    #[case("0.0.0.1", true)]
    #[case("::1", true)]
    #[case("::", true)]
    #[case("fc00::1", true)]
    #[case("fe80::1", true)]
    #[case("fec0::1", true)]
    #[case("ff02::1", true)]
    #[case("::ffff:127.0.0.1", true)]
    #[case("::127.0.0.1", true)]
    #[case("0:0:0:0:1::1", true)]
    #[case("4000::1", true)]
    #[case("3fff::1", true)]
    #[case("2001:db8::1", true)]
    #[case("2001:2::1", true)]
    #[case("2001:10::1", true)]
    #[case("2002:7f00:0001::1", true)]
    #[case("2001:0000:4136:e378:8000:63bf:a247:27dd", false)]
    #[case("2001:0000:4136:e378:8000:63bf:3fff:fdd2", true)]
    #[case("2001:0000:0a00:0001:8000:63bf:3fff:fdd2", true)]
    #[case("64:ff9b::7f00:1", true)]
    #[case("64:ff9b::5db8:d822", false)]
    #[case("64:ff9b:1::0808:0808", true)]
    #[case("64:ff9b:0:0:0:1:7f00:1", true)]
    #[case("93.184.216.34", false)]
    #[case("2606:2800:220:1:248:1893:25c8:1946", false)]
    fn classifies_forbidden_and_global_ips(#[case] raw_ip: &str, #[case] expected_forbidden: bool) {
        let parsed: IpAddr = raw_ip.parse().unwrap();
        assert_eq!(is_forbidden_ip(parsed), expected_forbidden);
    }

    #[rstest]
    #[case("http://localhost/paper.pdf")]
    #[case("http://localhost./paper.pdf")]
    #[case("http://ip6-localhost/paper.pdf")]
    #[case("http://intranet/paper.pdf")]
    #[case("http://metadata.cloud.internal/computeMetadata/v1/")]
    #[case("https://wiki.internal.corp/doc")]
    #[case("https://service.intranet/doc")]
    #[case("https://printer.lan/doc")]
    #[case("https://router.home/doc")]
    #[case("http://127.0.0.1.nip.io/paper.pdf")]
    #[case("http://app-7f000001.nip.io/paper.pdf")]
    #[case("http://--1.sslip.io/paper.pdf")]
    #[case("http://2130706433/paper.pdf")]
    #[case("http://0x7f000001/paper.pdf")]
    fn blocks_ssrf_hostnames_rebind_services_and_single_labels(#[case] raw_url: &str) {
        let url = HttpUrl::parse(raw_url).unwrap();
        assert!(matches!(
            validate_hostname_policy(&url, false).unwrap_err(),
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[rstest]
    #[case("https://corp.example.com/doc", true)]
    #[case("https://docs.corp.example.com/doc", true)]
    #[case("https://sub.private.example.org/spec", true)]
    #[case("https://notcorp.example.com/paper", false)]
    fn blocks_configured_host_suffixes_for_exact_and_subdomains_only(
        #[case] raw_url: &str,
        #[case] expected_blocked: bool,
    ) {
        let suffixes =
            parse_blocked_host_suffixes(" *.CORP.example.com. , .private.example.org , ");
        let url = HttpUrl::parse(raw_url).unwrap();

        let result = validate_hostname_policy_with_suffixes(&url, false, &suffixes);

        assert_eq!(
            matches!(result, Err(AdfontesError::SafetyBlock { .. })),
            expected_blocked
        );
    }

    #[rstest]
    #[case("*")]
    #[case("https://corp.example.com")]
    #[case("corp.*.example.com")]
    #[case("corp.example.com:443")]
    #[case("corp..example.com")]
    fn rejects_malformed_blocked_host_suffix_entries_fail_closed(#[case] bad_entry: &str) {
        let suffixes = parse_blocked_host_suffixes(bad_entry);
        let url = HttpUrl::parse("https://example.org/paper").unwrap();

        let error = validate_hostname_policy_with_suffixes(&url, false, &suffixes).unwrap_err();

        assert!(matches!(error, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn rejects_socks_proxy_url_cleanly_without_panicking() {
        let config = ProxyConfig {
            all_proxy: Some("socks5h://127.0.0.1:1080".to_owned()),
            ..ProxyConfig::none()
        };
        let url = HttpUrl::parse("https://example.org/paper.pdf").unwrap();
        assert!(matches!(
            config.resolve_proxy_for_url(&url, false).unwrap_err(),
            AdfontesError::Transport { .. }
        ));
    }

    #[test]
    fn resolves_proxies_per_scheme_and_matches_cidr_wildcard_and_ipv6_no_proxy() {
        let config = ProxyConfig {
            http_proxy: Some("http://proxy-http.example:8080".to_owned()),
            https_proxy: Some("http://proxy-https.example:8443".to_owned()),
            all_proxy: None,
            no_proxy: vec![
                "*.ArXiv.org.".to_owned(),
                "10.0.0.0/8".to_owned(),
                "[0:0:0:0:0:0:0:1]:8080".to_owned(),
                "[::1]:badport".to_owned(),
                "0.1".to_owned(),
            ],
            allow_proxy_remote_dns: false,
        };
        let excluded_domain = HttpUrl::parse("https://export.arxiv.org/pdf/2409.13740").unwrap();
        let excluded_cidr = HttpUrl::parse("http://10.1.2.3/x").unwrap();
        let excluded_ipv6 = HttpUrl::parse("http://[::1]:8080/x").unwrap();
        let non_matching_ipv4 = HttpUrl::parse("http://127.0.0.1/x").unwrap();

        assert_eq!(
            config
                .resolve_proxy_for_url(&excluded_domain, false)
                .unwrap(),
            None
        );
        assert_eq!(
            config.resolve_proxy_for_url(&excluded_cidr, false).unwrap(),
            None
        );
        assert_eq!(
            config.resolve_proxy_for_url(&excluded_ipv6, false).unwrap(),
            None
        );
        assert_eq!(
            config
                .resolve_proxy_for_url(&non_matching_ipv4, false)
                .unwrap()
                .as_deref(),
            Some("http://proxy-http.example:8080")
        );
    }

    #[test]
    fn strips_authorization_header_across_cross_origin_redirects() {
        let (dest_addr, dest_handle) = spawn_loopback_server(1, |request| {
            let body = if request.headers.contains_key("authorization") {
                b"LEAKED".to_vec()
            } else {
                b"STRIPPED".to_vec()
            };
            TestResponse {
                status_line: "200 OK",
                headers: vec![("Content-Type".to_owned(), "text/plain".to_owned())],
                body,
            }
        });
        let dest_url = format!("http://{dest_addr}/final");
        let (start_addr, start_handle) = spawn_loopback_server(1, move |request| {
            let status_line = if request.headers.get("authorization").map(String::as_str)
                == Some("Bearer secret")
            {
                "302 Found"
            } else {
                "401 Unauthorized"
            };
            TestResponse {
                status_line,
                headers: vec![("Location".to_owned(), dest_url.clone())],
                body: Vec::new(),
            }
        });

        let client = loopback_client(&[start_addr, dest_addr], 4096);
        let url = HttpUrl::parse(&format!("http://{start_addr}/start")).unwrap();
        let response = client
            .get(&url, None, &[("Authorization", "Bearer secret")])
            .unwrap();
        start_handle.join().unwrap();
        dest_handle.join().unwrap();

        assert_eq!(response.final_url.path(), "/final");
        assert_eq!(response.decode_text(), "STRIPPED");
    }

    #[test]
    fn decodes_deflate_with_windows_1252_charset_and_clears_content_encoding() {
        let latin1_bytes = encoding_rs::WINDOWS_1252.encode("Café").0.into_owned();
        let compressed = test_zlib_bytes(&latin1_bytes);
        let (addr, handle) = spawn_loopback_server(1, move |_| TestResponse {
            status_line: "200 OK",
            headers: vec![
                (
                    "Content-Type".to_owned(),
                    "text/plain; charset=windows-1252".to_owned(),
                ),
                ("Content-Encoding".to_owned(), "deflate".to_owned()),
            ],
            body: compressed.clone(),
        });

        let client = loopback_client(&[addr], 4096);
        let url = HttpUrl::parse(&format!("http://{addr}/latin1")).unwrap();
        let response = client.get(&url, None, &[]).unwrap();
        handle.join().unwrap();

        assert_eq!(response.decode_text(), "Café");
        assert!(!response.headers.contains_key("content-encoding"));
    }

    #[test]
    fn preserves_raw_gzip_bytes_for_archive_mime_types_only() {
        let gz_payload = test_gzip_bytes(b"\\documentclass{article}");
        let expected = gz_payload.clone();

        let (addr, handle) = spawn_loopback_server(1, move |_| TestResponse {
            status_line: "200 OK",
            headers: vec![
                ("Content-Type".to_owned(), "application/x-gzip".to_owned()),
                ("Content-Encoding".to_owned(), "x-gzip".to_owned()),
            ],
            body: gz_payload.clone(),
        });

        let client = loopback_client(&[addr], 4096);
        let url = HttpUrl::parse(&format!("http://{addr}/src/2409.13740")).unwrap();
        let response = client.get(&url, None, &[]).unwrap();
        handle.join().unwrap();

        assert_eq!(response.body, expected);
    }

    #[test]
    fn blocks_stacked_duplicate_content_encoding_and_oversized_uncompressed_body() {
        let double_gz = test_gzip_bytes(&test_gzip_bytes(b"nested"));
        let (addr, handle) = spawn_loopback_server(2, move |request| {
            if request.path == "/stacked" {
                TestResponse {
                    status_line: "200 OK",
                    headers: vec![
                        ("Content-Type".to_owned(), "text/plain".to_owned()),
                        ("Content-Encoding".to_owned(), "gzip".to_owned()),
                        ("Content-Encoding".to_owned(), "gzip".to_owned()),
                    ],
                    body: double_gz.clone(),
                }
            } else {
                TestResponse {
                    status_line: "200 OK",
                    headers: vec![("Content-Type".to_owned(), "text/plain".to_owned())],
                    body: vec![b'X'; 512],
                }
            }
        });

        let client = loopback_client(&[addr], 256);
        let stacked_url = HttpUrl::parse(&format!("http://{addr}/stacked")).unwrap();
        let big_url = HttpUrl::parse(&format!("http://{addr}/big")).unwrap();
        let stacked_failure = client.get(&stacked_url, None, &[]).unwrap_err();
        let oversized_failure = client.get(&big_url, None, &[]).unwrap_err();
        handle.join().unwrap();

        assert!(matches!(stacked_failure, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(
            oversized_failure,
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[test]
    fn blocks_gzip_bomb_over_loopback() {
        let bomb_bytes = test_gzip_bytes(&vec![b'B'; 4096]);
        let (addr, handle) = spawn_loopback_server(1, move |_| TestResponse {
            status_line: "200 OK",
            headers: vec![
                ("Content-Type".to_owned(), "text/html".to_owned()),
                ("Content-Encoding".to_owned(), "gzip".to_owned()),
            ],
            body: bomb_bytes.clone(),
        });

        let client = loopback_client(&[addr], 256);
        let url = HttpUrl::parse(&format!("http://{addr}/bomb")).unwrap();
        let failure = client.get(&url, None, &[]).unwrap_err();
        handle.join().unwrap();

        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn blocks_redirect_to_disallowed_loopback_address_and_sibling_127_ip() {
        let (addr, handle) = spawn_loopback_server(1, |_| TestResponse {
            status_line: "302 Found",
            headers: vec![(
                "Location".to_owned(),
                "http://127.0.0.2:8080/internal".to_owned(),
            )],
            body: Vec::new(),
        });

        let client = loopback_client(&[addr], 4096);
        let url = HttpUrl::parse(&format!("http://{addr}/hop")).unwrap();
        let failure = client.get(&url, None, &[]).unwrap_err();
        handle.join().unwrap();

        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[test]
    fn preserves_literal_second_bom_in_decode_text() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice("\u{FEFF}body".as_bytes());
        let response = HttpResponse {
            status: 200,
            redirect_chain: Vec::new(),
            final_url: HttpUrl::parse("https://example.org/doc").unwrap(),
            content_type: Some("text/plain".to_owned()),
            headers: BTreeMap::new(),
            body: bytes,
        };
        assert_eq!(response.decode_text(), "\u{FEFF}body");
    }
}
