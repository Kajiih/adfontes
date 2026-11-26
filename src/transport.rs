//! SSRF-pinned asynchronous `GET` HTTP(S) transport and WHATWG charset decoding.
//!
//! Encapsulates `reqwest` (`rustls` with `ring` + OS platform verifier) and `encoding_rs` behind
//! hop-by-hop redirect validation, DNS/IP pinning against rebinding attacks, strict
//! scheme-specific HTTP(S) proxy selection, end-to-end request deadlines, bounded body streaming,
//! and safe `gzip`/`deflate` content decoding.

architecture_component!(Transport);

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error as StdError;
use std::fmt;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use encoding_rs::Encoding;
use ipnet::{IpNet, Ipv4Net, Ipv6Net};
use mime::Mime;
use reqwest::StatusCode;
use reqwest::header::{
    ACCEPT, ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, HOST, HeaderMap,
    HeaderName, HeaderValue, LOCATION,
};
use tokio::sync::Semaphore;

use crate::archive::{is_gzip_magic, safe_gunzip_bytes, safe_inflate_bytes};
use crate::model::{AdfontesError, HttpUrl, UrlHost};

/// Default end-to-end HTTP request timeout (`20 s`, shared across redirect hops).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// Default maximum HTTP redirect hops followed per request.
pub const DEFAULT_MAX_REDIRECTS: usize = 6;

/// Default maximum response body size (`25 MiB`, covering large open-access academic PDFs).
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 25 * 1024 * 1024;

/// Default descriptive `User-Agent` sent on outbound scholarly API requests.
pub const DEFAULT_USER_AGENT: &str = concat!(
    "adfontes/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/Kajiih/adfontes)"
);

const MAX_PINNED_SOCKET_ADDRS: usize = 16;
const MAX_CONTENT_ENCODING_LAYERS: usize = 2;
const LOCALHOST_NAME: &str = "localhost";
const SCHEME_HTTP: &str = "http";
const SCHEME_HTTPS: &str = "https";
const ENCODING_GZIP: &str = "gzip";
const ENCODING_X_GZIP: &str = "x-gzip";

const FORBIDDEN_IPV4_NETS: [Ipv4Net; 6] = [
    Ipv4Net::new_assert(Ipv4Addr::UNSPECIFIED, 8),
    Ipv4Net::new_assert(Ipv4Addr::new(100, 64, 0, 0), 10),
    Ipv4Net::new_assert(Ipv4Addr::new(192, 0, 0, 0), 24),
    Ipv4Net::new_assert(Ipv4Addr::new(192, 88, 99, 0), 24),
    Ipv4Net::new_assert(Ipv4Addr::new(198, 18, 0, 0), 15),
    Ipv4Net::new_assert(Ipv4Addr::new(240, 0, 0, 0), 4),
];

const IPV6_NAT64_WKP: Ipv6Net =
    Ipv6Net::new_assert(Ipv6Addr::new(0x0064, 0xff9b, 0, 0, 0, 0, 0, 0), 96);
const IPV6_GLOBAL_UNICAST: Ipv6Net =
    Ipv6Net::new_assert(Ipv6Addr::new(0x2000, 0, 0, 0, 0, 0, 0, 0), 3);
const IPV6_TEREDO: Ipv6Net = Ipv6Net::new_assert(Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0), 32);
const IPV6_SIX_TO_FOUR: Ipv6Net =
    Ipv6Net::new_assert(Ipv6Addr::new(0x2002, 0, 0, 0, 0, 0, 0, 0), 16);
const FORBIDDEN_IPV6_NETS: [Ipv6Net; 3] = [
    Ipv6Net::new_assert(Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0, 0), 32),
    Ipv6Net::new_assert(Ipv6Addr::new(0x3fff, 0, 0, 0, 0, 0, 0, 0), 20),
    Ipv6Net::new_assert(Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0), 23),
];

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
    /// `reqwest` without the `socks` feature enabled.
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
            .map_or_default(parse_media_type)
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
            .map_or(encoding_rs::UTF_8, encoding_from_content_type);
        let (decoded, _, _) = encoding.decode(&self.body);
        decoded.into_owned()
    }
}

/// Asynchronous SSRF-pinned HTTP(S) client supporting only read-only `GET` requests.
#[derive(Debug, Clone)]
pub struct HttpClient {
    config: TransportConfig,
    tls_config: Arc<Result<rustls::ClientConfig, String>>,
    direct_client: Result<reqwest::Client, String>,
    proxy_clients: Arc<Mutex<HashMap<String, reqwest::Client>>>,
}

impl HttpClient {
    /// Constructs a new [`HttpClient`] caching the OS `PlatformVerifier` TLS configuration and
    /// SSRF-pinned HTTP connection pools.
    #[must_use]
    pub fn new(config: TransportConfig) -> Self {
        let tls_config = Arc::new(build_platform_tls_config());
        let direct_client = build_direct_client(&config, &tls_config);
        Self {
            config,
            tls_config,
            direct_client,
            proxy_clients: Arc::new(Mutex::new(HashMap::new())),
        }
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
    pub async fn get(
        &self,
        url: &HttpUrl,
        accept: Option<&str>,
        extra_headers: &[(&str, &str)],
    ) -> Result<HttpResponse, AdfontesError> {
        self.execute(url, accept, extra_headers).await
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

    async fn execute(
        &self,
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
        let mut same_origin = true;
        let mut current_url = initial_url.clone();
        let mut redirect_chain = Vec::new();

        for hop in 0..=self.config.max_redirects {
            same_origin &= url_origin(&current_url) == initial_origin;
            let remaining = remaining_timeout(deadline, current_url.as_str())?;
            let allow_loopback = self.is_loopback_target_allowed(&current_url);
            validate_hostname_policy_with_suffixes(
                &current_url,
                allow_loopback,
                &self.config.blocked_host_suffixes,
            )?;
            let proxy_url = self
                .config
                .proxy
                .resolve_proxy_for_url(&current_url, allow_loopback)?;
            if proxy_url.is_some() {
                validate_proxied_target_dns(&current_url, &self.config, remaining).await?;
            }

            let hop_timeout = remaining_timeout(deadline, current_url.as_str())?;
            let client = self.client_for_hop(&current_url, proxy_url.as_deref())?;
            let hop_headers = filter_hop_headers(extra_headers, same_origin);
            let response =
                send_single_hop(&client, &current_url, accept, &hop_headers, hop_timeout).await?;
            let response_status = response.status();
            let status = response_status.as_u16();

            if is_redirect_status(response_status) {
                let location =
                    extract_single_location_header(response.headers(), current_url.as_str())?;
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
                extract_content_encoding_tokens(response.headers(), current_url.as_str())?;
            let mut headers = collect_response_headers(response.headers());
            let content_type = headers.get(CONTENT_TYPE.as_str()).cloned();
            let raw_body = read_bounded_http_body(
                response,
                self.config.max_response_bytes,
                current_url.as_str(),
            )
            .await?;
            let decoded = decode_content_encoding(
                raw_body,
                &content_encodings,
                content_type.as_deref(),
                self.config.max_response_bytes,
            )?;
            if decoded.was_decoded {
                headers.remove(CONTENT_ENCODING.as_str());
                headers.remove(CONTENT_LENGTH.as_str());
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

    fn client_for_hop(
        &self,
        url: &HttpUrl,
        proxy_url: Option<&str>,
    ) -> Result<reqwest::Client, AdfontesError> {
        let Some(proxy_raw) = proxy_url else {
            return self
                .direct_client
                .clone()
                .map_err(|detail| AdfontesError::Transport {
                    url: url.to_string(),
                    detail,
                });
        };
        let mut cache = self
            .proxy_clients
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(client) = cache.get(proxy_raw) {
            return Ok(client.clone());
        }
        let client = build_proxy_client(&self.config, &self.tls_config, proxy_raw, url)?;
        cache.insert(proxy_raw.to_owned(), client.clone());
        drop(cache);
        Ok(client)
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
                domain
                    .trim_end_matches('.')
                    .eq_ignore_ascii_case(LOCALHOST_NAME)
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
pub fn is_forbidden_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(ipv4) => is_forbidden_ipv4(ipv4),
        IpAddr::V6(ipv6) => is_forbidden_ipv6(ipv6),
    }
}

/// Validates a URL's hostname against built-in and configured SSRF suffix blocklists.
///
/// Built-in checks reject `localhost`, metadata services, single-label domains, and wildcard DNS
/// rebinding domains; additional operator-configured suffixes match both exact domains and
/// subdomains.
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
            && HeaderName::from_bytes(name.as_bytes()).is_ok_and(|parsed| parsed != HOST);
        if !valid_name {
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
    if value.is_empty() || !value.is_ascii() || HeaderValue::from_str(value).is_err() {
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

async fn validate_proxied_target_dns(
    url: &HttpUrl,
    config: &TransportConfig,
    timeout: Duration,
) -> Result<(), AdfontesError> {
    let UrlHost::Domain(domain) = url.host() else {
        return Ok(());
    };
    match resolve_and_validate_host(
        domain,
        url.port_or_default(),
        false,
        &config.allowed_loopback_addrs,
        timeout,
    )
    .await
    {
        Ok(_) => Ok(()),
        Err(HostResolutionError::SafetyBlock(reason)) => Err(AdfontesError::SafetyBlock { reason }),
        Err(HostResolutionError::Transport(_)) if config.proxy.allow_proxy_remote_dns => Ok(()),
        Err(HostResolutionError::Transport(detail)) => Err(AdfontesError::Transport {
            url: url.to_string(),
            detail,
        }),
    }
}

fn build_platform_tls_config() -> Result<rustls::ClientConfig, String> {
    use rustls_platform_verifier::BuilderVerifierExt;

    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut tls_config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|error| format!("failed to configure TLS protocol versions: {error}"))?
        .with_platform_verifier()
        .map_err(|error| format!("failed to initialize platform TLS verifier: {error}"))?
        .with_no_client_auth();
    tls_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(tls_config)
}

fn build_direct_client(
    transport_config: &TransportConfig,
    tls_config: &Result<rustls::ClientConfig, String>,
) -> Result<reqwest::Client, String> {
    let builder = base_client_builder(transport_config, tls_config)?
        .no_proxy()
        .dns_resolver(Arc::new(SsrfDnsResolver::new(transport_config)));
    finish_client_builder(builder)
}

fn build_proxy_client(
    transport_config: &TransportConfig,
    tls_config: &Result<rustls::ClientConfig, String>,
    proxy_raw: &str,
    url: &HttpUrl,
) -> Result<reqwest::Client, AdfontesError> {
    let proxy = reqwest::Proxy::all(proxy_raw).map_err(|proxy_error| AdfontesError::Transport {
        url: proxy_raw.to_owned(),
        detail: format!("invalid proxy URL: {proxy_error}"),
    })?;
    let builder = base_client_builder(transport_config, tls_config)
        .map_err(|detail| AdfontesError::Transport {
            url: url.to_string(),
            detail,
        })?
        .proxy(proxy);
    finish_client_builder(builder).map_err(|detail| AdfontesError::Transport {
        url: url.to_string(),
        detail,
    })
}

fn base_client_builder(
    transport_config: &TransportConfig,
    tls_config: &Result<rustls::ClientConfig, String>,
) -> Result<reqwest::ClientBuilder, String> {
    let built_tls = tls_config.as_ref().map_err(Clone::clone)?;
    Ok(reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(transport_config.timeout)
        .connect_timeout(transport_config.timeout)
        .user_agent(&transport_config.user_agent)
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .tls_backend_preconfigured(built_tls.clone()))
}

fn finish_client_builder(builder: reqwest::ClientBuilder) -> Result<reqwest::Client, String> {
    builder
        .build()
        .map_err(|client_error| format!("failed to build HTTP client: {client_error}"))
}

#[derive(Debug)]
struct SsrfDnsResolver {
    timeout: Duration,
    allowed_loopback_addrs: Arc<BTreeSet<SocketAddr>>,
}

impl SsrfDnsResolver {
    fn new(config: &TransportConfig) -> Self {
        Self {
            timeout: config.timeout,
            allowed_loopback_addrs: Arc::new(config.allowed_loopback_addrs.clone()),
        }
    }
}

impl reqwest::dns::Resolve for SsrfDnsResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().trim_end_matches('.').to_owned();
        let timeout = self.timeout;
        let allowed_loopback_addrs = Arc::clone(&self.allowed_loopback_addrs);
        Box::pin(async move {
            let allow_localhost = host.eq_ignore_ascii_case(LOCALHOST_NAME);
            match resolve_and_validate_host(
                &host,
                0,
                allow_localhost,
                &allowed_loopback_addrs,
                timeout,
            )
            .await
            {
                Ok(addrs) => Ok(Box::new(addrs.into_iter()) as reqwest::dns::Addrs),
                Err(HostResolutionError::SafetyBlock(reason)) => {
                    Err(Box::new(SsrfBlockedError { reason }) as Box<dyn StdError + Send + Sync>)
                }
                Err(HostResolutionError::Transport(detail)) => {
                    Err(Box::new(io::Error::other(detail)) as Box<dyn StdError + Send + Sync>)
                }
            }
        })
    }
}

enum HostResolutionError {
    SafetyBlock(String),
    Transport(String),
}

async fn resolve_and_validate_host(
    host: &str,
    port: u16,
    allow_localhost_loopback: bool,
    allowed_loopback_addrs: &BTreeSet<SocketAddr>,
    timeout: Duration,
) -> Result<Vec<SocketAddr>, HostResolutionError> {
    let all_addrs = resolve_domain_with_timeout(host, port, timeout)
        .await
        .map_err(HostResolutionError::Transport)?;
    if all_addrs.is_empty() {
        return Err(HostResolutionError::Transport(format!(
            "DNS returned no addresses for `{host}`"
        )));
    }

    if allow_localhost_loopback {
        let mut allowed_addrs = Vec::new();
        for socket_addr in &all_addrs {
            let ip = socket_addr.ip();
            if !ip.is_loopback() && is_forbidden_ip(ip) {
                return Err(non_global_ip_block(host, ip));
            }
            if ip.is_loopback()
                && allowed_loopback_addrs
                    .iter()
                    .any(|allowed| allowed.ip() == ip && (port == 0 || allowed.port() == port))
            {
                allowed_addrs.push(*socket_addr);
            }
        }
        if allowed_addrs.is_empty() {
            return Err(HostResolutionError::SafetyBlock(format!(
                "host `{host}` resolved to non-allowlisted loopback address"
            )));
        }
        allowed_addrs.truncate(MAX_PINNED_SOCKET_ADDRS);
        return Ok(allowed_addrs);
    }

    for socket_addr in &all_addrs {
        let ip = socket_addr.ip();
        if is_forbidden_ip(ip) {
            return Err(non_global_ip_block(host, ip));
        }
    }

    Ok(all_addrs
        .into_iter()
        .take(MAX_PINNED_SOCKET_ADDRS)
        .collect())
}

fn non_global_ip_block(host: &str, ip: IpAddr) -> HostResolutionError {
    HostResolutionError::SafetyBlock(format!("host `{host}` resolved to non-global IP `{ip}`"))
}

const MAX_CONCURRENT_DNS_LOOKUPS: usize = 16;
static DNS_LOOKUP_SEMAPHORE: Semaphore = Semaphore::const_new(MAX_CONCURRENT_DNS_LOOKUPS);

async fn resolve_domain_with_timeout(
    domain: &str,
    port: u16,
    timeout: Duration,
) -> Result<Vec<SocketAddr>, String> {
    tokio::time::timeout(timeout, async move {
        let _permit = DNS_LOOKUP_SEMAPHORE
            .acquire()
            .await
            .map_err(|_| format!("DNS resolver semaphore closed for `{domain}`"))?;
        let addrs = tokio::net::lookup_host((domain, port))
            .await
            .map_err(|io_error| format!("DNS resolution failed for `{domain}`: {io_error}"))?;
        Ok(addrs.collect())
    })
    .await
    .unwrap_or_else(|_| Err(format!("DNS resolution timed out for `{domain}`")))
}

async fn send_single_hop(
    client: &reqwest::Client,
    url: &HttpUrl,
    accept: Option<&str>,
    extra_headers: &[(&str, &str)],
    hop_timeout: Duration,
) -> Result<reqwest::Response, AdfontesError> {
    let mut request = client
        .get(url.as_str())
        .timeout(hop_timeout)
        .header(ACCEPT_ENCODING, "gzip, deflate");
    if let Some(accept_value) = accept {
        request = request.header(ACCEPT, accept_value);
    }
    for &(name, value) in extra_headers {
        request = request.header(name, value);
    }
    request
        .send()
        .await
        .map_err(|http_error| map_reqwest_error(&http_error, url))
}

fn map_reqwest_error(http_error: &reqwest::Error, url: &HttpUrl) -> AdfontesError {
    let mut source: Option<&(dyn StdError + 'static)> = Some(http_error);
    while let Some(error) = source {
        if let Some(blocked) = error.downcast_ref::<SsrfBlockedError>() {
            return AdfontesError::SafetyBlock {
                reason: blocked.reason.clone(),
            };
        }
        source = error.source();
    }
    AdfontesError::Transport {
        url: url.to_string(),
        detail: format_error_chain(http_error),
    }
}

#[derive(Debug)]
struct SsrfBlockedError {
    reason: String,
}

impl fmt::Display for SsrfBlockedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl StdError for SsrfBlockedError {}

fn extract_single_location_header(headers: &HeaderMap, url: &str) -> Result<String, AdfontesError> {
    let mut locations = Vec::new();
    for raw_value in headers.get_all(LOCATION) {
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
    headers: &HeaderMap,
    url: &str,
) -> Result<Vec<String>, AdfontesError> {
    let mut tokens = Vec::new();
    for raw_value in headers.get_all(CONTENT_ENCODING) {
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

fn collect_response_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
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

async fn read_bounded_http_body(
    mut response: reqwest::Response,
    max_bytes: usize,
    url: &str,
) -> Result<Vec<u8>, AdfontesError> {
    if let Some(declared_len) = response.content_length()
        && declared_len > max_bytes as u64
    {
        return Err(AdfontesError::SafetyBlock {
            reason: format!("HTTP Content-Length {declared_len} exceeds {max_bytes}-byte limit"),
        });
    }

    let mut output = Vec::new();
    while let Some(chunk) =
        response
            .chunk()
            .await
            .map_err(|http_error| AdfontesError::Transport {
                url: url.to_owned(),
                detail: format!(
                    "failed reading HTTP body: {}",
                    format_error_chain(&http_error)
                ),
            })?
    {
        let remaining = max_bytes.saturating_sub(output.len());
        if chunk.len() > remaining {
            return Err(AdfontesError::SafetyBlock {
                reason: format!("HTTP response body exceeded {max_bytes}-byte limit"),
            });
        }
        output.extend_from_slice(&chunk);
    }
    Ok(output)
}

fn format_error_chain(error: &(dyn StdError + 'static)) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut current: Option<&(dyn StdError + 'static)> = Some(error);
    while let Some(frame) = current {
        let rendered = frame.to_string();
        if !rendered.is_empty() && parts.last().is_none_or(|prev| prev != &rendered) {
            parts.push(rendered);
        }
        current = frame.source();
    }
    parts.join(": ")
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

    let media_type = content_type.map_or_default(parse_media_type);
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

const fn is_redirect_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::MOVED_PERMANENTLY
            | StatusCode::FOUND
            | StatusCode::SEE_OTHER
            | StatusCode::TEMPORARY_REDIRECT
            | StatusCode::PERMANENT_REDIRECT
    )
}

fn is_forbidden_ipv6(ipv6: Ipv6Addr) -> bool {
    let [_seg0, seg1, seg2, seg3, _seg4, _seg5, seg6, seg7] = ipv6.segments();

    if IPV6_NAT64_WKP.contains(&ipv6) {
        return is_forbidden_ipv4(ipv4_from_segments(seg6, seg7));
    }

    if !IPV6_GLOBAL_UNICAST.contains(&ipv6) {
        return true;
    }

    if IPV6_TEREDO.contains(&ipv6) {
        let server_ip = ipv4_from_segments(seg2, seg3);
        let client_ip = ipv4_from_segments(!seg6, !seg7);
        return is_forbidden_ipv4(server_ip) || is_forbidden_ipv4(client_ip);
    }

    if IPV6_SIX_TO_FOUR.contains(&ipv6) {
        return is_forbidden_ipv4(ipv4_from_segments(seg1, seg2));
    }

    FORBIDDEN_IPV6_NETS.iter().any(|net| net.contains(&ipv6))
}

fn is_forbidden_ipv4(ipv4: Ipv4Addr) -> bool {
    ipv4.is_private()
        || ipv4.is_loopback()
        || ipv4.is_link_local()
        || ipv4.is_broadcast()
        || ipv4.is_documentation()
        || ipv4.is_multicast()
        || ipv4.is_unspecified()
        || FORBIDDEN_IPV4_NETS.iter().any(|net| net.contains(&ipv4))
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
    if rest.is_empty()
        || raw_proxy
            .chars()
            .any(|character| character.is_control() || character.is_ascii_whitespace())
    {
        return Err(AdfontesError::Transport {
            url: raw_proxy.to_owned(),
            detail: "proxy URL is empty or contains whitespace or control characters".to_owned(),
        });
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_userinfo, host_part)| host_part);
    if host_port.is_empty() || host_port.starts_with(':') {
        return Err(AdfontesError::Transport {
            url: raw_proxy.to_owned(),
            detail: "proxy URL authority host is empty".to_owned(),
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
    Cidr(IpNet),
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
            Self::Cidr(network) => match target_host {
                UrlHost::Ip(target_ip) => network.contains(&target_ip),
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

    if trimmed.contains('/') {
        return trimmed
            .parse::<IpNet>()
            .ok()
            .map(|net| NoProxyEntry::Cidr(net.trunc()));
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

fn parse_media_type(content_type: &str) -> String {
    let trimmed = content_type.trim();
    if let Ok(parsed) = trimmed.parse::<Mime>() {
        return parsed.essence_str().to_owned();
    }
    trimmed
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
}

fn encoding_from_content_type(content_type: &str) -> &'static Encoding {
    let trimmed = content_type.trim();
    if let Ok(parsed) = trimmed.parse::<Mime>() {
        return parsed
            .get_param(mime::CHARSET)
            .and_then(|charset| Encoding::for_label(charset.as_str().trim_matches('\'').as_bytes()))
            .unwrap_or(encoding_rs::UTF_8);
    }
    for parameter in trimmed.split(';').skip(1) {
        let Some((key, value)) = parameter.split_once('=') else {
            continue;
        };
        if key.trim().eq_ignore_ascii_case(mime::CHARSET.as_str()) {
            let label = value.trim().trim_matches('"').trim_matches('\'');
            return Encoding::for_label(label.as_bytes()).unwrap_or(encoding_rs::UTF_8);
        }
    }
    encoding_rs::UTF_8
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead as _, BufReader, Write as _};
    use std::net::{SocketAddr, TcpListener};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use rstest::rstest;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

    use super::*;
    use crate::archive::{test_gzip_bytes, test_zlib_bytes};

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
            validate_hostname_policy_with_suffixes(&url, false, &[]).unwrap_err(),
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

    #[rstest]
    #[case("https://export.arxiv.org/pdf/2409.13740", None)]
    #[case(
        "https://example.org/paper.pdf",
        Some("http://proxy-https.example:8443")
    )]
    #[case("http://10.1.2.3/x", None)]
    #[case("http://[fd00::42]/x", None)]
    #[case("http://[fd01::1]/x", Some("http://proxy-http.example:8080"))]
    #[case("http://[fd02::1]/x", Some("http://proxy-http.example:8080"))]
    #[case("http://[::1]:8080/x", None)]
    #[case("http://127.0.0.1/x", Some("http://proxy-http.example:8080"))]
    #[case("http://192.168.1.1/x", Some("http://proxy-http.example:8080"))]
    fn resolves_proxies_per_scheme_and_matches_cidr_wildcard_and_ipv6_no_proxy(
        #[case] raw_url: &str,
        #[case] expected_proxy: Option<&str>,
    ) {
        let config = ProxyConfig {
            http_proxy: Some("http://proxy-http.example:8080".to_owned()),
            https_proxy: Some("http://proxy-https.example:8443".to_owned()),
            all_proxy: None,
            no_proxy: vec![
                "*.ArXiv.org.".to_owned(),
                "10.9.8.7/8".to_owned(),
                "fd00::abcd/16".to_owned(),
                "fd02::/129".to_owned(),
                "192.168.1.1/33".to_owned(),
                "[0:0:0:0:0:0:0:1]:8080".to_owned(),
                "[::1]:badport".to_owned(),
                "0.1".to_owned(),
            ],
            allow_proxy_remote_dns: false,
        };
        let url = HttpUrl::parse(raw_url).unwrap();

        assert_eq!(
            config
                .resolve_proxy_for_url(&url, false)
                .unwrap()
                .as_deref(),
            expected_proxy
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn strips_authorization_header_across_cross_origin_redirects() {
        let dest_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/final"))
            .and(|request: &WiremockRequest| !request.headers.contains_key("authorization"))
            .respond_with(ResponseTemplate::new(200).set_body_string("STRIPPED"))
            .expect(1)
            .mount(&dest_server)
            .await;

        let start_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/start"))
            .and(header("authorization", "Bearer secret"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("Location", format!("{}/final", dest_server.uri())),
            )
            .expect(1)
            .mount(&start_server)
            .await;

        let client = loopback_client(&[*start_server.address(), *dest_server.address()], 4096);
        let url = HttpUrl::parse(&format!("{}/start", start_server.uri())).unwrap();
        let response = client
            .get(&url, None, &[("Authorization", "Bearer secret")])
            .await
            .unwrap();
        start_server.verify().await;
        dest_server.verify().await;

        assert_eq!(response.final_url.path(), "/final");
        assert_eq!(response.decode_text(), "STRIPPED");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn decodes_deflate_with_windows_1252_charset_and_clears_content_encoding() {
        let latin1_bytes = encoding_rs::WINDOWS_1252.encode("Café").0.into_owned();
        let compressed = test_zlib_bytes(&latin1_bytes);
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/latin1"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(compressed, "text/plain; charset=windows-1252")
                    .insert_header("Content-Encoding", "deflate"),
            )
            .mount(&server)
            .await;

        let client = loopback_client(&[*server.address()], 4096);
        let url = HttpUrl::parse(&format!("{}/latin1", server.uri())).unwrap();
        let response = client.get(&url, None, &[]).await.unwrap();

        assert_eq!(response.decode_text(), "Café");
        assert!(!response.headers.contains_key("content-encoding"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_raw_gzip_bytes_for_archive_mime_types_only() {
        let gz_payload = test_gzip_bytes(b"\\documentclass{article}");
        let expected = gz_payload.clone();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/src/2409.13740"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(gz_payload, "application/x-gzip")
                    .insert_header("Content-Encoding", "x-gzip"),
            )
            .mount(&server)
            .await;

        let client = loopback_client(&[*server.address()], 4096);
        let url = HttpUrl::parse(&format!("{}/src/2409.13740", server.uri())).unwrap();
        let response = client.get(&url, None, &[]).await.unwrap();

        assert_eq!(response.body, expected);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocks_stacked_duplicate_content_encoding_and_oversized_uncompressed_body() {
        let double_gz = test_gzip_bytes(&test_gzip_bytes(b"nested"));
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/stacked"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(double_gz, "text/plain")
                    .append_header("Content-Encoding", "gzip")
                    .append_header("Content-Encoding", "gzip"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/big"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(vec![b'X'; 512], "text/plain"))
            .mount(&server)
            .await;

        let client = loopback_client(&[*server.address()], 256);
        let stacked_url = HttpUrl::parse(&format!("{}/stacked", server.uri())).unwrap();
        let big_url = HttpUrl::parse(&format!("{}/big", server.uri())).unwrap();
        let stacked_failure = client.get(&stacked_url, None, &[]).await.unwrap_err();
        let oversized_failure = client.get(&big_url, None, &[]).await.unwrap_err();

        assert!(matches!(stacked_failure, AdfontesError::SafetyBlock { .. }));
        assert!(matches!(
            oversized_failure,
            AdfontesError::SafetyBlock { .. }
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocks_gzip_bomb_over_loopback() {
        let bomb_bytes = test_gzip_bytes(&vec![b'B'; 4096]);
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bomb"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(bomb_bytes, "text/html")
                    .insert_header("Content-Encoding", "gzip"),
            )
            .mount(&server)
            .await;

        let client = loopback_client(&[*server.address()], 256);
        let url = HttpUrl::parse(&format!("{}/bomb", server.uri())).unwrap();
        let failure = client.get(&url, None, &[]).await.unwrap_err();

        assert!(matches!(failure, AdfontesError::SafetyBlock { .. }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocks_redirect_to_disallowed_loopback_address_and_sibling_127_ip() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/hop"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("Location", "http://127.0.0.2:8080/internal"),
            )
            .mount(&server)
            .await;

        let client = loopback_client(&[*server.address()], 4096);
        let url = HttpUrl::parse(&format!("{}/hop", server.uri())).unwrap();
        let failure = client.get(&url, None, &[]).await.unwrap_err();

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

    #[rstest]
    #[case(Some("TEXT/HTML; charset=\"windows-1252\""), "text/html", "Café")]
    #[case(Some("text/html; charset='windows-1252'"), "text/html", "Café")]
    #[case(
        Some("text/html; foo=\"a;b\"; charset=\"windows-1252\""),
        "text/html",
        "Café"
    )]
    #[case(
        Some("text/html; foo=\"bar; charset=windows-1252\""),
        "text/html",
        "Caf\u{FFFD}"
    )]
    #[case(
        Some("text/html; charset=bogus; charset=windows-1252"),
        "text/html",
        "Caf\u{FFFD}"
    )]
    #[case(Some("text/html; invalid param"), "text/html", "Caf\u{FFFD}")]
    #[case(Some("application/jats+xml; "), "application/jats+xml", "Caf\u{FFFD}")]
    fn normalizes_mime_type_and_decodes_quoted_and_quirky_content_types(
        #[case] content_type: Option<&str>,
        #[case] expected_mime: &str,
        #[case] expected_text: &str,
    ) {
        let latin1_bytes = encoding_rs::WINDOWS_1252.encode("Café").0.into_owned();
        let response = HttpResponse {
            status: 200,
            redirect_chain: Vec::new(),
            final_url: HttpUrl::parse("https://example.org/doc").unwrap(),
            content_type: content_type.map(str::to_owned),
            headers: BTreeMap::new(),
            body: latin1_bytes,
        };
        assert_eq!(response.mime_type(), expected_mime);
        assert_eq!(response.decode_text(), expected_text);
    }

    #[rstest]
    #[case("127.0.0.1", &[])]
    #[case("localhost", &[])]
    #[case("127.0.0.1", &["127.0.0.1:45123", "[::1]:45123"])]
    #[tokio::test(flavor = "current_thread")]
    async fn ssrf_dns_resolver_blocks_forbidden_and_non_localhost_ips(
        #[case] query: &str,
        #[case] allowed_raw: &[&str],
    ) {
        let allowed_addrs = allowed_raw.iter().map(|raw| raw.parse().unwrap()).collect();
        let resolver = SsrfDnsResolver::new(&TransportConfig {
            allowed_loopback_addrs: allowed_addrs,
            ..TransportConfig::default()
        });
        let failure = reqwest::dns::Resolve::resolve(&resolver, query.parse().unwrap())
            .await
            .err()
            .unwrap();
        assert!(failure.downcast_ref::<SsrfBlockedError>().is_some());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ssrf_dns_resolver_allows_allowlisted_localhost() {
        let mut allowed_addrs = BTreeSet::new();
        allowed_addrs.insert("127.0.0.1:45123".parse().unwrap());
        allowed_addrs.insert("[::1]:45123".parse().unwrap());
        let resolver = SsrfDnsResolver::new(&TransportConfig {
            allowed_loopback_addrs: allowed_addrs,
            ..TransportConfig::default()
        });
        let addrs: Vec<SocketAddr> =
            reqwest::dns::Resolve::resolve(&resolver, "localhost".parse().unwrap())
                .await
                .unwrap()
                .collect();
        assert_ne!(addrs, [] as [SocketAddr; 0]);
        assert!(addrs.iter().all(|addr| addr.ip().is_loopback()));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dns_resolver_ssrf_block_propagates_as_safety_block() {
        struct BlockedTestResolver(SsrfDnsResolver);

        impl reqwest::dns::Resolve for BlockedTestResolver {
            fn resolve(&self, _name: reqwest::dns::Name) -> reqwest::dns::Resolving {
                self.0.resolve("127.0.0.1".parse().unwrap())
            }
        }

        let config = TransportConfig::default();
        let inner = SsrfDnsResolver::new(&config);
        let client = base_client_builder(&config, &build_platform_tls_config())
            .unwrap()
            .no_proxy()
            .dns_resolver(Arc::new(BlockedTestResolver(inner)))
            .build()
            .unwrap();
        let target = HttpUrl::parse("http://blocked-dns.example/paper").unwrap();
        let failure = send_single_hop(&client, &target, None, &[], Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(matches!(
            failure,
            AdfontesError::SafetyBlock { ref reason } if reason.contains("non-global IP")
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn reuses_keep_alive_connection_across_sequential_requests() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let accept_count = Arc::new(AtomicUsize::new(0));
        let request_count = Arc::new(AtomicUsize::new(0));
        let accepts_for_server = Arc::clone(&accept_count);
        let requests_for_server = Arc::clone(&request_count);

        let server = thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            accepts_for_server.fetch_add(1, Ordering::SeqCst);
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            for _ in 0..2 {
                let mut header_line = String::new();
                loop {
                    header_line.clear();
                    if reader.read_line(&mut header_line).unwrap_or(0) == 0 {
                        return;
                    }
                    if header_line == "\r\n" || header_line == "\n" {
                        break;
                    }
                }
                requests_for_server.fetch_add(1, Ordering::SeqCst);
                let response = [
                    "HTTP/1.1 200 OK",
                    "Content-Type: text/plain",
                    "Content-Length: 2",
                    "",
                    "ok",
                ]
                .join("\r\n");
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let client = loopback_client(&[addr], 4096);
        let url = HttpUrl::parse(&format!("http://{addr}/ping")).unwrap();
        let first = client.get(&url, None, &[]).await.unwrap();
        let second = client.get(&url, None, &[]).await.unwrap();
        drop(client);
        server.join().unwrap();

        assert_eq!(first.decode_text(), "ok");
        assert_eq!(second.decode_text(), "ok");
        assert_eq!(accept_count.load(Ordering::SeqCst), 1);
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn strips_authorization_permanently_across_redirect_bounce_back() {
        let origin_server = MockServer::start().await;
        let mid_server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/start"))
            .and(header("authorization", "Bearer secret"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("Location", format!("{}/mid", mid_server.uri())),
            )
            .expect(1)
            .mount(&origin_server)
            .await;

        Mock::given(method("GET"))
            .and(path("/mid"))
            .and(|request: &WiremockRequest| !request.headers.contains_key("authorization"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("Location", format!("{}/final", origin_server.uri())),
            )
            .expect(1)
            .mount(&mid_server)
            .await;

        let final_url = format!("{}/final", origin_server.uri());
        Mock::given(method("GET"))
            .and(path("/final"))
            .and(|request: &WiremockRequest| !request.headers.contains_key("authorization"))
            .respond_with(ResponseTemplate::new(200).set_body_string("STRIPPED"))
            .expect(1)
            .mount(&origin_server)
            .await;

        let client = loopback_client(&[*origin_server.address(), *mid_server.address()], 4096);
        let start_url = HttpUrl::parse(&format!("{}/start", origin_server.uri())).unwrap();
        let response = client
            .get(&start_url, None, &[("Authorization", "Bearer secret")])
            .await
            .unwrap();
        origin_server.verify().await;
        mid_server.verify().await;

        assert_eq!(response.status, 200);
        assert_eq!(response.final_url.as_str(), final_url);
        assert_eq!(response.decode_text(), "STRIPPED");
    }

    #[derive(Debug)]
    struct ChainedTestError {
        message: &'static str,
        source: Option<Box<Self>>,
    }

    impl fmt::Display for ChainedTestError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.message)
        }
    }

    impl StdError for ChainedTestError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            self.source
                .as_deref()
                .map(|inner| inner as &(dyn StdError + 'static))
        }
    }

    #[test]
    fn formats_error_chain_preserving_root_cause_and_deduplicating_consecutive_frames() {
        let root = ChainedTestError {
            message: "dns lookup timed out",
            source: None,
        };
        let duplicate = ChainedTestError {
            message: "tcp connect failed",
            source: Some(Box::new(root)),
        };
        let outer = ChainedTestError {
            message: "tcp connect failed",
            source: Some(Box::new(duplicate)),
        };
        assert_eq!(
            format_error_chain(&outer),
            "tcp connect failed: dns lookup timed out"
        );
    }

    #[tokio::test]
    async fn transport_error_preserves_reqwest_source_chain() {
        const LOOPBACK_BUFFER_BYTES: usize = 4096;
        const MIN_CHAIN_SEPARATORS: usize = 2;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let closed_addr = listener.local_addr().unwrap();
        drop(listener);

        let client = loopback_client(&[closed_addr], LOOPBACK_BUFFER_BYTES);
        let url = HttpUrl::parse(&format!("http://{closed_addr}/unreachable")).unwrap();
        let error_text = client.get(&url, None, &[]).await.unwrap_err().to_string();
        assert!(
            error_text.starts_with("transport failure for "),
            "unexpected error variant: {error_text}"
        );
        assert!(
            error_text.matches(": ").count() >= MIN_CHAIN_SEPARATORS,
            "expected chained source frames in {error_text}"
        );
    }

    #[rstest]
    #[case("text/html; charset=bogus; charset=shift_jis", encoding_rs::UTF_8)]
    #[case("text/html; ; charset=bogus; charset=shift_jis", encoding_rs::UTF_8)]
    #[case("text/html; ; charset=shift_jis", encoding_rs::SHIFT_JIS)]
    fn resolves_encoding_from_first_charset_only(
        #[case] content_type: &str,
        #[case] expected: &'static Encoding,
    ) {
        assert_eq!(encoding_from_content_type(content_type), expected);
    }

    #[rstest]
    #[case("http://proxy.example:8080", true)]
    #[case("https://user@proxy.example/path", true)]
    #[case("http://proxy .example:8080", false)]
    #[case("http://proxy.example:8080\t", false)]
    #[case("http:///path", false)]
    #[case("http://:8080", false)]
    #[case("http://user@", false)]
    #[case("http://user@:8080/path", false)]
    fn validates_http_proxy_url(#[case] raw_proxy: &str, #[case] expected_ok: bool) {
        assert_eq!(validate_http_proxy_url(raw_proxy).is_ok(), expected_ok);
    }

    #[test]
    fn allows_loopback_localhost_with_trailing_dot() {
        const LOOPBACK_BUFFER_BYTES: usize = 4096;
        let addr: SocketAddr = "127.0.0.1:8443".parse().unwrap();
        let client = loopback_client(&[addr], LOOPBACK_BUFFER_BYTES);
        let trailing_dot_url = HttpUrl::parse("http://localhost.:8443/paper").unwrap();
        assert!(client.is_loopback_target_allowed(&trailing_dot_url));
    }
}
