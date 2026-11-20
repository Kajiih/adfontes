//! Newline-delimited JSON-RPC 2.0 stdio Model Context Protocol (MCP) server.
//!
//! Supports MCP protocol revisions `2025-06-18`, `2025-03-26`, and `2024-11-05`, exposing
//! read-only `fetch` and `slice_pdf` tools over bounded (`<= 1 MiB`) newline-delimited frames.
//! Rejects JSON-RPC batch arrays before `initialize` and under `2025-06-18` (while accepting
//! bounded batches on `2025-03-26` and `2024-11-05`), distinguishes notifications (absent `id`)
//! from `"id": null`, retries interrupted (`EINTR`) reads unless signaled to stop, and treats
//! `BrokenPipe` (`EPIPE`) on stdout as a clean exit.

architecture_component!(Mcp);

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::Path;

use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::model::{AdfontesError, PageSpan, ParsedTarget};
use crate::resolve::{ResolverConfig, resolve_target};
use crate::stage::{StagingPolicy, slice_staged_pdf, stage_document, validate_staging_dir_policy};
use crate::transport::HttpClient;

const PROTOCOL_V2025_06_18: &str = "2025-06-18";
const PROTOCOL_V2025_03_26: &str = "2025-03-26";
const PROTOCOL_V2024_11_05: &str = "2024-11-05";

const JSONRPC_VERSION: &str = "2.0";
const SERVER_NAME: &str = "adfontes";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

const TOOL_FETCH: &str = "fetch";
const TOOL_SLICE_PDF: &str = "slice_pdf";

const ARG_TARGET: &str = "target";
const ARG_OUTPUT_DIR: &str = "output_dir";
const ARG_INCLUDE_PDF_INDEX: &str = "include_pdf_index";
const ARG_PDF_PATH: &str = "pdf_path";
const ARG_START_PAGE: &str = "start_page";
const ARG_END_PAGE: &str = "end_page";

const FIELD_JSONRPC: &str = "jsonrpc";
const FIELD_ID: &str = "id";
const FIELD_METHOD: &str = "method";
const FIELD_PARAMS: &str = "params";

const ERROR_PARSE: i64 = -32700;
const ERROR_INVALID_REQUEST: i64 = -32600;
const ERROR_METHOD_NOT_FOUND: i64 = -32601;
const ERROR_INVALID_PARAMS: i64 = -32602;
const ERROR_INTERNAL: i64 = -32603;

const MESSAGE_NOT_INITIALIZED: &str = "Server is not initialized; send `initialize` first";

const MAX_JSONRPC_LINE_BYTES: usize = 1024 * 1024;
const MAX_BATCH_ITEMS: usize = 32;

/// Dependency container injected into the stdio MCP server loop.
#[derive(Debug, Clone)]
pub struct McpRuntime {
    /// SSRF-validated HTTP client used by `fetch`.
    pub http_client: HttpClient,
    /// Upstream repository endpoints and optional polite-pool contact email.
    pub resolver_config: ResolverConfig,
    /// Filesystem confinement and PDF slicing policy.
    pub staging_policy: StagingPolicy,
}

/// Runs the stdio MCP server against process `stdin` and `stdout` until EOF or `BrokenPipe`.
///
/// # Errors
///
/// Returns [`AdfontesError::Io`] only on non-`BrokenPipe` I/O failures reading stdin or writing
/// stdout.
pub fn serve_stdio(runtime: &McpRuntime) -> Result<(), AdfontesError> {
    let stdin = io::stdin();
    serve_io(stdin.lock(), io::stdout(), runtime)
}

/// Runs the newline-delimited JSON-RPC 2.0 MCP server loop over arbitrary buffered streams.
///
/// # Errors
///
/// Returns [`AdfontesError::Io`] on non-`BrokenPipe` stream read/write failures.
pub fn serve_io<R: BufRead, W: Write>(
    reader: R,
    writer: W,
    runtime: &McpRuntime,
) -> Result<(), AdfontesError> {
    serve_io_until(reader, writer, runtime, || false)
}

/// Runs the stdio MCP server loop while consulting `should_stop` whenever a read is interrupted by
/// a signal (`io::ErrorKind::Interrupted`).
///
/// If `should_stop` returns `true` on `EINTR` (for example, a pending Python `SIGINT`), the server
/// returns `Ok(())` cleanly without writing an error frame; if `false` (for example, `SIGWINCH` or
/// `SIGCHLD`), the read is retried.
///
/// # Errors
///
/// Returns [`AdfontesError::Io`] on non-`BrokenPipe`, non-interrupted stream read/write failures.
pub fn serve_io_until<R, W, F>(
    mut reader: R,
    mut writer: W,
    runtime: &McpRuntime,
    mut should_stop: F,
) -> Result<(), AdfontesError>
where
    R: BufRead,
    W: Write,
    F: FnMut() -> bool,
{
    let mut state = SessionState::default();
    loop {
        if should_stop() {
            return Ok(());
        }
        match read_bounded_line(&mut reader, MAX_JSONRPC_LINE_BYTES, &mut should_stop)? {
            BoundedLine::Eof | BoundedLine::InterruptedStop => return Ok(()),
            BoundedLine::TooLong => {
                let response = jsonrpc_error(
                    &Value::Null,
                    ERROR_INVALID_REQUEST,
                    "JSON-RPC frame exceeds 1 MiB safety ceiling",
                );
                if !write_json_line(&mut writer, &response)? {
                    return Ok(());
                }
            }
            BoundedLine::Line(raw_bytes) => {
                let trimmed = trim_json_whitespace(&raw_bytes);
                if trimmed.is_empty() {
                    continue;
                }
                if let Some(reply) = handle_raw_frame(trimmed, &mut state, runtime)
                    && !write_json_line(&mut writer, &reply)?
                {
                    return Ok(());
                }
            }
        }
    }
}

#[derive(Debug, Default)]
struct SessionState {
    negotiated_version: Option<ProtocolVersion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtocolVersion {
    V2025_06_18,
    V2025_03_26,
    V2024_11_05,
}

impl ProtocolVersion {
    const fn as_str(self) -> &'static str {
        match self {
            Self::V2025_06_18 => PROTOCOL_V2025_06_18,
            Self::V2025_03_26 => PROTOCOL_V2025_03_26,
            Self::V2024_11_05 => PROTOCOL_V2024_11_05,
        }
    }

    const fn allows_jsonrpc_batch(self) -> bool {
        matches!(self, Self::V2025_03_26 | Self::V2024_11_05)
    }

    fn negotiate(requested: Option<&str>) -> Self {
        match requested {
            Some(PROTOCOL_V2025_03_26) => Self::V2025_03_26,
            Some(PROTOCOL_V2024_11_05) => Self::V2024_11_05,
            _ => Self::V2025_06_18,
        }
    }
}

enum BoundedLine {
    Eof,
    InterruptedStop,
    Line(Vec<u8>),
    TooLong,
}

fn read_bounded_line<R, F>(
    reader: &mut R,
    max_bytes: usize,
    should_stop: &mut F,
) -> Result<BoundedLine, AdfontesError>
where
    R: BufRead,
    F: FnMut() -> bool,
{
    let mut line = Vec::new();
    let mut exceeded = false;

    loop {
        let available = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                if should_stop() {
                    return Ok(BoundedLine::InterruptedStop);
                }
                continue;
            }
            Err(source) => {
                return Err(AdfontesError::Io {
                    context: "reading MCP request frame".to_owned(),
                    source,
                });
            }
        };
        if available.is_empty() {
            if exceeded {
                return Ok(BoundedLine::TooLong);
            }
            if line.is_empty() {
                return Ok(BoundedLine::Eof);
            }
            return Ok(BoundedLine::Line(line));
        }

        let newline_offset = available.iter().position(|&byte| byte == b'\n');
        let chunk_len = newline_offset.map_or(available.len(), |offset| offset + 1);
        let chunk = available.get(..chunk_len).unwrap_or_default();

        if !exceeded {
            if line.len() + chunk.len() > max_bytes {
                exceeded = true;
                line.clear();
            } else {
                line.extend_from_slice(chunk);
            }
        }

        reader.consume(chunk_len);
        if newline_offset.is_some() {
            return Ok(if exceeded {
                BoundedLine::TooLong
            } else {
                BoundedLine::Line(line)
            });
        }
    }
}

fn handle_raw_frame(
    frame_bytes: &[u8],
    state: &mut SessionState,
    runtime: &McpRuntime,
) -> Option<Value> {
    let parsed: Value = match serde_json::from_slice(frame_bytes) {
        Ok(value) => value,
        Err(parse_error) => {
            return Some(jsonrpc_error(
                &Value::Null,
                ERROR_PARSE,
                &format!("Parse error: {parse_error}"),
            ));
        }
    };

    match parsed {
        Value::Object(object) => handle_single_object(&object, state, runtime),
        Value::Array(items) => handle_batch_array(&items, state, runtime),
        _ => Some(jsonrpc_error(
            &Value::Null,
            ERROR_INVALID_REQUEST,
            "JSON-RPC message must be an object",
        )),
    }
}

fn handle_batch_array(
    items: &[Value],
    state: &mut SessionState,
    runtime: &McpRuntime,
) -> Option<Value> {
    let Some(version) = state.negotiated_version else {
        return Some(jsonrpc_error(
            &Value::Null,
            ERROR_INVALID_REQUEST,
            "JSON-RPC batch requests are not allowed before `initialize`",
        ));
    };
    if !version.allows_jsonrpc_batch() {
        return Some(jsonrpc_error(
            &Value::Null,
            ERROR_INVALID_REQUEST,
            "JSON-RPC batch requests are not supported in MCP protocol version 2025-06-18",
        ));
    }
    if items.is_empty() {
        return Some(jsonrpc_error(
            &Value::Null,
            ERROR_INVALID_REQUEST,
            "JSON-RPC batch array must not be empty",
        ));
    }
    if items.len() > MAX_BATCH_ITEMS {
        return Some(jsonrpc_error(
            &Value::Null,
            ERROR_INVALID_REQUEST,
            "JSON-RPC batch exceeds maximum of 32 items",
        ));
    }

    let mut replies = Vec::new();
    for item in items {
        let reply = match item {
            Value::Object(object) => handle_single_object(object, state, runtime),
            _ => Some(jsonrpc_error(
                &Value::Null,
                ERROR_INVALID_REQUEST,
                "JSON-RPC batch element must be an object",
            )),
        };
        if let Some(response) = reply {
            replies.push(response);
        }
    }
    if replies.is_empty() {
        None
    } else {
        Some(Value::Array(replies))
    }
}

fn handle_single_object(
    object: &Map<String, Value>,
    state: &mut SessionState,
    runtime: &McpRuntime,
) -> Option<Value> {
    let call = match parse_jsonrpc_call(object) {
        Ok(parsed) => parsed,
        Err(error_value) => return Some(error_value),
    };

    // Notifications (absent `id` key) must never receive a response.
    let Some(request_id) = call.id else {
        handle_notification(call.method);
        return None;
    };

    let response = match call.method {
        "initialize" => handle_initialize(&request_id, call.params, state),
        "ping" => handle_ping(&request_id, call.params),
        "tools/list" => handle_tools_list(&request_id, call.params, state),
        "tools/call" => handle_tools_call(&request_id, call.params, state, runtime),
        other => jsonrpc_error(
            &request_id,
            ERROR_METHOD_NOT_FOUND,
            &format!("Method not found: {other}"),
        ),
    };
    Some(response)
}

struct JsonRpcCall<'a> {
    id: Option<Value>,
    method: &'a str,
    params: Option<&'a Value>,
}

fn parse_jsonrpc_call(object: &Map<String, Value>) -> Result<JsonRpcCall<'_>, Value> {
    let has_id = object.contains_key(FIELD_ID);
    let validated_id = if has_id {
        let raw_id = object.get(FIELD_ID).cloned().unwrap_or(Value::Null);
        if !is_valid_jsonrpc_id(&raw_id) {
            return Err(jsonrpc_error(
                &Value::Null,
                ERROR_INVALID_REQUEST,
                "JSON-RPC `id` must be a string, integer, or null",
            ));
        }
        Some(raw_id)
    } else {
        None
    };

    if object.get(FIELD_JSONRPC).and_then(Value::as_str) != Some(JSONRPC_VERSION) {
        return Err(jsonrpc_error(
            validated_id.as_ref().unwrap_or(&Value::Null),
            ERROR_INVALID_REQUEST,
            "JSON-RPC `jsonrpc` field must be \"2.0\"",
        ));
    }

    let Some(method) = object.get(FIELD_METHOD).and_then(Value::as_str) else {
        return Err(jsonrpc_error(
            validated_id.as_ref().unwrap_or(&Value::Null),
            ERROR_INVALID_REQUEST,
            "JSON-RPC `method` field must be a string",
        ));
    };

    // JSON-RPC 2.0 §4.2: when present, `params` must be a structured value (Object or Array).
    let params = object.get(FIELD_PARAMS);
    if let Some(raw_params) = params
        && !matches!(raw_params, Value::Object(_) | Value::Array(_))
    {
        return Err(jsonrpc_error(
            validated_id.as_ref().unwrap_or(&Value::Null),
            ERROR_INVALID_REQUEST,
            "JSON-RPC `params` must be an object or array when present",
        ));
    }

    Ok(JsonRpcCall {
        id: validated_id,
        method,
        params,
    })
}

const fn handle_notification(method: &str) {
    let _ = method;
}

fn handle_initialize(
    request_id: &Value,
    params: Option<&Value>,
    state: &mut SessionState,
) -> Value {
    if state.negotiated_version.is_some() {
        return jsonrpc_error(
            request_id,
            ERROR_INVALID_REQUEST,
            "MCP session is already initialized",
        );
    }

    let requested_version = match params {
        None => None,
        Some(Value::Object(map)) => match map.get("protocolVersion") {
            None => None,
            Some(Value::String(version)) => Some(version.as_str()),
            Some(_) => {
                return jsonrpc_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    "`params.protocolVersion` must be a string",
                );
            }
        },
        Some(_) => {
            return jsonrpc_error(
                request_id,
                ERROR_INVALID_PARAMS,
                "`initialize` `params` must be an object",
            );
        }
    };

    let negotiated = ProtocolVersion::negotiate(requested_version);
    state.negotiated_version = Some(negotiated);

    let result = InitializeResult {
        protocol_version: negotiated.as_str(),
        capabilities: ServerCapabilities {
            tools: ToolCapabilities {
                list_changed: false,
            },
        },
        server_info: ServerInfo {
            name: SERVER_NAME,
            version: SERVER_VERSION,
        },
    };
    jsonrpc_serialized_result(request_id, &result)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InitializeResult {
    protocol_version: &'static str,
    capabilities: ServerCapabilities,
    server_info: ServerInfo,
}

#[derive(Serialize)]
struct ServerCapabilities {
    tools: ToolCapabilities,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolCapabilities {
    list_changed: bool,
}

#[derive(Serialize)]
struct ServerInfo {
    name: &'static str,
    version: &'static str,
}

fn handle_ping(request_id: &Value, params: Option<&Value>) -> Value {
    if matches!(params, Some(value) if !value.is_object()) {
        return jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            "`ping` `params` must be an object when present",
        );
    }
    jsonrpc_result(request_id, &json!({}))
}

fn handle_tools_list(request_id: &Value, params: Option<&Value>, state: &SessionState) -> Value {
    if state.negotiated_version.is_none() {
        return jsonrpc_error(request_id, ERROR_INVALID_REQUEST, MESSAGE_NOT_INITIALIZED);
    }
    if matches!(params, Some(value) if !value.is_object()) {
        return jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            "`tools/list` `params` must be an object when present",
        );
    }
    jsonrpc_serialized_result(request_id, &build_tools_list_result())
}

#[derive(Serialize)]
struct ToolsListResult {
    tools: Vec<ToolDefinition>,
}

fn build_tools_list_result() -> ToolsListResult {
    ToolsListResult {
        tools: vec![
            ToolDefinition {
                name: TOOL_FETCH,
                description: "Retrieve full-text academic papers, specifications, or articles via keyless open-access waterfalls, stage normalized Markdown or PDF locally, and return a compact (<= 3,600 B) provenance header and line/page Table of Contents for read_file.",
                input_schema: InputSchema::object(
                    &[ARG_TARGET],
                    [
                        (
                            ARG_TARGET,
                            PropertySchema::string(
                                "arXiv ID/URL, DOI, PMCID, PMID, IETF RFC, or http(s):// URL.",
                            ),
                        ),
                        (
                            ARG_OUTPUT_DIR,
                            PropertySchema::string(
                                "Optional confined staging directory inside temp, ADFONTES_EXTRA_ALLOWED_ROOTS, or ADFONTES_STAGING_DIR.",
                            ),
                        ),
                        (
                            ARG_INCLUDE_PDF_INDEX,
                            PropertySchema::boolean(
                                true,
                                "When the retrieved document is a PDF, write a companion <slug>.pages.md page map.",
                            ),
                        ),
                    ],
                ),
                annotations: ToolAnnotations {
                    title: "Fetch Literature",
                    read_only_hint: true,
                    destructive_hint: false,
                    idempotent_hint: true,
                    open_world_hint: true,
                },
            },
            ToolDefinition {
                name: TOOL_SLICE_PDF,
                description: "Extract a lossless 1..=8-page vector PDF slice from a staged PDF file so read_file can visually inspect target equations, plots, or tables without attaching the entire PDF.",
                input_schema: InputSchema::object(
                    &[ARG_PDF_PATH, ARG_START_PAGE],
                    [
                        (
                            ARG_PDF_PATH,
                            PropertySchema::string("Absolute path to a confined staged PDF file."),
                        ),
                        (
                            ARG_START_PAGE,
                            PropertySchema::positive_integer("1-indexed first page to extract."),
                        ),
                        (
                            ARG_END_PAGE,
                            PropertySchema::positive_integer(
                                "Optional 1-indexed inclusive last page (defaults to start_page; maximum span 8 pages).",
                            ),
                        ),
                        (
                            ARG_OUTPUT_DIR,
                            PropertySchema::string(
                                "Optional confined output directory (defaults to pdf_path's directory).",
                            ),
                        ),
                    ],
                ),
                annotations: ToolAnnotations {
                    title: "Slice Staged PDF",
                    read_only_hint: true,
                    destructive_hint: false,
                    idempotent_hint: true,
                    open_world_hint: false,
                },
            },
        ],
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolDefinition {
    name: &'static str,
    description: &'static str,
    input_schema: InputSchema,
    annotations: ToolAnnotations,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InputSchema {
    #[serde(rename = "type")]
    schema_type: &'static str,
    additional_properties: bool,
    required: &'static [&'static str],
    properties: BTreeMap<&'static str, PropertySchema>,
}

impl InputSchema {
    fn object(
        required: &'static [&'static str],
        properties: impl IntoIterator<Item = (&'static str, PropertySchema)>,
    ) -> Self {
        Self {
            schema_type: "object",
            additional_properties: false,
            required,
            properties: properties.into_iter().collect(),
        }
    }
}

#[derive(Serialize)]
struct PropertySchema {
    #[serde(rename = "type")]
    schema_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    minimum: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    default: Option<bool>,
    description: &'static str,
}

impl PropertySchema {
    const fn string(description: &'static str) -> Self {
        Self {
            schema_type: "string",
            minimum: None,
            default: None,
            description,
        }
    }

    const fn positive_integer(description: &'static str) -> Self {
        Self {
            schema_type: "integer",
            minimum: Some(1),
            default: None,
            description,
        }
    }

    const fn boolean(default_value: bool, description: &'static str) -> Self {
        Self {
            schema_type: "boolean",
            minimum: None,
            default: Some(default_value),
            description,
        }
    }
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "MCP ToolAnnotations wire schema defines four independent boolean hints"
)]
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolAnnotations {
    title: &'static str,
    read_only_hint: bool,
    destructive_hint: bool,
    idempotent_hint: bool,
    open_world_hint: bool,
}

fn handle_tools_call(
    request_id: &Value,
    params: Option<&Value>,
    state: &SessionState,
    runtime: &McpRuntime,
) -> Value {
    if state.negotiated_version.is_none() {
        return jsonrpc_error(request_id, ERROR_INVALID_REQUEST, MESSAGE_NOT_INITIALIZED);
    }
    let Some(call_params) = params.and_then(Value::as_object) else {
        return jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            "tools/call `params` must be an object",
        );
    };
    let Some(tool_name) = call_params.get("name").and_then(Value::as_str) else {
        return jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            "tools/call `params.name` must be a string",
        );
    };
    let empty_args = Map::new();
    let arguments = match call_params.get("arguments") {
        None | Some(Value::Null) => &empty_args,
        Some(Value::Object(map)) => map,
        Some(_) => {
            return jsonrpc_error(
                request_id,
                ERROR_INVALID_PARAMS,
                "tools/call `params.arguments` must be an object",
            );
        }
    };

    match tool_name {
        TOOL_FETCH => execute_fetch_tool(request_id, arguments, runtime),
        TOOL_SLICE_PDF => execute_slice_pdf_tool(request_id, arguments, runtime),
        unknown => jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            &format!("Unknown tool: {unknown}"),
        ),
    }
}

fn execute_fetch_tool(
    request_id: &Value,
    arguments: &Map<String, Value>,
    runtime: &McpRuntime,
) -> Value {
    if let Some(unexpected) = find_unexpected_key(
        arguments,
        &[ARG_TARGET, ARG_OUTPUT_DIR, ARG_INCLUDE_PDF_INDEX],
    ) {
        return jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            &format!("Unexpected argument `{unexpected}` for `{TOOL_FETCH}`"),
        );
    }

    let Some(raw_target) = arguments.get(ARG_TARGET).and_then(Value::as_str) else {
        return jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            "`target` must be a string",
        );
    };
    let output_dir = match parse_optional_string(arguments, ARG_OUTPUT_DIR) {
        Ok(value) => value,
        Err(message) => return jsonrpc_error(request_id, ERROR_INVALID_PARAMS, &message),
    };
    let include_pdf_index = match parse_optional_bool(arguments, ARG_INCLUDE_PDF_INDEX, true) {
        Ok(value) => value,
        Err(message) => return jsonrpc_error(request_id, ERROR_INVALID_PARAMS, &message),
    };

    let target = match ParsedTarget::parse(raw_target) {
        Ok(parsed) => parsed,
        Err(error) => return tool_text_response(request_id, &error.to_string(), true),
    };
    let output_path = output_dir.as_deref().map(Path::new);
    if let Err(error) = validate_staging_dir_policy(output_path, &runtime.staging_policy) {
        return tool_text_response(request_id, &error.to_string(), true);
    }
    let extracted = match resolve_target(&target, &runtime.http_client, &runtime.resolver_config) {
        Ok(document) => document,
        Err(error) => return tool_text_response(request_id, &error.to_string(), true),
    };
    match stage_document(
        &target,
        extracted,
        output_path,
        include_pdf_index,
        &runtime.staging_policy,
    ) {
        Ok(staged) => tool_text_response(request_id, &staged.compact_index, false),
        Err(error) => tool_text_response(request_id, &error.to_string(), true),
    }
}

fn execute_slice_pdf_tool(
    request_id: &Value,
    arguments: &Map<String, Value>,
    runtime: &McpRuntime,
) -> Value {
    if let Some(unexpected) = find_unexpected_key(
        arguments,
        &[ARG_PDF_PATH, ARG_START_PAGE, ARG_END_PAGE, ARG_OUTPUT_DIR],
    ) {
        return jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            &format!("Unexpected argument `{unexpected}` for `{TOOL_SLICE_PDF}`"),
        );
    }

    let Some(raw_pdf_path) = arguments.get(ARG_PDF_PATH).and_then(Value::as_str) else {
        return jsonrpc_error(
            request_id,
            ERROR_INVALID_PARAMS,
            "`pdf_path` must be a string",
        );
    };
    let start_page = match parse_required_page_number(arguments, ARG_START_PAGE) {
        Ok(page) => page,
        Err(message) => return jsonrpc_error(request_id, ERROR_INVALID_PARAMS, &message),
    };
    let end_page = match parse_optional_page_number(arguments, ARG_END_PAGE) {
        Ok(Some(page)) => page,
        Ok(None) => start_page,
        Err(message) => return jsonrpc_error(request_id, ERROR_INVALID_PARAMS, &message),
    };
    let output_dir = match parse_optional_string(arguments, ARG_OUTPUT_DIR) {
        Ok(value) => value,
        Err(message) => return jsonrpc_error(request_id, ERROR_INVALID_PARAMS, &message),
    };

    let span = match PageSpan::new(start_page, end_page) {
        Ok(valid_span) => valid_span,
        Err(error) => return tool_text_response(request_id, &error.to_string(), true),
    };
    let output_path = output_dir.as_deref().map(Path::new);
    match slice_staged_pdf(
        Path::new(raw_pdf_path),
        span,
        output_path,
        &runtime.staging_policy,
    ) {
        Ok(slice) => tool_text_response(request_id, &slice.summary, false),
        Err(error) => tool_text_response(request_id, &error.to_string(), true),
    }
}

fn find_unexpected_key<'a>(arguments: &'a Map<String, Value>, allowed: &[&str]) -> Option<&'a str> {
    arguments
        .keys()
        .find(|key| !allowed.contains(&key.as_str()))
        .map(String::as_str)
}

fn parse_optional_string(
    arguments: &Map<String, Value>,
    key: &str,
) -> Result<Option<String>, String> {
    match arguments.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(format!("`{key}` must be a string")),
    }
}

fn parse_optional_bool(
    arguments: &Map<String, Value>,
    key: &str,
    default_value: bool,
) -> Result<bool, String> {
    match arguments.get(key) {
        None | Some(Value::Null) => Ok(default_value),
        Some(Value::Bool(flag)) => Ok(*flag),
        Some(_) => Err(format!("`{key}` must be a boolean")),
    }
}

fn parse_required_page_number(arguments: &Map<String, Value>, key: &str) -> Result<u32, String> {
    let Some(raw) = arguments.get(key) else {
        return Err(format!(
            "`{key}` is required and must be a positive integer"
        ));
    };
    parse_strict_positive_u32(raw, key)
}

fn parse_optional_page_number(
    arguments: &Map<String, Value>,
    key: &str,
) -> Result<Option<u32>, String> {
    match arguments.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(raw) => parse_strict_positive_u32(raw, key).map(Some),
    }
}

fn parse_strict_positive_u32(raw: &Value, key: &str) -> Result<u32, String> {
    // Reject booleans, strings, floats (`1.0`), and negative values strictly.
    let Some(unsigned) = raw.as_u64() else {
        return Err(format!("`{key}` must be a positive integer"));
    };
    let page = u32::try_from(unsigned)
        .map_err(|_| format!("`{key}` exceeds supported 32-bit page range"))?;
    if page == 0 {
        return Err(format!("`{key}` must be >= 1"));
    }
    Ok(page)
}

fn is_valid_jsonrpc_id(value: &Value) -> bool {
    match value {
        Value::Null | Value::String(_) => true,
        Value::Number(number) => number.is_i64() || number.is_u64(),
        Value::Bool(_) | Value::Array(_) | Value::Object(_) => false,
    }
}

fn tool_text_response(request_id: &Value, text: &str, is_error: bool) -> Value {
    let payload = ToolCallResult {
        content: [TextContent {
            content_type: "text",
            text,
        }],
        is_error,
    };
    jsonrpc_serialized_result(request_id, &payload)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolCallResult<'a> {
    content: [TextContent<'a>; 1],
    is_error: bool,
}

#[derive(Serialize)]
struct TextContent<'a> {
    #[serde(rename = "type")]
    content_type: &'static str,
    text: &'a str,
}

fn jsonrpc_serialized_result<T: Serialize>(request_id: &Value, payload: &T) -> Value {
    match serde_json::to_value(payload) {
        Ok(value) => jsonrpc_result(request_id, &value),
        Err(error) => jsonrpc_error(
            request_id,
            ERROR_INTERNAL,
            &format!("Internal serialization error: {error}"),
        ),
    }
}

fn jsonrpc_result(request_id: &Value, result: &Value) -> Value {
    json!({
        FIELD_JSONRPC: JSONRPC_VERSION,
        FIELD_ID: request_id,
        "result": result
    })
}

fn jsonrpc_error(request_id: &Value, code: i64, message: &str) -> Value {
    json!({
        FIELD_JSONRPC: JSONRPC_VERSION,
        FIELD_ID: request_id,
        "error": {
            "code": code,
            "message": message
        }
    })
}

fn write_json_line<W: Write>(writer: &mut W, payload: &Value) -> Result<bool, AdfontesError> {
    let mut encoded =
        serde_json::to_string(payload).map_err(|error| AdfontesError::Extraction {
            format: "mcp-jsonrpc",
            detail: error.to_string(),
        })?;
    encoded.push('\n');
    match writer
        .write_all(encoded.as_bytes())
        .and_then(|()| writer.flush())
    {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(false),
        Err(source) => Err(AdfontesError::Io {
            context: "writing MCP response frame".to_owned(),
            source,
        }),
    }
}

fn trim_json_whitespace(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|&byte| !is_rfc8259_whitespace(byte))
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|&byte| !is_rfc8259_whitespace(byte))
        .map_or(start, |idx| idx + 1);
    bytes.get(start..end).unwrap_or_default()
}

const fn is_rfc8259_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n')
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::io::{Cursor, Read};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use indoc::{formatdoc, indoc};
    use rstest::rstest;

    use super::*;
    use crate::transport::TransportConfig;

    const INIT_2025_06_18: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#;

    struct BrokenPipeWriter;

    impl Write for BrokenPipeWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed pipe"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct InterruptingReader {
        interrupts_remaining: usize,
        inner: Cursor<Vec<u8>>,
    }

    impl Read for InterruptingReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.inner.read(buf)
        }
    }

    impl BufRead for InterruptingReader {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            if self.interrupts_remaining > 0 {
                self.interrupts_remaining -= 1;
                return Err(io::Error::new(io::ErrorKind::Interrupted, "EINTR"));
            }
            self.inner.fill_buf()
        }

        fn consume(&mut self, amt: usize) {
            self.inner.consume(amt);
        }
    }

    fn test_runtime(sandbox: &Path) -> McpRuntime {
        McpRuntime {
            http_client: HttpClient::new(TransportConfig::default()),
            resolver_config: ResolverConfig::default(),
            staging_policy: StagingPolicy::for_sandbox(sandbox),
        }
    }

    fn run_mcp_session(input: &str, runtime: &McpRuntime) -> Vec<Value> {
        let mut output = Vec::new();
        serve_io(Cursor::new(input.as_bytes()), &mut output, runtime).unwrap();
        let text = String::from_utf8(output).unwrap();
        text.lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect()
    }

    fn spawn_counting_markdown_server(
        body: &'static str,
    ) -> (
        std::net::SocketAddr,
        Arc<AtomicUsize>,
        thread::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let server_hits = Arc::clone(&hits);
        let handle = thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                server_hits.fetch_add(1, Ordering::SeqCst);
                let mut req_buf = [0u8; 4096];
                let _ = stream.read(&mut req_buf);
                let headers = [
                    "HTTP/1.1 200 OK".to_owned(),
                    "Content-Type: text/markdown; charset=utf-8".to_owned(),
                    format!("Content-Length: {}", body.len()),
                    "Connection: close".to_owned(),
                    String::new(),
                    String::new(),
                ]
                .join("\r\n");
                let _ = stream.write_all(headers.as_bytes());
                let _ = stream.write_all(body.as_bytes());
            }
        });
        (addr, hits, handle)
    }

    #[test]
    fn negotiates_2025_06_18_and_rejects_jsonrpc_batch_arrays() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let input = indoc! {r#"
            {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}
            {"jsonrpc":"2.0","method":"notifications/initialized"}
            [{"jsonrpc":"2.0","id":2,"method":"ping"}]
        "#};

        let responses = run_mcp_session(input, &runtime);
        assert_eq!(responses.len(), 2);
        assert_eq!(
            responses[0].pointer("/result/protocolVersion"),
            Some(&json!("2025-06-18"))
        );
        assert_eq!(responses[1].pointer("/error/code"), Some(&json!(-32600)));
        assert_eq!(responses[1].get("id"), Some(&Value::Null));
    }

    #[test]
    fn rejects_duplicate_initialize_mid_session() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let input = indoc! {r#"
            {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}
            {"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}
        "#};

        let responses = run_mcp_session(input, &runtime);
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[1].pointer("/error/code"), Some(&json!(-32600)));
        assert_eq!(responses[1].get("id"), Some(&json!(2)));
    }

    #[rstest]
    #[case("2025-03-26")]
    #[case("2024-11-05")]
    fn accepts_jsonrpc_batches_on_older_protocol_revisions(#[case] version: &str) {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let input = formatdoc! {r#"
            {{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"{version}"}}}}
            [{{"jsonrpc":"2.0","id":2,"method":"ping"}},{{"jsonrpc":"2.0","method":"notifications/initialized"}}]
            [{{"jsonrpc":"2.0","method":"notifications/initialized"}}]
        "#};

        let responses = run_mcp_session(&input, &runtime);
        assert_eq!(responses.len(), 2);
        assert_eq!(
            responses[0].pointer("/result/protocolVersion"),
            Some(&json!(version))
        );
        assert_eq!(
            responses[1],
            json!([{"jsonrpc": "2.0", "id": 2, "result": {}}])
        );
    }

    #[test]
    fn rejects_empty_oversized_and_non_object_batches_under_2025_03_26() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let oversized: Vec<Value> = (1..=33)
            .map(|idx| json!({"jsonrpc": "2.0", "id": idx, "method": "ping"}))
            .collect();
        let oversized_json = serde_json::to_string(&oversized).unwrap();
        let input = formatdoc! {r#"
            {{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-03-26"}}}}
            []
            [42]
            {oversized_json}
        "#};

        let responses = run_mcp_session(&input, &runtime);
        assert_eq!(responses.len(), 4);
        assert_eq!(responses[1].pointer("/error/code"), Some(&json!(-32600)));
        assert_eq!(responses[2].pointer("/0/error/code"), Some(&json!(-32600)));
        assert_eq!(responses[3].pointer("/error/code"), Some(&json!(-32600)));
    }

    #[test]
    fn distinguishes_null_id_request_from_absent_id_notification() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let input = indoc! {r#"
            {"jsonrpc":"2.0","method":"ping"}
            {"jsonrpc":"2.0","id":null,"method":"ping"}
        "#};

        let responses = run_mcp_session(input, &runtime);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].get("id"), Some(&Value::Null));
        assert_eq!(responses[0].get("result"), Some(&json!({})));
    }

    #[test]
    fn rejects_pre_init_batch_and_tool_methods() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let input = indoc! {r#"
            [{"jsonrpc":"2.0","id":2,"method":"ping"}]
            {"jsonrpc":"2.0","id":3,"method":"tools/list"}
            {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"fetch","arguments":{"target":"2409.13740"}}}
        "#};

        let responses = run_mcp_session(input, &runtime);
        let codes: Vec<Option<i64>> = responses
            .iter()
            .map(|item| item.pointer("/error/code").and_then(Value::as_i64))
            .collect();
        assert_eq!(codes, vec![Some(-32600), Some(-32600), Some(-32600)]);
        assert!(
            responses[0]
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap()
                .contains("before `initialize`")
        );
    }

    #[test]
    fn lists_fetch_and_slice_pdf_with_read_only_and_open_world_annotations() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let input = format!(
            "{INIT_2025_06_18}\r\n{{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/list\"}}\r\n"
        );

        let responses = run_mcp_session(&input, &runtime);
        let tools = responses[1]
            .pointer("/result/tools")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(
            tools[0].pointer("/annotations/openWorldHint"),
            Some(&json!(true))
        );
        assert_eq!(
            tools[1].pointer("/annotations/openWorldHint"),
            Some(&json!(false))
        );
        assert_eq!(
            tools[1].pointer("/annotations/readOnlyHint"),
            Some(&json!(true))
        );
    }

    #[rstest]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":true}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1.5}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":"1"}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":0}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1,"unexpected":1}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":{"target":42}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":{"target":"2409.13740","include_pdf_index":"yes"}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":{"target":"2409.13740","extra":1}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"unknown_tool","arguments":{}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"unknown/method"}"#, -32601)]
    #[case(r#"{"jsonrpc":"1.0","id":2,"method":"ping"}"#, -32600)]
    #[case(r#"{"jsonrpc":"2.0","id":true,"method":"ping"}"#, -32600)]
    #[case(r#"{"jsonrpc":"2.0","id":1.5,"method":"ping"}"#, -32600)]
    #[case(r#"{"jsonrpc":"2.0","id":[],"method":"ping"}"#, -32600)]
    #[case(r#"{"jsonrpc":"2.0","id":{},"method":"ping"}"#, -32600)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"ping","params":123}"#, -32600)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"ping","params":null}"#, -32600)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"ping","params":[1]}"#, -32602)]
    #[case(r#"{"not valid json"#, -32700)]
    fn rejects_invalid_jsonrpc_and_tool_arguments(#[case] frame: &str, #[case] expected_code: i64) {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let input = format!("{INIT_2025_06_18}\n{frame}\n");

        let responses = run_mcp_session(&input, &runtime);
        assert_eq!(
            responses[1].pointer("/error/code"),
            Some(&json!(expected_code))
        );
    }

    #[test]
    fn rejects_form_feed_prefixed_frame_and_invalid_initialize_protocol_type() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let input = formatdoc! {"
            \x0c{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}}
            {{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"initialize\",\"params\":{{\"protocolVersion\":123}}}}
            {{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"initialize\",\"params\":{{\"protocolVersion\":\"2099-01-01\"}}}}
        "};

        let responses = run_mcp_session(&input, &runtime);
        assert_eq!(responses[0].pointer("/error/code"), Some(&json!(-32700)));
        assert_eq!(responses[1].pointer("/error/code"), Some(&json!(-32602)));
        assert_eq!(
            responses[2].pointer("/result/protocolVersion"),
            Some(&json!("2025-06-18"))
        );
    }

    #[test]
    fn executes_slice_pdf_happy_path() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let pdf_dir = sandbox.path().join("staged");
        let pdf_path = runtime
            .staging_policy
            .write_test_pdf(
                &pdf_dir,
                "sample.pdf",
                &["Page one", "Page two", "Page three"],
            )
            .unwrap();
        let pdf_location = pdf_path.display().to_string();
        let ok_call = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "slice_pdf",
                "arguments": {
                    "pdf_path": pdf_location,
                    "start_page": 2,
                    "end_page": 3
                }
            }
        });
        let input = formatdoc! {"
            {INIT_2025_06_18}
            {ok_call}
        "};

        let responses = run_mcp_session(&input, &runtime);
        let text = responses[1]
            .pointer("/result/content/0/text")
            .and_then(Value::as_str)
            .unwrap();
        assert_eq!(responses[1].pointer("/result/isError"), Some(&json!(false)));
        assert!(text.contains("STATUS: OK"));
        assert!(text.contains("PAGES: 2-3 (of 3)"));
    }

    #[test]
    fn returns_tool_error_for_invalid_slice_pdf_page_span() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let pdf_dir = sandbox.path().join("staged");
        let pdf_path = runtime
            .staging_policy
            .write_test_pdf(&pdf_dir, "sample.pdf", &["Page one", "Page two"])
            .unwrap();
        let bad_span = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "slice_pdf",
                "arguments": {
                    "pdf_path": pdf_path.display().to_string(),
                    "start_page": 5,
                    "end_page": 2
                }
            }
        });
        let input = formatdoc! {"
            {INIT_2025_06_18}
            {bad_span}
        "};

        let responses = run_mcp_session(&input, &runtime);
        assert_eq!(responses[1].pointer("/result/isError"), Some(&json!(true)));
    }

    #[test]
    fn executes_fetch_happy_path_and_blocks_forbidden_output_dir_before_network() {
        let sandbox = tempfile::tempdir().unwrap();
        let body = indoc! {"
            # Grounded Paper

            ## Section 1

            Detailed prose with equation $E = mc^2$ and verifiable empirical results across all benchmarks.
        "};
        let (addr, hits, handle) = spawn_counting_markdown_server(body);
        let mut allowed_loopback_addrs = BTreeSet::new();
        allowed_loopback_addrs.insert(addr);
        let runtime = McpRuntime {
            http_client: HttpClient::new(TransportConfig {
                allowed_loopback_addrs,
                ..TransportConfig::default()
            }),
            resolver_config: ResolverConfig::default(),
            staging_policy: StagingPolicy::for_sandbox(sandbox.path()),
        };

        let blocked_out = sandbox.path().join("home/.ssh/blocked");
        let blocked_call = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "fetch",
                "arguments": {
                    "target": format!("http://{addr}/paper.md"),
                    "output_dir": blocked_out.display().to_string()
                }
            }
        });
        let valid_out = sandbox.path().join("lit");
        let ok_call = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "fetch",
                "arguments": {
                    "target": format!("http://{addr}/paper.md"),
                    "output_dir": valid_out.display().to_string()
                }
            }
        });
        let input = formatdoc! {"
            {INIT_2025_06_18}
            {blocked_call}
            {ok_call}
        "};

        let responses = run_mcp_session(&input, &runtime);
        handle.join().unwrap();
        let summary = responses[2]
            .pointer("/result/content/0/text")
            .and_then(Value::as_str)
            .unwrap();

        assert_eq!(responses[1].pointer("/result/isError"), Some(&json!(true)));
        assert_eq!(responses[2].pointer("/result/isError"), Some(&json!(false)));
        assert!(summary.contains("Grounded Paper"));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn retries_eintr_when_not_stopping() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let reader = InterruptingReader {
            interrupts_remaining: 2,
            inner: Cursor::new(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n".to_vec()),
        };
        let mut output = Vec::new();

        serve_io_until(reader, &mut output, &runtime, || false).unwrap();
        let reply: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(reply.get("result"), Some(&json!({})));
    }

    #[test]
    fn exits_cleanly_on_eintr_when_stop_is_signaled() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let reader = InterruptingReader {
            interrupts_remaining: 1,
            inner: Cursor::new(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n".to_vec()),
        };
        let mut output = Vec::new();

        serve_io_until(reader, &mut output, &runtime, || true).unwrap();
        assert_eq!(output, b"");
    }

    #[test]
    fn treats_broken_pipe_stdout_as_clean_exit() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let ping = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;

        let result = serve_io(Cursor::new(ping.as_bytes()), BrokenPipeWriter, &runtime);
        assert!(result.is_ok());
    }

    #[test]
    fn recovers_after_oversized_frame_and_rejects_unterminated_oversized_eof() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let oversized_a = "x".repeat(MAX_JSONRPC_LINE_BYTES + 16);
        let oversized_b = "y".repeat(MAX_JSONRPC_LINE_BYTES + 16);
        let recovery_input = formatdoc! {"
            {oversized_a}
            {{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}}
            {oversized_b}"};

        let responses = run_mcp_session(&recovery_input, &runtime);
        assert_eq!(responses.len(), 3);
        assert_eq!(responses[0].pointer("/error/code"), Some(&json!(-32600)));
        assert_eq!(responses[1].get("result"), Some(&json!({})));
        assert_eq!(responses[2].pointer("/error/code"), Some(&json!(-32600)));
    }

    #[test]
    fn stops_between_frames_when_stop_signal_becomes_true() {
        let sandbox = tempfile::tempdir().unwrap();
        let runtime = test_runtime(sandbox.path());
        let checks = std::cell::Cell::new(0usize);
        let input = indoc! {r#"
            {"jsonrpc":"2.0","id":1,"method":"ping"}
            {"jsonrpc":"2.0","id":2,"method":"ping"}
        "#};
        let mut output = Vec::new();

        serve_io_until(Cursor::new(input.as_bytes()), &mut output, &runtime, || {
            let count = checks.get();
            checks.set(count + 1);
            count >= 1
        })
        .unwrap();

        let reply_count = String::from_utf8(output).unwrap().lines().count();
        assert_eq!(reply_count, 1);
    }
}
