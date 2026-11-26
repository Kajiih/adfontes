//! Newline-delimited JSON-RPC 2.0 stdio Model Context Protocol (MCP) server.
//!
//! Supports MCP protocol revisions `2025-06-18`, `2025-03-26`, and `2024-11-05`, exposing
//! read-only `fetch` and `slice_pdf` tools over bounded (`<= 1 MiB`) newline-delimited frames.
//! Rejects JSON-RPC batch arrays before `initialize` and under `2025-06-18` (while accepting
//! bounded batches on `2025-03-26` and `2024-11-05`), distinguishes notifications (absent `id`)
//! from `"id": null`, retries interrupted (`EINTR`) reads unless signaled to stop, and treats
//! `BrokenPipe` (`EPIPE`) on stdout as a clean exit.

architecture_component!(Mcp);

use std::io::{self, BufRead, Write};
use std::num::NonZeroU32;
use std::path::Path;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::model::{AdfontesError, ExtractedDocument, PageSpan, ParsedTarget};
use crate::stage::{StagingPolicy, slice_staged_pdf, stage_document, validate_staging_dir_policy};

const PROTOCOL_V2025_06_18: &str = "2025-06-18";
const PROTOCOL_V2025_03_26: &str = "2025-03-26";
const PROTOCOL_V2024_11_05: &str = "2024-11-05";

const JSONRPC_VERSION: &str = "2.0";
const SERVER_NAME: &str = "adfontes";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

const METHOD_INITIALIZE: &str = "initialize";
const METHOD_PING: &str = "ping";
const METHOD_LIST_TOOLS: &str = "tools/list";
const METHOD_CALL_TOOL: &str = "tools/call";

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
const FIELD_NAME: &str = "name";
const FIELD_TOOLS: &str = "tools";
const FIELD_TYPE: &str = "type";
const FIELD_TEXT: &str = "text";
const FIELD_DESCRIPTION: &str = "description";

const SCHEMA_STRING: &str = "string";
const SCHEMA_INTEGER: &str = "integer";
const CONTENT_TYPE_TEXT: &str = "text";

const ERROR_PARSE: i64 = -32700;
const ERROR_INVALID_REQUEST: i64 = -32600;
const ERROR_METHOD_NOT_FOUND: i64 = -32601;
const ERROR_INVALID_PARAMS: i64 = -32602;

const MESSAGE_NOT_INITIALIZED: &str = "Server is not initialized; send `initialize` first";

const MAX_JSONRPC_LINE_BYTES: usize = 1024 * 1024;
const MAX_BATCH_ITEMS: usize = 32;

/// Runs the stdio MCP server loop while consulting `should_stop` whenever a read is interrupted by
/// a signal (`io::ErrorKind::Interrupted`) or during async retrieval polling inside `fetcher`.
///
/// If `should_stop` returns `true` on `EINTR` or `fetcher` signals cancellation by returning
/// [`AdfontesError::Io`] with `io::ErrorKind::Interrupted` (for example, a pending Python `SIGINT`),
/// the server returns `Ok(())` cleanly without writing an error or partial batch frame; if
/// `should_stop` returns `false` on `EINTR` (for example, `SIGWINCH` or `SIGCHLD`), the read is
/// retried.
///
/// # Errors
///
/// Returns [`AdfontesError::Io`] on non-`BrokenPipe`, non-interrupted stream read/write failures.
pub fn serve_io_until<R, W, F, S>(
    mut reader: R,
    mut writer: W,
    staging_policy: &StagingPolicy,
    mut fetcher: F,
    mut should_stop: S,
) -> Result<(), AdfontesError>
where
    R: BufRead,
    W: Write,
    F: FnMut(&ParsedTarget, &mut dyn FnMut() -> bool) -> Result<ExtractedDocument, AdfontesError>,
    S: FnMut() -> bool,
{
    let mut session = McpSession {
        negotiated_version: None,
        staging_policy,
        fetcher: &mut fetcher,
        should_stop: &mut should_stop,
    };

    loop {
        if (session.should_stop)() {
            return Ok(());
        }
        match read_bounded_line(
            &mut reader,
            MAX_JSONRPC_LINE_BYTES,
            &mut *session.should_stop,
        )? {
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
                let Ok(maybe_reply) = session.handle_raw_frame(trimmed) else {
                    return Ok(());
                };
                if let Some(reply) = maybe_reply
                    && !write_json_line(&mut writer, &reply)?
                {
                    return Ok(());
                }
            }
        }
    }
}

struct McpSession<'a, F, S> {
    negotiated_version: Option<ProtocolVersion>,
    staging_policy: &'a StagingPolicy,
    fetcher: &'a mut F,
    should_stop: &'a mut S,
}

impl<F, S> McpSession<'_, F, S>
where
    F: FnMut(&ParsedTarget, &mut dyn FnMut() -> bool) -> Result<ExtractedDocument, AdfontesError>,
    S: FnMut() -> bool,
{
    fn handle_raw_frame(&mut self, frame_bytes: &[u8]) -> Result<Option<Value>, McpInterrupted> {
        let parsed: Value = match serde_json::from_slice(frame_bytes) {
            Ok(value) => value,
            Err(parse_error) => {
                return Ok(Some(jsonrpc_error(
                    &Value::Null,
                    ERROR_PARSE,
                    &format!("Parse error: {parse_error}"),
                )));
            }
        };

        match parsed {
            Value::Object(object) => self.handle_single_object(&object),
            Value::Array(items) => self.handle_batch_array(&items),
            _ => Ok(Some(jsonrpc_error(
                &Value::Null,
                ERROR_INVALID_REQUEST,
                "JSON-RPC message must be an object",
            ))),
        }
    }

    fn handle_batch_array(&mut self, items: &[Value]) -> Result<Option<Value>, McpInterrupted> {
        let Some(version) = self.negotiated_version else {
            return Ok(Some(jsonrpc_error(
                &Value::Null,
                ERROR_INVALID_REQUEST,
                "JSON-RPC batch requests are not allowed before `initialize`",
            )));
        };
        if !version.allows_jsonrpc_batch() {
            return Ok(Some(jsonrpc_error(
                &Value::Null,
                ERROR_INVALID_REQUEST,
                "JSON-RPC batch requests are not supported in MCP protocol version 2025-06-18",
            )));
        }
        if items.is_empty() {
            return Ok(Some(jsonrpc_error(
                &Value::Null,
                ERROR_INVALID_REQUEST,
                "JSON-RPC batch array must not be empty",
            )));
        }
        if items.len() > MAX_BATCH_ITEMS {
            return Ok(Some(jsonrpc_error(
                &Value::Null,
                ERROR_INVALID_REQUEST,
                "JSON-RPC batch exceeds maximum of 32 items",
            )));
        }

        let mut replies = Vec::new();
        for item in items {
            if (self.should_stop)() {
                return Err(McpInterrupted);
            }
            let reply = match item {
                Value::Object(object) => self.handle_single_object(object)?,
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
        Ok((!replies.is_empty()).then_some(Value::Array(replies)))
    }

    fn handle_single_object(
        &mut self,
        object: &Map<String, Value>,
    ) -> Result<Option<Value>, McpInterrupted> {
        let call = match parse_jsonrpc_call(object) {
            Ok(parsed) => parsed,
            Err(error_value) => return Ok(Some(error_value)),
        };

        // Notifications (absent `id` key) must never receive a response.
        let Some(request_id) = call.id else {
            return Ok(None);
        };

        let response = match call.method {
            METHOD_INITIALIZE => self.handle_initialize(&request_id, call.params),
            METHOD_PING => handle_ping(&request_id, call.params),
            METHOD_LIST_TOOLS => self.handle_tools_list(&request_id, call.params),
            METHOD_CALL_TOOL => self.handle_tools_call(&request_id, call.params)?,
            other => jsonrpc_error(
                &request_id,
                ERROR_METHOD_NOT_FOUND,
                &format!("Method not found: {other}"),
            ),
        };
        Ok(Some(response))
    }

    fn handle_initialize(&mut self, request_id: &Value, params: Option<&Value>) -> Value {
        if self.negotiated_version.is_some() {
            return jsonrpc_error(
                request_id,
                ERROR_INVALID_REQUEST,
                "MCP session is already initialized",
            );
        }

        let parsed: InitializeParams<'_> =
            match parse_optional_object_params(params, METHOD_INITIALIZE) {
                Ok(val) => val,
                Err(detail) => return jsonrpc_error(request_id, ERROR_INVALID_PARAMS, &detail),
            };

        let negotiated = ProtocolVersion::negotiate(parsed.protocol_version);
        self.negotiated_version = Some(negotiated);

        jsonrpc_result(
            request_id,
            &json!({
                "protocolVersion": negotiated.as_str(),
                "capabilities": { FIELD_TOOLS: { "listChanged": false } },
                "serverInfo": {
                    FIELD_NAME: SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            }),
        )
    }

    fn handle_tools_list(&self, request_id: &Value, params: Option<&Value>) -> Value {
        if self.negotiated_version.is_none() {
            return jsonrpc_error(request_id, ERROR_INVALID_REQUEST, MESSAGE_NOT_INITIALIZED);
        }
        if let Err(detail) = parse_optional_object_params::<EmptyParams>(params, METHOD_LIST_TOOLS)
        {
            return jsonrpc_error(request_id, ERROR_INVALID_PARAMS, &detail);
        }
        jsonrpc_result(request_id, &build_tools_list_result())
    }

    fn handle_tools_call(
        &mut self,
        request_id: &Value,
        params: Option<&Value>,
    ) -> Result<Value, McpInterrupted> {
        if self.negotiated_version.is_none() {
            return Ok(jsonrpc_error(
                request_id,
                ERROR_INVALID_REQUEST,
                MESSAGE_NOT_INITIALIZED,
            ));
        }
        let Some(call_params) = params.and_then(Value::as_object) else {
            return Ok(jsonrpc_error(
                request_id,
                ERROR_INVALID_PARAMS,
                &format!("`{METHOD_CALL_TOOL}` `params` must be an object"),
            ));
        };
        let parsed = match ToolsCallParams::deserialize(call_params) {
            Ok(valid) => valid,
            Err(error) => {
                return Ok(jsonrpc_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    &format!("Invalid `{METHOD_CALL_TOOL}` params: {error}"),
                ));
            }
        };
        let arguments = parsed.arguments.unwrap_or_default();

        match parsed.name {
            TOOL_FETCH => self.execute_fetch_tool(request_id, &arguments),
            TOOL_SLICE_PDF => Ok(self.execute_slice_pdf_tool(request_id, &arguments)),
            unknown => Ok(jsonrpc_error(
                request_id,
                ERROR_INVALID_PARAMS,
                &format!("Unknown tool: {unknown}"),
            )),
        }
    }

    fn execute_fetch_tool(
        &mut self,
        request_id: &Value,
        arguments: &Map<String, Value>,
    ) -> Result<Value, McpInterrupted> {
        let args = match FetchArgs::deserialize(arguments) {
            Ok(parsed) => parsed,
            Err(error) => {
                return Ok(jsonrpc_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    &format!("Invalid arguments for `{TOOL_FETCH}`: {error}"),
                ));
            }
        };

        let target = match ParsedTarget::parse(args.target) {
            Ok(parsed) => parsed,
            Err(error) => return Ok(tool_text_response(request_id, &error.to_string(), true)),
        };
        let output_path = args.output_dir.map(Path::new);
        if let Err(error) = validate_staging_dir_policy(output_path, self.staging_policy) {
            return Ok(tool_text_response(request_id, &error.to_string(), true));
        }
        let extracted = match (self.fetcher)(&target, &mut *self.should_stop) {
            Ok(document) => document,
            Err(AdfontesError::Io { source, .. })
                if source.kind() == io::ErrorKind::Interrupted =>
            {
                return Err(McpInterrupted);
            }
            Err(error) => return Ok(tool_text_response(request_id, &error.to_string(), true)),
        };
        Ok(
            match stage_document(
                &target,
                extracted,
                output_path,
                args.include_pdf_index.unwrap_or(true),
                self.staging_policy,
            ) {
                Ok(staged) => tool_text_response(request_id, &staged.compact_index, false),
                Err(error) => tool_text_response(request_id, &error.to_string(), true),
            },
        )
    }

    fn execute_slice_pdf_tool(&self, request_id: &Value, arguments: &Map<String, Value>) -> Value {
        let args = match SlicePdfArgs::deserialize(arguments) {
            Ok(parsed) => parsed,
            Err(error) => {
                return jsonrpc_error(
                    request_id,
                    ERROR_INVALID_PARAMS,
                    &format!("Invalid arguments for `{TOOL_SLICE_PDF}`: {error}"),
                );
            }
        };

        let end_page = args.end_page.unwrap_or(args.start_page);
        let span = match PageSpan::new(args.start_page.get(), end_page.get()) {
            Ok(valid_span) => valid_span,
            Err(error) => return tool_text_response(request_id, &error.to_string(), true),
        };
        let output_path = args.output_dir.map(Path::new);
        match slice_staged_pdf(
            Path::new(args.pdf_path),
            span,
            output_path,
            self.staging_policy,
        ) {
            Ok(slice) => tool_text_response(request_id, &slice.summary, false),
            Err(error) => tool_text_response(request_id, &error.to_string(), true),
        }
    }
}

struct McpInterrupted;

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

fn is_valid_jsonrpc_id(value: &Value) -> bool {
    match value {
        Value::Null | Value::String(_) => true,
        Value::Number(number) => number.is_i64() || number.is_u64(),
        Value::Bool(_) | Value::Array(_) | Value::Object(_) => false,
    }
}

fn handle_ping(request_id: &Value, params: Option<&Value>) -> Value {
    if let Err(detail) = parse_optional_object_params::<EmptyParams>(params, METHOD_PING) {
        return jsonrpc_error(request_id, ERROR_INVALID_PARAMS, &detail);
    }
    jsonrpc_result(request_id, &json!({}))
}

fn parse_optional_object_params<'a, T>(params: Option<&'a Value>, method: &str) -> Result<T, String>
where
    T: Default + Deserialize<'a>,
{
    let Some(value) = params else {
        return Ok(T::default());
    };
    let Some(object) = value.as_object() else {
        return Err(format!(
            "`{method}` `params` must be an object when present"
        ));
    };
    T::deserialize(object).map_err(|error| format!("Invalid `{method}` params: {error}"))
}

#[derive(Default, Deserialize)]
struct InitializeParams<'a> {
    #[serde(default, rename = "protocolVersion")]
    protocol_version: Option<&'a str>,
}

#[derive(Default, Deserialize)]
struct EmptyParams {}

#[derive(Deserialize)]
struct ToolsCallParams<'a> {
    name: &'a str,
    #[serde(default)]
    arguments: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FetchArgs<'a> {
    target: &'a str,
    #[serde(default)]
    output_dir: Option<&'a str>,
    #[serde(default)]
    include_pdf_index: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SlicePdfArgs<'a> {
    pdf_path: &'a str,
    start_page: NonZeroU32,
    #[serde(default)]
    end_page: Option<NonZeroU32>,
    #[serde(default)]
    output_dir: Option<&'a str>,
}

fn build_tools_list_result() -> Value {
    json!({
        FIELD_TOOLS: [
            tool_definition(
                TOOL_FETCH,
                "Retrieve full-text academic papers, specifications, or articles via keyless open-access waterfalls, stage normalized Markdown or PDF locally, and return a compact (<= 3,600 B) provenance header and line/page Table of Contents for read_file.",
                &[ARG_TARGET],
                &json!({
                    ARG_TARGET: string_property(
                        "arXiv ID/URL, DOI, PMCID, PMID, IETF RFC, or http(s):// URL.",
                    ),
                    ARG_OUTPUT_DIR: string_property(
                        "Optional confined staging directory inside temp, ADFONTES_EXTRA_ALLOWED_ROOTS, or ADFONTES_STAGING_DIR.",
                    ),
                    ARG_INCLUDE_PDF_INDEX: bool_property(
                        true,
                        "When the retrieved document is a PDF, write a companion <slug>.pages.md page map.",
                    ),
                }),
                "Fetch Literature",
                true,
            ),
            tool_definition(
                TOOL_SLICE_PDF,
                "Extract a lossless 1..=8-page vector PDF slice from a staged PDF file so read_file can visually inspect target equations, plots, or tables without attaching the entire PDF.",
                &[ARG_PDF_PATH, ARG_START_PAGE],
                &json!({
                    ARG_PDF_PATH: string_property(
                        "Absolute path to a confined staged PDF file.",
                    ),
                    ARG_START_PAGE: positive_int_property(
                        "1-indexed first page to extract.",
                    ),
                    ARG_END_PAGE: positive_int_property(
                        "Optional 1-indexed inclusive last page (defaults to start_page; maximum span 8 pages).",
                    ),
                    ARG_OUTPUT_DIR: string_property(
                        "Optional confined output directory (defaults to pdf_path's directory).",
                    ),
                }),
                "Slice Staged PDF",
                false,
            ),
        ]
    })
}

fn tool_definition(
    name: &str,
    description: &str,
    required: &[&str],
    properties: &Value,
    title: &str,
    open_world_hint: bool,
) -> Value {
    json!({
        FIELD_NAME: name,
        FIELD_DESCRIPTION: description,
        "inputSchema": {
            FIELD_TYPE: "object",
            "additionalProperties": false,
            "required": required,
            "properties": properties,
        },
        "annotations": {
            "title": title,
            "readOnlyHint": true,
            "destructiveHint": false,
            "idempotentHint": true,
            "openWorldHint": open_world_hint,
        },
    })
}

fn string_property(description: &str) -> Value {
    json!({
        FIELD_TYPE: SCHEMA_STRING,
        FIELD_DESCRIPTION: description,
    })
}

fn positive_int_property(description: &str) -> Value {
    json!({
        FIELD_TYPE: SCHEMA_INTEGER,
        "minimum": 1,
        FIELD_DESCRIPTION: description,
    })
}

fn bool_property(default: bool, description: &str) -> Value {
    json!({
        FIELD_TYPE: "boolean",
        "default": default,
        FIELD_DESCRIPTION: description,
    })
}

fn tool_text_response(request_id: &Value, text: &str, is_error: bool) -> Value {
    jsonrpc_result(
        request_id,
        &json!({
            "content": [{ FIELD_TYPE: CONTENT_TYPE_TEXT, FIELD_TEXT: text }],
            "isError": is_error,
        }),
    )
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
    use std::io::{Cursor, Read};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::{formatdoc, indoc};
    use rstest::rstest;

    use super::*;
    use crate::model::{ExtractedPayload, HttpUrl, TextFormat, WorkMetadata};
    use crate::stage::TestSandbox;

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

    fn unexpected_fetcher(
        target: &ParsedTarget,
        _should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<ExtractedDocument, AdfontesError> {
        Err(AdfontesError::Transport {
            url: format!("{target:?}"),
            detail: "unexpected fetch call in unit test".to_owned(),
        })
    }

    fn run_mcp_session_with<F>(input: &str, policy: &StagingPolicy, fetcher: F) -> Vec<Value>
    where
        F: FnMut(
            &ParsedTarget,
            &mut dyn FnMut() -> bool,
        ) -> Result<ExtractedDocument, AdfontesError>,
    {
        let mut output = Vec::new();
        serve_io_until(
            Cursor::new(input.as_bytes()),
            &mut output,
            policy,
            fetcher,
            || false,
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        text.lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect()
    }

    fn run_sandbox_session(input: &str) -> Vec<Value> {
        let sandbox = TestSandbox::new();
        run_mcp_session_with(input, &sandbox.policy(), unexpected_fetcher)
    }

    fn setup_staged_sample_pdf(sandbox: &Path, pages: &[&str]) -> (StagingPolicy, String) {
        let policy = StagingPolicy::for_sandbox(sandbox);
        let pdf_dir = sandbox.join("staged");
        let pdf_path = policy
            .write_test_pdf(&pdf_dir, "sample.pdf", pages)
            .unwrap();
        (policy, pdf_path.display().to_string())
    }

    #[test]
    fn negotiates_2025_06_18_and_rejects_jsonrpc_batch_arrays() {
        let input = indoc! {r#"
            {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}
            {"jsonrpc":"2.0","method":"notifications/initialized"}
            [{"jsonrpc":"2.0","id":2,"method":"ping"}]
        "#};

        let responses = run_sandbox_session(input);
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
        let input = indoc! {r#"
            {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}
            {"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}
        "#};

        let responses = run_sandbox_session(input);
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[1].pointer("/error/code"), Some(&json!(-32600)));
        assert_eq!(responses[1].get("id"), Some(&json!(2)));
    }

    #[rstest]
    #[case("2025-03-26")]
    #[case("2024-11-05")]
    fn accepts_jsonrpc_batches_on_older_protocol_revisions(#[case] version: &str) {
        let input = formatdoc! {r#"
            {{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"{version}"}}}}
            [{{"jsonrpc":"2.0","id":2,"method":"ping"}},{{"jsonrpc":"2.0","method":"notifications/initialized"}}]
            [{{"jsonrpc":"2.0","method":"notifications/initialized"}}]
        "#};

        let responses = run_sandbox_session(&input);
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

        let responses = run_sandbox_session(&input);
        assert_eq!(responses.len(), 4);
        assert_eq!(responses[1].pointer("/error/code"), Some(&json!(-32600)));
        assert_eq!(responses[2].pointer("/0/error/code"), Some(&json!(-32600)));
        assert_eq!(responses[3].pointer("/error/code"), Some(&json!(-32600)));
    }

    #[test]
    fn distinguishes_null_id_request_from_absent_id_notification() {
        let input = indoc! {r#"
            {"jsonrpc":"2.0","method":"ping"}
            {"jsonrpc":"2.0","id":null,"method":"ping"}
        "#};

        let responses = run_sandbox_session(input);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].get("id"), Some(&Value::Null));
        assert_eq!(responses[0].get("result"), Some(&json!({})));
    }

    #[test]
    fn rejects_pre_init_batch_and_tool_methods() {
        let input = indoc! {r#"
            [{"jsonrpc":"2.0","id":2,"method":"ping"}]
            {"jsonrpc":"2.0","id":3,"method":"tools/list"}
            {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"fetch","arguments":{"target":"2409.13740"}}}
        "#};

        let responses = run_sandbox_session(input);
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
        let input = format!(
            "{INIT_2025_06_18}\r\n{{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/list\"}}\r\n"
        );

        let responses = run_sandbox_session(&input);
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
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1.0}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1.5}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":"1"}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":0}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":-1}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":4294967296}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1,"end_page":0}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1,"end_page":-1}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1,"end_page":1.0}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1,"end_page":"2"}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1,"end_page":4294967296}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf"}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slice_pdf","arguments":{"pdf_path":"/tmp/a.pdf","start_page":1,"unexpected":1}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":{"target":42}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":{"target":"2409.13740","output_dir":123}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":{"target":"2409.13740","include_pdf_index":"yes"}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":{"target":"2409.13740","extra":1}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":{}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch"}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":null}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":["2409.13740"]}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":[]}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"arguments":{}}}"#, -32602)]
    #[case(r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":123,"arguments":{}}}"#, -32602)]
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
        let input = format!("{INIT_2025_06_18}\n{frame}\n");
        let responses = run_sandbox_session(&input);
        assert_eq!(
            responses[1].pointer("/error/code"),
            Some(&json!(expected_code))
        );
    }

    #[test]
    fn rejects_form_feed_prefixed_frame_and_invalid_initialize_protocol_type() {
        let input = formatdoc! {"
            \x0c{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}}
            {{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"initialize\",\"params\":{{\"protocolVersion\":123}}}}
            {{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"initialize\",\"params\":{{\"protocolVersion\":\"2099-01-01\"}}}}
        "};

        let responses = run_sandbox_session(&input);
        assert_eq!(responses[0].pointer("/error/code"), Some(&json!(-32700)));
        assert_eq!(responses[1].pointer("/error/code"), Some(&json!(-32602)));
        assert_eq!(
            responses[2].pointer("/result/protocolVersion"),
            Some(&json!("2025-06-18"))
        );
    }

    #[test]
    fn executes_slice_pdf_happy_path() {
        let sandbox = TestSandbox::new();
        let (policy, pdf_location) =
            setup_staged_sample_pdf(sandbox.path(), &["Page one", "Page two", "Page three"]);
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

        let responses = run_mcp_session_with(&input, &policy, unexpected_fetcher);
        let text = responses[1]
            .pointer("/result/content/0/text")
            .and_then(Value::as_str)
            .unwrap();
        assert_eq!(responses[1].pointer("/result/isError"), Some(&json!(false)));
        assert!(text.contains("STATUS: OK"));
        assert!(text.contains("PAGES: 2-3 (of 3)"));
    }

    #[test]
    fn executes_slice_pdf_with_omitted_and_null_optional_arguments() {
        let sandbox = TestSandbox::new();
        let (policy, pdf_location) =
            setup_staged_sample_pdf(sandbox.path(), &["Page one", "Page two", "Page three"]);
        let omitted_end = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "slice_pdf",
                "arguments": {
                    "pdf_path": pdf_location,
                    "start_page": 2
                }
            }
        });
        let null_optionals = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "slice_pdf",
                "arguments": {
                    "pdf_path": pdf_location,
                    "start_page": 2,
                    "end_page": null,
                    "output_dir": null
                }
            }
        });
        let input = formatdoc! {"
            {INIT_2025_06_18}
            {omitted_end}
            {null_optionals}
        "};

        let responses = run_mcp_session_with(&input, &policy, unexpected_fetcher);
        let text_a = responses[1]
            .pointer("/result/content/0/text")
            .and_then(Value::as_str)
            .unwrap();
        let text_b = responses[2]
            .pointer("/result/content/0/text")
            .and_then(Value::as_str)
            .unwrap();
        assert_eq!(responses[1].pointer("/result/isError"), Some(&json!(false)));
        assert!(text_a.contains("PAGES: 2 (of 3)"));
        assert_eq!(responses[2].pointer("/result/isError"), Some(&json!(false)));
        assert!(text_b.contains("PAGES: 2 (of 3)"));
    }

    #[test]
    fn deserializes_fetch_args_with_null_defaults_and_explicit_false_index() {
        let with_nulls = json!({
            "target": "2409.13740",
            "output_dir": null,
            "include_pdf_index": null
        });
        let with_false = json!({
            "target": "2409.13740",
            "include_pdf_index": false
        });
        let parsed_nulls = FetchArgs::deserialize(with_nulls.as_object().unwrap()).unwrap();
        let parsed_false = FetchArgs::deserialize(with_false.as_object().unwrap()).unwrap();

        assert_eq!(parsed_nulls.target, "2409.13740");
        assert_eq!(parsed_nulls.output_dir, None);
        assert_eq!(parsed_nulls.include_pdf_index, None);
        assert_eq!(parsed_false.include_pdf_index, Some(false));
    }

    #[test]
    fn returns_tool_error_for_invalid_slice_pdf_page_span() {
        let sandbox = TestSandbox::new();
        let (policy, pdf_location) =
            setup_staged_sample_pdf(sandbox.path(), &["Page one", "Page two"]);
        let bad_span = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "slice_pdf",
                "arguments": {
                    "pdf_path": pdf_location,
                    "start_page": 5,
                    "end_page": 2
                }
            }
        });
        let input = formatdoc! {"
            {INIT_2025_06_18}
            {bad_span}
        "};

        let responses = run_mcp_session_with(&input, &policy, unexpected_fetcher);
        assert_eq!(responses[1].pointer("/result/isError"), Some(&json!(true)));
    }

    #[test]
    fn executes_fetch_happy_path_and_blocks_forbidden_output_dir_before_network() {
        let sandbox = TestSandbox::new();
        let body = indoc! {"
            # Grounded Paper

            ## Section 1

            Detailed prose with equation $E = mc^2$ and verifiable empirical results across all benchmarks.
        "};
        let policy = sandbox.policy();
        let hits = Arc::new(AtomicUsize::new(0));
        let fetcher_hits = Arc::clone(&hits);

        let blocked_out = sandbox.path().join("home/.ssh/blocked");
        let blocked_call = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "fetch",
                "arguments": {
                    "target": "https://example.org/paper.md",
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
                    "target": "https://example.org/paper.md",
                    "output_dir": valid_out.display().to_string()
                }
            }
        });
        let input = formatdoc! {"
            {INIT_2025_06_18}
            {blocked_call}
            {ok_call}
        "};

        let responses = run_mcp_session_with(&input, &policy, |_target, _should_stop| {
            fetcher_hits.fetch_add(1, Ordering::SeqCst);
            Ok(ExtractedDocument {
                metadata: WorkMetadata {
                    title: Some("Grounded Paper".to_owned()),
                    source_url: HttpUrl::parse("https://example.org/paper.md").ok(),
                    ..WorkMetadata::default()
                },
                payload: ExtractedPayload::FullText {
                    format: TextFormat::Markdown,
                    markdown: body.to_owned(),
                },
                raw_bytes_fetched: body.len(),
            })
        });
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
    fn exits_cleanly_when_fetcher_reports_interrupted() {
        let sandbox = TestSandbox::new();
        let policy = sandbox.policy();
        let valid_out = sandbox.path().join("lit");
        let interrupted_call = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "fetch",
                "arguments": {
                    "target": "https://example.org/slow.md",
                    "output_dir": valid_out.display().to_string()
                }
            }
        });
        let input = formatdoc! {"
            {INIT_2025_06_18}
            {interrupted_call}
        "};

        let responses = run_mcp_session_with(&input, &policy, |_target, _should_stop| {
            Err(AdfontesError::Io {
                context: "interrupted during network I/O".to_owned(),
                source: io::Error::new(io::ErrorKind::Interrupted, "operation interrupted"),
            })
        });

        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].get("id"), Some(&json!(1)));
    }

    #[test]
    fn retries_eintr_when_not_stopping() {
        let sandbox = TestSandbox::new();
        let policy = sandbox.policy();
        let reader = InterruptingReader {
            interrupts_remaining: 2,
            inner: Cursor::new(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n".to_vec()),
        };
        let mut output = Vec::new();

        serve_io_until(reader, &mut output, &policy, unexpected_fetcher, || false).unwrap();
        let reply: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(reply.get("result"), Some(&json!({})));
    }

    #[test]
    fn exits_cleanly_on_eintr_when_stop_is_signaled() {
        let sandbox = TestSandbox::new();
        let policy = sandbox.policy();
        let reader = InterruptingReader {
            interrupts_remaining: 1,
            inner: Cursor::new(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n".to_vec()),
        };
        let mut output = Vec::new();

        serve_io_until(reader, &mut output, &policy, unexpected_fetcher, || true).unwrap();
        assert_eq!(output, b"");
    }

    #[test]
    fn treats_broken_pipe_stdout_as_clean_exit() {
        let sandbox = TestSandbox::new();
        let policy = sandbox.policy();
        let ping = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;

        let result = serve_io_until(
            Cursor::new(ping.as_bytes()),
            BrokenPipeWriter,
            &policy,
            unexpected_fetcher,
            || false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn recovers_after_oversized_frame_and_rejects_unterminated_oversized_eof() {
        let oversized_a = "x".repeat(MAX_JSONRPC_LINE_BYTES + 16);
        let oversized_b = "y".repeat(MAX_JSONRPC_LINE_BYTES + 16);
        let recovery_input = formatdoc! {"
            {oversized_a}
            {{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}}
            {oversized_b}"};

        let responses = run_sandbox_session(&recovery_input);
        assert_eq!(responses.len(), 3);
        assert_eq!(responses[0].pointer("/error/code"), Some(&json!(-32600)));
        assert_eq!(responses[1].get("result"), Some(&json!({})));
        assert_eq!(responses[2].pointer("/error/code"), Some(&json!(-32600)));
    }

    #[test]
    fn stops_between_frames_when_stop_signal_becomes_true() {
        let sandbox = TestSandbox::new();
        let policy = sandbox.policy();
        let checks = std::cell::Cell::new(0usize);
        let input = indoc! {r#"
            {"jsonrpc":"2.0","id":1,"method":"ping"}
            {"jsonrpc":"2.0","id":2,"method":"ping"}
        "#};
        let mut output = Vec::new();

        serve_io_until(
            Cursor::new(input.as_bytes()),
            &mut output,
            &policy,
            unexpected_fetcher,
            || {
                let count = checks.get();
                checks.set(count + 1);
                count >= 1
            },
        )
        .unwrap();

        let reply_count = String::from_utf8(output).unwrap().lines().count();
        assert_eq!(reply_count, 1);
    }

    #[test]
    fn stops_cleanly_without_partial_output_when_fetcher_interrupted_in_batch() {
        let sandbox = TestSandbox::new();
        let policy = sandbox.policy();
        let batch_call = indoc! {r#"
            {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}
            {"jsonrpc":"2.0","method":"notifications/initialized"}
            [{"jsonrpc":"2.0","id":2,"method":"ping"},{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fetch","arguments":{"target":"2409.13740"}}}]
        "#};
        let mut output = Vec::new();

        serve_io_until(
            Cursor::new(batch_call.as_bytes()),
            &mut output,
            &policy,
            |_target, _should_stop| {
                Err(AdfontesError::Io {
                    context: "polling interrupted fetch".to_owned(),
                    source: io::Error::new(io::ErrorKind::Interrupted, "SIGINT"),
                })
            },
            || false,
        )
        .unwrap();

        let text = String::from_utf8(output).unwrap();
        let responses: Vec<Value> = text
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].get("id"), Some(&json!(1)));
    }
}
