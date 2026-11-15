//! A small MCP server for the tests, on rmcp's server side.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
    Implementation, JsonObject, ListToolsResult, PaginatedRequestParams, ServerCapabilities,
    ServerConfig, Tool, ToolAnnotations,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler};
use serde_json::{json, Value};
use tokio::sync::Notify;

/// What the tests watch and steer.
#[derive(Default)]
pub struct State {
    /// Whether `install` was called: the list then has `extra` and no longer `slow`.
    pub installed: AtomicBool,
    /// `slow` waits for this.
    pub release: Notify,
    /// Notified when `slow` started.
    pub started: Notify,
}

/// The server: weather, files and charts for a household.
#[derive(Clone, Default)]
pub struct TestServer {
    pub state: Arc<State>,
}

fn object(value: Value) -> JsonObject {
    match value {
        Value::Object(map) => map,
        _ => JsonObject::new(),
    }
}

fn city_schema() -> JsonObject {
    object(json!({
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The city"}},
        "required": ["city"]
    }))
}

impl TestServer {
    fn tools(&self) -> Vec<Tool> {
        let mut tools = vec![
            Tool::new("get_weather", "The weather in a city.", city_schema())
                .with_annotations(ToolAnnotations::new().read_only(true)),
            Tool::new(
                "delete_file",
                "Delete a file.",
                object(json!({"type": "object", "properties": {"path": {"type": "string"}}})),
            )
            .with_annotations(ToolAnnotations::new().destructive(true)),
            Tool::new("find_city", "Find a city by name.", city_schema()),
            Tool::new(
                "chart",
                "A chart of the temperature.",
                object(json!({"type": "object"})),
            ),
            // No `type`, a `const` and a definition no `$ref` reaches.
            Tool::new(
                "search",
                "Search the recipes.",
                object(json!({
                    "properties": {"kind": {"const": "recipe"}, "query": {"type": "string"}},
                    "$defs": {"Unused": {"type": "string"}}
                })),
            ),
            // Two names that are the same once cleaned up.
            Tool::new(
                "look.up",
                "Look up a word.",
                object(json!({"type": "object"})),
            ),
            Tool::new(
                "look-up",
                "Look up a place.",
                object(json!({"type": "object"})),
            ),
            Tool::new(
                "trace",
                "Say the trace context.",
                object(json!({"type": "object"})),
            ),
            Tool::new(
                "install",
                "Install the extra tool.",
                object(json!({"type": "object"})),
            ),
        ];
        if self.state.installed.load(Ordering::SeqCst) {
            tools.push(Tool::new(
                "extra",
                "The extra tool.",
                object(json!({"type": "object"})),
            ));
        } else {
            tools.push(Tool::new(
                "slow",
                "Wait to be released.",
                object(json!({"type": "object"})),
            ));
        }
        tools
    }
}

fn text(text: impl Into<String>) -> CallToolResponse {
    CallToolResponse::Complete(CallToolResult::success(vec![ContentBlock::text(text)]))
}

impl ServerHandler for TestServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_tool_list_changed()
                .build(),
        )
        .with_server_info(
            Implementation::new("household", "1.0.0")
                .with_description("Weather, files and charts for the household."),
        )
        .with_instructions("Ask for the weather before planning outside.")
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(self.tools()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let arguments = Value::Object(request.arguments.clone().unwrap_or_default());
        let city = arguments["city"].as_str().unwrap_or("nowhere").to_string();
        match request.name.as_ref() {
            "get_weather" => Ok(text(format!("12 degrees in {city}"))),
            "delete_file" => Ok(text("deleted")),
            "find_city" => Ok(CallToolResponse::Complete(CallToolResult::error(vec![
                ContentBlock::text(format!("no such city: {city}")),
            ]))),
            "chart" => Ok(CallToolResponse::Complete(CallToolResult::success(vec![
                ContentBlock::text("the temperature this week"),
                ContentBlock::image("iVBORw0KGgo=", "image/png"),
            ]))),
            "search" => Ok(text(format!("found {}", arguments["kind"]))),
            "look.up" => Ok(text("a word")),
            "look-up" => Ok(text("a place")),
            "trace" => {
                let traceparent = request
                    .meta
                    .as_ref()
                    .and_then(|meta| meta.get_traceparent())
                    .or_else(|| context.meta.get_traceparent())
                    .unwrap_or("none")
                    .to_string();
                Ok(text(traceparent))
            }
            "install" => {
                self.state.installed.store(true, Ordering::SeqCst);
                context
                    .peer
                    .notify_tool_list_changed()
                    .await
                    .map_err(|error| ErrorData::internal_error(error.to_string(), None))?;
                Ok(text("installed"))
            }
            "slow" => {
                let released = self.state.release.notified();
                self.state.started.notify_one();
                released.await;
                Ok(text("done"))
            }
            "extra" => Ok(text("extra")),
            other => Err(ErrorData::invalid_params(
                format!("unknown tool {other}"),
                None,
            )),
        }
    }
}
