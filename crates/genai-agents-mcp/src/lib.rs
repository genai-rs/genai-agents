//! The tools of [MCP](https://modelcontextprotocol.io) servers as
//! [genai-agents](https://docs.rs/genai-agents) tools, on the official Rust SDK
//! [rmcp](https://docs.rs/rmcp).
//!
//! [`McpServer`] connects to a server (a child process over stdio, streamable HTTP, or any rmcp
//! transport) and [registers](McpServer::register) its tools in a
//! [`ToolRegistry`](genai_agents::tools::ToolRegistry) as dynamic tools:
//!
//! - in a namespace named after the server, described with the server's description (or
//!   prefixed with its name, for models without namespaces: [`Grouping`]);
//! - [deferred](genai_agents::tools::Exposure::Deferred) by default, so that a tool search finds
//!   them and a large server does not fill the context;
//! - with names and schemas cleaned up for the model APIs, and a hash in the names of tools that
//!   would collide;
//! - running at the same time as other tools when the server says they are read-only
//!   (`readOnlyHint`); the other hints, such as `destructiveHint`, are on [`McpTool`] for an
//!   approval policy;
//! - with outputs marked untrusted (content from outside the application), unless
//!   [`McpOptions`] says otherwise.
//!
//! A call's text comes back as text, images as images, embedded files as files; a result the
//! server marks as an error (`isError`) is a failure the model sees.
//!
//! ```no_run
//! use genai_agents::tools::ToolRegistry;
//! use genai_agents_mcp::{McpOptions, McpServer};
//! use tokio::process::Command;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut command = Command::new("npx");
//! command.args(["-y", "@modelcontextprotocol/server-filesystem", "."]);
//! let files = McpServer::stdio("files", command, McpOptions::new()).await?;
//!
//! let mut tools = ToolRegistry::new();
//! for tool in files.register(&mut tools).await? {
//!     println!("{} (destructive: {})", tool.key.name, tool.destructive());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # A changed tool list
//!
//! When the server says its tool list changed (`notifications/tools/list_changed`),
//! [`tools_changed`](McpServer::tools_changed) turns true; [`register`](McpServer::register)
//! again replaces the server's tools. A run keeps the tools it started with, so change a
//! runner's tools between runs:
//!
//! ```no_run
//! # use genai_agents::{Conversation, Model, Runner};
//! # use genai_agents_mcp::McpServer;
//! # async fn example(mut runner: Runner<impl Model>, files: McpServer) -> Result<(), Box<dyn std::error::Error>> {
//! if files.tools_changed() {
//!     files.register(runner.registry_mut()).await?;
//! }
//! runner.run(&mut Conversation::user("Which files changed today?")).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Trace context
//!
//! A call can carry the W3C trace context to the server in its `_meta` (`traceparent`,
//! `tracestate`). This crate does not depend on OpenTelemetry, and a `tracing` span has no
//! trace id of its own, so the application says where the context comes from
//! ([`McpOptions::trace_context`]); with `tracing-opentelemetry`:
//!
//! ```ignore
//! use opentelemetry::propagation::TextMapPropagator;
//! use opentelemetry_sdk::propagation::TraceContextPropagator;
//! use tracing_opentelemetry::OpenTelemetrySpanExt;
//!
//! let options = McpOptions::new().trace_context(|| {
//!     let mut headers = std::collections::HashMap::new();
//!     TraceContextPropagator::new()
//!         .inject_context(&tracing::Span::current().context(), &mut headers);
//!     let context = TraceContext::new(headers.remove("traceparent")?);
//!     Some(match headers.remove("tracestate") {
//!         Some(state) if !state.is_empty() => context.with_tracestate(state),
//!         _ => context,
//!     })
//! });
//! ```
//!
//! Not on WebAssembly: the transports are native, and the crate is empty there.

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(not(target_family = "wasm"))]
mod convert;
#[cfg(not(target_family = "wasm"))]
mod server;

#[cfg(not(target_family = "wasm"))]
pub use rmcp;
#[cfg(not(target_family = "wasm"))]
pub use server::{Error, Grouping, McpOptions, McpServer, McpTool, TraceContext};

/// The examples of the MCP guide (`docs/mcp.md`), compiled as doc tests.
#[cfg(all(doctest, feature = "stdio"))]
#[doc = include_str!("../../../docs/mcp.md")]
pub struct McpGuide;
