//! A connection to an MCP server, and its tools in a [`ToolRegistry`].

use std::sync::{Arc, Mutex, PoisonError};

use genai_agents::tools::{
    DynamicTool, Exposure, OutputBudget, ToolError, ToolKey, ToolNamer, ToolRegistry,
};
use genai_agents::ToolOutputBody;
use rmcp::model::{
    CallToolRequestParams, ClientCapabilities, ClientConfig, Implementation, JsonObject,
    RequestMetaObject, Tool, ToolAnnotations,
};
use rmcp::service::{NotificationContext, RunningService};
use rmcp::transport::IntoTransport;
use rmcp::{ClientHandler, RoleClient, ServiceError, ServiceExt};
use serde_json::Value;
use tokio::sync::watch;

use crate::convert;

/// A failure to connect to a server or to register its tools.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The server could not be started, such as a command that is not found.
    #[error("cannot start the MCP server: {0}")]
    Io(#[from] std::io::Error),
    /// The connection or the MCP handshake failed.
    #[error("cannot connect to the MCP server: {0}")]
    Connect(#[source] Box<rmcp::service::ClientInitializeError>),
    /// A request to the server failed, such as listing its tools.
    #[error("MCP request failed: {0}")]
    Service(#[from] ServiceError),
    /// The registry refused the server's namespace (its name is taken by a tool, or invalid).
    #[error(transparent)]
    Registry(#[from] ToolError),
}

/// How the tools of a server are grouped in the registry.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Grouping {
    /// In a namespace named after the server, with the server's description: the model calls
    /// them with the namespace. Only models that take namespaces
    /// ([`ModelInfo::namespaces`](genai_agents::ModelInfo::namespaces)) see them.
    #[default]
    Namespace,
    /// Outside namespaces, each name prefixed with the server's name (`server__tool`): for
    /// models without namespaces, such as Chat Completions.
    Prefix,
}

/// The W3C trace context of a call, sent to the server in the request's `_meta`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TraceContext {
    /// The `traceparent` value, such as `00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01`.
    pub traceparent: String,
    /// The `tracestate` value, if any.
    pub tracestate: Option<String>,
}

impl TraceContext {
    /// A trace context with this `traceparent`.
    #[must_use]
    pub fn new(traceparent: impl Into<String>) -> Self {
        Self {
            traceparent: traceparent.into(),
            tracestate: None,
        }
    }

    /// The same trace context with a `tracestate`.
    #[must_use]
    pub fn with_tracestate(mut self, tracestate: impl Into<String>) -> Self {
        self.tracestate = Some(tracestate.into());
        self
    }
}

type TraceFn = dyn Fn() -> Option<TraceContext> + Send + Sync;

/// How a server's tools are registered and called.
#[derive(Clone)]
#[non_exhaustive]
pub struct McpOptions {
    /// When the model sees the tools. Default [`Exposure::Deferred`]: a tool search finds
    /// them, so that a large server does not fill the context.
    pub exposure: Exposure,
    /// Whether the outputs are marked [untrusted](genai_agents::ToolOutput::untrusted).
    /// Default `true`: a server's output is content from outside the application.
    pub untrusted_output: bool,
    /// How the tools are grouped (default [`Grouping::Namespace`]).
    pub grouping: Grouping,
    /// The namespace's description, instead of the server's own.
    pub description: Option<String>,
    /// How much of an output the model sees, instead of the runner's default.
    pub output_budget: Option<OutputBudget>,
    trace_context: Option<Arc<TraceFn>>,
}

impl std::fmt::Debug for McpOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpOptions")
            .field("exposure", &self.exposure)
            .field("untrusted_output", &self.untrusted_output)
            .field("grouping", &self.grouping)
            .field("description", &self.description)
            .field("output_budget", &self.output_budget)
            .field("trace_context", &self.trace_context.is_some())
            .finish()
    }
}

impl Default for McpOptions {
    fn default() -> Self {
        Self {
            exposure: Exposure::Deferred,
            untrusted_output: true,
            grouping: Grouping::Namespace,
            description: None,
            output_budget: None,
            trace_context: None,
        }
    }
}

impl McpOptions {
    /// The defaults: deferred, untrusted output, in a namespace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// When the model sees the tools.
    #[must_use]
    pub fn exposure(mut self, exposure: Exposure) -> Self {
        self.exposure = exposure;
        self
    }

    /// Whether the outputs are marked untrusted.
    #[must_use]
    pub fn untrusted_output(mut self, untrusted: bool) -> Self {
        self.untrusted_output = untrusted;
        self
    }

    /// How the tools are grouped.
    #[must_use]
    pub fn grouping(mut self, grouping: Grouping) -> Self {
        self.grouping = grouping;
        self
    }

    /// The namespace's description, instead of the server's own.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// How much of an output the model sees.
    #[must_use]
    pub fn output_budget(mut self, budget: OutputBudget) -> Self {
        self.output_budget = Some(budget);
        self
    }

    /// Where the trace context of a call comes from: `current` runs as the call starts, inside
    /// the call's span, and its answer goes to the server in the request's `_meta`
    /// (`traceparent`, `tracestate`). With `tracing-opentelemetry`, it reads the current span's
    /// OpenTelemetry context (see the crate documentation).
    #[must_use]
    pub fn trace_context(
        mut self,
        current: impl Fn() -> Option<TraceContext> + Send + Sync + 'static,
    ) -> Self {
        self.trace_context = Some(Arc::new(current));
        self
    }
}

/// A tool of a server, as it is registered.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct McpTool {
    /// The tool in the registry: the name the model calls, and the namespace.
    pub key: ToolKey,
    /// The server's name for the tool.
    pub name: String,
    /// The server's title for the tool, if any.
    pub title: Option<String>,
    /// The server's description, if any.
    pub description: Option<String>,
    /// The server's hints about the tool, if any. They come from the server: an approval
    /// policy should not trust a server it does not trust otherwise.
    pub annotations: Option<ToolAnnotations>,
}

impl McpTool {
    fn new(key: ToolKey, tool: &Tool) -> Self {
        Self {
            key,
            name: tool.name.to_string(),
            title: tool.title.clone(),
            description: tool.description.as_ref().map(ToString::to_string),
            annotations: tool.annotations.clone(),
        }
    }

    /// Whether the server says the tool does not change its environment (`readOnlyHint`,
    /// default `false`). Such a tool may run at the same time as others.
    #[must_use]
    pub fn read_only(&self) -> bool {
        self.annotations
            .as_ref()
            .and_then(|hints| hints.read_only_hint)
            .unwrap_or(false)
    }

    /// Whether the tool may destroy something (`destructiveHint`): `true` unless the server
    /// says it is read-only or not destructive.
    #[must_use]
    pub fn destructive(&self) -> bool {
        !self.read_only()
            && self
                .annotations
                .as_ref()
                .and_then(|hints| hints.destructive_hint)
                .unwrap_or(true)
    }

    /// Whether calling the tool again with the same arguments has no further effect
    /// (`idempotentHint`, default `false`).
    #[must_use]
    pub fn idempotent(&self) -> bool {
        self.annotations
            .as_ref()
            .and_then(|hints| hints.idempotent_hint)
            .unwrap_or(false)
    }

    /// Whether the tool reaches outside entities, such as the web (`openWorldHint`, default
    /// `true`).
    #[must_use]
    pub fn open_world(&self) -> bool {
        self.annotations
            .as_ref()
            .and_then(|hints| hints.open_world_hint)
            .unwrap_or(true)
    }
}

/// The client side of the connection: it notes when the server's tool list changed.
struct Client {
    changes: watch::Sender<u64>,
}

impl ClientHandler for Client {
    async fn on_tool_list_changed(&self, _context: NotificationContext<RoleClient>) {
        self.changes.send_modify(|count| *count += 1);
    }

    fn get_info(&self) -> ClientConfig {
        ClientConfig::new(
            ClientCapabilities::default(),
            Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
        )
    }
}

/// What the last registration did.
#[derive(Default)]
struct Registered {
    /// The `tools/list_changed` notifications counted when the list was read.
    seen: Option<u64>,
    tools: Vec<McpTool>,
}

struct Inner {
    name: String,
    description: String,
    instructions: Option<String>,
    options: McpOptions,
    service: RunningService<RoleClient, Client>,
    changes: watch::Receiver<u64>,
    registered: Mutex<Registered>,
}

/// A connection to an MCP server: register its tools in a [`ToolRegistry`] with
/// [`register`](Self::register), and again when its list changed.
///
/// Cloning it shares the connection. The registered tools hold it too: it closes when the
/// handles and the tools are dropped, or with [`close`](Self::close).
#[derive(Clone)]
pub struct McpServer {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for McpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServer")
            .field("name", &self.inner.name)
            .field("options", &self.inner.options)
            .finish_non_exhaustive()
    }
}

impl McpServer {
    /// Connect to a server over any rmcp transport, such as an in-process pipe
    /// (`tokio::io::duplex`) or a [`StreamableHttpClientTransport`](rmcp::transport::StreamableHttpClientTransport)
    /// with an authorization header.
    ///
    /// `name` names the server's namespace (or the prefix of its tools): it is cleaned up into
    /// a valid tool name of at most 64 characters.
    ///
    /// # Errors
    ///
    /// [`Error::Connect`] when the transport or the MCP handshake fails.
    pub async fn connect<T, E, A>(
        name: &str,
        transport: T,
        options: McpOptions,
    ) -> Result<Self, Error>
    where
        T: IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let (sender, changes) = watch::channel(0);
        let service = Client { changes: sender }
            .serve(transport)
            .await
            .map_err(|error| Error::Connect(Box::new(error)))?;
        let info = service.peer().peer_info();
        let instructions = info
            .as_ref()
            .and_then(|info| info.instructions.clone())
            .filter(|text| !text.trim().is_empty());
        let name = ToolNamer::standard().unique(None, name);
        let description = options
            .description
            .clone()
            .or_else(|| {
                let server = info.as_ref()?.server_info.as_ref()?;
                server.description.clone().or_else(|| server.title.clone())
            })
            .or_else(|| instructions.clone())
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| format!("The tools of the MCP server {name}."));
        Ok(Self {
            inner: Arc::new(Inner {
                name,
                description,
                instructions,
                options,
                service,
                changes,
                registered: Mutex::new(Registered::default()),
            }),
        })
    }

    /// Start `command` and connect to it over its standard input and output.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] when the command cannot be started, [`Error::Connect`] when the handshake
    /// fails.
    #[cfg(feature = "stdio")]
    #[cfg_attr(docsrs, doc(cfg(feature = "stdio")))]
    pub async fn stdio(
        name: &str,
        command: tokio::process::Command,
        options: McpOptions,
    ) -> Result<Self, Error> {
        let transport = rmcp::transport::TokioChildProcess::new(command)?;
        Self::connect(name, transport, options).await
    }

    /// Connect to a server over streamable HTTP at `url`. For headers (such as an
    /// authorization), build the transport yourself and use [`connect`](Self::connect).
    ///
    /// # Errors
    ///
    /// [`Error::Connect`] when the connection or the handshake fails.
    #[cfg(feature = "http")]
    #[cfg_attr(docsrs, doc(cfg(feature = "http")))]
    pub async fn http(name: &str, url: &str, options: McpOptions) -> Result<Self, Error> {
        let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(url);
        Self::connect(name, transport, options).await
    }

    /// The server's namespace in the registry (or the prefix of its tools' names).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// The namespace's description: the one in the options, else the server's description,
    /// title or instructions.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.inner.description
    }

    /// The server's instructions, if it gave any. Put them in the agent's instructions if the
    /// model needs them.
    #[must_use]
    pub fn instructions(&self) -> Option<&str> {
        self.inner.instructions.as_deref()
    }

    /// The rmcp connection, for what this crate does not do (resources, prompts).
    #[must_use]
    pub fn peer(&self) -> &rmcp::Peer<RoleClient> {
        self.inner.service.peer()
    }

    /// Read the server's tools and register them in `registry`, in place of those this server
    /// registered before; returns them.
    ///
    /// With [`Grouping::Namespace`], the namespace is replaced with the server's tools; with
    /// [`Grouping::Prefix`], the tools of the previous registration (of this server, in any
    /// registry) are removed first. Names are cleaned up and kept unique with a hash; schemas are
    /// rewritten into ones models accept ([`ToolRegistry::register_dynamic`]). A tool whose
    /// schema is not an object is left out, with a warning.
    ///
    /// A registry a run already holds (a clone, see [`ToolRegistry`]) keeps the tools it had: to
    /// change a [`Runner`](genai_agents::Runner)'s tools between runs, pass
    /// [`Runner::registry_mut`](genai_agents::Runner::registry_mut).
    ///
    /// # Errors
    ///
    /// [`Error::Service`] when the list cannot be read (the registry is then unchanged), and
    /// [`Error::Registry`] when the namespace's name is taken by a tool.
    pub async fn register(&self, registry: &mut ToolRegistry) -> Result<Vec<McpTool>, Error> {
        let seen = *self.inner.changes.borrow();
        let listed = self.inner.service.peer().list_all_tools().await?;
        let previous = std::mem::take(&mut self.lock().tools);
        let options = &self.inner.options;
        let mut tools = Vec::with_capacity(listed.len());
        match options.grouping {
            Grouping::Prefix => {
                for tool in &previous {
                    registry.remove(&tool.key);
                }
                for tool in &listed {
                    let dynamic = self.dynamic(tool).prefix(self.name());
                    if let Some(name) = skip_invalid(tool, registry.register_dynamic(dynamic))? {
                        tools.push(McpTool::new(ToolKey::new(name), tool));
                    }
                }
            }
            _ => {
                registry.remove_namespace(self.name());
                let mut namespace = registry.namespace(self.name(), self.description())?;
                for tool in &listed {
                    let registered = namespace.register_dynamic(self.dynamic(tool));
                    if let Some(name) = skip_invalid(tool, registered)? {
                        tools.push(McpTool::new(ToolKey::namespaced(self.name(), name), tool));
                    }
                }
            }
        }
        let mut registered = self.lock();
        registered.seen = Some(seen);
        registered.tools.clone_from(&tools);
        Ok(tools)
    }

    /// The tools of the last [`register`](Self::register), with the server's hints
    /// (`readOnlyHint`, `destructiveHint`, …).
    #[must_use]
    pub fn tools(&self) -> Vec<McpTool> {
        self.lock().tools.clone()
    }

    /// The tool registered under `key`, if it is one of this server's.
    #[must_use]
    pub fn tool(&self, key: &ToolKey) -> Option<McpTool> {
        self.lock()
            .tools
            .iter()
            .find(|tool| &tool.key == key)
            .cloned()
    }

    /// Whether the server said its tool list changed (`notifications/tools/list_changed`)
    /// since the last [`register`](Self::register) read it, or nothing was registered yet.
    #[must_use]
    pub fn tools_changed(&self) -> bool {
        let now = *self.inner.changes.borrow();
        self.lock().seen != Some(now)
    }

    /// Wait until [`tools_changed`](Self::tools_changed) is true (at once if it is), or the
    /// connection closes.
    pub async fn changed(&self) {
        let seen = self.lock().seen;
        let mut changes = self.inner.changes.clone();
        // An error means the connection closed: nothing changes any more.
        let _ = changes.wait_for(|count| Some(*count) != seen).await;
    }

    /// Close the connection (a child process is ended). Calls of its tools then fail.
    pub fn close(&self) {
        self.inner.service.cancellation_token().cancel();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Registered> {
        self.inner
            .registered
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The registry's form of a tool: its handler calls the server.
    fn dynamic(&self, tool: &Tool) -> DynamicTool {
        let options = &self.inner.options;
        let description = tool
            .description
            .as_ref()
            .map(ToString::to_string)
            .or_else(|| tool.title.clone())
            .unwrap_or_default();
        let read_only = tool
            .annotations
            .as_ref()
            .and_then(|hints| hints.read_only_hint)
            .unwrap_or(false);
        let inner = Arc::clone(&self.inner);
        let name = tool.name.to_string();
        let mut dynamic = DynamicTool::new(
            name.clone(),
            description,
            Value::Object((*tool.input_schema).clone()),
            move |arguments: Value| {
                let inner = Arc::clone(&inner);
                let name = name.clone();
                let trace = inner
                    .options
                    .trace_context
                    .as_ref()
                    .and_then(|current| current());
                async move { inner.call(name, arguments, trace).await }
            },
        )
        .supports_parallel(read_only)
        .exposure(options.exposure)
        .untrusted_output(options.untrusted_output);
        if let Some(budget) = options.output_budget {
            dynamic = dynamic.output_budget(budget);
        }
        dynamic
    }
}

/// A tool the registry refused for its schema is left out; other failures are returned.
fn skip_invalid(
    tool: &Tool,
    registered: Result<String, ToolError>,
) -> Result<Option<String>, Error> {
    match registered {
        Ok(name) => Ok(Some(name)),
        Err(ToolError::InvalidDefinition { reason, .. }) => {
            tracing::warn!(tool = %tool.name, %reason, "an MCP tool was left out");
            Ok(None)
        }
        Err(error) => Err(error.into()),
    }
}

/// A failed call, as the model sees it: the server's error text, or the request's failure.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct CallFailed(String);

impl Inner {
    async fn call(
        &self,
        name: String,
        arguments: Value,
        trace: Option<TraceContext>,
    ) -> genai_agents::Result<ToolOutputBody> {
        let arguments = match arguments {
            Value::Object(arguments) => arguments,
            _ => JsonObject::new(),
        };
        let mut params = CallToolRequestParams::new(name).with_arguments(arguments);
        if let Some(trace) = trace {
            let mut meta = RequestMetaObject::new();
            meta.set_traceparent(trace.traceparent);
            if let Some(tracestate) = trace.tracestate {
                meta.set_tracestate(tracestate);
            }
            params.meta = Some(meta);
        }
        let result = self
            .service
            .call_tool(params)
            .await
            .map_err(|error| genai_agents::Error::application(CallFailed(error.to_string())))?;
        convert::output(result).map_err(|text| genai_agents::Error::application(CallFailed(text)))
    }
}
