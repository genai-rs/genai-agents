//! The model: what the [`Runner`](crate::Runner) sends a provider and what it gets back.
//!
//! A provider implements [`Model`]: it turns a [`ModelRequest`] (the conversation, the tools,
//! a few settings) into its own request, sends it, and returns the response as a stream of
//! [`ModelEvent`]s. Everything else (running the tools, retries, limits, cancellation) is the
//! runner's.

use std::future::Future;
use std::time::Duration;

use serde_json::Value;

use crate::error::ModelError;
use crate::item::Item;
use crate::platform::{BoxStream, MaybeSend, MaybeSync};
use crate::tools::{CustomFormat, ToolKey};

/// A model of a provider.
///
/// [`stream`](Self::stream) sends one request and returns its response as events: text deltas
/// while the model writes, each output item once it is done ([`ModelEvent::ItemDone`], so that
/// the runner can start a tool call before the response ends), and
/// [`ModelEvent::Completed`] at the end. A provider that does not stream sends the items and
/// `Completed` once the whole response arrived.
///
/// Natively the futures and the stream must be `Send`; on WebAssembly they need not be.
pub trait Model: MaybeSend + MaybeSync {
    /// What the model is and can take, which changes what the runner sends it.
    fn info(&self) -> ModelInfo;

    /// Send one request. A failure to send it, or an error status, is the `Err` of the future
    /// (before the body); a failure while the response streams is an `Err` of the stream.
    ///
    /// The stream may borrow the model, not the request: convert the items before sending.
    fn stream<'a>(
        &'a self,
        request: ModelRequest<'_>,
    ) -> impl Future<Output = Result<ModelStream<'a>, ModelError>> + MaybeSend;

    /// Whether the provider's HTTP layer already retries a request that fails before its body
    /// (a refused connection, a rate limit or server error status). The runner then leaves those
    /// failures to it and retries only what it cannot, so a request is not retried by both.
    /// Default `false`.
    fn retries_before_body(&self) -> bool {
        false
    }
}

/// The response of a request: what its headers said, and its events.
#[non_exhaustive]
pub struct ModelStream<'a> {
    /// The rate limits the response's headers reported, if any.
    pub rate_limits: Option<RateLimits>,
    /// The events of the response, ending with [`ModelEvent::Completed`].
    pub events: BoxStream<'a, Result<ModelEvent, ModelError>>,
}

impl std::fmt::Debug for ModelStream<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelStream")
            .field("rate_limits", &self.rate_limits)
            .finish_non_exhaustive()
    }
}

impl<'a> ModelStream<'a> {
    /// A response with these events.
    pub fn new(events: BoxStream<'a, Result<ModelEvent, ModelError>>) -> Self {
        Self {
            rate_limits: None,
            events,
        }
    }

    /// A response whose events are known already, such as one that was not streamed.
    #[must_use]
    pub fn from_events(events: Vec<ModelEvent>) -> Self {
        Self::new(Box::pin(futures::stream::iter(events.into_iter().map(Ok))))
    }

    /// The same response, with the rate limits its headers reported.
    #[must_use]
    pub fn with_rate_limits(mut self, rate_limits: Option<RateLimits>) -> Self {
        self.rate_limits = rate_limits;
        self
    }
}

/// One request of a run.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ModelRequest<'a> {
    /// The conversation, [repaired](crate::Conversation::normalize) for the model.
    pub items: &'a [Item],
    /// The tools to advertise.
    pub tools: &'a [ToolDefinition],
    /// Whether the model may call several tools at once (sent only when there are tools).
    pub parallel_tool_calls: bool,
    /// The prompt cache key, the same for every request of a runner.
    pub prompt_cache_key: Option<&'a str>,
}

impl<'a> ModelRequest<'a> {
    /// A request of these items, without tools.
    #[must_use]
    pub fn new(items: &'a [Item]) -> Self {
        Self {
            items,
            tools: &[],
            parallel_tool_calls: true,
            prompt_cache_key: None,
        }
    }

    /// The same request, with these tools.
    #[must_use]
    pub fn with_tools(mut self, tools: &'a [ToolDefinition]) -> Self {
        self.tools = tools;
        self
    }

    /// The same request, with this prompt cache key.
    #[must_use]
    pub fn with_prompt_cache_key(mut self, key: Option<&'a str>) -> Self {
        self.prompt_cache_key = key;
        self
    }

    /// The same request, with or without parallel tool calls.
    #[must_use]
    pub fn with_parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = parallel;
        self
    }
}

/// A tool as the runner advertises it to the model.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ToolDefinition {
    /// A function tool: JSON arguments that match `parameters`.
    Function {
        /// The name.
        name: String,
        /// What the tool does and when to use it.
        description: String,
        /// The JSON Schema of the arguments, an object.
        parameters: Value,
        /// Whether the model's own tool search loads it ([`ModelInfo::tool_search`]).
        defer_loading: bool,
    },
    /// A custom (freeform) tool: text input.
    Custom {
        /// The name.
        name: String,
        /// What the tool does, when to use it and what to send.
        description: String,
        /// The input format.
        format: CustomFormat,
        /// Whether the model's own tool search loads it.
        defer_loading: bool,
    },
    /// Tools grouped under a namespace ([`ModelInfo::namespaces`]).
    Namespace {
        /// The namespace.
        name: String,
        /// What its tools are for.
        description: String,
        /// Its tools.
        tools: Vec<ToolDefinition>,
    },
    /// The provider's own tool search over the deferred tools ([`ModelInfo::tool_search`]).
    ToolSearch,
}

impl ToolDefinition {
    /// The name of a function, custom or namespace definition.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Function { name, .. }
            | Self::Custom { name, .. }
            | Self::Namespace { name, .. } => Some(name),
            Self::ToolSearch => None,
        }
    }
}

/// What the model is and can take, which changes what the runner sends it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(clippy::struct_excessive_bools)] // independent capabilities
pub struct ModelInfo {
    /// The tokens the model reads at most (input and output together), when known.
    pub context_window: Option<u32>,
    /// The tokens the model writes at most in one response, when known.
    pub max_output_tokens: Option<u32>,
    /// Whether the model accepts images. When it does not,
    /// [`Conversation::normalize`](crate::Conversation::normalize) replaces the images of the
    /// history (in messages and tool outputs) with a short text. Default `true`.
    pub images: bool,
    /// Whether tool outputs may hold images and files. When they may not, an output with them is
    /// sent as a failure. Default `true`.
    pub rich_tool_outputs: bool,
    /// Whether the model takes custom (freeform) tools. Default `true`.
    pub custom_tools: bool,
    /// Whether the model takes tools grouped in namespaces. When it does not, namespaced tools
    /// are not advertised. Default `true`.
    pub namespaces: bool,
    /// Whether the model searches deferred tools itself: the runner then advertises them with
    /// `defer_loading` and a [`ToolDefinition::ToolSearch`], and records the tools the model
    /// loaded ([`ModelEvent::ToolsLoaded`]); otherwise it offers a `tool_search` function it
    /// answers itself. Default `false`.
    pub tool_search: bool,
    /// Whether the model reasons, so that its reasoning items are worth keeping. Default
    /// `false`.
    pub reasoning: bool,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            context_window: None,
            max_output_tokens: None,
            images: true,
            rich_tool_outputs: true,
            custom_tools: true,
            namespaces: true,
            tool_search: false,
            reasoning: false,
        }
    }
}

impl ModelInfo {
    /// The defaults: images, rich tool outputs, custom tools and namespaces; no tool search of
    /// its own, no reasoning.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The model's context window, in tokens.
    #[must_use]
    pub fn with_context_window(mut self, tokens: u32) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// The most tokens the model writes in one response.
    #[must_use]
    pub fn with_max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Whether the model accepts images.
    #[must_use]
    pub fn with_images(mut self, images: bool) -> Self {
        self.images = images;
        self
    }

    /// Whether tool outputs may hold images and files.
    #[must_use]
    pub fn with_rich_tool_outputs(mut self, rich: bool) -> Self {
        self.rich_tool_outputs = rich;
        self
    }

    /// Whether the model takes custom tools.
    #[must_use]
    pub fn with_custom_tools(mut self, custom: bool) -> Self {
        self.custom_tools = custom;
        self
    }

    /// Whether the model takes namespaces.
    #[must_use]
    pub fn with_namespaces(mut self, namespaces: bool) -> Self {
        self.namespaces = namespaces;
        self
    }

    /// Whether the model searches deferred tools itself.
    #[must_use]
    pub fn with_tool_search(mut self, tool_search: bool) -> Self {
        self.tool_search = tool_search;
        self
    }

    /// Whether the model reasons.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }
}

/// Something the model's response produced.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ModelEvent {
    /// Text the model is writing.
    TextDelta(String),
    /// Reasoning (a summary of it, as far as the provider shows it) the model is writing.
    ReasoningDelta(String),
    /// Arguments (or the input of a custom tool) the model is writing for a call; the call itself
    /// comes with [`ItemDone`](Self::ItemDone).
    ToolCallArgsDelta {
        /// The provider's id of the call's item, or its call id.
        id: String,
        /// The next part of the arguments.
        delta: String,
    },
    /// An output item, complete: a message, a tool call, reasoning or a provider item. The runner
    /// appends it to the conversation and starts a call at once.
    ItemDone(Item),
    /// The model's own tool search loaded these deferred tools: the runner advertises them from
    /// the next request on.
    ToolsLoaded(Vec<ToolKey>),
    /// An event of the provider the other variants do not cover, with its type and its JSON.
    Other {
        /// The provider's event type, such as `response.created`.
        kind: String,
        /// The event.
        data: Value,
    },
    /// The response is complete (or stopped early: [`ModelResponse::incomplete`]).
    Completed(ModelResponse),
}

impl ModelEvent {
    /// The text of a [`TextDelta`](Self::TextDelta).
    #[must_use]
    pub fn text_delta(&self) -> Option<&str> {
        match self {
            Self::TextDelta(text) => Some(text),
            _ => None,
        }
    }
}

/// Why the model stopped writing.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StopReason {
    /// It answered.
    Stop,
    /// It called tools.
    ToolUse,
    /// It reached the output token limit.
    Length,
    /// The provider's content filter stopped it.
    ContentFilter,
    /// The request was cancelled.
    Aborted,
    /// Another reason, as the provider names it.
    Other(String),
}

/// The end of a response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelResponse {
    /// The id the provider gave the response.
    pub id: Option<String>,
    /// The tokens it used.
    pub usage: Option<Usage>,
    /// Why the model stopped, when the provider says.
    pub stop: Option<StopReason>,
    /// Why it stopped early (`max_output_tokens`, `length`, `content_filter`), if it did, as the
    /// provider names it: the run then ends with
    /// [`RunStop::Incomplete`](crate::RunStop::Incomplete).
    pub incomplete: Option<String>,
}

impl ModelResponse {
    /// A complete response.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The same response, with its id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The same response, with its usage.
    #[must_use]
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// The same response, with why the model stopped.
    #[must_use]
    pub fn with_stop(mut self, stop: StopReason) -> Self {
        self.stop = Some(stop);
        self
    }

    /// The same response, stopped early for `reason`.
    #[must_use]
    pub fn with_incomplete(mut self, reason: impl Into<String>) -> Self {
        self.incomplete = Some(reason.into());
        self
    }
}

/// The tokens of a response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Usage {
    /// The input tokens, cached ones included.
    pub input_tokens: i64,
    /// The output tokens, reasoning ones included.
    pub output_tokens: i64,
    /// The input tokens read from the prompt cache, when the provider says.
    pub cache_read_tokens: Option<i64>,
    /// The input tokens written to the prompt cache, when the provider says.
    pub cache_write_tokens: Option<i64>,
    /// The output tokens spent on reasoning, when the provider says.
    pub reasoning_tokens: Option<i64>,
}

impl Usage {
    /// This many input and output tokens.
    #[must_use]
    pub fn new(input_tokens: i64, output_tokens: i64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        }
    }

    /// The same usage, with the tokens read from the prompt cache.
    #[must_use]
    pub fn with_cache_read_tokens(mut self, tokens: i64) -> Self {
        self.cache_read_tokens = Some(tokens);
        self
    }

    /// The same usage, with the tokens written to the prompt cache.
    #[must_use]
    pub fn with_cache_write_tokens(mut self, tokens: i64) -> Self {
        self.cache_write_tokens = Some(tokens);
        self
    }

    /// The same usage, with the reasoning tokens.
    #[must_use]
    pub fn with_reasoning_tokens(mut self, tokens: i64) -> Self {
        self.reasoning_tokens = Some(tokens);
        self
    }
}

/// The rate limits a response reported (OpenAI's `x-ratelimit-*` headers).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct RateLimits {
    /// The requests window.
    pub requests: Option<RateLimitWindow>,
    /// The tokens window.
    pub tokens: Option<RateLimitWindow>,
}

impl RateLimits {
    /// These windows.
    #[must_use]
    pub fn new(requests: Option<RateLimitWindow>, tokens: Option<RateLimitWindow>) -> Self {
        Self { requests, tokens }
    }
}

/// One rate limit window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct RateLimitWindow {
    /// The limit.
    pub limit: Option<u64>,
    /// What is left of it.
    pub remaining: Option<u64>,
    /// When it resets.
    pub reset: Option<Duration>,
}

impl RateLimitWindow {
    /// A window.
    #[must_use]
    pub fn new(limit: Option<u64>, remaining: Option<u64>, reset: Option<Duration>) -> Self {
        Self {
            limit,
            remaining,
            reset,
        }
    }
}
