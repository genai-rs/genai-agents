//! [`OpenAiResponses`]: the Responses API as a [`Model`].

use std::collections::BTreeSet;
use std::future::Future;

use genai_agents::{
    MaybeSend, MaybeSync, Model, ModelError, ModelInfo, ModelRequest, ModelStream, Origin,
    ReasoningLevel,
};
use openai_ergonomic::semantic_conventions::operation_names;
use openai_ergonomic::{Builder, Client, ResponsesBuilder};
use serde_json::Value;

use crate::{convert, http};

/// The Responses API (`POST /responses`) as a [`Model`].
///
/// `template` sets the model and the options of every request, and the hosted tools (web
/// search, file search, code interpreter, remote MCP servers: [`ResponsesBuilder::tools`]); its
/// input, if any, is sent before the conversation every time. The runner's tools are added to
/// the template's.
///
/// The requests are stateless by default:
///
/// - `store: false`, and every request carries the whole conversation (its items without their
///   ids, which the server did not keep); requests are never chained with
///   `previous_response_id`;
/// - with a reasoning model ([`ModelInfo::reasoning`], or reasoning options on the template), the
///   encrypted reasoning is asked for (`include: ["reasoning.encrypted_content"]`) and sent back,
///   so the model keeps its reasoning between requests;
/// - responses are streamed, so the runner starts a tool call as soon as the model finished
///   writing it.
///
/// The request's [settings](genai_agents::ModelSettings) override the template's: the reasoning
/// level becomes `reasoning.effort` ([`ReasoningLevel::Off`] is `none`), the output limit
/// `max_output_tokens`.
///
/// The requests go the way of the client's own calls: through its interceptors (such as
/// Langfuse's) and its HTTP stack, with its retries, its provider (OpenAI, Azure OpenAI, a
/// compatible API) and credentials, the layers of the application, and its rate limits
/// ([`Client::rate_limits`]).
///
/// Its items record the model as their [origin](genai_agents::ItemMeta::origin): API
/// `openai-responses`, provider [`PROVIDER`](crate::PROVIDER), and the template's model.
///
/// ```rust,no_run
/// use genai_agents::{Conversation, Runner, ToolRegistry};
/// use genai_agents_openai::OpenAiResponses;
/// use openai_ergonomic::{Client, ResponsesBuilder};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::from_env()?.build();
/// let model = OpenAiResponses::new(
///     client,
///     ResponsesBuilder::new("gpt-5-mini").instructions("You are a household assistant."),
/// );
/// let mut runner = Runner::new(model, ToolRegistry::new());
/// let outcome = runner.run(&mut Conversation::user("Hello!")).await?;
/// println!("{}", outcome.output_text());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct OpenAiResponses<T = ()> {
    client: Client<T>,
    template: ResponsesBuilder,
    info: ModelInfo,
    store: bool,
    stream: bool,
    include_encrypted_reasoning: bool,
    origin: Origin,
}

/// The API name of the Responses API in item origins.
pub const RESPONSES_API: &str = "openai-responses";

/// The `effort` of OpenAI's reasoning for a level.
pub(crate) fn effort(level: ReasoningLevel) -> &'static str {
    match level {
        ReasoningLevel::Off => "none",
        ReasoningLevel::Minimal => "minimal",
        ReasoningLevel::Low => "low",
        ReasoningLevel::High => "high",
        ReasoningLevel::XHigh => "xhigh",
        _ => "medium",
    }
}

/// The `model` of a built request template, or nothing.
pub(crate) fn template_model(built: Option<Value>) -> String {
    built
        .and_then(|body| {
            body.get("model")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default()
}

impl<T> OpenAiResponses<T> {
    /// The Responses API of `client`, with this request template.
    pub fn new(client: Client<T>, template: ResponsesBuilder) -> Self {
        let built = template.clone().user("placeholder").build().ok();
        let model = template_model(built.and_then(|built| serde_json::to_value(built).ok()));
        Self {
            client,
            template,
            origin: Origin::new(RESPONSES_API, crate::PROVIDER, model),
            info: ModelInfo::default(),
            store: false,
            stream: true,
            include_encrypted_reasoning: true,
        }
    }

    /// The same model, with what it can take (default: images, rich tool outputs, custom tools,
    /// namespaces; no tool search of its own, no reasoning). With
    /// [`ModelInfo::tool_search`], deferred tools are sent with `defer_loading` and the hosted
    /// `tool_search` tool. Without an [origin](ModelInfo::origin), the model's own is used.
    #[must_use]
    pub fn with_info(mut self, info: ModelInfo) -> Self {
        self.info = info;
        self
    }

    /// Whether the server stores the responses (`store`). Every request still carries the whole
    /// conversation, with the items' ids. Default `false`.
    #[must_use]
    pub fn with_store(mut self, store: bool) -> Self {
        self.store = store;
        self
    }

    /// Whether responses are streamed. Default `true`.
    #[must_use]
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Whether to ask for the encrypted reasoning, and send it back, when nothing is stored and
    /// the model reasons. Default `true`.
    #[must_use]
    pub fn with_encrypted_reasoning(mut self, include: bool) -> Self {
        self.include_encrypted_reasoning = include;
        self
    }

    /// The client.
    pub fn client(&self) -> &Client<T> {
        &self.client
    }

    /// The request template.
    pub fn template(&self) -> &ResponsesBuilder {
        &self.template
    }

    /// The JSON body of the request for `request`.
    fn body(&self, request: &ModelRequest<'_>) -> Result<Value, ModelError> {
        let stateless = !self.store;
        // A placeholder message, so that the template builds without input of its own; it is
        // replaced with the conversation.
        let built = self
            .template
            .clone()
            .user("placeholder")
            .build()
            .map_err(|error| http::model_error(error, false, None))?;
        let mut body = serde_json::to_value(built)
            .map_err(|error| http::model_error(error.into(), false, None))?;
        let mut input = match body.get_mut("input").map(Value::take) {
            Some(Value::Array(items)) => items,
            _ => Vec::new(),
        };
        input.pop();
        input.extend(convert::input_items(request.items, stateless));
        body["input"] = Value::Array(input);

        let mut tools = match body.get_mut("tools").map(Value::take) {
            Some(Value::Array(tools)) => tools,
            _ => Vec::new(),
        };
        tools.extend(request.tools.iter().filter_map(convert::responses_tool));
        let map = body
            .as_object_mut()
            .ok_or_else(|| ModelError::new("the request is not a JSON object"))?;
        if tools.is_empty() {
            map.remove("tools");
        } else {
            map.insert("tools".into(), Value::Array(tools));
            map.insert(
                "parallel_tool_calls".into(),
                Value::Bool(request.parallel_tool_calls),
            );
        }
        map.insert("store".into(), Value::Bool(self.store));
        if let Some(key) = request.prompt_cache_key {
            map.insert("prompt_cache_key".into(), Value::String(key.to_string()));
        }
        if let Some(level) = request.settings.reasoning {
            let reasoning = map
                .entry("reasoning")
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            if !reasoning.is_object() {
                *reasoning = Value::Object(serde_json::Map::new());
            }
            reasoning["effort"] = Value::String(effort(level).to_string());
        }
        if let Some(tokens) = request.settings.max_output_tokens {
            map.insert("max_output_tokens".into(), Value::from(tokens));
        }
        let reasoning = self.info.reasoning || map.get("reasoning").is_some_and(Value::is_object);
        if stateless && self.include_encrypted_reasoning && reasoning {
            let include = map
                .entry("include")
                .or_insert_with(|| Value::Array(Vec::new()));
            if !include.is_array() {
                *include = Value::Array(Vec::new());
            }
            if let Value::Array(include) = include {
                let wanted = Value::String("reasoning.encrypted_content".into());
                if !include.contains(&wanted) {
                    include.push(wanted);
                }
            }
        }
        if self.stream {
            map.insert("stream".into(), Value::Bool(true));
        } else {
            map.remove("stream");
        }
        Ok(body)
    }
}

impl<T> Model for OpenAiResponses<T>
where
    Client<T>: MaybeSend + MaybeSync,
    T: Default + MaybeSend + MaybeSync + 'static,
{
    fn info(&self) -> ModelInfo {
        let mut info = self.info.clone();
        if info.origin.is_none() {
            info.origin = Some(self.origin.clone());
        }
        info
    }

    fn stream<'a>(
        &'a self,
        request: ModelRequest<'_>,
    ) -> impl Future<Output = Result<ModelStream<'a>, ModelError>> + MaybeSend {
        let body = self.body(&request);
        async move {
            let body = body?;
            let operation = operation_names::RESPONSES;
            if self.stream {
                let (events, rate_limits) =
                    http::event_call(&self.client, operation, "/responses", &body).await?;
                return Ok(
                    ModelStream::new(http::response_stream(events)).with_rate_limits(rate_limits)
                );
            }
            let (response, rate_limits) =
                http::call(&self.client, operation, "/responses", &body).await?;
            let events = http::response_events(&response, &mut BTreeSet::new());
            Ok(ModelStream::new(Box::pin(futures::stream::iter(events)))
                .with_rate_limits(rate_limits))
        }
    }

    fn retries_before_body(&self) -> bool {
        http::retries_before_body(&self.client)
    }
}
