//! [`OpenAiChat`]: Chat Completions as a [`Model`].

use std::future::Future;

use genai_agents::{
    MaybeSend, MaybeSync, Model, ModelError, ModelEvent, ModelInfo, ModelRequest, ModelResponse,
    ModelStream,
};
use openai_ergonomic::{Builder, ChatCompletionBuilder, Client, Error};
use serde_json::Value;

use crate::{convert, http};

/// Chat Completions (`POST /chat/completions`) as a [`Model`]: OpenAI's, and that of the
/// providers that speak it (Azure OpenAI, OpenRouter, Groq, Ollama, gateways).
///
/// `template` sets the model and the options of every request; its messages (such as a system
/// message) are sent before the conversation every time. The conversation becomes chat messages:
/// consecutive calls in one assistant message, outputs as tool messages; reasoning and provider
/// items are left out.
///
/// Chat Completions takes function tools only, without namespaces, and text in tool messages:
/// its [information](Model::info) says so, so the runner advertises only such tools and sends an output
/// with images or files as a failure. Responses are not streamed.
///
/// The requests go through the client's HTTP stack, as for
/// [`OpenAiResponses`](crate::OpenAiResponses); the client's interceptors are not called.
///
/// ```rust,no_run
/// use genai_agents::{Conversation, Runner, ToolRegistry};
/// use genai_agents_openai::OpenAiChat;
/// use openai_ergonomic::{ChatCompletionBuilder, Client};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::from_env()?.build();
/// let model = OpenAiChat::new(client, ChatCompletionBuilder::new("gpt-4.1-mini").system("Be brief."));
/// let mut runner = Runner::new(model, ToolRegistry::new());
/// let outcome = runner.run(&mut Conversation::user("Hello!")).await?;
/// println!("{}", outcome.output_text());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct OpenAiChat<T = ()> {
    client: Client<T>,
    template: ChatCompletionBuilder,
    images: bool,
}

impl<T> OpenAiChat<T> {
    /// Chat Completions of `client`, with this request template.
    pub fn new(client: Client<T>, template: ChatCompletionBuilder) -> Self {
        Self {
            client,
            template,
            images: true,
        }
    }

    /// Whether the model accepts images (default `true`). When it does not, the images of the
    /// history are replaced with a short text.
    #[must_use]
    pub fn with_images(mut self, images: bool) -> Self {
        self.images = images;
        self
    }

    /// The client.
    pub fn client(&self) -> &Client<T> {
        &self.client
    }

    /// The request template.
    pub fn template(&self) -> &ChatCompletionBuilder {
        &self.template
    }

    /// The JSON body of the request for `request`.
    fn body(&self, request: &ModelRequest<'_>) -> Result<Value, ModelError> {
        // A placeholder message, so that a template without messages builds; it is replaced
        // with the conversation.
        let built = self
            .template
            .clone()
            .user("placeholder")
            .build()
            .map_err(|error| http::model_error(error, false, None))?;
        let mut body = serde_json::to_value(built)
            .map_err(|error| http::model_error(error.into(), false, None))?;
        let map = body
            .as_object_mut()
            .ok_or_else(|| ModelError::new("the request is not a JSON object"))?;
        let mut messages = match map.get_mut("messages").map(Value::take) {
            Some(Value::Array(messages)) => messages,
            _ => Vec::new(),
        };
        messages.pop();
        messages.extend(convert::chat_messages(request.items));
        map.insert("messages".into(), Value::Array(messages));
        let mut tools = match map.get_mut("tools").map(Value::take) {
            Some(Value::Array(tools)) => tools,
            _ => Vec::new(),
        };
        tools.extend(request.tools.iter().filter_map(convert::chat_tool));
        if tools.is_empty() {
            map.remove("tools");
        } else {
            map.insert("tools".into(), Value::Array(tools));
            map.insert(
                "parallel_tool_calls".into(),
                Value::Bool(request.parallel_tool_calls),
            );
        }
        if let Some(key) = request.prompt_cache_key {
            map.insert("prompt_cache_key".into(), Value::String(key.to_string()));
        }
        map.remove("stream");
        map.remove("stream_options");
        Ok(body)
    }
}

/// The events of a chat completion: its message's items, then its end.
fn completion_events(completion: &Value) -> Result<Vec<ModelEvent>, ModelError> {
    let Some(choice) = completion
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    else {
        let error = Error::api(200, "the chat completion has no choice");
        return Err(http::model_error(error, false, None));
    };
    let mut events: Vec<ModelEvent> = convert::chat_items(&choice["message"])
        .into_iter()
        .map(ModelEvent::ItemDone)
        .collect();
    let mut end = ModelResponse::new();
    if let Some(id) = completion.get("id").and_then(Value::as_str) {
        end = end.with_id(id);
    }
    if let Some(usage) = completion.get("usage").filter(|usage| usage.is_object()) {
        end = end.with_usage(http::usage_of(
            usage,
            ["prompt_tokens", "completion_tokens"],
            ["prompt_tokens_details", "completion_tokens_details"],
        ));
    }
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        end = end.with_stop(http::stop_reason(reason));
        if matches!(reason, "length" | "content_filter") {
            end = end.with_incomplete(reason);
        }
    }
    events.push(ModelEvent::Completed(end));
    Ok(events)
}

impl<T> Model for OpenAiChat<T>
where
    Client<T>: MaybeSend + MaybeSync,
    T: MaybeSend + MaybeSync,
{
    fn info(&self) -> ModelInfo {
        ModelInfo::new()
            .with_images(self.images)
            .with_rich_tool_outputs(false)
            .with_custom_tools(false)
            .with_namespaces(false)
            .with_tool_search(false)
    }

    fn stream<'a>(
        &'a self,
        request: ModelRequest<'_>,
    ) -> impl Future<Output = Result<ModelStream<'a>, ModelError>> + MaybeSend {
        let body = self.body(&request);
        async move {
            let body = body?;
            let response = http::post(&self.client, "/chat/completions", &body, false).await?;
            let rate_limits = http::rate_limits(response.headers());
            let completion = http::json_body(response).await?;
            let events = completion_events(&completion)?;
            Ok(ModelStream::from_events(events).with_rate_limits(rate_limits))
        }
    }

    fn retries_before_body(&self) -> bool {
        http::retries_before_body(&self.client)
    }
}
