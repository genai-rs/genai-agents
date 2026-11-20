//! [`OpenAiChat`]: Chat Completions as a [`Model`].

use std::future::Future;

use genai_agents::{
    Item, MaybeSend, MaybeSync, Model, ModelError, ModelEvent, ModelInfo, ModelRequest,
    ModelResponse, ModelStream, Origin,
};
use openai_ergonomic::semantic_conventions::operation_names;
use openai_ergonomic::{Builder, ChatCompletionBuilder, Client, Error};
use serde_json::Value;

use crate::responses::{effort, template_model};
use crate::{convert, http, Compat};

/// The API name of Chat Completions in item origins.
pub const CHAT_API: &str = "openai-chat";

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
/// What the server takes besides is its [`Compat`]: by default the preset of the client's
/// provider. The request's [settings](genai_agents::ModelSettings) override the template's: the
/// reasoning level becomes `reasoning_effort` (when the server takes it), the output limit
/// `max_completion_tokens` or `max_tokens`. When a completion stops at the output limit, its tool
/// calls are marked as cut off and not run.
///
/// The requests go the way of the client's own calls, through its interceptors, as for
/// [`OpenAiResponses`](crate::OpenAiResponses). Its items record API `openai-chat`, provider
/// [`PROVIDER`](crate::PROVIDER) and the template's model as their origin.
///
/// ```rust,no_run
/// use genai_agents::{Conversation, Runner, ToolRegistry};
/// use genai_agents_openai::OpenAiChat;
/// use openai_ergonomic::{ChatCompletionBuilder, Client};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::from_env()?.build();
/// let model = OpenAiChat::new(client, ChatCompletionBuilder::new("gpt-5.4-mini").system("Be brief."));
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
    compat: Compat,
    origin: Origin,
}

impl<T> OpenAiChat<T> {
    /// Chat Completions of `client`, with this request template.
    pub fn new(client: Client<T>, template: ChatCompletionBuilder) -> Self {
        let compat = Compat::for_provider(client.config().provider().kind());
        let built = template.clone().user("placeholder").build().ok();
        let model = template_model(built.and_then(|built| serde_json::to_value(built).ok()));
        Self {
            client,
            template,
            images: true,
            compat,
            origin: Origin::new(CHAT_API, crate::PROVIDER, model),
        }
    }

    /// The same model, for a server that takes what `compat` says.
    #[must_use]
    pub fn with_compat(mut self, compat: Compat) -> Self {
        self.compat = compat;
        self
    }

    /// What the server takes.
    pub fn compat(&self) -> &Compat {
        &self.compat
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
        if let Some(instructions) = request.instructions {
            // The template's own instructions give way to the request's.
            messages.retain(|message| {
                !matches!(message["role"].as_str(), Some("system" | "developer"))
            });
            let role = if self.compat.supports_developer_role {
                "developer"
            } else {
                "system"
            };
            messages.insert(
                0,
                serde_json::json!({"role": role, "content": instructions}),
            );
        }
        messages.extend(convert::chat_messages(request.items, &self.compat));
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
        if let Some(output) = request.output {
            map.insert(
                "response_format".into(),
                serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": output.name,
                        "schema": output.schema,
                        "strict": output.strict,
                    },
                }),
            );
        }
        if let Some(level) = request.settings.reasoning {
            if self.compat.supports_reasoning_effort {
                map.insert(
                    "reasoning_effort".into(),
                    Value::String(effort(level).to_string()),
                );
            }
        }
        if let Some(tokens) = request.settings.max_output_tokens {
            map.remove("max_tokens");
            map.remove("max_completion_tokens");
            map.insert(
                self.compat.max_tokens_field.as_str().into(),
                Value::from(tokens),
            );
        }
        if !self.compat.supports_store {
            map.remove("store");
        }
        if !self.compat.supports_reasoning_effort {
            map.remove("reasoning_effort");
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
    let cut_off = choice.get("finish_reason").and_then(Value::as_str) == Some("length");
    let mut events: Vec<ModelEvent> = convert::chat_items(&choice["message"])
        .into_iter()
        .map(|mut item| {
            // The completion stopped at the output limit: its calls may be cut off.
            if let (true, Item::ToolCall(call)) = (cut_off, &mut item) {
                call.meta.interrupted = true;
            }
            ModelEvent::ItemDone(item)
        })
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
    T: Default + MaybeSend + MaybeSync + 'static,
{
    fn info(&self) -> ModelInfo {
        ModelInfo::new()
            .with_origin(self.origin.clone())
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
            let (completion, rate_limits) = http::call(
                &self.client,
                operation_names::CHAT,
                "/chat/completions",
                &body,
            )
            .await?;
            let events = completion_events(&completion)?;
            Ok(ModelStream::from_events(events).with_rate_limits(rate_limits))
        }
    }

    fn retries_before_body(&self) -> bool {
        http::retries_before_body(&self.client)
    }
}
