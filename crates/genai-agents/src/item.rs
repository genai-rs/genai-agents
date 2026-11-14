//! The items of a [`Conversation`](crate::Conversation): messages, tool calls, their outputs,
//! reasoning and the provider's own items, independent of the model provider.
//!
//! Every item is `Serialize` and `Deserialize`, so a conversation can be stored and sent again,
//! also to another provider or model. Items the model wrote record where they came from
//! ([`ItemMeta::origin`]), so that the [projection](crate::Conversation::for_request) for another
//! model leaves out the reasoning and provider state only their own model can read.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One item of a conversation.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Item {
    /// A message: of the user, the assistant (the model), the system or the developer.
    Message(Message),
    /// A tool call of the model.
    ToolCall(ToolCall),
    /// The answer to a tool call.
    ToolOutput(ToolOutput),
    /// The model's reasoning, as far as the provider returns it.
    Reasoning(Reasoning),
    /// An item only its provider understands, such as a call of a hosted tool (web search, a
    /// tool search, code interpreter), kept as the provider's JSON.
    Provider(ProviderItem),
}

impl Item {
    /// A user message with this text.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self::Message(Message::text(Role::User, text))
    }

    /// An assistant message with this text.
    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::Message(Message::text(Role::Assistant, text))
    }

    /// What is known about the item: where it came from, the turn that produced it, whether it
    /// is injected context, whether it was cut off.
    #[must_use]
    pub fn meta(&self) -> &ItemMeta {
        match self {
            Self::Message(message) => &message.meta,
            Self::ToolCall(call) => &call.meta,
            Self::ToolOutput(output) => &output.meta,
            Self::Reasoning(reasoning) => &reasoning.meta,
            Self::Provider(item) => &item.meta,
        }
    }

    /// The item's [metadata](ItemMeta), to change it.
    pub fn meta_mut(&mut self) -> &mut ItemMeta {
        match self {
            Self::Message(message) => &mut message.meta,
            Self::ToolCall(call) => &mut call.meta,
            Self::ToolOutput(output) => &mut output.meta,
            Self::Reasoning(reasoning) => &mut reasoning.meta,
            Self::Provider(item) => &mut item.meta,
        }
    }

    /// The id the provider gave the item, if any.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        match self {
            Self::Message(message) => message.id.as_deref(),
            Self::ToolCall(call) => call.id.as_deref(),
            Self::Reasoning(reasoning) => reasoning.id.as_deref(),
            Self::ToolOutput(output) => output.id.as_deref(),
            Self::Provider(_) => None,
        }
    }
}

impl From<Message> for Item {
    fn from(message: Message) -> Self {
        Self::Message(message)
    }
}

impl From<ToolCall> for Item {
    fn from(call: ToolCall) -> Self {
        Self::ToolCall(call)
    }
}

impl From<ToolOutput> for Item {
    fn from(output: ToolOutput) -> Self {
        Self::ToolOutput(output)
    }
}

impl From<Reasoning> for Item {
    fn from(reasoning: Reasoning) -> Self {
        Self::Reasoning(reasoning)
    }
}

impl From<ProviderItem> for Item {
    fn from(item: ProviderItem) -> Self {
        Self::Provider(item)
    }
}

/// Where an item came from: the API, the provider and the model that wrote it.
///
/// A [`Model`](crate::Model) names itself the same way ([`ModelInfo::origin`](crate::ModelInfo::origin)),
/// so that the [projection](crate::Conversation::for_request) for a model can tell its own
/// reasoning and provider state from another model's.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Origin {
    /// The API, such as `openai-responses` or `openai-chat`.
    pub api: String,
    /// The provider, such as `openai` or `azure`.
    pub provider: String,
    /// The model, such as `gpt-5-mini`.
    pub model: String,
}

impl Origin {
    /// The `model` of `provider`, through `api`.
    #[must_use]
    pub fn new(
        api: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            api: api.into(),
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// Whether an item of this origin can be read by `other`: the same provider, API and model.
    #[must_use]
    pub fn same_model(&self, other: &Self) -> bool {
        self == other
    }
}

/// What is known about an item besides its content. Every field is optional: an item made by
/// the application has none of them.
///
/// It is serialized as the item's `meta` field, left out when empty, so stored conversations
/// without it still load.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ItemMeta {
    /// The model that wrote the item; `None` for an item of the application.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
    /// The model response that produced the item, counted over the whole conversation from 1
    /// ([`Conversation::next_turn`](crate::Conversation::next_turn)); `None` for an item of the
    /// application.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<u32>,
    /// Whether the item is context the application injected (instructions, a skill, a note),
    /// not input of the user or output of the model; later compaction can take it out and put it
    /// back.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub injected: bool,
    /// Whether the item was cut off: the partial text of a response that was cancelled or
    /// failed, or a call the response stopped in the middle of. It is kept for the application;
    /// the [projection](crate::Conversation::for_request) does not send it, and the runner does
    /// not run such a call.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub interrupted: bool,
}

impl ItemMeta {
    /// No metadata.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether nothing is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    /// The same metadata, with the model that wrote the item.
    #[must_use]
    pub fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = Some(origin);
        self
    }

    /// The same metadata, with the turn that produced the item.
    #[must_use]
    pub fn with_turn(mut self, turn: u32) -> Self {
        self.turn = Some(turn);
        self
    }

    /// The same metadata, marked as injected context (or not).
    #[must_use]
    pub fn with_injected(mut self, injected: bool) -> Self {
        self.injected = injected;
        self
    }

    /// The same metadata, marked as cut off (or not).
    #[must_use]
    pub fn with_interrupted(mut self, interrupted: bool) -> Self {
        self.interrupted = interrupted;
        self
    }
}

/// Who wrote a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Instructions of the application. Providers without system messages send them as their
    /// instructions.
    System,
    /// Instructions within the conversation (OpenAI's `developer` role); providers without it
    /// send a system message.
    Developer,
    /// The user.
    User,
    /// The model.
    Assistant,
}

/// A message: its role and its content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Message {
    /// The id the provider gave the message, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Who wrote it.
    pub role: Role,
    /// Text, images and files, in order.
    pub content: Vec<Content>,
    /// Where it came from, and whether it is injected context.
    #[serde(default, skip_serializing_if = "ItemMeta::is_empty")]
    pub meta: ItemMeta,
}

impl Message {
    /// A message of `role` with this text.
    #[must_use]
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self::new(role, vec![Content::text(text)])
    }

    /// A message of `role` with this content.
    #[must_use]
    pub fn new(role: Role, content: Vec<Content>) -> Self {
        Self {
            id: None,
            role,
            content,
            meta: ItemMeta::default(),
        }
    }

    /// The same message, with this metadata.
    #[must_use]
    pub fn with_meta(mut self, meta: ItemMeta) -> Self {
        self.meta = meta;
        self
    }

    /// The same message, marked as context the application injected (instructions, a skill,
    /// a note) rather than input of the user.
    #[must_use]
    pub fn injected(mut self) -> Self {
        self.meta.injected = true;
        self
    }

    /// The same message, with the id the provider gave it.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The text of the message: its text parts, concatenated.
    #[must_use]
    pub fn text_content(&self) -> String {
        self.content.iter().filter_map(Content::as_text).collect()
    }
}

/// The detail level a model sees an image at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImageDetail {
    /// Low resolution.
    Low,
    /// High resolution.
    High,
    /// The model chooses.
    Auto,
    /// The image as it is.
    Original,
}

impl ImageDetail {
    /// The name OpenAI uses: `low`, `high`, `auto`, `original`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high",
            Self::Auto => "auto",
            Self::Original => "original",
        }
    }
}

/// One part of a message or of a tool output: text, an image or a file.
///
/// New kinds may be added in minor releases: build parts with the constructors and match with a
/// wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Content {
    /// Text.
    Text {
        /// The text.
        text: String,
    },
    /// An image, by URL (or data URL) or by uploaded file id.
    #[non_exhaustive]
    Image {
        /// The image URL, or a `data:` URL with the base64 image.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        /// The id of a file uploaded to the provider.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// The detail level the model sees the image at.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    /// A file, by uploaded file id, URL or base64 data.
    #[non_exhaustive]
    File {
        /// The id of a file uploaded to the provider.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// The URL of the file.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        /// The base64 content of the file.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        /// The file name the model sees.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

impl Content {
    /// A text part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// An image at a URL (or a `data:` URL).
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            image_url: Some(url.into()),
            file_id: None,
            detail: None,
        }
    }

    /// An image uploaded to the provider as a file.
    #[must_use]
    pub fn image_file(file_id: impl Into<String>) -> Self {
        Self::Image {
            image_url: None,
            file_id: Some(file_id.into()),
            detail: None,
        }
    }

    /// A file uploaded to the provider.
    #[must_use]
    pub fn file_id(file_id: impl Into<String>) -> Self {
        Self::File {
            file_id: Some(file_id.into()),
            file_url: None,
            file_data: None,
            filename: None,
        }
    }

    /// A file at a URL.
    #[must_use]
    pub fn file_url(url: impl Into<String>) -> Self {
        Self::File {
            file_id: None,
            file_url: Some(url.into()),
            file_data: None,
            filename: None,
        }
    }

    /// A file sent inline: its name and its base64 content.
    #[must_use]
    pub fn file_data(filename: impl Into<String>, base64: impl Into<String>) -> Self {
        Self::File {
            file_id: None,
            file_url: None,
            file_data: Some(base64.into()),
            filename: Some(filename.into()),
        }
    }

    /// The same image with a detail level; other parts are returned unchanged.
    #[must_use]
    pub fn with_detail(mut self, level: ImageDetail) -> Self {
        if let Self::Image { detail, .. } = &mut self {
            *detail = Some(level);
        }
        self
    }

    /// The text of a text part.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Whether this is an image.
    #[must_use]
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image { .. })
    }
}

/// The kind of a tool, or of a call the model made.
///
/// New kinds may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolKind {
    /// A function tool: JSON arguments.
    Function,
    /// A custom (freeform) tool: text input, optionally constrained by a grammar.
    Custom,
    /// A call of a kind this crate does not dispatch, with the kind the provider gave it.
    Other(String),
}

/// A tool call of the model, ready for
/// [`ToolRegistry::answer_call`](crate::tools::ToolRegistry::answer_call).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCall {
    /// The id the provider gave the call item, if any (not the call id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The identifier the answer must carry.
    pub call_id: String,
    /// The namespace of the tool, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// The tool name.
    pub name: String,
    /// The kind of call.
    pub kind: ToolKind,
    /// The JSON arguments of a function call, or the text input of a custom call.
    pub input: String,
    /// State the provider attached to the call and wants back with it (such as a signature of
    /// the model's reasoning), opaque to everyone else. The projection for another model leaves
    /// it out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_state: Option<Value>,
    /// Where it came from, and whether it was cut off.
    #[serde(default, skip_serializing_if = "ItemMeta::is_empty")]
    pub meta: ItemMeta,
}

impl ToolCall {
    /// A function call with JSON `arguments`.
    #[must_use]
    pub fn function(
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: None,
            call_id: call_id.into(),
            namespace: None,
            name: name.into(),
            kind: ToolKind::Function,
            input: arguments.into(),
            provider_state: None,
            meta: ItemMeta::default(),
        }
    }

    /// A custom (freeform) tool call with its text `input`.
    #[must_use]
    pub fn custom(
        call_id: impl Into<String>,
        name: impl Into<String>,
        input: impl Into<String>,
    ) -> Self {
        Self {
            kind: ToolKind::Custom,
            ..Self::function(call_id, name, input)
        }
    }

    /// A call of a kind this crate does not dispatch.
    #[must_use]
    pub fn other(call_id: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            kind: ToolKind::Other(kind.into()),
            ..Self::function(call_id, "", "")
        }
    }

    /// The same call, of a tool in `namespace`.
    #[must_use]
    pub fn in_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// The same call, with the id the provider gave its item.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The same call, with the provider's opaque state for it.
    #[must_use]
    pub fn with_provider_state(mut self, state: Value) -> Self {
        self.provider_state = Some(state);
        self
    }

    /// The same call, with this metadata.
    #[must_use]
    pub fn with_meta(mut self, meta: ItemMeta) -> Self {
        self.meta = meta;
        self
    }
}

impl From<&ToolCall> for ToolCall {
    fn from(call: &ToolCall) -> Self {
        call.clone()
    }
}

/// What the model sees of a tool's output: text, or content parts (text, images, files).
///
/// [`FunctionTool`](crate::tools::FunctionTool)s produce JSON text;
/// [`RichTool`](crate::tools::RichTool)s, [`CustomTool`](crate::tools::CustomTool)s and dynamic
/// tools return a body directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ToolOutputBody {
    /// Text, such as the JSON encoding of a function tool's output.
    Text(String),
    /// Content parts.
    Items(Vec<Content>),
}

impl ToolOutputBody {
    /// A text body.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// A text body with the JSON encoding of `value` (strings are quoted, as for function tools).
    pub fn json<T: Serialize + ?Sized>(value: &T) -> crate::Result<Self> {
        Ok(Self::Text(serde_json::to_string(value)?))
    }

    /// The text of a text body.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Items(_) => None,
        }
    }

    /// The body as text, for a model that takes only text in tool outputs: the text, or the text
    /// parts concatenated. `None` when there is an image or a file.
    #[must_use]
    pub fn to_text(&self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text.clone()),
            Self::Items(items) => items
                .iter()
                .map(Content::as_text)
                .collect::<Option<Vec<_>>>()
                .map(|parts| parts.concat()),
        }
    }
}

impl From<String> for ToolOutputBody {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for ToolOutputBody {
    fn from(text: &str) -> Self {
        Self::Text(text.to_string())
    }
}

impl From<Vec<Content>> for ToolOutputBody {
    fn from(items: Vec<Content>) -> Self {
        Self::Items(items)
    }
}

impl From<Content> for ToolOutputBody {
    fn from(item: Content) -> Self {
        Self::Items(vec![item])
    }
}

/// The output of a call that was never answered, as history repair inserts it.
pub const ABORTED_OUTPUT: &str = "aborted";

/// The answer to one tool call, to send back to the model.
///
/// [`ToolRegistry::answer_call`](crate::tools::ToolRegistry::answer_call) gives one for every
/// call it can answer, failed calls included (`success: false`, with the error text as output).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolOutput {
    /// The id of the output item, if it has one: the provider's, or for an answer the
    /// projection made up, one derived from the call id (so that it is the same every time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The identifier of the call this answers.
    pub call_id: String,
    /// The kind of call it answers.
    pub kind: ToolKind,
    /// What the model sees.
    pub body: ToolOutputBody,
    /// Whether the tool succeeded. Not sent to the model; for logging and policies.
    pub success: bool,
    /// Whether the output holds content from outside the application, such as a web page or an
    /// email, which may carry instructions the model should not follow. Set from the tool
    /// ([`FunctionTool::untrusted_output`](crate::tools::FunctionTool::untrusted_output)); not
    /// sent to the model.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub untrusted: bool,
    /// What the tool reported for the application only, such as the full data behind a short
    /// answer or a diff to show. Never sent to the model. Set by
    /// [`FunctionTool::details`](crate::tools::FunctionTool::details) or by the application.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Where it came from.
    #[serde(default, skip_serializing_if = "ItemMeta::is_empty")]
    pub meta: ItemMeta,
}

impl ToolOutput {
    /// An answer to `call`.
    #[must_use]
    pub fn new(call: &ToolCall, body: impl Into<ToolOutputBody>, success: bool) -> Self {
        Self {
            id: None,
            call_id: call.call_id.clone(),
            kind: call.kind.clone(),
            body: body.into(),
            success,
            untrusted: false,
            details: None,
            meta: ItemMeta::default(),
        }
    }

    /// The same answer, with details for the application, which the model never sees.
    #[must_use]
    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    /// The same answer, with the id of its item.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The same answer, marked as holding content from outside the application (or not).
    #[must_use]
    pub fn with_untrusted(mut self, untrusted: bool) -> Self {
        self.untrusted = untrusted;
        self
    }

    /// A failed answer to `call`, with the text the model sees.
    #[must_use]
    pub fn failure(call: &ToolCall, message: impl Into<String>) -> Self {
        Self::new(call, ToolOutputBody::Text(message.into()), false)
    }

    /// The answer to a call that was not run, such as a denied or cancelled one: `"aborted"`.
    #[must_use]
    pub fn aborted(call: &ToolCall) -> Self {
        Self::failure(call, ABORTED_OUTPUT)
    }
}

/// The model's reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Reasoning {
    /// The id the provider gave the item, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The provider that produced it: only that provider can read its encrypted content.
    pub provider: String,
    /// The summary the provider returned, in parts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summary: Vec<String>,
    /// The reasoning, encrypted by the provider, to send back so that the model keeps it
    /// between requests without the provider storing it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    /// Where it came from: only that model can read its encrypted content.
    #[serde(default, skip_serializing_if = "ItemMeta::is_empty")]
    pub meta: ItemMeta,
}

impl Reasoning {
    /// Reasoning of `provider`.
    #[must_use]
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            id: None,
            provider: provider.into(),
            summary: Vec::new(),
            encrypted_content: None,
            meta: ItemMeta::default(),
        }
    }

    /// The same reasoning, with this metadata.
    #[must_use]
    pub fn with_meta(mut self, meta: ItemMeta) -> Self {
        self.meta = meta;
        self
    }

    /// The same reasoning, with an id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The same reasoning, with its encrypted content.
    #[must_use]
    pub fn with_encrypted_content(mut self, content: impl Into<String>) -> Self {
        self.encrypted_content = Some(content.into());
        self
    }

    /// The same reasoning, with its summary.
    #[must_use]
    pub fn with_summary(mut self, summary: Vec<String>) -> Self {
        self.summary = summary;
        self
    }
}

/// An item only its provider understands, kept as the provider's JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderItem {
    /// The provider, such as `openai`.
    pub provider: String,
    /// The item, as the provider sent it.
    pub data: Value,
    /// Where it came from.
    #[serde(default, skip_serializing_if = "ItemMeta::is_empty")]
    pub meta: ItemMeta,
}

impl ProviderItem {
    /// An item of `provider`.
    #[must_use]
    pub fn new(provider: impl Into<String>, data: Value) -> Self {
        Self {
            provider: provider.into(),
            data,
            meta: ItemMeta::default(),
        }
    }

    /// The same item, with this metadata.
    #[must_use]
    pub fn with_meta(mut self, meta: ItemMeta) -> Self {
        self.meta = meta;
        self
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn items_round_trip_through_json() {
        let items = vec![
            Item::user("Weather in Ghent?"),
            Item::ToolCall(ToolCall::function("a", "get_weather", "{}").with_id("fc_a")),
            Item::ToolOutput(ToolOutput::new(
                &ToolCall::function("a", "get_weather", "{}"),
                vec![
                    Content::text("12"),
                    Content::image_url("https://example.com/a.png").with_detail(ImageDetail::Low),
                ],
                true,
            )),
            Item::Reasoning(Reasoning::new("openai").with_encrypted_content("gAAA")),
            Item::Provider(ProviderItem::new(
                "openai",
                json!({"type": "web_search_call", "id": "ws_1"}),
            )),
        ];
        let value = serde_json::to_value(&items).unwrap();
        assert_eq!(value[0]["type"], "message");
        assert_eq!(value[0]["role"], "user");
        assert_eq!(
            value[0]["content"][0],
            json!({"type": "text", "text": "Weather in Ghent?"})
        );
        assert_eq!(value[1]["type"], "tool_call");
        assert_eq!(value[1]["kind"], "function");
        assert_eq!(value[2]["body"][1]["detail"], "low");
        let back: Vec<Item> = serde_json::from_value(value).unwrap();
        assert_eq!(back, items);
    }

    #[test]
    fn bodies_become_text_when_they_are() {
        let body = ToolOutputBody::Items(vec![Content::text("a"), Content::text("b")]);
        assert_eq!(body.to_text().as_deref(), Some("ab"));
        let body: ToolOutputBody = Content::image_file("file_1").into();
        assert_eq!(body.to_text(), None);
    }
}
