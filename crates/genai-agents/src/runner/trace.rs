//! What the runner records in its spans: messages in the form of OpenTelemetry's GenAI
//! conventions, and why a request could not reuse the prompt cache.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::item::{Content, Item, Role};
use crate::model::{fnv1a, ModelSettings, ToolDefinition};

/// Why a request could not reuse the prompt cache of the request before it: it no longer
/// starts with what that one sent ([`TurnRecord::cache_break`](super::TurnRecord::cache_break)).
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CacheBreak {
    /// Another model.
    Model,
    /// The tools changed (one was loaded by a tool search, say).
    Tools,
    /// The model settings changed (the reasoning level, say).
    Settings,
    /// The prompt cache key changed.
    Key,
    /// The history no longer starts with what the request before sent (a compaction, an edit,
    /// reasoning of another model left out).
    History,
}

impl CacheBreak {
    /// Its name, as the spans record it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Tools => "tools",
            Self::Settings => "settings",
            Self::Key => "key",
            Self::History => "history",
        }
    }
}

/// What the prompt cache depends on of a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RequestShape {
    model: u64,
    tools: u64,
    settings: ModelSettings,
    key: Option<String>,
    items: Vec<u64>,
}

impl RequestShape {
    pub(super) fn new(
        model: &str,
        tools: &[ToolDefinition],
        settings: ModelSettings,
        key: Option<&str>,
        items: &[Item],
    ) -> Self {
        Self {
            model: fnv1a(model.as_bytes()),
            tools: hash(tools),
            settings,
            key: key.map(str::to_owned),
            items: items.iter().map(hash).collect(),
        }
    }

    /// Why `self` cannot reuse the cache of `previous`, if it cannot.
    pub(super) fn break_from(&self, previous: &Self) -> Option<CacheBreak> {
        if self.model != previous.model {
            Some(CacheBreak::Model)
        } else if self.key != previous.key {
            Some(CacheBreak::Key)
        } else if self.tools != previous.tools {
            Some(CacheBreak::Tools)
        } else if self.settings != previous.settings {
            Some(CacheBreak::Settings)
        } else if !self.items.starts_with(&previous.items) {
            Some(CacheBreak::History)
        } else {
            None
        }
    }
}

/// The hash of a value's JSON.
fn hash<T: Serialize + ?Sized>(value: &T) -> u64 {
    fnv1a(&serde_json::to_vec(value).unwrap_or_default())
}

/// Items as the `gen_ai.input.messages` / `gen_ai.output.messages` attributes hold them: a JSON
/// array of `{role, parts}`, with consecutive items of one role in one message.
pub(super) fn messages(items: &[Item]) -> String {
    let mut messages: Vec<(String, Vec<Value>)> = Vec::new();
    for item in items {
        let (role, part) = match item {
            Item::Message(message) => {
                let parts: Vec<Value> = message.content.iter().map(content_part).collect();
                push(&mut messages, role_name(message.role), parts);
                continue;
            }
            Item::ToolCall(call) => (
                "assistant",
                json!({
                    "type": "tool_call",
                    "id": call.call_id,
                    "name": call.name,
                    "arguments": arguments(&call.input),
                }),
            ),
            Item::ToolOutput(output) => (
                "tool",
                json!({
                    "type": "tool_call_response",
                    "id": output.call_id,
                    "response": output.body.to_text().unwrap_or_default(),
                }),
            ),
            Item::Reasoning(reasoning) => (
                "assistant",
                json!({"type": "reasoning", "content": reasoning.summary.join("\n")}),
            ),
            _ => continue,
        };
        push(&mut messages, role, vec![part]);
    }
    let messages: Vec<Value> = messages
        .into_iter()
        .map(|(role, parts)| json!({"role": role, "parts": parts}))
        .collect();
    Value::Array(messages).to_string()
}

fn push(messages: &mut Vec<(String, Vec<Value>)>, role: &str, parts: Vec<Value>) {
    match messages.last_mut() {
        Some((last, existing)) if last == role => existing.extend(parts),
        _ => messages.push((role.to_owned(), parts)),
    }
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System | Role::Developer => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn content_part(content: &Content) -> Value {
    match content {
        Content::Text { text } => json!({"type": "text", "content": text}),
        Content::Image { .. } => json!({"type": "blob", "modality": "image"}),
        _ => json!({"type": "blob"}),
    }
}

fn arguments(input: &str) -> Value {
    serde_json::from_str(input).unwrap_or_else(|_| Value::String(input.to_owned()))
}

/// The name of a span: `"{operation} {name}"`.
pub(super) fn span_name(operation: &str, name: &str) -> String {
    if name.is_empty() {
        operation.to_owned()
    } else {
        format!("{operation} {name}")
    }
}
