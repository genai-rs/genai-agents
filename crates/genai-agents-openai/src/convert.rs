//! Between genai-agents' items and tool definitions and OpenAI's JSON: the Responses API's input
//! and output items and tools, and Chat Completions messages and tools.

use genai_agents::tools::{CustomFormat, ToolKey};
use genai_agents::{
    Content, Item, Message, ProviderItem, Reasoning, Role, ToolCall, ToolDefinition, ToolKind,
    ToolOutputBody,
};
use serde_json::{json, Map, Value};

/// The provider name of OpenAI's reasoning and provider items.
pub const PROVIDER: &str = "openai";

fn text(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn optional(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

// ---------------------------------------------------------------------------------------------
// Responses API.

/// The input items of a Responses API request. A stateless request (`store: false`) sends the
/// items without their ids (the server kept none) and without the reasoning it cannot read back
/// (reasoning without encrypted content). Reasoning and provider items of other providers are
/// left out, and so are calls of a kind this crate does not know, with their outputs.
pub fn input_items(items: &[Item], stateless: bool) -> Vec<Value> {
    let mut input = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Item::Message(message) => input.push(input_message(message, stateless)),
            Item::ToolCall(call) => {
                let mut value = match &call.kind {
                    ToolKind::Function => json!({
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.input,
                    }),
                    ToolKind::Custom => json!({
                        "type": "custom_tool_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "input": call.input,
                    }),
                    _ => continue,
                };
                if let Some(namespace) = &call.namespace {
                    value["namespace"] = Value::String(namespace.clone());
                }
                if let (false, Some(id)) = (stateless, &call.id) {
                    value["id"] = Value::String(id.clone());
                }
                input.push(value);
            }
            Item::ToolOutput(output) => {
                let kind = match output.kind {
                    ToolKind::Function => "function_call_output",
                    ToolKind::Custom => "custom_tool_call_output",
                    _ => continue,
                };
                input.push(json!({
                    "type": kind,
                    "call_id": output.call_id,
                    "output": output_json(&output.body),
                }));
            }
            Item::Reasoning(reasoning) if reasoning.provider == PROVIDER => {
                let encrypted = reasoning
                    .encrypted_content
                    .as_deref()
                    .filter(|content| !content.is_empty());
                if stateless && encrypted.is_none() {
                    continue;
                }
                let summary: Vec<Value> = reasoning
                    .summary
                    .iter()
                    .map(|text| json!({"type": "summary_text", "text": text}))
                    .collect();
                let mut value = json!({"type": "reasoning", "summary": summary});
                if let Some(content) = encrypted {
                    value["encrypted_content"] = Value::String(content.to_string());
                }
                if let (false, Some(id)) = (stateless, &reasoning.id) {
                    value["id"] = Value::String(id.clone());
                }
                input.push(value);
            }
            Item::Provider(item) if item.provider == PROVIDER => {
                let mut value = item.data.clone();
                let reference = value.get("type").and_then(Value::as_str) == Some("item_reference");
                if stateless && !reference {
                    if let Some(map) = value.as_object_mut() {
                        map.remove("id");
                    }
                }
                input.push(value);
            }
            _ => {}
        }
    }
    input
}

fn input_message(message: &Message, stateless: bool) -> Value {
    if message.role == Role::Assistant {
        return match (&message.id, stateless) {
            (Some(id), false) => {
                let parts: Vec<Value> = message
                    .content
                    .iter()
                    .filter_map(Content::as_text)
                    .map(|text| json!({"type": "output_text", "text": text, "annotations": []}))
                    .collect();
                json!({
                    "type": "message",
                    "id": id,
                    "role": "assistant",
                    "status": "completed",
                    "content": parts,
                })
            }
            _ => json!({"type": "message", "role": "assistant", "content": message.text_content()}),
        };
    }
    let content = match message.content.as_slice() {
        [Content::Text { text }] => Value::String(text.clone()),
        parts => Value::Array(parts.iter().map(input_part).collect()),
    };
    let mut value = json!({"type": "message", "role": role_name(message.role), "content": content});
    if let (false, Some(id)) = (stateless, &message.id) {
        value["id"] = Value::String(id.clone());
    }
    value
}

/// A content part as the Responses API takes it: `input_text`, `input_image`, `input_file`.
pub fn input_part(part: &Content) -> Value {
    let mut value = match part {
        Content::Image {
            image_url,
            file_id,
            detail,
            ..
        } => json!({
            "type": "input_image",
            "image_url": image_url,
            "file_id": file_id,
            "detail": detail.map(|detail| detail.as_str()),
        }),
        Content::File {
            file_id,
            file_url,
            file_data,
            filename,
            ..
        } => json!({
            "type": "input_file",
            "file_id": file_id,
            "file_url": file_url,
            "file_data": file_data,
            "filename": filename,
        }),
        other => json!({"type": "input_text", "text": other.as_text().unwrap_or_default()}),
    };
    if let Some(map) = value.as_object_mut() {
        map.retain(|_, value| !value.is_null());
    }
    value
}

/// The `output` of a `function_call_output` or `custom_tool_call_output`: a string, or content
/// parts.
pub fn output_json(body: &ToolOutputBody) -> Value {
    match body {
        ToolOutputBody::Text(text) => Value::String(text.clone()),
        ToolOutputBody::Items(items) => Value::Array(items.iter().map(input_part).collect()),
        _ => Value::Null,
    }
}

/// An output item of a response as an [`Item`]: messages, function and custom tool calls and
/// reasoning become theirs; every other item (a hosted tool's call or output, …) is kept as a
/// [`ProviderItem`].
pub fn output_item(value: Value) -> Item {
    match value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "message" => {
            let role = match value.get("role").and_then(Value::as_str) {
                Some("user") => Role::User,
                Some("system") => Role::System,
                Some("developer") => Role::Developer,
                _ => Role::Assistant,
            };
            let content = value
                .get("content")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|part| {
                            part.get("text")
                                .or_else(|| part.get("refusal"))
                                .and_then(Value::as_str)
                                .map(Content::text)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let mut message = Message::new(role, content);
            message.id = optional(&value, "id");
            Item::Message(message)
        }
        kind @ ("function_call" | "custom_tool_call") => {
            let mut call = if kind == "function_call" {
                ToolCall::function(
                    text(&value, "call_id"),
                    text(&value, "name"),
                    text(&value, "arguments"),
                )
            } else {
                ToolCall::custom(
                    text(&value, "call_id"),
                    text(&value, "name"),
                    text(&value, "input"),
                )
            };
            call.namespace = optional(&value, "namespace");
            call.id = optional(&value, "id");
            // A call the response stopped in the middle of (at the output limit): not run.
            call.meta.interrupted =
                value.get("status").and_then(Value::as_str) == Some("incomplete");
            Item::ToolCall(call)
        }
        "reasoning" => {
            let mut reasoning = Reasoning::new(PROVIDER).with_summary(
                value
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect(),
            );
            reasoning.id = optional(&value, "id");
            reasoning.encrypted_content = optional(&value, "encrypted_content");
            Item::Reasoning(reasoning)
        }
        _ => Item::Provider(ProviderItem::new(PROVIDER, value)),
    }
}

/// The tools a hosted tool search output (`tool_search_output`) loaded.
pub fn loaded_tools(item: &Value) -> Vec<ToolKey> {
    if item.get("type").and_then(Value::as_str) != Some("tool_search_output") {
        return Vec::new();
    }
    let mut keys = Vec::new();
    for tool in item
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        collect_loaded(tool, None, &mut keys);
    }
    keys
}

fn collect_loaded(tool: &Value, namespace: Option<&str>, keys: &mut Vec<ToolKey>) {
    let Some(name) = tool.get("name").and_then(Value::as_str) else {
        return;
    };
    if tool.get("type").and_then(Value::as_str) == Some("namespace") {
        for member in tool
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            collect_loaded(member, Some(name), keys);
        }
        return;
    }
    keys.push(namespace.map_or_else(
        || ToolKey::new(name),
        |namespace| ToolKey::namespaced(namespace, name),
    ));
}

/// A tool definition as the Responses API takes it.
pub fn responses_tool(definition: &ToolDefinition) -> Option<Value> {
    let defer = |mut value: Value, defer_loading: bool| {
        if defer_loading {
            value["defer_loading"] = Value::Bool(true);
        }
        value
    };
    Some(match definition {
        ToolDefinition::Function {
            name,
            description,
            parameters,
            defer_loading,
        } => defer(
            json!({"type": "function", "name": name, "description": description, "parameters": parameters}),
            *defer_loading,
        ),
        ToolDefinition::Custom {
            name,
            description,
            format,
            defer_loading,
        } => defer(
            json!({"type": "custom", "name": name, "description": description, "format": format_json(format)}),
            *defer_loading,
        ),
        ToolDefinition::Namespace {
            name,
            description,
            tools,
        } => json!({
            "type": "namespace",
            "name": name,
            "description": description,
            "tools": tools.iter().filter_map(responses_tool).collect::<Vec<_>>(),
        }),
        ToolDefinition::ToolSearch => json!({"type": "tool_search"}),
        _ => return None,
    })
}

fn format_json(format: &CustomFormat) -> Value {
    match format {
        CustomFormat::Grammar { syntax, definition } => {
            json!({"type": "grammar", "syntax": syntax.as_str(), "definition": definition})
        }
        _ => json!({"type": "text"}),
    }
}

// ---------------------------------------------------------------------------------------------
// Chat Completions.

/// A tool definition as Chat Completions takes it: function tools only.
pub fn chat_tool(definition: &ToolDefinition) -> Option<Value> {
    match definition {
        ToolDefinition::Function {
            name,
            description,
            parameters,
            ..
        } => Some(json!({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        })),
        _ => None,
    }
}

/// The conversation as chat messages: messages by role; consecutive calls (with the assistant
/// text before them) as one assistant message with tool calls; outputs as tool messages (images
/// and files as a short text). Reasoning and provider items are left out.
pub fn chat_messages(items: &[Item], compat: &crate::Compat) -> Vec<Value> {
    let names: std::collections::BTreeMap<&str, &str> = items
        .iter()
        .filter_map(|item| match item {
            Item::ToolCall(call) => Some((call.call_id.as_str(), call.name.as_str())),
            _ => None,
        })
        .collect();
    let mut messages = Vec::new();
    let mut assistant: Option<(Option<String>, Vec<Value>)> = None;
    let flush = |assistant: &mut Option<(Option<String>, Vec<Value>)>,
                 messages: &mut Vec<Value>| {
        if let Some((content, calls)) = assistant.take() {
            let mut message = Map::new();
            message.insert("role".into(), "assistant".into());
            message.insert("content".into(), content.map_or(Value::Null, Value::String));
            if !calls.is_empty() {
                message.insert("tool_calls".into(), Value::Array(calls));
            }
            messages.push(Value::Object(message));
        }
    };
    for item in items {
        match item {
            Item::Message(message) if message.role == Role::Assistant => {
                flush(&mut assistant, &mut messages);
                let text = message.text_content();
                assistant = Some(((!text.is_empty()).then_some(text), Vec::new()));
            }
            Item::Message(message) => {
                flush(&mut assistant, &mut messages);
                let role = match message.role {
                    Role::Developer if !compat.supports_developer_role => "system",
                    role => role_name(role),
                };
                messages.push(json!({"role": role, "content": chat_content(message)}));
            }
            Item::ToolCall(call) => {
                let value = match &call.kind {
                    ToolKind::Function => json!({
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.input},
                    }),
                    ToolKind::Custom => json!({
                        "id": call.call_id,
                        "type": "custom",
                        "custom": {"name": call.name, "input": call.input},
                    }),
                    _ => continue,
                };
                assistant
                    .get_or_insert_with(|| (None, Vec::new()))
                    .1
                    .push(value);
            }
            Item::ToolOutput(output) if !matches!(output.kind, ToolKind::Other(_)) => {
                flush(&mut assistant, &mut messages);
                let mut message = json!({
                    "role": "tool",
                    "tool_call_id": output.call_id,
                    "content": chat_text(&output.body),
                });
                if compat.requires_tool_result_name {
                    if let Some(name) = names.get(output.call_id.as_str()) {
                        message["name"] = Value::String((*name).to_string());
                    }
                }
                messages.push(message);
            }
            _ => {}
        }
    }
    flush(&mut assistant, &mut messages);
    messages
}

/// The content of a chat message: text, or (for a user message with images) text and image
/// parts.
fn chat_content(message: &Message) -> Value {
    let has_media = message.content.iter().any(|part| part.as_text().is_none());
    if message.role != Role::User || !has_media {
        return Value::String(message.text_content());
    }
    Value::Array(
        message
            .content
            .iter()
            .filter_map(|part| match part {
                Content::Text { text } => Some(json!({"type": "text", "text": text})),
                Content::Image {
                    image_url: Some(url),
                    detail,
                    ..
                } => {
                    let mut image = json!({"url": url});
                    if let Some(detail) = detail {
                        image["detail"] = Value::String(detail.as_str().to_string());
                    }
                    Some(json!({"type": "image_url", "image_url": image}))
                }
                _ => None,
            })
            .collect(),
    )
}

/// The text of a tool output for a chat tool message: images and files as a short text.
fn chat_text(body: &ToolOutputBody) -> String {
    match body {
        ToolOutputBody::Text(text) => text.clone(),
        ToolOutputBody::Items(parts) => parts
            .iter()
            .map(|part| match part {
                Content::Text { text } => text.clone(),
                Content::Image { .. } => "[image omitted: chat tool messages are text]".into(),
                _ => "[file omitted: chat tool messages are text]".into(),
            })
            .collect(),
        _ => String::new(),
    }
}

/// The items of a chat completion's message: its text (or refusal), then its tool calls.
pub fn chat_items(message: &Value) -> Vec<Item> {
    let mut items = Vec::new();
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
        .or_else(|| message.get("refusal").and_then(Value::as_str));
    if let Some(content) = content.filter(|content| !content.is_empty()) {
        items.push(Item::assistant(content));
    }
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = text(call, "id");
        let call = match call.get("type").and_then(Value::as_str) {
            Some("function") => ToolCall::function(
                id,
                text(&call["function"], "name"),
                text(&call["function"], "arguments"),
            ),
            Some("custom") => ToolCall::custom(
                id,
                text(&call["custom"], "name"),
                text(&call["custom"], "input"),
            ),
            other => ToolCall::other(id, other.unwrap_or("unknown")),
        };
        items.push(Item::ToolCall(call));
    }
    items
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;
    use genai_agents::{ImageDetail, ToolOutput};

    #[test]
    fn stateless_input_drops_ids_and_unreadable_reasoning() {
        let items = vec![
            output_item(json!({"type": "reasoning", "id": "rs_1", "summary": []})),
            output_item(
                json!({"type": "reasoning", "id": "rs_2", "summary": [{"type": "summary_text", "text": "hm"}], "encrypted_content": "gAAA"}),
            ),
            output_item(
                json!({"type": "function_call", "id": "fc_1", "call_id": "a", "name": "f", "arguments": "{}"}),
            ),
            output_item(json!({"type": "item_reference", "id": "msg_1"})),
            output_item(json!({"type": "web_search_call", "id": "ws_1", "status": "completed"})),
            Item::Reasoning(Reasoning::new("anthropic").with_encrypted_content("other")),
        ];
        let input = input_items(&items, true);
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["encrypted_content"], "gAAA");
        assert_eq!(input[0]["summary"][0]["text"], "hm");
        assert!(input[0].get("id").is_none());
        assert!(input[1].get("id").is_none());
        assert_eq!(input[2]["id"], "msg_1");
        assert!(input[3].get("id").is_none());
        // Stored: the ids stay.
        let input = input_items(&items, false);
        assert_eq!(input.len(), 6 - 1);
        assert_eq!(input[0]["id"], "rs_1");
        assert_eq!(input[2]["id"], "fc_1");
    }

    #[test]
    fn messages_and_outputs_become_input_items() {
        let call = ToolCall::custom("c", "patch", "*** Begin").in_namespace("fs");
        let items = vec![
            Item::user("hi"),
            Item::Message(Message::new(
                Role::User,
                vec![
                    Content::text("look"),
                    Content::image_url("https://x/a.png").with_detail(ImageDetail::Low),
                ],
            )),
            Item::Message(Message::text(Role::Assistant, "Done").with_id("msg_1")),
            Item::ToolCall(call.clone()),
            Item::ToolOutput(ToolOutput::new(
                &call,
                vec![Content::text("ok"), Content::file_id("file_1")],
                true,
            )),
        ];
        let input = input_items(&items, false);
        assert_eq!(
            input[0],
            json!({"type": "message", "role": "user", "content": "hi"})
        );
        assert_eq!(
            input[1]["content"][1],
            json!({"type": "input_image", "image_url": "https://x/a.png", "detail": "low"})
        );
        assert_eq!(input[2]["content"][0]["type"], "output_text");
        assert_eq!(input[2]["id"], "msg_1");
        assert_eq!(input[3]["type"], "custom_tool_call");
        assert_eq!(input[3]["namespace"], "fs");
        assert_eq!(input[4]["type"], "custom_tool_call_output");
        assert_eq!(
            input[4]["output"],
            json!([{"type": "input_text", "text": "ok"}, {"type": "input_file", "file_id": "file_1"}])
        );
        // Stateless, the assistant message is plain text.
        assert_eq!(
            input_items(&items[2..3], true)[0],
            json!({"type": "message", "role": "assistant", "content": "Done"})
        );
    }

    #[test]
    fn output_items_and_loaded_tools() {
        let message = output_item(json!({
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "Hel", "annotations": []},
                        {"type": "output_text", "text": "lo", "annotations": []}]
        }));
        let Item::Message(message) = message else {
            panic!("a message expected");
        };
        assert_eq!(message.text_content(), "Hello");
        assert_eq!(message.id.as_deref(), Some("msg_1"));
        let search = json!({"type": "tool_search_output", "tools": [
            {"type": "function", "name": "convert"},
            {"type": "namespace", "name": "files", "tools": [{"type": "function", "name": "read"}]}
        ]});
        assert_eq!(
            loaded_tools(&search),
            [
                ToolKey::new("convert"),
                ToolKey::namespaced("files", "read")
            ]
        );
        assert!(matches!(output_item(search), Item::Provider(_)));
    }

    #[test]
    fn tools_for_each_api() {
        let definitions = [
            ToolDefinition::Function {
                name: "f".into(),
                description: "F".into(),
                parameters: json!({"type": "object"}),
                defer_loading: true,
            },
            ToolDefinition::Custom {
                name: "c".into(),
                description: "C".into(),
                format: CustomFormat::regex("a+"),
                defer_loading: false,
            },
            ToolDefinition::Namespace {
                name: "n".into(),
                description: "N".into(),
                tools: vec![ToolDefinition::Function {
                    name: "g".into(),
                    description: "G".into(),
                    parameters: json!({"type": "object"}),
                    defer_loading: false,
                }],
            },
            ToolDefinition::ToolSearch,
        ];
        let responses: Vec<Value> = definitions.iter().filter_map(responses_tool).collect();
        assert_eq!(responses[0]["defer_loading"], true);
        assert_eq!(
            responses[1]["format"],
            json!({"type": "grammar", "syntax": "regex", "definition": "a+"})
        );
        assert_eq!(responses[2]["tools"][0]["name"], "g");
        assert!(responses[2]["tools"][0].get("defer_loading").is_none());
        assert_eq!(responses[3], json!({"type": "tool_search"}));
        let chat: Vec<Value> = definitions.iter().filter_map(chat_tool).collect();
        assert_eq!(chat.len(), 1);
        assert_eq!(chat[0]["function"]["name"], "f");
    }

    #[test]
    fn the_conversation_becomes_chat_messages() {
        let a = ToolCall::function("a", "get_weather", r#"{"city":"Ghent"}"#);
        let b = ToolCall::function("b", "get_weather", r#"{"city":"Paris"}"#);
        let items = vec![
            Item::Message(Message::text(Role::System, "Be brief")),
            Item::user("Weather in Ghent and Paris?"),
            Item::assistant("Let me check."),
            Item::ToolCall(a.clone()),
            Item::ToolCall(b.clone()),
            Item::ToolOutput(ToolOutput::new(&a, "12", true)),
            Item::ToolOutput(ToolOutput::new(
                &b,
                vec![Content::text("14"), Content::image_url("https://x/y.png")],
                true,
            )),
            Item::Reasoning(Reasoning::new(PROVIDER)),
        ];
        let messages = chat_messages(&items, &crate::Compat::openai());
        assert_eq!(messages.len(), 5);
        assert_eq!(
            messages[0],
            json!({"role": "system", "content": "Be brief"})
        );
        assert_eq!(messages[2]["content"], "Let me check.");
        assert_eq!(messages[2]["tool_calls"][1]["id"], "b");
        assert_eq!(
            messages[2]["tool_calls"][1]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            messages[3],
            json!({"role": "tool", "tool_call_id": "a", "content": "12"})
        );
        assert_eq!(
            messages[4]["content"],
            "14[image omitted: chat tool messages are text]"
        );
        // A user message with an image keeps its parts.
        let items = vec![Item::Message(Message::new(
            Role::User,
            vec![
                Content::text("what is this"),
                Content::image_url("https://x/z.png"),
            ],
        ))];
        assert_eq!(
            chat_messages(&items, &crate::Compat::openai())[0]["content"][1],
            json!({"type": "image_url", "image_url": {"url": "https://x/z.png"}})
        );
    }

    #[test]
    fn chat_completions_become_items() {
        let items = chat_items(&json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "type": "custom", "custom": {"name": "p", "input": "x"}},
                {"id": "c", "type": "brand_new"}
            ]
        }));
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], Item::ToolCall(ToolCall::function("a", "f", "{}")));
        assert_eq!(items[1], Item::ToolCall(ToolCall::custom("b", "p", "x")));
        assert_eq!(items[2], Item::ToolCall(ToolCall::other("c", "brand_new")));
        let items = chat_items(&json!({"role": "assistant", "content": "Hi", "refusal": null}));
        assert_eq!(items, [Item::assistant("Hi")]);
        let items = chat_items(&json!({"role": "assistant", "content": null, "refusal": "No."}));
        assert_eq!(items, [Item::assistant("No.")]);
    }
}
