//! The history of a run: [`Item`]s, repaired before every request.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::item::{Content, Item, Message, Role, ToolCall, ToolOutput};
use crate::model::ModelInfo;

/// The text that replaces an image when the model takes none ([`ModelInfo::images`]).
pub const IMAGE_OMITTED: &str = "[image omitted: the model does not accept images]";

/// A conversation: messages, the model's output items, tool calls and their outputs, in order.
///
/// A [`Runner`](crate::Runner) appends the model's items and the answers to its tool calls, and
/// [repairs](Self::normalize) the history before every request. It is `Serialize` and
/// `Deserialize`: store it and run on it again later.
///
/// ```
/// use genai_agents::{Conversation, ModelInfo, ToolCall};
///
/// let mut conversation = Conversation::new();
/// conversation.push_user("What is the weather in Ghent?");
/// conversation.push_call(&ToolCall::function("call_1", "get_weather", r#"{"city":"Ghent"}"#));
/// // The call was never answered (say, the application stopped): repair the history.
/// let report = conversation.normalize(ModelInfo::default());
/// assert_eq!(report.aborted, ["call_1"]);
/// assert!(conversation.unanswered_calls().is_empty());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Conversation {
    items: Vec<Item>,
}

/// What [`Conversation::normalize`] changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct NormalizeReport {
    /// The calls that had no output and were answered with `"aborted"`, by call id.
    pub aborted: Vec<String>,
    /// The outputs dropped because no earlier call has their call id, by call id.
    pub orphans: Vec<String>,
    /// The images replaced with a text because the model takes none.
    pub images: usize,
}

impl NormalizeReport {
    /// Whether nothing changed.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.aborted.is_empty() && self.orphans.is_empty() && self.images == 0
    }
}

/// How an item relates to tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Link<'a> {
    Call(&'a str),
    Output(&'a str),
    None,
}

fn link(item: &Item) -> Link<'_> {
    match item {
        Item::ToolCall(call) => Link::Call(&call.call_id),
        Item::ToolOutput(output) => Link::Output(&output.call_id),
        _ => Link::None,
    }
}

impl Conversation {
    /// An empty conversation.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A conversation that starts with a user message.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        let mut conversation = Self::new();
        conversation.push_user(text);
        conversation
    }

    /// A conversation of these items.
    #[must_use]
    pub fn from_items(items: impl IntoIterator<Item = Item>) -> Self {
        Self {
            items: items.into_iter().collect(),
        }
    }

    /// The items, in order.
    #[must_use]
    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// The items, by value.
    #[must_use]
    pub fn into_items(self) -> Vec<Item> {
        self.items
    }

    /// The number of items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether there is no item.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Append an item.
    pub fn push(&mut self, item: impl Into<Item>) {
        self.items.push(item.into());
    }

    /// Append a user message.
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.push(Message::text(Role::User, text));
    }

    /// Append a developer message (instructions within the conversation).
    pub fn push_developer(&mut self, text: impl Into<String>) {
        self.push(Message::text(Role::Developer, text));
    }

    /// Append a system message.
    pub fn push_system(&mut self, text: impl Into<String>) {
        self.push(Message::text(Role::System, text));
    }

    /// Append an assistant message (an earlier answer of the model).
    pub fn push_assistant(&mut self, text: impl Into<String>) {
        self.push(Message::text(Role::Assistant, text));
    }

    /// Append a tool call of the model.
    pub fn push_call(&mut self, call: &ToolCall) {
        self.push(call.clone());
    }

    /// Append the answer to a call.
    pub fn push_output(&mut self, output: &ToolOutput) {
        self.push(output.clone());
    }

    /// The calls that no later item answers, in order.
    #[must_use]
    pub fn unanswered_calls(&self) -> Vec<ToolCall> {
        let answered: BTreeSet<&str> = self
            .items
            .iter()
            .filter_map(|item| match link(item) {
                Link::Output(id) => Some(id),
                _ => None,
            })
            .collect();
        self.items
            .iter()
            .filter_map(|item| match item {
                Item::ToolCall(call) if !answered.contains(call.call_id.as_str()) => {
                    Some(call.clone())
                }
                _ => None,
            })
            .collect()
    }

    /// The tool calls of the conversation, in order.
    #[must_use]
    pub fn calls(&self) -> Vec<ToolCall> {
        self.items
            .iter()
            .filter_map(|item| match item {
                Item::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect()
    }

    /// The text of the last assistant message, if any.
    #[must_use]
    pub fn last_assistant_text(&self) -> Option<String> {
        self.items.iter().rev().find_map(|item| match item {
            Item::Message(message) if message.role == Role::Assistant => {
                Some(message.text_content())
            }
            _ => None,
        })
    }

    /// Remove the item at `index`, and its partner: the outputs of a call, or the call of an
    /// output (and that call's other outputs). Returns the removed items in order; nothing if
    /// `index` is out of range.
    pub fn remove(&mut self, index: usize) -> Vec<Item> {
        let Some(item) = self.items.get(index) else {
            return Vec::new();
        };
        let id = match link(item) {
            Link::Call(id) | Link::Output(id) => Some(id.to_string()),
            Link::None => None,
        };
        let mut removed = Vec::new();
        let mut kept = Vec::with_capacity(self.items.len());
        for (position, item) in std::mem::take(&mut self.items).into_iter().enumerate() {
            let partner = id.as_deref().is_some_and(|id| match link(&item) {
                Link::Call(other) | Link::Output(other) => other == id,
                Link::None => false,
            });
            if position == index || partner {
                removed.push(item);
            } else {
                kept.push(item);
            }
        }
        self.items = kept;
        removed
    }

    /// Repair the history so that a model accepts it; idempotent:
    ///
    /// - a call without an output gets one, `"aborted"`, after the call and the calls and outputs
    ///   that directly follow it (so that a chat assistant message keeps its tool messages
    ///   together);
    /// - an output whose call is not earlier in the conversation is dropped;
    /// - when the model takes no images (`caps.images` is false), the images of messages and
    ///   tool outputs are replaced with a short text ([`IMAGE_OMITTED`]).
    ///
    /// The synthetic outputs carry no id, so the repaired history is the same every time: the
    /// prompt cache still applies.
    pub fn normalize(&mut self, caps: ModelInfo) -> NormalizeReport {
        let mut report = NormalizeReport::default();

        // Orphan outputs.
        let mut calls = BTreeSet::new();
        self.items.retain(|item| match link(item) {
            Link::Call(id) => {
                calls.insert(id.to_string());
                true
            }
            Link::Output(id) if calls.contains(id) => true,
            Link::Output(id) => {
                report.orphans.push(id.to_string());
                false
            }
            Link::None => true,
        });

        // Missing outputs.
        let answered: BTreeSet<String> = self
            .items
            .iter()
            .filter_map(|item| match link(item) {
                Link::Output(id) => Some(id.to_string()),
                _ => None,
            })
            .collect();
        let mut index = 0;
        while index < self.items.len() {
            if let Item::ToolCall(call) = &self.items[index] {
                if !answered.contains(&call.call_id) {
                    let answer = ToolOutput::aborted(call);
                    report.aborted.push(call.call_id.clone());
                    let mut at = index + 1;
                    while at < self.items.len() && link(&self.items[at]) != Link::None {
                        at += 1;
                    }
                    self.items.insert(at, Item::ToolOutput(answer));
                }
            }
            index += 1;
        }

        if !caps.images {
            for item in &mut self.items {
                let parts = match item {
                    Item::Message(message) => Some(&mut message.content),
                    Item::ToolOutput(output) => match &mut output.body {
                        crate::item::ToolOutputBody::Items(items) => Some(items),
                        crate::item::ToolOutputBody::Text(_) => None,
                    },
                    _ => None,
                };
                for part in parts.into_iter().flatten() {
                    if part.is_image() {
                        *part = Content::text(IMAGE_OMITTED);
                        report.images += 1;
                    }
                }
            }
        }
        report
    }
}

impl From<Vec<Item>> for Conversation {
    fn from(items: Vec<Item>) -> Self {
        Self { items }
    }
}

impl Extend<Item> for Conversation {
    fn extend<I: IntoIterator<Item = Item>>(&mut self, items: I) {
        self.items.extend(items);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use crate::item::{ToolOutputBody, ABORTED_OUTPUT};

    fn kinds(conversation: &Conversation) -> Vec<String> {
        conversation
            .items()
            .iter()
            .map(|item| match item {
                Item::Message(_) => "message".to_string(),
                Item::ToolCall(call) => format!("call:{}", call.call_id),
                Item::ToolOutput(output) => format!("output:{}", output.call_id),
                _ => "other".to_string(),
            })
            .collect()
    }

    #[test]
    fn answers_unanswered_calls_after_their_group() {
        let mut conversation = Conversation::user("hi");
        let a = ToolCall::function("a", "f", "{}");
        let b = ToolCall::function("b", "f", "{}");
        conversation.push_call(&a);
        conversation.push_call(&b);
        conversation.push_output(&ToolOutput::new(&b, "2", true));
        conversation.push_user("next");
        let report = conversation.normalize(ModelInfo::default());
        assert_eq!(report.aborted, ["a"]);
        assert_eq!(
            kinds(&conversation),
            ["message", "call:a", "call:b", "output:b", "output:a", "message"]
        );
        let Item::ToolOutput(aborted) = &conversation.items()[4] else {
            panic!("an output expected");
        };
        assert_eq!(aborted.body, ToolOutputBody::text(ABORTED_OUTPUT));
        assert!(!aborted.success);
        // Idempotent.
        assert!(conversation.normalize(ModelInfo::default()).is_clean());
    }

    #[test]
    fn custom_calls_get_custom_outputs() {
        let mut conversation = Conversation::new();
        conversation.push_call(&ToolCall::custom("c", "patch", "*** Begin").in_namespace("fs"));
        conversation.normalize(ModelInfo::default());
        let Item::ToolOutput(output) = &conversation.items()[1] else {
            panic!("an output expected");
        };
        assert_eq!(output.kind, crate::item::ToolKind::Custom);
        assert_eq!(output.call_id, "c");
        assert_eq!(conversation.calls()[0].namespace.as_deref(), Some("fs"));
    }

    #[test]
    fn drops_orphan_outputs() {
        let mut conversation = Conversation::user("hi");
        let a = ToolCall::function("a", "f", "{}");
        conversation.push_output(&ToolOutput::new(&a, "early", true));
        conversation.push_call(&a);
        conversation.push_output(&ToolOutput::new(&a, "1", true));
        let ghost = ToolCall::function("ghost", "f", "{}");
        conversation.push_output(&ToolOutput::new(&ghost, "?", true));
        let report = conversation.normalize(ModelInfo::default());
        assert_eq!(report.orphans, ["a", "ghost"]);
        assert_eq!(report.aborted, Vec::<String>::new());
        assert_eq!(kinds(&conversation), ["message", "call:a", "output:a"]);
    }

    #[test]
    fn strips_images_when_the_model_takes_none() {
        let mut conversation = Conversation::new();
        conversation.push(Message::new(
            Role::User,
            vec![
                Content::text("look"),
                Content::image_url("https://example.com/b.png"),
            ],
        ));
        let call = ToolCall::function("a", "screenshot", "{}");
        conversation.push_call(&call);
        conversation.push_output(&ToolOutput::new(
            &call,
            vec![
                Content::text("here"),
                Content::image_url("https://example.com/a.png"),
            ],
            true,
        ));
        assert!(conversation.normalize(ModelInfo::default()).is_clean());
        let report = conversation.normalize(ModelInfo::new().with_images(false));
        assert_eq!(report.images, 2);
        let Item::ToolOutput(output) = &conversation.items()[2] else {
            panic!("an output expected");
        };
        assert_eq!(
            output.body,
            ToolOutputBody::Items(vec![Content::text("here"), Content::text(IMAGE_OMITTED)])
        );
        assert!(conversation
            .normalize(ModelInfo::new().with_images(false))
            .is_clean());
    }

    #[test]
    fn remove_takes_the_partner_along() {
        let mut conversation = Conversation::user("hi");
        let a = ToolCall::function("a", "f", "{}");
        conversation.push_call(&a);
        conversation.push_output(&ToolOutput::new(&a, "1", true));
        let removed = conversation.remove(2);
        assert_eq!(removed.len(), 2);
        assert_eq!(kinds(&conversation), ["message"]);
        assert_eq!(conversation.remove(5), Vec::<Item>::new());
    }

    #[test]
    fn round_trips_through_json() {
        let mut conversation = Conversation::user("hi");
        let a = ToolCall::function("a", "f", "{}");
        conversation.push_call(&a);
        conversation.push_output(&ToolOutput::new(&a, "1", true));
        conversation.push_assistant("Done");
        let json = serde_json::to_string(&conversation).unwrap();
        assert!(json.starts_with('['));
        let back: Conversation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, conversation);
        assert_eq!(back.last_assistant_text().as_deref(), Some("Done"));
    }
}
