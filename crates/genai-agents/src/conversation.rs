//! The history of a run: [`Item`]s, kept as they came, and projected for each request.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::item::ItemMeta;
use crate::item::{Content, Item, Message, Role, ToolCall, ToolOutput};
use crate::model::{fnv1a, ModelInfo};

/// The text that replaces an image when the model takes none ([`ModelInfo::images`]).
pub const IMAGE_OMITTED: &str = "[image omitted: the model does not accept images]";

/// A conversation: messages, the model's output items, tool calls and their outputs, in order.
///
/// A [`Runner`](crate::Runner) appends the model's items and the answers to its tool calls, as
/// they came, and sends each request a [projection](Self::for_request) of them for its model. It
/// is `Serialize` and `Deserialize`: store it and run on it again later.
///
/// ```
/// use genai_agents::{Conversation, Item, ModelInfo, ToolCall};
///
/// let mut conversation = Conversation::new();
/// conversation.push_user("What is the weather in Ghent?");
/// conversation.push_call(&ToolCall::function("call_1", "get_weather", r#"{"city":"Ghent"}"#));
/// // The call was never answered (say, the application stopped). The request gets an answer
/// // for it; the conversation stays as it is.
/// let request = conversation.for_request(&ModelInfo::default());
/// assert!(matches!(&request.items()[2], Item::ToolOutput(output) if output.call_id == "call_1"));
/// assert_eq!(conversation.unanswered_calls().len(), 1);
/// // Or repair the conversation itself.
/// let report = conversation.normalize(&ModelInfo::default());
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

/// What [`Conversation::for_request`] left out or changed, besides what
/// [`normalize`](Conversation::normalize) does.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProjectionReport {
    /// What the repair changed: answers made up, orphan outputs dropped, images replaced.
    pub normalized: NormalizeReport,
    /// The items left out because they were cut off ([`ItemMeta::interrupted`]).
    pub interrupted: usize,
    /// The reasoning and provider items left out because another model wrote them.
    pub foreign: usize,
    /// The calls whose provider state was left out because another model wrote them.
    pub provider_states: usize,
    /// The call ids rewritten for the model ([`ModelInfo::call_ids`]).
    pub call_ids: usize,
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

    /// The items, to change them in place (a projection's, say).
    pub(crate) fn items_mut(&mut self) -> &mut [Item] {
        &mut self.items
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

    /// The turn the next model response is: one more than the highest
    /// [`ItemMeta::turn`] of the conversation, or 1.
    #[must_use]
    pub fn next_turn(&self) -> u32 {
        self.items
            .iter()
            .filter_map(|item| item.meta().turn)
            .max()
            .map_or(1, |turn| turn.saturating_add(1))
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

    /// Repair the conversation itself so that a model accepts it; idempotent:
    ///
    /// - a call without an output gets one, `"aborted"`, after the call and the calls and outputs
    ///   that directly follow it (so that a chat assistant message keeps its tool messages
    ///   together);
    /// - an output whose call is not earlier in the conversation is dropped;
    /// - when the model takes no images ([`ModelInfo::images`] is false), the images of messages
    ///   and tool outputs are replaced with a short text ([`IMAGE_OMITTED`]).
    ///
    /// The runner does not change the stored conversation this way: it sends a
    /// [projection](Self::for_request), which repairs a copy.
    pub fn normalize(&mut self, info: &ModelInfo) -> NormalizeReport {
        repair(&mut self.items, info)
    }

    /// The index of the latest [compaction](Item::Compaction), if any.
    #[must_use]
    pub fn latest_compaction(&self) -> Option<usize> {
        self.items
            .iter()
            .rposition(|item| matches!(item, Item::Compaction(_)))
    }

    /// The items a request is built from: from the latest [compaction](Item::Compaction) on,
    /// with the compaction [expanded](crate::Compaction::expand) into the items it sends; every
    /// item when there is none.
    #[must_use]
    pub fn active_items(&self) -> Vec<Item> {
        let Some(at) = self.latest_compaction() else {
            return self.items.clone();
        };
        let Item::Compaction(compaction) = &self.items[at] else {
            return self.items.clone();
        };
        let mut items = compaction.expand();
        items.extend(self.items[at + 1..].iter().cloned());
        items
    }

    /// The conversation as a request to the model `info` describes: a copy, repaired and fitted
    /// to the model; the conversation itself does not change. The same conversation gives the
    /// same projection, byte for byte, so the provider's prompt cache holds.
    ///
    /// - It starts at the latest [compaction](Item::Compaction): the items before it are not
    ///   sent, the compaction sends what it kept and its summary ([`Self::active_items`]).
    /// - Items cut off ([`ItemMeta::interrupted`]: the partial text of a cancelled or failed
    ///   response) are left out; a cut-off call stays, with its answer.
    /// - Reasoning and provider items another model wrote are left out, and so is the provider
    ///   state of its calls ([`ToolCall::provider_state`]): only the model that wrote them can
    ///   read them. An item without an [origin](ItemMeta::origin) is kept when its provider is
    ///   the model's. When [`ModelInfo::origin`] is `None`, everything is kept.
    /// - Then it is [repaired](Self::normalize): an answer that the model's call never got is
    ///   made up, `"aborted"`, with an id derived from the call id; orphan outputs are dropped;
    ///   images are replaced for a model without vision.
    /// - Call ids the model does not take ([`ModelInfo::call_ids`]) are rewritten, in calls and
    ///   outputs alike.
    #[must_use]
    pub fn for_request(&self, info: &ModelInfo) -> Self {
        self.project(info).0
    }

    /// [`for_request`](Self::for_request), with what it left out or changed.
    #[must_use]
    pub fn project(&self, info: &ModelInfo) -> (Self, ProjectionReport) {
        let mut report = ProjectionReport::default();
        let target = info.origin.as_ref();
        let active = self.active_items();
        let mut items = Vec::with_capacity(active.len());
        for item in &active {
            let meta = item.meta();
            if meta.interrupted && !matches!(item, Item::ToolCall(_)) {
                report.interrupted += 1;
                continue;
            }
            let readable = |provider: &str| match (target, &meta.origin) {
                (None, _) => true,
                (Some(target), Some(origin)) => origin.same_model(target),
                (Some(target), None) => provider == target.provider,
            };
            match item {
                Item::Reasoning(reasoning) if !readable(&reasoning.provider) => {
                    report.foreign += 1;
                }
                Item::Provider(provider)
                    if !same_provider(target, &provider.provider, &provider.meta) =>
                {
                    report.foreign += 1;
                }
                Item::ToolCall(call)
                    if call.provider_state.is_some()
                        && matches!((target, &call.meta.origin), (Some(target), Some(origin)) if !origin.same_model(target)) =>
                {
                    let mut call = call.clone();
                    call.provider_state = None;
                    report.provider_states += 1;
                    items.push(Item::ToolCall(call));
                }
                other => items.push(other.clone()),
            }
        }
        report.normalized = repair(&mut items, info);
        if info.call_ids != crate::model::CallIds::Any {
            for item in &mut items {
                let id = match item {
                    Item::ToolCall(call) => &mut call.call_id,
                    Item::ToolOutput(output) => &mut output.call_id,
                    _ => continue,
                };
                if !info.call_ids.accepts(id) {
                    *id = info.call_ids.rewrite(id);
                    report.call_ids += 1;
                }
            }
        }
        (Self { items }, report)
    }
}

/// Whether a provider item of `provider` can go to the model `target`: the same provider, and
/// the same API when its origin is known.
fn same_provider(target: Option<&crate::item::Origin>, provider: &str, meta: &ItemMeta) -> bool {
    let Some(target) = target else {
        return true;
    };
    match &meta.origin {
        Some(origin) => origin.provider == target.provider && origin.api == target.api,
        None => provider == target.provider,
    }
}

/// The id of an answer the repair made up for the call `call_id`: the same every time.
fn synthetic_output_id(call_id: &str) -> String {
    format!("aborted_{:016x}", fnv1a(call_id.as_bytes()))
}

/// Repair `items` so that a model accepts them: see [`Conversation::normalize`].
fn repair(items: &mut Vec<Item>, info: &ModelInfo) -> NormalizeReport {
    let mut report = NormalizeReport::default();

    // Orphan outputs.
    let mut calls = BTreeSet::new();
    items.retain(|item| match link(item) {
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
    let answered: BTreeSet<String> = items
        .iter()
        .filter_map(|item| match link(item) {
            Link::Output(id) => Some(id.to_string()),
            _ => None,
        })
        .collect();
    let mut index = 0;
    while index < items.len() {
        if let Item::ToolCall(call) = &items[index] {
            if !answered.contains(&call.call_id) {
                let answer = ToolOutput::aborted(call).with_id(synthetic_output_id(&call.call_id));
                report.aborted.push(call.call_id.clone());
                let mut at = index + 1;
                while at < items.len() && link(&items[at]) != Link::None {
                    at += 1;
                }
                items.insert(at, Item::ToolOutput(answer));
            }
        }
        index += 1;
    }

    if !info.images {
        for item in items.iter_mut() {
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
        let report = conversation.normalize(&ModelInfo::default());
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
        assert!(conversation.normalize(&ModelInfo::default()).is_clean());
    }

    #[test]
    fn custom_calls_get_custom_outputs() {
        let mut conversation = Conversation::new();
        conversation.push_call(&ToolCall::custom("c", "patch", "*** Begin").in_namespace("fs"));
        conversation.normalize(&ModelInfo::default());
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
        let report = conversation.normalize(&ModelInfo::default());
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
        assert!(conversation.normalize(&ModelInfo::default()).is_clean());
        let report = conversation.normalize(&ModelInfo::new().with_images(false));
        assert_eq!(report.images, 2);
        let Item::ToolOutput(output) = &conversation.items()[2] else {
            panic!("an output expected");
        };
        assert_eq!(
            output.body,
            ToolOutputBody::Items(vec![Content::text("here"), Content::text(IMAGE_OMITTED)])
        );
        assert!(conversation
            .normalize(&ModelInfo::new().with_images(false))
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
