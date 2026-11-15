//! Context management: what keeps a long run within the model's context window (token
//! accounting, compaction, pruning), input that arrives while a run is going, and the guards
//! against a model that loops.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use serde_json::Value;

use crate::item::{Content, Item, ItemMeta, Message, Role, ToolOutputBody};
use crate::model::ModelSettings;
use crate::tools::{truncate_middle, OutputBudget};

/// What a compaction asks the model for.
pub const COMPACTION_PROMPT: &str = "\
Write a summary of this conversation for whoever continues it, who will see nothing of it but \
your summary and the user's last messages. Cover:
- what the user wants, with their constraints and preferences;
- what was done so far, what was found, and the decisions taken with their reasons;
- the facts, names, paths, identifiers and figures that the rest of the work needs;
- what is left to do, in order, starting with the next step.
Be specific and concise. Write the summary only.";

/// The note a cancelled run leaves in the conversation, so that the model knows tools may have
/// done part of their work.
pub const INTERRUPTED_NOTE: &str = "The previous run was interrupted before it finished. Tools \
that were running may have done part of their work; check before relying on their results.";

/// The note sent when the model wrote a tool call as text instead of calling the tool.
pub const TEXT_CALL_NOTE: &str = "You wrote a tool call as text; it was not run. To use a tool, \
call it as a tool call.";

/// The answer to a call that already failed with the same arguments
/// [`ContextConfig::max_repeated_failures`] times.
pub const REPEATED_FAILURE: &str = "this call already failed with these same arguments several \
times; it was not run again. Change the arguments or try another way.";

/// What replaces a tool output that [pruning](PruneConfig) left out.
pub const PRUNED_OUTPUT: &str = "[an older tool output, left out to save context]";

/// How the runner keeps a conversation within the model's context window, and guards against a
/// model that loops. See the [context guide](https://github.com/genai-rs/genai-agents/blob/main/docs/context.md).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContextConfig {
    /// Compact before a request whose estimated tokens pass this percentage of the model's
    /// [context window](crate::ModelInfo::context_window); `None` to compact only when the model
    /// answers that the context is too long. Default 90.
    pub compact_at: Option<u8>,
    /// Compact and send the request again when the model answers that the context is too long
    /// ([`ErrorKind::ContextOverflow`](crate::ErrorKind::ContextOverflow)), once per request.
    /// Default `true`.
    pub compact_on_overflow: bool,
    /// What a compaction asks the model for. Default [`COMPACTION_PROMPT`].
    pub compaction_prompt: String,
    /// The settings of the compaction request. Default: the run's.
    pub compaction_settings: Option<ModelSettings>,
    /// The tokens of the user's most recent messages that a compaction keeps word for word,
    /// before its summary. Default 20 000.
    pub keep_user_tokens: u64,
    /// The pruning of old tool outputs; `None` to keep them whole. Default on.
    pub prune: Option<PruneConfig>,
    /// A call that failed this many times with the same arguments in a run is not run again:
    /// it is answered with [`REPEATED_FAILURE`]. 0 for no limit. Default 3.
    pub max_repeated_failures: u32,
    /// When the model writes a tool call as text instead of calling the tool, tell it so and
    /// send one more request (once per run); the text is never run. Default `true`.
    pub nudge_text_calls: bool,
    /// A cancelled run leaves [`INTERRUPTED_NOTE`] in the conversation. Default `true`.
    pub note_interruptions: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            compact_at: Some(90),
            compact_on_overflow: true,
            compaction_prompt: COMPACTION_PROMPT.to_owned(),
            compaction_settings: None,
            keep_user_tokens: 20_000,
            prune: Some(PruneConfig::default()),
            max_repeated_failures: 3,
            nudge_text_calls: true,
            note_interruptions: true,
        }
    }
}

impl ContextConfig {
    /// The defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compact past this percentage of the context window; `None` only on overflow.
    #[must_use]
    pub fn with_compact_at(mut self, percent: Option<u8>) -> Self {
        self.compact_at = percent;
        self
    }

    /// Whether to compact when the model answers that the context is too long.
    #[must_use]
    pub fn with_compact_on_overflow(mut self, compact: bool) -> Self {
        self.compact_on_overflow = compact;
        self
    }

    /// What a compaction asks the model for.
    #[must_use]
    pub fn with_compaction_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.compaction_prompt = prompt.into();
        self
    }

    /// The settings of the compaction request.
    #[must_use]
    pub fn with_compaction_settings(mut self, settings: ModelSettings) -> Self {
        self.compaction_settings = Some(settings);
        self
    }

    /// The tokens of recent user messages a compaction keeps.
    #[must_use]
    pub fn with_keep_user_tokens(mut self, tokens: u64) -> Self {
        self.keep_user_tokens = tokens;
        self
    }

    /// The pruning of old tool outputs; `None` to keep them whole.
    #[must_use]
    pub fn with_prune(mut self, prune: Option<PruneConfig>) -> Self {
        self.prune = prune;
        self
    }

    /// The failures of one call with the same arguments before it is no longer run; 0 for no
    /// limit.
    #[must_use]
    pub fn with_max_repeated_failures(mut self, failures: u32) -> Self {
        self.max_repeated_failures = failures;
        self
    }

    /// Whether to tell a model that wrote a tool call as text to call the tool.
    #[must_use]
    pub fn with_nudge_text_calls(mut self, nudge: bool) -> Self {
        self.nudge_text_calls = nudge;
        self
    }

    /// Whether a cancelled run leaves a note in the conversation.
    #[must_use]
    pub fn with_note_interruptions(mut self, note: bool) -> Self {
        self.note_interruptions = note;
        self
    }
}

/// The pruning of old tool outputs: what the request sends of them, not what the conversation
/// keeps.
///
/// Changing an earlier part of a request breaks the provider's prompt cache, so pruning happens
/// only once the cache has expired anyway: when [`cache_ttl`](Self::cache_ttl) has passed since
/// the previous request. It then takes effect up to that point of the conversation, and stays so
/// for the requests after it, which keeps their prefix stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PruneConfig {
    /// How long the provider keeps a prompt in its cache. Default 5 minutes.
    pub cache_ttl: Duration,
    /// The most recent tool outputs that are never pruned. Default 4.
    pub keep_recent: usize,
    /// What is left of an older output: its head and tail, this many bytes. Default 2 000.
    pub soft_bytes: usize,
    /// Outputs older than this many outputs are replaced with [`PRUNED_OUTPUT`]. Default 16.
    pub hard_after: usize,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            cache_ttl: Duration::from_secs(300),
            keep_recent: 4,
            soft_bytes: 2_000,
            hard_after: 16,
        }
    }
}

impl PruneConfig {
    /// The defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How long the provider keeps a prompt in its cache.
    #[must_use]
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// The most recent tool outputs that are never pruned.
    #[must_use]
    pub fn with_keep_recent(mut self, outputs: usize) -> Self {
        self.keep_recent = outputs;
        self
    }

    /// The bytes left of an older output.
    #[must_use]
    pub fn with_soft_bytes(mut self, bytes: usize) -> Self {
        self.soft_bytes = bytes;
        self
    }

    /// The outputs after which an older one is replaced with a placeholder.
    #[must_use]
    pub fn with_hard_after(mut self, outputs: usize) -> Self {
        self.hard_after = outputs;
        self
    }
}

/// Prune the tool outputs of `items` (a request's projection) that come before `boundary`
/// outputs: the oldest, beyond [`PruneConfig::hard_after`] from the boundary, replaced with
/// [`PRUNED_OUTPUT`]; the others cut to [`PruneConfig::soft_bytes`]. Returns how many it changed.
pub(super) fn prune(items: &mut [Item], boundary: usize, config: &PruneConfig) -> usize {
    let total = items
        .iter()
        .filter(|item| matches!(item, Item::ToolOutput(_)))
        .count();
    let boundary = boundary.min(total);
    let hard = boundary.saturating_sub(config.hard_after.saturating_sub(config.keep_recent));
    let mut changed = 0;
    let mut ordinal = 0;
    for item in items.iter_mut() {
        let Item::ToolOutput(output) = item else {
            continue;
        };
        let index = ordinal;
        ordinal += 1;
        if index >= boundary {
            break;
        }
        let pruned = if index < hard {
            ToolOutputBody::text(PRUNED_OUTPUT)
        } else {
            match output.body.to_text() {
                Some(text) if text.len() > config.soft_bytes => ToolOutputBody::text(
                    truncate_middle(&text, OutputBudget::Bytes(config.soft_bytes)).into_owned(),
                ),
                Some(_) => continue,
                // Images and files: only their text is kept.
                None => ToolOutputBody::text(PRUNED_OUTPUT),
            }
        };
        if output.body != pruned {
            output.body = pruned;
            changed += 1;
        }
    }
    changed
}

/// The number of tool outputs in `items`.
pub(super) fn outputs(items: &[Item]) -> usize {
    items
        .iter()
        .filter(|item| matches!(item, Item::ToolOutput(_)))
        .count()
}

/// The estimated tokens of items: their JSON at four bytes a token, and a flat 1 000 per image
/// or file. An estimate, for deciding when to compact; the server's count is the real one.
#[must_use]
pub fn estimate_tokens(items: &[Item]) -> u64 {
    items.iter().map(estimate_item).sum()
}

fn estimate_item(item: &Item) -> u64 {
    let mut media = 0_u64;
    let mut count = |parts: &[Content]| {
        media += 1_000
            * parts
                .iter()
                .filter(|part| part.is_image() || is_file(part))
                .count() as u64;
    };
    match item {
        Item::Message(message) => count(&message.content),
        Item::ToolOutput(output) => {
            if let ToolOutputBody::Items(parts) = &output.body {
                count(parts);
            }
        }
        _ => {}
    }
    let text = serde_json::to_string(item).map_or(0, |json| json.len());
    (text as u64).div_ceil(4) + media
}

fn is_file(part: &Content) -> bool {
    matches!(part, Content::File { .. })
}

/// What a compaction keeps word for word: the user's most recent messages, newest first until
/// `budget` tokens, in their order; and the injected context, to put back after the summary.
pub(super) fn kept_items(active: &[Item], budget: u64) -> (Vec<Item>, Vec<Item>) {
    let mut kept = Vec::new();
    let mut spent = 0_u64;
    for item in active.iter().rev() {
        let Item::Message(message) = item else {
            continue;
        };
        if message.role != Role::User || message.meta.injected || message.meta.interrupted {
            continue;
        }
        let tokens = estimate_item(item);
        if spent + tokens > budget {
            break;
        }
        spent += tokens;
        kept.push(item.clone());
    }
    kept.reverse();
    let context = active
        .iter()
        .filter(|item| {
            matches!(item, Item::Message(message)
                if message.meta.injected && !is_note(message))
        })
        .cloned()
        .collect();
    (kept, context)
}

/// Whether an injected message is one the runner or a compaction wrote, not context to keep.
fn is_note(message: &Message) -> bool {
    let text = message.text_content();
    text.starts_with(crate::item::SUMMARY_PREFIX)
        || text == INTERRUPTED_NOTE
        || text == TEXT_CALL_NOTE
}

/// An injected user message with this text.
pub(super) fn note(text: &str) -> Item {
    Item::Message(Message::text(Role::User, text).with_meta(ItemMeta::new().with_injected(true)))
}

/// Whether `text`, the whole of an answer, is a tool call written as text: a JSON object that
/// names a tool (`{"name": "get_weather", "arguments": …}`), or a `<tool_call>` block.
pub(super) fn looks_like_text_call(text: &str, is_tool: impl Fn(&str) -> bool) -> bool {
    let text = text.trim();
    if text.contains("<tool_call>") || text.contains("<function_call>") {
        return true;
    }
    let text = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .map_or(text, |inner| inner.trim_end_matches("```"))
        .trim();
    let Ok(Value::Object(object)) = serde_json::from_str::<Value>(text) else {
        return false;
    };
    let named = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(&is_tool)
    };
    let function = object
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .is_some_and(&is_tool);
    (named("name") || named("tool") || function)
        && ["arguments", "parameters", "input", "args", "function"]
            .iter()
            .any(|key| object.contains_key(*key))
}

/// Input for a run that is going: messages the application sends while the model works.
///
/// - [`steer`](Self::steer): a message the model should take into account now. The runner
///   sends it with the next request, once the calls of the current one are answered.
/// - [`follow_up`](Self::follow_up): a message for when the model is done. When the run would
///   stop because the model answered, the runner sends the follow-ups and goes on.
///
/// Clone it and push from anywhere (another task, a UI's handler); the runner takes the
/// messages at those points, so it still spawns nothing ([`Runner::input`](super::Runner::input)).
#[derive(Debug, Clone, Default)]
pub struct RunInput {
    queue: Arc<Mutex<Queues>>,
}

#[derive(Debug, Default)]
struct Queues {
    steering: VecDeque<Item>,
    follow_ups: VecDeque<Item>,
}

impl RunInput {
    /// An empty input.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A message for the next request (a user message, say: `Item::user("Use metric units")`).
    pub fn steer(&self, item: impl Into<Item>) {
        self.lock().steering.push_back(item.into());
    }

    /// A message for when the model is done.
    pub fn follow_up(&self, item: impl Into<Item>) {
        self.lock().follow_ups.push_back(item.into());
    }

    /// Whether no message waits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let queues = self.lock();
        queues.steering.is_empty() && queues.follow_ups.is_empty()
    }

    /// Drop the messages that wait.
    pub fn clear(&self) {
        let mut queues = self.lock();
        queues.steering.clear();
        queues.follow_ups.clear();
    }

    pub(super) fn take_steering(&self) -> Vec<Item> {
        self.lock().steering.drain(..).collect()
    }

    pub(super) fn take_follow_ups(&self) -> Vec<Item> {
        self.lock().follow_ups.drain(..).collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Queues> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use crate::item::{ToolCall, ToolOutput};

    #[test]
    fn text_calls_are_recognised() {
        let tool = |name: &str| name == "get_weather";
        assert!(looks_like_text_call(
            r#"{"name": "get_weather", "arguments": {"city": "Ghent"}}"#,
            tool
        ));
        assert!(looks_like_text_call(
            "```json\n{\"function\": {\"name\": \"get_weather\"}, \"arguments\": {}}\n```",
            tool
        ));
        assert!(looks_like_text_call(
            "<tool_call>get_weather</tool_call>",
            tool
        ));
        // An answer about a tool, or JSON the user asked for, is not a call.
        assert!(!looks_like_text_call("I will call get_weather now.", tool));
        assert!(!looks_like_text_call(
            r#"{"name": "Ghent", "arguments": 2}"#,
            tool
        ));
        assert!(!looks_like_text_call(r#"{"name": "get_weather"}"#, tool));
    }

    #[test]
    fn pruning_cuts_older_outputs_and_keeps_the_boundary() {
        let mut items = Vec::new();
        for index in 0..6 {
            let call = ToolCall::function(format!("c{index}"), "f", "{}");
            items.push(Item::ToolCall(call.clone()));
            items.push(Item::ToolOutput(ToolOutput::new(
                &call,
                "x".repeat(5_000),
                true,
            )));
        }
        let config = PruneConfig::new()
            .with_keep_recent(1)
            .with_hard_after(3)
            .with_soft_bytes(100);
        let original = items.clone();
        // Outputs 0..4 are before the boundary: 0 and 1 replaced, 2 and 3 cut.
        let changed = prune(&mut items, 4, &config);
        assert_eq!(changed, 4);
        let bodies: Vec<usize> = items
            .iter()
            .filter_map(|item| match item {
                Item::ToolOutput(output) => Some(output.body.to_text().unwrap().len()),
                _ => None,
            })
            .collect();
        assert_eq!(bodies[0], PRUNED_OUTPUT.len());
        assert_eq!(bodies[1], PRUNED_OUTPUT.len());
        assert!(bodies[2] < 200 && bodies[3] < 200);
        assert_eq!(&bodies[4..], [5_000, 5_000]);
        // Each request prunes its own projection: the same history gives the same request.
        let mut again = original;
        prune(&mut again, 4, &config);
        assert_eq!(again, items);
    }

    #[test]
    fn compaction_keeps_recent_user_messages_and_context() {
        let instructions = Item::Message(
            Message::text(Role::Developer, "Be brief.")
                .with_meta(ItemMeta::new().with_injected(true)),
        );
        let items = vec![
            instructions.clone(),
            Item::user("first ".repeat(100)),
            Item::assistant("ok"),
            Item::user("second"),
            note(INTERRUPTED_NOTE),
            Item::user("third"),
        ];
        let budget = estimate_tokens(&[Item::user("second"), Item::user("third")]);
        let (kept, context) = kept_items(&items, budget);
        assert_eq!(kept, [Item::user("second"), Item::user("third")]);
        assert_eq!(context, [instructions]);
    }

    #[test]
    fn input_queues() {
        let input = RunInput::new();
        input.steer(Item::user("now"));
        input.follow_up(Item::user("later"));
        let clone = input.clone();
        assert!(!clone.is_empty());
        assert_eq!(clone.take_steering(), [Item::user("now")]);
        assert_eq!(input.take_follow_ups(), [Item::user("later")]);
        assert!(input.is_empty());
    }
}
