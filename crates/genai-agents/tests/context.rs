//! Context management: compaction (past a threshold, on overflow, by hand), pruning of old tool
//! outputs, input while a run is going, and the guards (repeated failures, calls written as text,
//! a note after a cancelled run).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(not(target_family = "wasm"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use genai_agents::session::{MemoryStore, Session, SessionId};
use genai_agents::testing::{Reply, ScriptedModel, SeenRequest};
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{
    CancellationSignal, ContextConfig, Conversation, Error, ErrorKind, Item, ItemMeta, Message,
    ModelError, ModelEvent, ModelInfo, ModelResponse, PruneConfig, Result, Role, RunConfig,
    RunEvent, RunStop, Runner, StopReason, ToolCall, COMPACTION_PROMPT, PRUNED_OUTPUT,
    REPEATED_FAILURE, SUMMARY_PREFIX, TEXT_CALL_NOTE,
};
use serde_json::{json, Value};

/// Answers with a long text, and counts its runs.
struct Fetch {
    runs: Arc<AtomicUsize>,
    size: usize,
}

impl FunctionTool for Fetch {
    type Input = Value;
    type Output = String;
    fn name(&self) -> &str {
        "fetch"
    }
    fn description(&self) -> &str {
        "Fetch a page"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"url": {"type": "string"}}})
    }
    async fn execute(&self, _: Value) -> Result<String> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        Ok("page ".repeat(self.size / 5))
    }
}

/// Always fails.
struct Broken {
    runs: Arc<AtomicUsize>,
}

impl FunctionTool for Broken {
    type Input = Value;
    type Output = Value;
    fn name(&self) -> &str {
        "broken"
    }
    fn description(&self) -> &str {
        "Never works"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _: Value) -> Result<Value> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        Err(Error::application(std::io::Error::other("disk on fire")))
    }
}

fn fetch_tools(size: usize) -> (ToolRegistry, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools
        .register(Fetch {
            runs: Arc::clone(&runs),
            size,
        })
        .unwrap();
    (tools, runs)
}

fn call(id: &str, name: &str) -> Reply {
    Reply::items(vec![Item::ToolCall(ToolCall::function(id, name, "{}"))])
}

/// The same reply without a token count, so that the runner estimates the context itself
/// (the scripted model's count is made up).
fn uncounted(mut reply: Reply) -> Reply {
    for (_, event) in reply.events_mut().unwrap() {
        if let Ok(ModelEvent::Completed(end)) = event {
            end.usage = None;
        }
    }
    reply
}

/// Whether this request is a compaction's: its last item is the compaction prompt.
fn is_compaction(request: &SeenRequest) -> bool {
    matches!(request.items.last(), Some(Item::Message(message))
        if message.text_content() == COMPACTION_PROMPT && message.meta.injected)
}

fn texts(items: &[Item]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| match item {
            Item::Message(message) => Some(message.text_content()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_long_run_is_compacted_past_the_threshold() {
    let (tools, _) = fetch_tools(4_000);
    // Fetches until the compaction, then answers.
    let model = ScriptedModel::new(|index, request| {
        uncounted(if is_compaction(request) {
            Reply::text("The user wants the news; three pages were fetched.")
        } else if texts(&request.items)
            .iter()
            .any(|text| text.starts_with(SUMMARY_PREFIX))
        {
            Reply::text("Here is the news.")
        } else {
            call(&format!("c{index}"), "fetch")
        })
    })
    // About 3 000 tokens: a fetched page is about 1 000.
    .with_info(ModelInfo::new().with_context_window(3_000));
    let mut runner = Runner::new(model.clone(), tools);
    let mut conversation = Conversation::new();
    conversation.push(
        Message::text(Role::Developer, "Be brief.").with_meta(ItemMeta::new().with_injected(true)),
    );
    conversation.push_user("What is the news?");
    let mut compactions = Vec::new();
    let outcome = runner
        .run_with(&mut conversation, &CancellationSignal::new(), |event| {
            if let RunEvent::Compacted {
                tokens_before,
                tokens_after,
            } = event
            {
                compactions.push((tokens_before, tokens_after));
            }
        })
        .await
        .unwrap();
    assert_eq!(outcome.output_text(), "Here is the news.");
    assert_eq!(compactions.len(), 1);
    let (before, after) = compactions[0];
    assert!(before >= 2_700 && after < 200, "{before} → {after}");

    // The compaction request carried the whole history and the prompt, and no tools.
    let requests = model.requests();
    let compaction = requests
        .iter()
        .find(|request| is_compaction(request))
        .unwrap();
    assert!(compaction.tools.is_empty());
    assert!(compaction.outputs().len() >= 2);

    // The request after it: the user's message, the summary, the instructions put back.
    let last = requests.last().unwrap();
    assert_eq!(
        texts(&last.items),
        [
            "What is the news?".to_owned(),
            format!("{SUMMARY_PREFIX}\n\nThe user wants the news; three pages were fetched."),
            "Be brief.".to_owned(),
        ]
    );
    assert!(last.outputs().is_empty());

    // The conversation keeps everything before the compaction.
    let compaction_at = conversation.latest_compaction().unwrap();
    assert!(compaction_at > 4);
    let Item::Compaction(stored) = &conversation.items()[compaction_at] else {
        panic!()
    };
    assert_eq!(stored.kept, [Item::user("What is the news?")]);
    assert_eq!(stored.tokens_before, Some(before));
}

#[tokio::test]
async fn a_context_overflow_is_compacted_and_sent_again() {
    let model = ScriptedModel::new(|index, request| match index {
        0 => Reply::fail(ModelError::new("too long").with_kind(ErrorKind::ContextOverflow)),
        _ if is_compaction(request) => Reply::text("Earlier: small talk."),
        _ => Reply::text("Fits now."),
    });
    let mut runner = Runner::new(model.clone(), ToolRegistry::new());
    let mut conversation = Conversation::user("Hello");
    let outcome = runner.run(&mut conversation).await.unwrap();
    assert_eq!(outcome.output_text(), "Fits now.");
    // Not counted as a retry.
    assert_eq!(outcome.turns[0].retries, 0);
    assert_eq!(model.requests().len(), 3);
    assert!(conversation.latest_compaction().is_some());
}

#[tokio::test]
async fn a_summary_cut_off_is_rejected() {
    let mut cut = Reply::text("The user wanted");
    for (_, event) in cut.events_mut().unwrap() {
        if let Ok(ModelEvent::Completed(end)) = event {
            *end = ModelResponse::new().with_stop(StopReason::Length);
        }
    }
    let mut runner = Runner::new(ScriptedModel::replies(vec![cut]), ToolRegistry::new());
    let mut conversation = Conversation::user("Hello");
    let error = runner
        .compact(&mut conversation, &CancellationSignal::new())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Compaction(_)), "{error}");
    assert_eq!(conversation.latest_compaction(), None);
}

#[tokio::test]
async fn old_tool_outputs_are_pruned_once_the_cache_expired() {
    let (tools, _) = fetch_tools(5_000);
    let model = ScriptedModel::new(|index, _| match index {
        0..=3 => call(&format!("c{index}"), "fetch"),
        _ => Reply::text("Done."),
    });
    let prune = PruneConfig::new()
        .with_cache_ttl(Duration::ZERO)
        .with_keep_recent(1)
        .with_soft_bytes(200)
        .with_hard_after(2);
    let mut runner = Runner::new(model.clone(), tools)
        .with_config(RunConfig::new().with_context(ContextConfig::new().with_prune(Some(prune))));
    let mut conversation = Conversation::user("Read four pages.");
    runner.run(&mut conversation).await.unwrap();

    // The last request: the oldest outputs replaced, the next cut, the newest whole.
    let requests = model.requests();
    let sizes: Vec<usize> = requests
        .last()
        .unwrap()
        .outputs()
        .iter()
        .map(|output| output.body.to_text().unwrap().len())
        .collect();
    assert_eq!(sizes.len(), 4);
    assert_eq!(sizes[0], PRUNED_OUTPUT.len());
    assert_eq!(sizes[1], PRUNED_OUTPUT.len());
    assert!(sizes[2] < 400, "{sizes:?}");
    assert_eq!(sizes[3], 5_002); // a JSON string, quoted
                                 // The conversation keeps every output whole.
    let kept: Vec<usize> = conversation
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::ToolOutput(output) => Some(output.body.to_text().unwrap().len()),
            _ => None,
        })
        .collect();
    assert_eq!(kept, [5_002; 4]);

    // With a cache that has not expired, nothing is pruned.
    let (tools, _) = fetch_tools(5_000);
    let model = ScriptedModel::new(|index, _| match index {
        0..=3 => call(&format!("c{index}"), "fetch"),
        _ => Reply::text("Done."),
    });
    let mut runner = Runner::new(model.clone(), tools);
    runner
        .run(&mut Conversation::user("Read four pages."))
        .await
        .unwrap();
    assert!(model
        .requests()
        .last()
        .unwrap()
        .outputs()
        .iter()
        .all(|output| output.body.to_text().unwrap().len() == 5_002));
}

#[tokio::test]
async fn steering_and_follow_ups_go_into_the_run() {
    let (tools, _) = fetch_tools(10);
    let model = ScriptedModel::new(|index, _| match index {
        0 => call("c0", "fetch"),
        _ => Reply::text(format!("answer {index}")),
    });
    let mut runner = Runner::new(model.clone(), tools);
    let input = runner.input();
    let steer = input.clone();
    let mut conversation = Conversation::user("Fetch it.");
    let mut seen = Vec::new();
    let outcome = runner
        .run_with(
            &mut conversation,
            &CancellationSignal::new(),
            |event| match event {
                // While the call runs: steering for the next request, and a follow-up for the end.
                RunEvent::ToolCalled(_) => {
                    steer.steer(Item::user("Only the headline, please."));
                    steer.follow_up(Item::user("And the weather?"));
                }
                RunEvent::Input(item) => seen.push(texts(std::slice::from_ref(item))[0].clone()),
                _ => {}
            },
        )
        .await
        .unwrap();
    assert_eq!(outcome.stop, RunStop::Completed);
    assert_eq!(seen, ["Only the headline, please.", "And the weather?"]);
    let requests = model.requests();
    assert_eq!(requests.len(), 3);
    // The steering went in after the call's answer.
    assert!(matches!(
        &requests[1].items[3..],
        [Item::Message(message)] if message.text_content() == "Only the headline, please."
    ));
    // The follow-up, after the answer.
    assert_eq!(
        texts(&requests[2].items).last().unwrap(),
        "And the weather?"
    );
    assert_eq!(outcome.output_text(), "answer 2");
    assert!(input.is_empty());
}

#[tokio::test]
async fn a_call_that_keeps_failing_is_not_run_again() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools
        .register(Broken {
            runs: Arc::clone(&runs),
        })
        .unwrap();
    let model = ScriptedModel::new(|index, _| match index {
        0..=3 => call(&format!("c{index}"), "broken"),
        _ => Reply::text("I give up."),
    });
    let mut runner = Runner::new(model.clone(), tools);
    runner.run(&mut Conversation::user("Try.")).await.unwrap();
    assert_eq!(runs.load(Ordering::SeqCst), 3);
    let last = model.requests().last().unwrap().outputs();
    assert_eq!(last[3].body.as_text(), Some(REPEATED_FAILURE));

    // Each run starts afresh.
    let model = ScriptedModel::new(|index, _| match index {
        0 => call("again", "broken"),
        _ => Reply::text("Still broken."),
    });
    *runner.model_mut() = model;
    runner
        .run(&mut Conversation::user("Try once more."))
        .await
        .unwrap();
    assert_eq!(runs.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn a_call_written_as_text_is_not_run_and_asked_for_again() {
    let (tools, runs) = fetch_tools(10);
    let text_call = r#"{"name": "fetch", "arguments": {"url": "https://example.com"}}"#;
    let model = ScriptedModel::new(move |index, _| match index {
        0 | 2 => Reply::text(text_call),
        _ => call("c1", "fetch"),
    });
    let mut runner =
        Runner::new(model.clone(), tools).with_config(RunConfig::new().with_max_iterations(4));
    let mut conversation = Conversation::user("Fetch example.com");
    let outcome = runner.run(&mut conversation).await.unwrap();
    // The note once; the second time the text is the answer.
    let requests = model.requests();
    assert_eq!(
        texts(&requests[1].items).last().map(String::as_str),
        Some(TEXT_CALL_NOTE)
    );
    assert_eq!(outcome.stop, RunStop::Completed);
    assert_eq!(outcome.output_text(), text_call);
    assert_eq!(requests.len(), 3);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_compaction_goes_into_the_session() {
    let model = ScriptedModel::new(|index, request| match index {
        0 => Reply::fail(ModelError::new("too long").with_kind(ErrorKind::ContextOverflow)),
        _ if is_compaction(request) => Reply::text("Summary."),
        _ => Reply::text("Answer."),
    });
    let store = MemoryStore::new();
    let mut session = Session::open(&store, SessionId::new("s")).await.unwrap();
    session.append_item(Item::user("q1")).await.unwrap();
    session.append_item(Item::assistant("a1")).await.unwrap();
    session.append_item(Item::user("q2")).await.unwrap();
    let mut runner = Runner::new(model, ToolRegistry::new());
    runner
        .run_session(&mut session, &CancellationSignal::new(), |_| {})
        .await
        .unwrap();
    let reopened = Session::open(&store, SessionId::new("s")).await.unwrap();
    let conversation = reopened.conversation();
    // From the compaction on: it, then the answer.
    assert_eq!(conversation.len(), 2);
    let Item::Compaction(compaction) = &conversation.items()[0] else {
        panic!("{conversation:?}")
    };
    assert_eq!(compaction.summary, "Summary.");
    assert_eq!(compaction.kept, [Item::user("q1"), Item::user("q2")]);
}
