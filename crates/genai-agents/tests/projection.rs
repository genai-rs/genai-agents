//! What each request carries: the projection of the stored conversation for the model (stable
//! answers, other models' reasoning left out, call ids rewritten, cut-off items left out), item
//! origins and turns, the partial text of a cancelled response, calls of a response cut off at
//! the output limit, the request settings, and details for the application.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(unknown_lints, clippy::unused_async_trait_impl)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{
    CallIds, CancellationSignal, Conversation, Item, ItemMeta, Message, ModelEvent, ModelInfo,
    ModelResponse, ModelSettings, Origin, ProviderItem, Reasoning, ReasoningLevel, Result, Role,
    RunConfig, RunStop, Runner, StopReason, ToolCall, ToolOutput, ToolOutputBody, CUT_OFF_OUTPUT,
};
use serde_json::{json, Value};

fn model(name: &str) -> Origin {
    Origin::new("test-api", "test", name)
}

fn from(origin: &Origin) -> ItemMeta {
    ItemMeta::new().with_origin(origin.clone())
}

// ---------------------------------------------------------------------------------------------
// The projection.

#[test]
fn the_projection_is_the_same_every_time_and_leaves_the_conversation_as_it_is() {
    let mut conversation = Conversation::user("Weather in Ghent and Paris?");
    conversation.push_call(&ToolCall::function(
        "call_a",
        "weather",
        r#"{"city":"Ghent"}"#,
    ));
    conversation.push_call(&ToolCall::function(
        "call_b",
        "weather",
        r#"{"city":"Paris"}"#,
    ));
    let stored = serde_json::to_string(&conversation).unwrap();

    let info = ModelInfo::default();
    let first = serde_json::to_string(&conversation.for_request(&info)).unwrap();
    let second = serde_json::to_string(&conversation.for_request(&info)).unwrap();
    assert_eq!(first, second);
    // The stored conversation did not change.
    assert_eq!(serde_json::to_string(&conversation).unwrap(), stored);

    let (projection, report) = conversation.project(&info);
    assert_eq!(report.normalized.aborted, ["call_a", "call_b"]);
    let outputs: Vec<&ToolOutput> = projection
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::ToolOutput(output) => Some(output),
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 2);
    // Made-up answers have ids derived from their calls: stable and distinct.
    let ids: Vec<&str> = outputs.iter().map(|o| o.id.as_deref().unwrap()).collect();
    assert_ne!(ids[0], ids[1]);
    assert!(ids.iter().all(|id| id.starts_with("aborted_")));
}

#[test]
fn reasoning_and_provider_state_of_another_model_are_left_out() {
    let mini = model("mini");
    let large = model("large");
    let mut conversation = Conversation::user("Think about it.");
    conversation.push(
        Reasoning::new("test")
            .with_encrypted_content("only-mini-reads-this")
            .with_meta(from(&mini)),
    );
    conversation.push(
        ToolCall::function("call_1", "look", "{}")
            .with_provider_state(json!({"signature": "mini"}))
            .with_meta(from(&mini)),
    );
    conversation.push_output(&ToolOutput::new(
        &ToolCall::function("call_1", "look", "{}"),
        "seen",
        true,
    ));
    conversation.push(
        ProviderItem::new("test", json!({"type": "web_search_call"}))
            .with_meta(ItemMeta::new().with_origin(model("mini"))),
    );
    // Reasoning of another provider, without an origin.
    conversation.push(Reasoning::new("elsewhere").with_encrypted_content("x"));

    // The model that wrote them gets them all, but not the other provider's reasoning.
    let (same, report) = conversation.project(&ModelInfo::new().with_origin(mini.clone()));
    assert_eq!(report.foreign, 1);
    assert_eq!(report.provider_states, 0);
    assert!(same
        .items()
        .iter()
        .any(|item| matches!(item, Item::Reasoning(r) if r.provider == "test")));

    // Another model of the same provider and API: no reasoning, no provider state, but the
    // provider item (a hosted tool call) and the call itself stay.
    let (other, report) = conversation.project(&ModelInfo::new().with_origin(large));
    assert_eq!(report.foreign, 2);
    assert_eq!(report.provider_states, 1);
    assert!(!other
        .items()
        .iter()
        .any(|item| matches!(item, Item::Reasoning(_))));
    let Some(Item::ToolCall(call)) = other.items().get(1) else {
        panic!("the call expected, got {:?}", other.items());
    };
    assert_eq!(call.provider_state, None);
    assert!(other
        .items()
        .iter()
        .any(|item| matches!(item, Item::Provider(_))));

    // Without a known target, everything is kept.
    let (all, report) = conversation.project(&ModelInfo::default());
    assert_eq!(report.foreign, 0);
    assert_eq!(all.len(), conversation.len());
}

#[test]
fn call_ids_the_model_refuses_are_rewritten_in_calls_and_outputs() {
    let long = format!("fc_{}", "x".repeat(200));
    let call = ToolCall::function(&long, "f", "{}");
    let mut conversation = Conversation::user("go");
    conversation.push_call(&call);
    conversation.push_output(&ToolOutput::new(&call, "ok", true));
    conversation.push_call(&ToolCall::function("call.with|chars", "f", "{}"));

    let info = ModelInfo::new().with_call_ids(CallIds::Restricted { max_len: 64 });
    let (projection, report) = conversation.project(&info);
    assert_eq!(report.call_ids, 4); // two calls and two outputs (one made up)
    let ids: Vec<String> = projection
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::ToolCall(call) => Some(call.call_id.clone()),
            Item::ToolOutput(output) => Some(output.call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ids.len(), 4);
    assert_eq!(ids[0], ids[1]);
    assert_eq!(ids[2], ids[3]);
    assert_ne!(ids[0], ids[2]);
    for id in &ids {
        assert!(info.call_ids.accepts(id), "{id}");
    }
    assert_eq!(
        serde_json::to_string(&projection).unwrap(),
        serde_json::to_string(&conversation.for_request(&info)).unwrap()
    );
    // Ids it takes stay as they are.
    assert_eq!(info.call_ids.rewrite("call_1"), "call_1");
}

#[test]
fn cut_off_text_is_kept_but_not_sent() {
    let mut conversation = Conversation::user("Tell me a story.");
    conversation.push(
        Message::text(Role::Assistant, "Once upon")
            .with_meta(ItemMeta::new().with_interrupted(true)),
    );
    conversation.push_user("Shorter, please.");
    let (projection, report) = conversation.project(&ModelInfo::default());
    assert_eq!(report.interrupted, 1);
    assert_eq!(projection.len(), 2);
    assert_eq!(conversation.len(), 3);
}

#[test]
fn metadata_round_trips_and_is_left_out_when_empty() {
    let item = Item::from(
        Message::text(Role::Developer, "Use metric units.")
            .injected()
            .with_meta(ItemMeta::new().with_injected(true).with_turn(3)),
    );
    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(json["meta"], json!({"turn": 3, "injected": true}));
    assert_eq!(serde_json::from_value::<Item>(json).unwrap(), item);
    // A conversation stored before metadata existed still loads.
    let old =
        json!([{"type": "message", "role": "user", "content": [{"type": "text", "text": "hi"}]}]);
    let conversation: Conversation = serde_json::from_value(old.clone()).unwrap();
    assert!(conversation.items()[0].meta().is_empty());
    assert_eq!(serde_json::to_value(&conversation).unwrap(), old);
}

// ---------------------------------------------------------------------------------------------
// The runner.

struct Counter {
    runs: Arc<AtomicUsize>,
}

impl FunctionTool for Counter {
    type Input = Value;
    type Output = Value;
    fn name(&self) -> &str {
        "count"
    }
    fn description(&self) -> &str {
        "Counts"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _input: Value) -> Result<Value> {
        let run = self.runs.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(json!({"run": run, "rows": ["a", "b", "c"]}))
    }
    fn details(&self, output: &Value) -> Option<Value> {
        Some(json!({"rows": output["rows"].as_array().map_or(0, Vec::len)}))
    }
}

fn counter() -> (ToolRegistry, Arc<AtomicUsize>) {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools
        .register(Counter {
            runs: Arc::clone(&runs),
        })
        .unwrap();
    (tools, runs)
}

#[tokio::test]
async fn the_runner_sends_projections_and_records_origin_turn_and_details() {
    let origin = model("mini");
    let (tools, _) = counter();
    let model = ScriptedModel::replies(vec![
        Reply::items(vec![Item::ToolCall(ToolCall::function(
            "c1", "count", "{}",
        ))]),
        Reply::text("Three rows."),
    ])
    .with_info(ModelInfo::new().with_origin(origin.clone()));
    let mut runner = Runner::new(model, tools).with_config(
        RunConfig::new().with_model_settings(
            ModelSettings::new()
                .with_reasoning(ReasoningLevel::Low)
                .with_max_output_tokens(500),
        ),
    );
    // A call left unanswered by an earlier run: the request answers it, the conversation not.
    let mut conversation = Conversation::user("Count.");
    conversation.push_call(&ToolCall::function("old", "count", "{}"));

    let outcome = runner.run(&mut conversation).await.unwrap();
    assert!(outcome.is_completed());

    let requests = runner.model().requests();
    assert_eq!(requests[0].settings.reasoning, Some(ReasoningLevel::Low));
    assert_eq!(requests[0].settings.max_output_tokens, Some(500));
    assert!(requests[0]
        .outputs()
        .iter()
        .any(|output| output.call_id == "old"));
    assert!(conversation
        .unanswered_calls()
        .iter()
        .any(|call| call.call_id == "old"));

    // The model's items carry its origin and their turn; the answers their details.
    let calls: Vec<&ToolCall> = conversation
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::ToolCall(call) if call.call_id == "c1" => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls[0].meta.origin.as_ref(), Some(&origin));
    assert_eq!(calls[0].meta.turn, Some(1));
    let answer = conversation
        .items()
        .iter()
        .find_map(|item| match item {
            Item::ToolOutput(output) if output.call_id == "c1" => Some(output),
            _ => None,
        })
        .unwrap();
    assert_eq!(answer.details, Some(json!({"rows": 3})));
    let Some(Item::Message(last)) = conversation.items().last() else {
        panic!("the answer expected");
    };
    assert_eq!(last.meta.turn, Some(2));
    assert_eq!(conversation.next_turn(), 3);
    // The details stay with the application: the model saw the body only.
    let seen = requests[1]
        .outputs()
        .into_iter()
        .find(|output| output.call_id == "c1")
        .unwrap();
    assert!(matches!(&seen.body, ToolOutputBody::Text(text) if text.contains("\"run\":1")));
}

#[tokio::test]
async fn a_cancelled_response_keeps_its_partial_text_cut_off() {
    let mut reply = Reply::items(vec![]);
    {
        let events = reply.events_mut().unwrap();
        events.clear();
        events.push((Duration::ZERO, Ok(ModelEvent::TextDelta("Once ".into()))));
        events.push((Duration::ZERO, Ok(ModelEvent::TextDelta("upon".into()))));
        events.push((
            Duration::from_secs(30),
            Ok(ModelEvent::Completed(ModelResponse::new())),
        ));
    }
    let model = ScriptedModel::replies(vec![reply, Reply::text("Done.")]);
    let mut runner = Runner::new(model, ToolRegistry::new());
    let mut conversation = Conversation::user("Tell me a story.");
    let cancel = CancellationSignal::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        trigger.cancel();
    });
    let outcome = runner
        .run_with(&mut conversation, &cancel, |_| {})
        .await
        .unwrap();
    assert_eq!(outcome.stop, RunStop::Cancelled);
    assert_eq!(outcome.output_text(), "Once upon");
    // The partial text, then the note that the run was interrupted.
    let items = conversation.items();
    let Item::Message(partial) = &items[items.len() - 2] else {
        panic!("the partial text expected");
    };
    assert_eq!(partial.text_content(), "Once upon");
    assert!(partial.meta.interrupted);
    let Some(Item::Message(note)) = items.last() else {
        panic!("the note expected");
    };
    assert_eq!(note.text_content(), genai_agents::INTERRUPTED_NOTE);
    assert!(note.meta.injected);

    // The next run does not send it.
    conversation.push_user("Go on.");
    runner.run(&mut conversation).await.unwrap();
    let requests = runner.model().requests();
    assert!(!requests[1].items.iter().any(|item| matches!(
        item,
        Item::Message(message) if message.text_content() == "Once upon"
    )));
}

#[tokio::test]
async fn calls_of_a_response_cut_off_at_the_output_limit_are_not_run() {
    let (tools, runs) = counter();
    let cut = ToolCall::function("c2", "count", r#"{"partial":"#)
        .with_meta(ItemMeta::new().with_interrupted(true));
    let mut reply = Reply::items(vec![Item::ToolCall(cut)]);
    {
        let events = reply.events_mut().unwrap();
        let last = events.len() - 1;
        events[last].1 = Ok(ModelEvent::Completed(
            ModelResponse::new()
                .with_stop(StopReason::Length)
                .with_incomplete("max_output_tokens"),
        ));
    }
    let model = ScriptedModel::replies(vec![reply]);
    let mut runner = Runner::new(model, tools);
    let mut conversation = Conversation::user("Count.");
    let outcome = runner.run(&mut conversation).await.unwrap();
    assert!(matches!(outcome.stop, RunStop::Incomplete { .. }));
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    let answer = conversation
        .items()
        .iter()
        .find_map(|item| match item {
            Item::ToolOutput(output) => Some(output),
            _ => None,
        })
        .unwrap();
    assert!(!answer.success);
    assert_eq!(answer.body, ToolOutputBody::text(CUT_OFF_OUTPUT));
}

struct Slow {
    runs: Arc<AtomicUsize>,
}

impl FunctionTool for Slow {
    type Input = Value;
    type Output = &'static str;
    fn name(&self) -> &str {
        "slow"
    }
    fn description(&self) -> &str {
        "Takes a while; one at a time"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _input: Value) -> Result<&'static str> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(150)).await;
        Ok("done")
    }
}

#[tokio::test]
async fn calls_queued_when_the_response_stops_at_the_output_limit_are_not_run() {
    let runs = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools
        .register(Slow {
            runs: Arc::clone(&runs),
        })
        .unwrap();
    let mut reply = Reply::items(vec![
        Item::ToolCall(ToolCall::function("s1", "slow", "{}")),
        Item::ToolCall(ToolCall::function("s2", "slow", "{}")),
    ]);
    {
        let events = reply.events_mut().unwrap();
        let last = events.len() - 1;
        events[last].1 = Ok(ModelEvent::Completed(
            ModelResponse::new()
                .with_stop(StopReason::Length)
                .with_incomplete("max_output_tokens"),
        ));
    }
    let mut runner = Runner::new(ScriptedModel::replies(vec![reply]), tools);
    let mut conversation = Conversation::user("Go.");
    runner.run(&mut conversation).await.unwrap();
    // The first call had started and finished; the second, still queued, was not run.
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    let answers: Vec<(String, ToolOutputBody)> = conversation
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::ToolOutput(output) => Some((output.call_id.clone(), output.body.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        answers,
        [
            ("s1".to_string(), ToolOutputBody::text("\"done\"")),
            ("s2".to_string(), ToolOutputBody::text(CUT_OFF_OUTPUT)),
        ]
    );
}
