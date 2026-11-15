//! Recorded runs: a tool loop recorded and replayed, failures and retries replayed, requests
//! that no longer match.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(not(target_family = "wasm"))]

use genai_agents::record::{RecordedResponse, RecordingModel, ReplayModel};
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{
    Conversation, ErrorKind, Item, Model, ModelError, ModelInfo, Origin, Result, RetryClass,
    RetryConfig, RunConfig, Runner, ToolCall,
};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct City {
    city: String,
}

struct Weather;

impl FunctionTool for Weather {
    type Input = City;
    type Output = Value;
    fn name(&self) -> &str {
        "get_weather"
    }
    fn description(&self) -> &str {
        "Current weather of a city"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
    }
    async fn execute(&self, input: City) -> Result<Value> {
        Ok(json!({"city": input.city, "sky": "light rain"}))
    }
}

fn tools() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(Weather).unwrap();
    tools
}

fn info() -> ModelInfo {
    ModelInfo::new()
        .with_origin(Origin::new("responses", "openai", "gpt-test"))
        .with_context_window(128_000)
}

fn live() -> ScriptedModel {
    ScriptedModel::replies(vec![
        Reply::items(vec![Item::ToolCall(ToolCall::function(
            "call_1",
            "get_weather",
            r#"{"city":"Ghent"}"#,
        ))])
        .with_id("resp_1")
        .with_remaining_requests(99),
        Reply::text("Take an umbrella.").with_id("resp_2"),
    ])
    .with_info(info())
}

fn quick_retries() -> RunConfig {
    RunConfig::new().with_retry(
        RetryConfig::default()
            .with_base_delay(std::time::Duration::from_millis(1))
            .with_max_delay(std::time::Duration::from_millis(1)),
    )
}

#[tokio::test]
async fn a_tool_loop_replays_as_it_was_recorded() {
    let mut runner = Runner::new(RecordingModel::new(live()), tools());
    let mut live_conversation = Conversation::user("Umbrella in Ghent?");
    let recorded = runner.run(&mut live_conversation).await.unwrap();
    let recording = runner.model().recording();
    assert_eq!(recording.exchanges.len(), 2);
    assert_eq!(recording.info, Some(info()));
    let jsonl = recording.to_jsonl().unwrap();
    assert_eq!(jsonl.lines().count(), 3);

    let replay = ReplayModel::from_jsonl(&jsonl).unwrap();
    assert_eq!(replay.info(), info());
    let mut runner = Runner::new(replay, tools());
    let mut conversation = Conversation::user("Umbrella in Ghent?");
    let replayed = runner.run(&mut conversation).await.unwrap();
    assert_eq!(replayed, recorded);
    assert_eq!(conversation, live_conversation);
    assert_eq!(runner.model().remaining(), 0);
}

#[tokio::test]
async fn failures_and_retries_replay() {
    let flaky = ScriptedModel::replies(vec![
        Reply::fail(
            ModelError::new("overloaded")
                .with_status(503)
                .with_kind(ErrorKind::Server)
                .with_retry_class(RetryClass::Backoff),
        ),
        Reply::text("Fine."),
    ]);
    let mut runner =
        Runner::new(RecordingModel::new(flaky), ToolRegistry::new()).with_config(quick_retries());
    let recorded = runner.run(&mut Conversation::user("Hi")).await.unwrap();
    assert_eq!(recorded.turns[0].retries, 1);
    let recording = runner.model().recording();
    let RecordedResponse::Failed { error } = &recording.exchanges[0].response else {
        panic!("{recording:?}")
    };
    assert_eq!(error.status, Some(503));
    assert_eq!(error.kind, ErrorKind::Server);

    let mut runner =
        Runner::new(ReplayModel::new(recording), ToolRegistry::new()).with_config(quick_retries());
    let replayed = runner.run(&mut Conversation::user("Hi")).await.unwrap();
    assert_eq!(replayed, recorded);
}

#[tokio::test]
async fn a_changed_request_fails_the_replay_with_the_difference() {
    let mut runner = Runner::new(RecordingModel::new(live()), tools());
    runner
        .run(&mut Conversation::user("Umbrella in Ghent?"))
        .await
        .unwrap();
    let recording = runner.model().recording();

    // Another tool description.
    struct Renamed;
    impl FunctionTool for Renamed {
        type Input = City;
        type Output = Value;
        fn name(&self) -> &str {
            "get_weather"
        }
        fn description(&self) -> &str {
            "The weather, now"
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object", "properties": {"city": {"type": "string"}}})
        }
        async fn execute(&self, _: City) -> Result<Value> {
            Ok(Value::Null)
        }
    }
    let mut renamed = ToolRegistry::new();
    renamed.register(Renamed).unwrap();
    let mut runner = Runner::new(ReplayModel::new(recording.clone()), renamed);
    let error = runner
        .run(&mut Conversation::user("Umbrella in Ghent?"))
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("request 1 differs"), "{message}");
    assert!(message.contains("/tools/0/description"), "{message}");
    assert!(message.contains("\"The weather, now\""), "{message}");
    // Not retried: a replay that does not match will not match on a retry either.
    assert_eq!(error.as_model().unwrap().retry_class(), RetryClass::Never);

    // A run that asks for more than was recorded.
    let mut runner =
        Runner::new(ReplayModel::new(recording), tools()).with_config(RunConfig::new());
    let mut conversation = Conversation::user("Umbrella in Ghent?");
    runner.run(&mut conversation).await.unwrap();
    conversation.push_user("And tomorrow?");
    let error = runner.run(&mut conversation).await.unwrap_err();
    assert!(
        error.to_string().contains("beyond the recording"),
        "{error}"
    );
}

#[tokio::test]
async fn a_recording_survives_a_file() {
    let mut runner = Runner::new(RecordingModel::new(live()), tools());
    runner
        .run(&mut Conversation::user("Umbrella in Ghent?"))
        .await
        .unwrap();
    let path = std::env::temp_dir().join(format!(
        "genai-agents-recording-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    runner.model().save(&path).unwrap();
    let replay = ReplayModel::load(&path);
    let _ = std::fs::remove_file(&path);
    let mut runner = Runner::new(replay.unwrap(), tools());
    let outcome = runner
        .run(&mut Conversation::user("Umbrella in Ghent?"))
        .await
        .unwrap();
    assert_eq!(outcome.output_text(), "Take an umbrella.");
}
