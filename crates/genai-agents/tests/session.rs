//! Sessions: the entry log, its guards (writers, input keys), rollback, fork, compaction, the
//! stores, and runs that write into a session.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(not(target_family = "wasm"))]

use std::time::Duration;

use genai_agents::session::{
    Compaction, EntryId, MemoryStore, Session, SessionEntry, SessionError, SessionId, SessionStore,
};
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{
    CancellationSignal, Error, Item, ModelEvent, ModelInfo, Origin, ReasoningLevel, Result, Role,
    RunConfig, RunStop, Runner, ToolCall, Usage,
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

/// Calls the weather tool, then answers.
fn weather_model() -> ScriptedModel {
    ScriptedModel::replies(vec![
        Reply::items(vec![Item::ToolCall(ToolCall::function(
            "call_1",
            "get_weather",
            r#"{"city":"Ghent"}"#,
        ))]),
        Reply::text("Take an umbrella."),
    ])
    .with_info(ModelInfo::new().with_origin(Origin::new("responses", "openai", "gpt-test")))
}

fn texts(session: &Session<impl SessionStore>) -> Vec<String> {
    session
        .conversation()
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::Message(message) => Some(message.text_content()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_run_writes_its_turns_into_the_session() {
    let store = MemoryStore::new();
    let mut session = Session::open(&store, SessionId::new("s1")).await.unwrap();
    session
        .append_input("in-1", Item::user("Umbrella in Ghent?"))
        .await
        .unwrap();
    let mut runner =
        Runner::new(weather_model(), tools()).with_config(RunConfig::new().with_model_settings(
            genai_agents::ModelSettings::new().with_reasoning(ReasoningLevel::Low),
        ));
    let outcome = runner
        .run_session(&mut session, &CancellationSignal::new(), |_| {})
        .await
        .unwrap();
    assert_eq!(outcome.output_text(), "Take an umbrella.");

    // question, call, output, answer
    assert_eq!(session.conversation().len(), 4);
    let entries: Vec<&SessionEntry> = session.path().iter().map(|record| &record.entry).collect();
    // The settings once, before the first item of the run; a usage after each response.
    assert!(matches!(entries[0], SessionEntry::Item(_)));
    let SessionEntry::TurnSettings(settings) = entries[1] else {
        panic!("{entries:?}")
    };
    assert_eq!(settings.settings.reasoning, Some(ReasoningLevel::Low));
    assert_eq!(settings.model.as_ref().unwrap().model, "gpt-test");
    let settings_count = entries
        .iter()
        .filter(|entry| matches!(entry, SessionEntry::TurnSettings(_)))
        .count();
    assert_eq!(settings_count, 1);
    assert_eq!(session.usage(), Usage::new(20, 10));

    // Another handle on the same store sees the same conversation.
    let reopened = Session::open(&store, SessionId::new("s1")).await.unwrap();
    assert_eq!(reopened.conversation(), session.conversation());

    // Opening it claimed the session: the first handle can no longer write.
    assert!(session
        .append_input("in-2", Item::user("And tomorrow?"))
        .await
        .is_err());
}

#[tokio::test]
async fn an_earlier_writer_is_superseded() {
    let store = MemoryStore::new();
    let id = SessionId::new("s");
    let mut first = Session::open(&store, id.clone()).await.unwrap();
    first.append_item(Item::user("one")).await.unwrap();
    let mut second = Session::open(&store, id.clone()).await.unwrap();
    let error = first.append_item(Item::user("two")).await.unwrap_err();
    assert!(matches!(error, SessionError::Superseded { session } if session == id));
    second.append_item(Item::user("two")).await.unwrap();
    assert_eq!(texts(&second), ["one", "two"]);

    // A run on a superseded handle fails with the session error, and the session keeps
    // what the newer writer wrote.
    let mut runner = Runner::new(ScriptedModel::replies(vec![Reply::text("hi")]), tools());
    let error = runner
        .run_session(&mut first, &CancellationSignal::new(), |_| {})
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Session(SessionError::Superseded { .. })
    ));
    let reopened = Session::open(&store, id).await.unwrap();
    assert_eq!(texts(&reopened), ["one", "two"]);
}

#[tokio::test]
async fn an_input_key_is_stored_once() {
    let store = MemoryStore::new();
    let mut session = Session::open(&store, SessionId::new("s")).await.unwrap();
    let first = session
        .append_input("m1", Item::user("hello"))
        .await
        .unwrap();
    assert!(first.is_some());
    let again = session
        .append_input("m1", Item::user("hello"))
        .await
        .unwrap();
    assert_eq!(again, None);
    assert_eq!(texts(&session), ["hello"]);
}

#[tokio::test]
async fn rollback_and_fork() {
    let store = MemoryStore::new();
    let mut session = Session::open(&store, SessionId::new("main")).await.unwrap();
    session.append_item(Item::user("q1")).await.unwrap();
    session.append_item(Item::assistant("a1")).await.unwrap();
    session.append_item(Item::user("q2")).await.unwrap();
    session.append_item(Item::assistant("a2")).await.unwrap();
    let inputs = session.user_inputs();
    assert_eq!(inputs.len(), 2);

    // A fork from the second question holds q1, a1, q2; the main session is untouched.
    let mut fork = session
        .fork(SessionId::new("fork"), Some(&inputs[1]))
        .await
        .unwrap();
    assert_eq!(texts(&fork), ["q1", "a1", "q2"]);
    fork.append_item(Item::assistant("other a2")).await.unwrap();
    assert_eq!(texts(&session), ["q1", "a1", "q2", "a2"]);

    // Roll back to the first answer, and take another way.
    let a1 = session.path()[1].id.clone();
    session.rollback(Some(&a1)).await.unwrap();
    assert_eq!(texts(&session), ["q1", "a1"]);
    session.append_item(Item::user("q2'")).await.unwrap();
    assert_eq!(texts(&session), ["q1", "a1", "q2'"]);
    // Nothing was removed.
    let items = session
        .records()
        .iter()
        .filter(|record| matches!(record.entry, SessionEntry::Item(_)))
        .count();
    assert_eq!(items, 5);

    // A rolled-back session reopens at its head.
    let mut reopened = Session::open(&store, SessionId::new("main")).await.unwrap();
    assert_eq!(texts(&reopened), ["q1", "a1", "q2'"]);
    // A fork of a fork from a rolled-back path has no rollback records.
    let again = reopened.fork(SessionId::new("again"), None).await.unwrap();
    assert_eq!(texts(&again), ["q1", "a1", "q2'"]);
    assert!(again
        .records()
        .iter()
        .all(|record| !matches!(record.entry, SessionEntry::Rollback)));

    // Back to before the first record (on the handle that opened the session last).
    reopened.rollback(None).await.unwrap();
    assert!(reopened.conversation().is_empty());

    let unknown = reopened.rollback(Some(&EntryId::new("nope"))).await;
    assert!(matches!(unknown, Err(SessionError::UnknownEntry(_))));
}

#[tokio::test]
async fn the_conversation_starts_at_the_latest_compaction() {
    let store = MemoryStore::new();
    let mut session = Session::open(&store, SessionId::new("s")).await.unwrap();
    session.append_item(Item::user("q1")).await.unwrap();
    session.append_item(Item::assistant("a1")).await.unwrap();
    let q2 = session.append_item(Item::user("q2")).await.unwrap();
    session.append_item(Item::assistant("a2")).await.unwrap();
    session
        .append(SessionEntry::Compaction(
            Compaction::new("They asked twice.").with_first_kept(q2),
        ))
        .await
        .unwrap();
    session.append_item(Item::user("q3")).await.unwrap();

    let conversation = session.conversation();
    assert_eq!(texts(&session), ["They asked twice.", "q2", "a2", "q3"]);
    let Item::Message(summary) = &conversation.items()[0] else {
        panic!()
    };
    assert_eq!(summary.role, Role::User);
    assert!(summary.meta.injected);
    // The summary is not a user input to roll back to.
    assert_eq!(session.user_inputs().len(), 3);

    // A compaction that keeps nothing.
    session
        .append(SessionEntry::Compaction(Compaction::new("All of it.")))
        .await
        .unwrap();
    assert_eq!(texts(&session), ["All of it."]);
}

#[tokio::test]
async fn a_cancelled_run_leaves_its_partial_turn_in_the_session() {
    let mut slow = Reply::text("A long answer");
    // The response stalls after its text delta: the run is cancelled then.
    slow.events_mut().unwrap()[1].0 = Duration::from_secs(60);
    let store = MemoryStore::new();
    let mut session = Session::open(&store, SessionId::new("s")).await.unwrap();
    session.append_item(Item::user("Tell me")).await.unwrap();
    let cancel = CancellationSignal::new();
    let mut runner = Runner::new(ScriptedModel::replies(vec![slow]), tools());
    let trigger = cancel.clone();
    let outcome = runner
        .run_session(&mut session, &cancel, move |event| {
            if let genai_agents::RunEvent::Model(ModelEvent::TextDelta(_)) = event {
                trigger.cancel();
            }
        })
        .await
        .unwrap();
    assert_eq!(outcome.stop, RunStop::Cancelled);
    let reopened = Session::open(&store, SessionId::new("s")).await.unwrap();
    let items = reopened.conversation();
    assert_eq!(items.len(), 2);
    let Item::Message(partial) = &items.items()[1] else {
        panic!()
    };
    assert!(partial.meta.interrupted);
}

#[tokio::test]
async fn sessions_round_trip_through_serde() {
    let store = MemoryStore::new();
    let mut session = Session::open(&store, SessionId::new("s")).await.unwrap();
    session.append_item(Item::user("q")).await.unwrap();
    session
        .append(SessionEntry::Custom {
            kind: "note".into(),
            data: json!({"pinned": true}),
        })
        .await
        .unwrap();
    session
        .append(SessionEntry::Label("checkpoint".into()))
        .await
        .unwrap();
    for record in session.records() {
        let text = serde_json::to_string(record).unwrap();
        let back: genai_agents::session::Record = serde_json::from_str(&text).unwrap();
        assert_eq!(&back, record);
    }
    // Entries of the application's own are not sent to the model.
    assert_eq!(texts(&session), ["q"]);
}

#[cfg(feature = "jsonl")]
mod jsonl {
    use super::*;
    use genai_agents::session::JsonlStore;

    struct Directory(std::path::PathBuf);

    impl Directory {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "genai-agents-{name}-{}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            Self(path)
        }
    }

    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[tokio::test]
    async fn a_jsonl_store_keeps_the_log_across_stores() {
        let directory = Directory::new("jsonl");
        let id = SessionId::new("user/42: chat");
        {
            let store = JsonlStore::new(&directory.0).unwrap();
            let mut session = Session::open(&store, id.clone()).await.unwrap();
            session
                .append_input("m1", Item::user("Umbrella in Ghent?"))
                .await
                .unwrap();
            let mut runner = Runner::new(weather_model(), tools());
            runner
                .run_session(&mut session, &CancellationSignal::new(), |_| {})
                .await
                .unwrap();
            // The file name is safe whatever the id.
            let path = store.path(&id);
            assert_eq!(
                path.file_name().unwrap().to_str().unwrap(),
                "user%2F42%3A%20chat.jsonl"
            );
            assert!(path.exists());
        }
        // A new store on the same directory: the same session, and its keys still hold.
        let store = JsonlStore::new(&directory.0).unwrap();
        let mut session = Session::open(&store, id.clone()).await.unwrap();
        assert_eq!(session.conversation().len(), 4);
        assert_eq!(
            session
                .append_input("m1", Item::user("Umbrella in Ghent?"))
                .await
                .unwrap(),
            None
        );
        // A second handle in this process supersedes the first.
        let mut newer = Session::open(&store, id.clone()).await.unwrap();
        assert!(session.append_item(Item::user("late")).await.is_err());
        newer
            .append_item(Item::user("And tomorrow?"))
            .await
            .unwrap();
        let text = std::fs::read_to_string(store.path(&id)).unwrap();
        assert!(text
            .lines()
            .all(|line| serde_json::from_str::<Value>(line).is_ok()));
    }
}
