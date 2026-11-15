//! The spans of a run: `invoke_agent`, `chat` and `execute_tool`, their GenAI attributes, the
//! content only when asked for, and why the prompt cache broke.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(not(target_family = "wasm"))]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{
    CacheBreak, Conversation, Item, ModelInfo, ModelSettings, Origin, ReasoningLevel, Result,
    RunConfig, Runner, ToolCall,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// A span as the test layer saw it.
#[derive(Debug, Clone, Default)]
struct Seen {
    name: String,
    parent: Option<String>,
    fields: BTreeMap<String, String>,
}

impl Seen {
    fn field(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<(u64, Seen)>>>);

impl Capture {
    fn spans(&self) -> Vec<Seen> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .map(|(_, seen)| seen.clone())
            .collect()
    }

    fn named(&self, name: &str) -> Vec<Seen> {
        self.spans()
            .into_iter()
            .filter(|seen| seen.name == name)
            .collect()
    }
}

struct Fields<'a>(&'a mut BTreeMap<String, String>);

impl Visit for Fields<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(
            field.name().to_owned(),
            format!("{value:?}").trim_matches('"').to_owned(),
        );
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_owned(), value.to_owned());
    }
}

impl<S> Layer<S> for Capture
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut seen = Seen {
            name: attrs.metadata().name().to_owned(),
            parent: ctx
                .span(id)
                .and_then(|span| span.parent())
                .map(|parent| parent.name().to_owned()),
            fields: BTreeMap::new(),
        };
        attrs.record(&mut Fields(&mut seen.fields));
        self.0.lock().unwrap().push((id.into_u64(), seen));
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut spans = self.0.lock().unwrap();
        if let Some((_, seen)) = spans.iter_mut().find(|(seen, _)| *seen == id.into_u64()) {
            values.record(&mut Fields(&mut seen.fields));
        }
    }
}

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

fn model() -> ScriptedModel {
    ScriptedModel::new(|index, _| match index % 2 {
        0 => Reply::items(vec![Item::ToolCall(ToolCall::function(
            format!("call_{index}"),
            "get_weather",
            r#"{"city":"Ghent"}"#,
        ))])
        .with_id(format!("resp_{index}")),
        _ => Reply::text("Take an umbrella.").with_id(format!("resp_{index}")),
    })
    .with_info(ModelInfo::new().with_origin(Origin::new("responses", "openai", "gpt-test")))
}

fn runner(config: RunConfig) -> Runner<ScriptedModel> {
    let mut tools = ToolRegistry::new();
    tools.register(Weather).unwrap();
    Runner::new(model(), tools).with_config(config)
}

async fn traced<T>(capture: &Capture, run: impl std::future::Future<Output = T>) -> T {
    let subscriber = tracing_subscriber::registry().with(capture.clone());
    let _guard = tracing::subscriber::set_default(subscriber);
    run.await
}

#[tokio::test(flavor = "current_thread")]
async fn a_run_has_agent_chat_and_tool_spans() {
    let capture = Capture::default();
    let mut runner = runner(
        RunConfig::new()
            .with_name("household")
            .with_model_settings(ModelSettings::new().with_reasoning(ReasoningLevel::Low)),
    );
    traced(&capture, async {
        runner
            .run(&mut Conversation::user("Umbrella?"))
            .await
            .unwrap();
    })
    .await;

    let agent = capture.named("invoke_agent");
    assert_eq!(agent.len(), 1);
    let agent = &agent[0];
    assert_eq!(agent.field("otel.name"), Some("invoke_agent household"));
    assert_eq!(agent.field("gen_ai.agent.name"), Some("household"));
    assert_eq!(agent.field("gen_ai.provider.name"), Some("openai"));
    assert_eq!(agent.field("gen_ai.usage.input_tokens"), Some("20"));
    assert_eq!(agent.field("gen_ai.usage.output_tokens"), Some("10"));

    let chats = capture.named("chat");
    assert_eq!(chats.len(), 2);
    for (index, chat) in chats.iter().enumerate() {
        assert_eq!(chat.parent.as_deref(), Some("invoke_agent"));
        assert_eq!(chat.field("otel.name"), Some("chat gpt-test"));
        assert_eq!(chat.field("otel.kind"), Some("client"));
        assert_eq!(chat.field("gen_ai.request.model"), Some("gpt-test"));
        assert_eq!(chat.field("gen_ai.request.reasoning"), Some("low"));
        assert_eq!(
            chat.field("gen_ai.response.id"),
            Some(format!("resp_{index}").as_str())
        );
        assert_eq!(chat.field("gen_ai.usage.input_tokens"), Some("10"));
        assert!(chat.field("genai_agents.time_to_first_token_ms").is_some());
        // The second request extends the first: the cache holds.
        assert_eq!(chat.field("genai_agents.cache_break"), None);
        // No content unless asked for.
        assert_eq!(chat.field("gen_ai.input.messages"), None);
    }

    let tools = capture.named("execute_tool");
    assert_eq!(tools.len(), 1);
    let tool = &tools[0];
    assert_eq!(tool.parent.as_deref(), Some("invoke_agent"));
    assert_eq!(tool.field("otel.name"), Some("execute_tool get_weather"));
    assert_eq!(tool.field("gen_ai.tool.name"), Some("get_weather"));
    assert_eq!(tool.field("gen_ai.tool.call.id"), Some("call_0"));
    assert_eq!(tool.field("gen_ai.tool.call.arguments"), None);
    assert_eq!(tool.field("gen_ai.tool.call.result"), None);
}

#[tokio::test(flavor = "current_thread")]
async fn content_is_recorded_when_asked_for() {
    let capture = Capture::default();
    let mut runner = runner(RunConfig::new().with_trace_content(true));
    traced(&capture, async {
        runner
            .run(&mut Conversation::user("Umbrella?"))
            .await
            .unwrap();
    })
    .await;
    let chats = capture.named("chat");
    let input: Value =
        serde_json::from_str(chats[1].field("gen_ai.input.messages").unwrap()).unwrap();
    assert_eq!(
        input[0],
        json!({"role": "user", "parts": [{"type": "text", "content": "Umbrella?"}]})
    );
    assert_eq!(input[1]["parts"][0]["type"], "tool_call");
    assert_eq!(input[1]["parts"][0]["arguments"], json!({"city": "Ghent"}));
    assert_eq!(input[2]["role"], "tool");
    assert_eq!(input[2]["parts"][0]["type"], "tool_call_response");
    let output: Value =
        serde_json::from_str(chats[1].field("gen_ai.output.messages").unwrap()).unwrap();
    assert_eq!(
        output,
        json!([{"role": "assistant", "parts": [{"type": "text", "content": "Take an umbrella."}]}])
    );
    let tool = &capture.named("execute_tool")[0];
    assert_eq!(
        tool.field("gen_ai.tool.call.arguments"),
        Some(r#"{"city":"Ghent"}"#)
    );
    assert!(tool
        .field("gen_ai.tool.call.result")
        .unwrap()
        .contains("light rain"));
}

#[tokio::test(flavor = "current_thread")]
async fn a_changed_setting_breaks_the_cache() {
    let mut runner = runner(RunConfig::new());
    let mut conversation = Conversation::user("Umbrella?");
    let first = runner.run(&mut conversation).await.unwrap();
    assert_eq!(first.turns[0].cache_break, None);
    assert_eq!(first.turns[1].cache_break, None);

    conversation.push_user("And tomorrow?");
    runner.config_mut().model_settings = ModelSettings::new().with_reasoning(ReasoningLevel::High);
    let second = runner.run(&mut conversation).await.unwrap();
    assert_eq!(second.turns[0].cache_break, Some(CacheBreak::Settings));
    assert_eq!(second.turns[1].cache_break, None);

    // A history that no longer starts with what was sent.
    let mut edited = Conversation::user("Sunscreen?");
    let third = runner.run(&mut edited).await.unwrap();
    assert_eq!(third.turns[0].cache_break, Some(CacheBreak::History));
}
