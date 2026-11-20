//! A run on a session, traced to Langfuse: the `invoke_agent`, `chat` and `execute_tool` spans
//! of genai-agents, exported with OpenTelemetry to Langfuse's OTLP endpoint.
//!
//! [opentelemetry-langfuse](https://docs.rs/opentelemetry-langfuse) builds this exporter in one
//! line (`ExporterBuilder::from_env()?.build()?`); it needs a newer Rust than this workspace's
//! minimum, so the example builds it with opentelemetry-otlp.
//!
//! ```bash
//! export OPENAI_API_KEY=...
//! export LANGFUSE_PUBLIC_KEY=pk-lf-...
//! export LANGFUSE_SECRET_KEY=sk-lf-...
//! export LANGFUSE_HOST=https://cloud.langfuse.com
//! cargo run -p genai-agents-openai --example langfuse
//! ```
//!
//! The trace shows the run, its two model requests (with their tokens, the cached ones
//! included) and the tool call between them; the messages too, since the example turns
//! `trace_content` on.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use base64::Engine as _;
use genai_agents::session::{MemoryStore, Session, SessionId};
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{CancellationSignal, Item, Result, RunConfig, Runner};
use genai_agents_openai::OpenAiResponses;
use openai_ergonomic::{Client, ResponsesBuilder};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{WithExportConfig as _, WithHttpConfig as _};
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use serde_json::{json, Value};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

struct Weather;

impl FunctionTool for Weather {
    type Input = Value;
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
    async fn execute(&self, input: Value) -> Result<Value> {
        Ok(json!({"city": input["city"], "sky": "light rain", "celsius": 11}))
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Spans go to Langfuse's OTLP endpoint, in batches, with its keys as basic authentication.
    let host =
        std::env::var("LANGFUSE_HOST").unwrap_or_else(|_| "https://cloud.langfuse.com".into());
    let keys = format!(
        "{}:{}",
        std::env::var("LANGFUSE_PUBLIC_KEY")?,
        std::env::var("LANGFUSE_SECRET_KEY")?
    );
    let authorization = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(keys)
    );
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(format!(
            "{}/api/public/otel/v1/traces",
            host.trim_end_matches('/')
        ))
        .with_headers([("authorization".to_owned(), authorization)].into())
        .build()?;
    let provider = SdkTracerProvider::builder()
        .with_resource(
            Resource::builder()
                .with_service_name("genai-agents-example")
                .build(),
        )
        .with_batch_exporter(exporter)
        .build();
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("genai-agents")))
        .init();

    let model = OpenAiResponses::new(
        Client::from_env()?.build(),
        ResponsesBuilder::new("gpt-5.4-mini").instructions("You are a household assistant."),
    );
    let mut tools = ToolRegistry::new();
    tools.register(Weather)?;
    let mut runner = Runner::new(model, tools).with_config(
        RunConfig::new()
            .with_name("household")
            .with_trace_content(true),
    );

    let store = MemoryStore::new();
    let mut session = Session::open(&store, SessionId::random()).await?;
    session
        .append_input("msg-1", Item::user("Do I need an umbrella in Ghent today?"))
        .await?;
    let outcome = runner
        .run_session(&mut session, &CancellationSignal::new(), |_| {})
        .await?;
    println!("{}", outcome.output_text());

    // Send the last batch before the process ends.
    provider.shutdown()?;
    Ok(())
}
