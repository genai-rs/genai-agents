//! The model, from the environment, and tracing to Langfuse.

use std::env;

use genai_agents::fallback::Fallback;
use genai_agents::Model as _;
use genai_agents_openai::{Compat, OpenAiChat, OpenAiResponses};
use openai_ergonomic::{ChatCompletionBuilder, Client, Config, Provider, ResponsesBuilder};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::runtime::Tokio;
use opentelemetry_sdk::trace::span_processor_with_async_runtime::BatchSpanProcessor;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// The model: OpenAI's Responses API when `OPENAI_API_KEY` is set (`DEMO_MODEL`, else
/// `OPENAI_DEFAULT_MODEL`, else openai-ergonomic's default model), with Ollama behind it as a
/// fallback; Ollama alone otherwise
/// (`DEMO_OLLAMA_MODEL`, default `qwen2.5:3b`, at `OLLAMA_BASE_URL`). Returns the model and a
/// line that says which it is.
pub fn from_env() -> Result<(Fallback, String), BoxError> {
    let ollama_model = env::var("DEMO_OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:3b".into());
    let ollama_url =
        env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let ollama = OpenAiChat::new(
        Client::builder(
            Config::builder()
                .provider(Provider::ollama().base_url(ollama_url.clone()))
                .build(),
        )?
        .build(),
        ChatCompletionBuilder::new(ollama_model.clone()),
    )
    .with_compat(Compat::ollama());
    if env::var("OPENAI_API_KEY").is_ok_and(|key| !key.is_empty()) {
        let client = Client::builder(Config::from_env()?)?.build();
        let template = match env::var("DEMO_MODEL") {
            Ok(model) if !model.is_empty() => ResponsesBuilder::new(model),
            _ => client.responses(),
        };
        let openai = OpenAiResponses::new(client, template);
        let model = openai
            .info()
            .origin
            .map(|origin| origin.model)
            .unwrap_or_default();
        let label = format!("OpenAI {model}, falling back to Ollama {ollama_model}");
        Ok((Fallback::new(openai).then(ollama), label))
    } else {
        let label = format!("Ollama {ollama_model} at {ollama_url}");
        Ok((Fallback::new(ollama), label))
    }
}

/// Tracing to Langfuse, when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
/// (`LANGFUSE_HOST`, default Langfuse Cloud).
pub struct Tracing {
    provider: Option<SdkTracerProvider>,
}

impl Tracing {
    pub fn init() -> Result<Self, BoxError> {
        if env::var("LANGFUSE_PUBLIC_KEY").is_err() || env::var("LANGFUSE_SECRET_KEY").is_err() {
            return Ok(Self { provider: None });
        }
        let exporter = openai_ergonomic::langfuse_exporter_from_env()?;
        let provider = SdkTracerProvider::builder()
            .with_resource(
                Resource::builder()
                    .with_service_name("genai-agents-demo")
                    .build(),
            )
            .with_span_processor(BatchSpanProcessor::builder(exporter, Tokio).build())
            .build();
        tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("genai-agents-demo")))
            .try_init()?;
        Ok(Self {
            provider: Some(provider),
        })
    }

    pub fn enabled(&self) -> bool {
        self.provider.is_some()
    }

    /// The trace id of a span, as Langfuse shows it.
    pub fn trace_id(span: &tracing::Span) -> String {
        use opentelemetry::trace::TraceContextExt as _;
        use tracing_opentelemetry::OpenTelemetrySpanExt as _;
        span.context().span().span_context().trace_id().to_string()
    }

    /// Send what is left, and stop.
    pub fn shutdown(self) {
        if let Some(provider) = self.provider {
            if let Err(error) = provider.shutdown() {
                eprintln!("tracing: {error}");
            }
        }
    }
}
