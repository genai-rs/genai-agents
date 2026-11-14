//! The OpenAI provider for [genai-agents](https://docs.rs/genai-agents), on
//! [openai-ergonomic](https://docs.rs/openai-ergonomic): OpenAI's APIs as
//! [`Model`](genai_agents::Model)s for a [`Runner`](genai_agents::Runner).
//!
//! - [`OpenAiResponses`]: the Responses API, streamed and stateless by default, with hosted tools
//!   and the encrypted reasoning of reasoning models.
//! - [`OpenAiChat`]: Chat Completions, which also serves the providers that speak it (Azure
//!   OpenAI, OpenRouter, Groq, Ollama, gateways).
//!
//! Both take an [`openai_ergonomic::Client`], whose configuration (provider, credentials,
//! retries, timeouts, layers) applies to their requests, and a request template.
//!
//! ```rust,no_run
//! use genai_agents::tools::{FunctionTool, ToolRegistry};
//! use genai_agents::{Conversation, Result, Runner};
//! use genai_agents_openai::OpenAiResponses;
//! use openai_ergonomic::{Client, ResponsesBuilder};
//! use serde_json::{json, Value};
//!
//! struct Weather;
//!
//! impl FunctionTool for Weather {
//!     type Input = Value;
//!     type Output = Value;
//!     fn name(&self) -> &str {
//!         "get_weather"
//!     }
//!     fn description(&self) -> &str {
//!         "Current weather of a city"
//!     }
//!     fn parameters_schema(&self) -> Value {
//!         json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
//!     }
//!     async fn execute(&self, input: Value) -> Result<Value> {
//!         Ok(json!({"city": input["city"], "sky": "light rain"}))
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! let client = Client::from_env()?.build();
//! let model = OpenAiResponses::new(
//!     client,
//!     ResponsesBuilder::new("gpt-5-mini").instructions("You are a household assistant."),
//! );
//! let mut tools = ToolRegistry::new();
//! tools.register(Weather)?;
//! let mut runner = Runner::new(model, tools);
//! let mut conversation = Conversation::user("Umbrella in Ghent today?");
//! let outcome = runner.run(&mut conversation).await?;
//! println!("{}", outcome.output_text());
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

mod chat;
mod compat;
mod convert;
mod http;
mod responses;

pub use chat::{OpenAiChat, CHAT_API};
pub use compat::{Compat, MaxTokensField};
pub use convert::PROVIDER;
pub use responses::{OpenAiResponses, RESPONSES_API};

/// The examples of the runner guide (`docs/runner.md`), compiled and run as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../../../docs/runner.md")]
pub struct RunnerGuide;
