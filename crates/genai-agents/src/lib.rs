//! Agents for Rust: a model, tools, and a [`Runner`] that calls the model in a loop, runs the
//! tools it asks for and sends their outputs back, until the model answers. Independent of the
//! model provider: a provider crate (such as
//! [genai-agents-openai](https://docs.rs/genai-agents-openai)) implements [`Model`].
//!
//! - [`Conversation`]: the history, as provider-independent [`Item`]s that record where they came
//!   from ([`ItemMeta`]); `Serialize` and `Deserialize`, and projected for the model before every
//!   request ([`Conversation::for_request`]).
//! - [`tools`]: typed tools ([`FunctionTool`](tools::FunctionTool),
//!   [`RichTool`](tools::RichTool), [`CustomTool`](tools::CustomTool), dynamic tools), their
//!   [`ToolRegistry`], and when the model sees them
//!   ([`Exposure`](tools::Exposure)).
//! - [`Runner`]: the loop, with concurrent tool calls started while the response streams,
//!   cancellation, retries, iteration limits and tool search.
//!
//! ```rust
//! use genai_agents::tools::{FunctionTool, ToolRegistry};
//! use genai_agents::{Conversation, Item, Model, ModelInfo, ModelError, ModelEvent};
//! use genai_agents::{ModelRequest, ModelResponse, ModelStream, Result, Runner, ToolCall};
//! use serde::Deserialize;
//! use serde_json::{json, Value};
//!
//! #[derive(Deserialize)]
//! struct City {
//!     city: String,
//! }
//!
//! struct Weather;
//!
//! impl FunctionTool for Weather {
//!     type Input = City;
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
//!     async fn execute(&self, input: City) -> Result<Value> {
//!         Ok(json!({"city": input.city, "sky": "light rain"}))
//!     }
//! }
//!
//! /// A model that calls the tool once, then answers: a provider crate gives a real one.
//! struct Scripted;
//!
//! impl Model for Scripted {
//!     fn info(&self) -> ModelInfo {
//!         ModelInfo::default()
//!     }
//!     async fn stream<'a>(
//!         &'a self,
//!         request: ModelRequest<'_>,
//!     ) -> std::result::Result<ModelStream<'a>, ModelError> {
//!         let answered = request.items.iter().any(|item| matches!(item, Item::ToolOutput(_)));
//!         let item = if answered {
//!             Item::assistant("Take an umbrella.")
//!         } else {
//!             Item::ToolCall(ToolCall::function("call_1", "get_weather", r#"{"city":"Ghent"}"#))
//!         };
//!         Ok(ModelStream::from_events(vec![
//!             ModelEvent::ItemDone(item),
//!             ModelEvent::Completed(ModelResponse::new()),
//!         ]))
//!     }
//! }
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<()> {
//! let mut tools = ToolRegistry::new();
//! tools.register(Weather)?;
//! let mut runner = Runner::new(Scripted, tools);
//! let mut conversation = Conversation::user("Umbrella in Ghent today?");
//! let outcome = runner.run(&mut conversation).await?;
//! assert_eq!(outcome.output_text(), "Take an umbrella.");
//! assert_eq!(conversation.len(), 4); // question, call, its output, answer
//! # Ok(())
//! # }
//! ```
//!
//! The [runner](https://github.com/genai-rs/genai-agents/blob/main/docs/runner.md) and
//! [tools](https://github.com/genai-rs/genai-agents/blob/main/docs/tools.md) guides go further.

#![cfg_attr(docsrs, feature(doc_cfg))]

mod conversation;
mod error;
mod item;
mod model;
mod platform;
pub mod runner;
#[cfg(feature = "testing")]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
pub mod testing;
pub mod tools;

pub use conversation::{Conversation, NormalizeReport, ProjectionReport, IMAGE_OMITTED};
pub use error::{BoxError, Error, ErrorKind, ModelError, Result, RetryClass};
pub use item::{
    Content, ImageDetail, Item, ItemMeta, Message, Origin, ProviderItem, Reasoning, Role, ToolCall,
    ToolKind, ToolOutput, ToolOutputBody, ABORTED_OUTPUT,
};
pub use model::{
    CallIds, Model, ModelEvent, ModelInfo, ModelRequest, ModelResponse, ModelSettings, ModelStream,
    RateLimitWindow, RateLimits, ReasoningLevel, StopReason, ToolDefinition, Usage,
};
pub use platform::{BoxFuture, BoxStream, MaybeSend, MaybeSync};
pub use runner::{
    CancellationSignal, PromptCacheKey, RetryConfig, RunConfig, RunEvent, RunOutcome, RunStop,
    Runner, ToolSearchConfig, TurnRecord, CUT_OFF_OUTPUT, TOOL_SEARCH,
};
pub use tools::ToolRegistry;

/// The examples of the tools guide (`docs/tools.md`), compiled and run as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../../../docs/tools.md")]
pub struct ToolsGuide;
