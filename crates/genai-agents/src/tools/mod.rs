//! Typed, asynchronous tools: define them, register them, answer the model's calls.
//!
//! Implement [`FunctionTool`] and register instances with [`ToolRegistry::register`]. Answer
//! each approved call with [`ToolRegistry::answer_call`], which turns failures the model can act
//! on (bad arguments, an unknown tool, a handler error) into an answer for the model.
//! Registration grants no execution authority: the application (or the
//! [`Runner`](crate::Runner)) decides which calls to run.
//!
//! Beyond JSON function tools: [`RichTool`]s return images and files, [`CustomTool`]s take
//! freeform text (optionally constrained by a grammar), [`ToolRegistry::namespace`] groups tools,
//! and [`ToolRegistry::register_dynamic`] registers tools only known at runtime, with their names
//! and schemas cleaned up ([`sanitize_tool_name`], [`schema`]).
//!
//! ```
//! use genai_agents::tools::{FunctionTool, ToolRegistry};
//! use genai_agents::Result;
//! use serde::Deserialize;
//! use serde_json::{json, Value};
//!
//! #[derive(Deserialize)]
//! #[serde(deny_unknown_fields)]
//! struct Greeting { name: String }
//! struct Greet;
//! impl FunctionTool for Greet {
//!     type Input = Greeting;
//!     type Output = String;
//!     fn name(&self) -> &str { "greet" }
//!     fn description(&self) -> &str { "Return a greeting for a person." }
//!     fn parameters_schema(&self) -> Value {
//!         json!({"type": "object", "properties": {
//!             "name": {"type": "string", "description": "Person to greet"}
//!         }, "required": ["name"], "additionalProperties": false})
//!     }
//!     async fn execute(&self, input: Greeting) -> Result<String> {
//!         Ok(format!("Hello, {}!", input.name))
//!     }
//! }
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! let mut tools = ToolRegistry::new();
//! tools.register(Greet)?;
//! let result = tools.execute("greet", r#"{"name":"Tim"}"#).await?;
//! assert_eq!(result, "Hello, Tim!");
//! # Ok(())
//! # }
//! ```

use std::collections::{btree_map::Entry, BTreeMap};
use std::future::Future;
use std::sync::Arc;

use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};

use crate::item::{ToolCall, ToolKind, ToolOutput, ToolOutputBody};
use crate::model::ToolDefinition;
use crate::platform::{BoxFuture, MaybeSend, MaybeSync};
use crate::{Error, Result};

mod names;
mod output;
pub mod schema;
mod search;

pub use crate::item::{Content, ImageDetail, ABORTED_OUTPUT};
pub use names::{sanitize_tool_name, ToolNamer, TOOL_NAME_MAX};
pub use output::{truncate_middle, OutputBudget, TruncationConfig};
/// A function tool. Store application state (for example an `Arc` or HTTP client)
/// in the implementing struct and access it through `&self` during execution.
///
/// Implement [`FunctionTool::execute`] with a plain `async fn`; no attribute macro
/// is needed. Natively the returned future must be `Send` ([`MaybeSend`]); on WebAssembly neither
/// the tool nor its future needs to be `Send` or `Sync`.
pub trait FunctionTool: MaybeSend + MaybeSync {
    /// Arguments decoded with Serde. Use `Value` for a dynamic JSON tool.
    type Input: DeserializeOwned + MaybeSend;
    /// Result encoded as JSON, including quotes for string outputs.
    type Output: Serialize + MaybeSend;
    /// Stable name: 1–64 ASCII letters, digits, underscores or hyphens.
    fn name(&self) -> &str;
    /// Describe what the tool does and when the model should use it.
    fn description(&self) -> &str;
    /// An object JSON Schema matching `Input`, including property descriptions. With the
    /// `schemars` feature, [`schema::for_type`] derives it from `Input`.
    ///
    /// The registry checks only the root object type, not full JSON Schema
    /// validity or conformance. Serde handles argument decoding; constraints such
    /// as numeric bounds must also be enforced by the implementation. Strict mode
    /// is not enabled. Optional fields may be omitted. For exact object fields,
    /// pair `additionalProperties: false` with `#[serde(deny_unknown_fields)]`.
    fn parameters_schema(&self) -> Value;
    /// Execute after the caller has made any required authorization decisions.
    /// Use `Error::application` to preserve custom domain errors, and `Error::fatal` for a
    /// failure that must not be reported to the model ([`ToolRegistry::answer_call`]).
    fn execute(
        &self,
        input: Self::Input,
    ) -> impl std::future::Future<Output = Result<Self::Output>> + MaybeSend;
    /// Whether this tool may run at the same time as other tools that allow it. Read once, at
    /// registration ([`ToolRegistry::supports_parallel`]); the application or a tool loop
    /// decides what to run concurrently. Default `false`.
    fn supports_parallel(&self) -> bool {
        false
    }
    /// How much of this tool's output the model should see, instead of the default of the
    /// [`TruncationConfig`]. Read once, at registration ([`ToolRegistry::output_budget`]).
    fn output_budget(&self) -> Option<OutputBudget> {
        None
    }
    /// Whether the model sees this tool from the start, after a tool search, or never. Read
    /// once, at registration ([`ToolRegistry::exposure`]). Default [`Exposure::Direct`].
    fn exposure(&self) -> Exposure {
        Exposure::Direct
    }
    /// Whether this tool's output holds content from outside the application (a web page, an
    /// email, a file of a user), which may carry instructions the model should not follow. Its
    /// answers are marked [`untrusted`](ToolOutput::untrusted); the runner records it. Read once,
    /// at registration. Default `false`.
    fn untrusted_output(&self) -> bool {
        false
    }
    /// What to report about `output` to the application only: it goes into the answer's
    /// [`details`](ToolOutput::details), which the model never sees (the full data behind a
    /// short answer, a diff to show, timings). Default `None`.
    fn details(&self, output: &Self::Output) -> Option<Value> {
        let _ = output;
        None
    }
}

/// A function tool whose output is more than JSON text.
///
/// It returns a [`ToolOutputBody`], with content items such as images or files. Register it with
/// [`ToolRegistry::register_rich`]; everything else is as for [`FunctionTool`].
///
/// A model that takes only text in tool outputs ([`ModelInfo::rich_tool_outputs`](crate::ModelInfo::rich_tool_outputs))
/// gets an output with images or files as a failure.
pub trait RichTool: MaybeSend + MaybeSync {
    /// Arguments decoded with Serde.
    type Input: DeserializeOwned + MaybeSend;
    /// Stable name: 1–64 ASCII letters, digits, underscores or hyphens.
    fn name(&self) -> &str;
    /// Describe what the tool does and when the model should use it.
    fn description(&self) -> &str;
    /// An object JSON Schema matching `Input` (see [`FunctionTool::parameters_schema`]).
    fn parameters_schema(&self) -> Value;
    /// Execute after the caller has made any required authorization decisions.
    fn execute(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<ToolOutputBody>> + MaybeSend;
    /// See [`FunctionTool::supports_parallel`].
    fn supports_parallel(&self) -> bool {
        false
    }
    /// See [`FunctionTool::output_budget`].
    fn output_budget(&self) -> Option<OutputBudget> {
        None
    }
    /// See [`FunctionTool::exposure`].
    fn exposure(&self) -> Exposure {
        Exposure::Direct
    }
    /// See [`FunctionTool::untrusted_output`].
    fn untrusted_output(&self) -> bool {
        false
    }
}

/// The syntax of a [`CustomFormat::Grammar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GrammarSyntax {
    /// A [Lark](https://lark-parser.readthedocs.io/) grammar.
    Lark,
    /// A regular expression.
    Regex,
}

impl GrammarSyntax {
    /// The name of the syntax: `lark` or `regex`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lark => "lark",
            Self::Regex => "regex",
        }
    }
}

/// The input format of a [`CustomTool`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CustomFormat {
    /// Any text.
    Text,
    /// Text that matches a grammar.
    Grammar {
        /// The grammar's syntax.
        syntax: GrammarSyntax,
        /// The grammar.
        definition: String,
    },
}

impl CustomFormat {
    /// Text that matches a Lark grammar.
    #[must_use]
    pub fn lark(definition: impl Into<String>) -> Self {
        Self::Grammar {
            syntax: GrammarSyntax::Lark,
            definition: definition.into(),
        }
    }

    /// Text that matches a regular expression.
    #[must_use]
    pub fn regex(definition: impl Into<String>) -> Self {
        Self::Grammar {
            syntax: GrammarSyntax::Regex,
            definition: definition.into(),
        }
    }
}

/// A custom (freeform) tool: the model sends text instead of JSON arguments.
///
/// The text can be a patch, a query or a program, optionally constrained by a grammar
/// ([`CustomFormat`]). Register it with [`ToolRegistry::register_custom`]; it is advertised to
/// models that take custom tools ([`ModelInfo::custom_tools`](crate::ModelInfo::custom_tools)).
pub trait CustomTool: MaybeSend + MaybeSync {
    /// Stable name: 1–64 ASCII letters, digits, underscores or hyphens.
    fn name(&self) -> &str;
    /// Describe what the tool does, when the model should use it and what to send.
    fn description(&self) -> &str;
    /// The input format. Default: any text.
    fn format(&self) -> CustomFormat {
        CustomFormat::Text
    }
    /// Execute with the model's input text.
    fn execute(&self, input: String) -> impl Future<Output = Result<ToolOutputBody>> + MaybeSend;
    /// See [`FunctionTool::supports_parallel`].
    fn supports_parallel(&self) -> bool {
        false
    }
    /// See [`FunctionTool::output_budget`].
    fn output_budget(&self) -> Option<OutputBudget> {
        None
    }
    /// See [`FunctionTool::exposure`].
    fn exposure(&self) -> Exposure {
        Exposure::Direct
    }
    /// See [`FunctionTool::untrusted_output`].
    fn untrusted_output(&self) -> bool {
        false
    }
}

/// Whether and when the model sees a registered tool.
///
/// The registry's [definitions](ToolRegistry::definitions) list only [`Direct`](Self::Direct)
/// tools. A [`Runner`](crate::Runner) also offers the deferred ones through a tool search, and
/// answers calls of every registered tool, hidden ones included.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Exposure {
    /// Advertised in every request.
    #[default]
    Direct,
    /// Advertised once a tool search found it (or with `defer_loading` to a model that searches
    /// tools itself): for large tool sets, whose definitions would fill the context.
    Deferred,
    /// Never advertised, but a call of it is answered (for example a tool the application
    /// advertises itself, or one an earlier conversation used).
    Hidden,
}

#[cfg(not(target_family = "wasm"))]
type DynamicHandler =
    Box<dyn Fn(Value) -> BoxFuture<'static, Result<ToolOutputBody>> + Send + Sync>;
#[cfg(target_family = "wasm")]
type DynamicHandler = Box<dyn Fn(Value) -> BoxFuture<'static, Result<ToolOutputBody>>>;

/// A function tool only known at runtime, such as one of an MCP server or a plugin.
///
/// It has a name, a description, a JSON Schema and a handler that takes the JSON arguments.
/// Register it with [`ToolRegistry::register_dynamic`], which cleans up its name and schema.
///
/// Natively the handler must be `Send` and `Sync` and its future `Send`; on WebAssembly neither
/// needs to be.
pub struct DynamicTool {
    raw_name: String,
    prefix: Option<String>,
    description: String,
    schema: Value,
    handler: DynamicHandler,
    supports_parallel: bool,
    output_budget: Option<OutputBudget>,
    exposure: Exposure,
    untrusted: bool,
}

impl std::fmt::Debug for DynamicTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicTool")
            .field("raw_name", &self.raw_name)
            .field("prefix", &self.prefix)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl DynamicTool {
    /// A tool named `name` (any text) whose arguments match `schema`, run by `handler`.
    #[cfg(not(target_family = "wasm"))]
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutputBody>> + Send + 'static,
    {
        Self::from_handler(
            name.into(),
            description.into(),
            schema,
            Box::new(move |arguments| Box::pin(handler(arguments))),
        )
    }

    /// A tool named `name` (any text) whose arguments match `schema`, run by `handler`.
    #[cfg(target_family = "wasm")]
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: Value,
        handler: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + 'static,
        Fut: Future<Output = Result<ToolOutputBody>> + 'static,
    {
        Self::from_handler(
            name.into(),
            description.into(),
            schema,
            Box::new(move |arguments| Box::pin(handler(arguments))),
        )
    }

    fn from_handler(
        raw_name: String,
        description: String,
        schema: Value,
        handler: DynamicHandler,
    ) -> Self {
        Self {
            raw_name,
            prefix: None,
            description,
            schema,
            handler,
            supports_parallel: false,
            output_budget: None,
            exposure: Exposure::Direct,
            untrusted: false,
        }
    }

    /// Prefix the advertised name with `prefix__`, such as the MCP server's name.
    #[must_use]
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Whether the tool may run at the same time as other tools that allow it.
    #[must_use]
    pub fn supports_parallel(mut self, parallel: bool) -> Self {
        self.supports_parallel = parallel;
        self
    }

    /// How much of the tool's output the model should see.
    #[must_use]
    pub fn output_budget(mut self, budget: OutputBudget) -> Self {
        self.output_budget = Some(budget);
        self
    }

    /// Whether and when the model sees the tool (default [`Exposure::Direct`]).
    #[must_use]
    pub fn exposure(mut self, exposure: Exposure) -> Self {
        self.exposure = exposure;
        self
    }

    /// Whether the tool's output holds content from outside the application
    /// ([`FunctionTool::untrusted_output`]).
    #[must_use]
    pub fn untrusted_output(mut self, untrusted: bool) -> Self {
        self.untrusted = untrusted;
        self
    }
}

/// The size above which [`ToolRegistry::register_dynamic`] compacts a schema
/// ([`schema::compact`]).
pub const DYNAMIC_SCHEMA_COMPACT_BYTES: usize = 5_000;
/// The size above which [`ToolRegistry::register_dynamic`] replaces a schema that compaction
/// could not shrink with one that accepts any object.
pub const DYNAMIC_SCHEMA_MAX_BYTES: usize = 8_000;

/// Registration or execution failure, with the tool name and original source.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolError {
    /// An existing registration was retained.
    #[error("Tool {name:?} is already registered")]
    Duplicate {
        /// Conflicting tool name.
        name: String,
    },
    /// Tool metadata is unsuitable for a function definition.
    #[error("Invalid definition for tool {name:?}: {reason}")]
    InvalidDefinition {
        /// Tool name supplied by the implementation.
        name: String,
        /// Actionable reason for rejection.
        reason: String,
    },
    /// No tool with this name is registered.
    #[error("Unknown tool {name:?}")]
    Unknown {
        /// Requested tool name.
        name: String,
    },
    /// Arguments could not be decoded as JSON or as the tool's input type.
    #[error("Tool {name:?} argument decoding failed: {source}")]
    Arguments {
        /// Registered tool name.
        name: String,
        /// Original Serde error.
        #[source]
        source: serde_json::Error,
    },
    /// The handler's output could not be encoded as JSON.
    #[error("Tool {name:?} output encoding failed: {source}")]
    Output {
        /// Registered tool name.
        name: String,
        /// Original Serde error.
        #[source]
        source: serde_json::Error,
    },
    /// The arguments are JSON but not an object.
    #[error("Tool {name:?} arguments must be a JSON object")]
    InvalidArguments {
        /// Registered tool name.
        name: String,
    },
    /// The tool handler returned an error.
    #[error("Tool {name:?} execution failed: {source}")]
    Execution {
        /// Registered tool name.
        name: String,
        /// Original application error.
        #[source]
        source: Error,
    },
    /// The call cannot be dispatched as a function call.
    #[error("Invalid tool call: {reason}")]
    InvalidCall {
        /// Actionable reason for rejection.
        reason: String,
    },
    /// Call context retained on dispatch failure, including unsupported calls.
    #[error("Tool call {call_id:?} failed: {source}")]
    Call {
        /// Identifier supplied by the model.
        call_id: String,
        /// Underlying dispatch or execution failure.
        #[source]
        source: Box<Self>,
    },
    /// The output cannot be sent where it was asked for, such as an image to a model that takes
    /// only text in tool outputs.
    #[error("Unsupported tool output: {reason}")]
    UnsupportedOutput {
        /// Actionable reason for rejection.
        reason: String,
    },
}

/// What to do with a failed tool call ([`ToolError::classify`]).
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum ToolFailure {
    /// Send this text to the model as the call's output (the answer is `success: false`), so
    /// it can correct the call or carry on.
    RespondToModel(String),
    /// Stop: the error is returned to the application, and nothing is sent for this call.
    Fatal(ToolError),
}

impl ToolError {
    /// The default classification of a call failure, the one
    /// [`ToolRegistry::answer_call`] applies:
    ///
    /// - arguments that are not JSON, not an object or not the tool's input
    ///   (`"failed to parse function arguments: …"`), an unknown tool (`"unsupported call: …"`),
    ///   an invalid call and an unsupported output: [`ToolFailure::RespondToModel`];
    /// - a handler error ([`ToolError::Execution`]): `RespondToModel` with the error's message,
    ///   unless the handler returned [`Error::fatal`];
    /// - an output that cannot be encoded, and registration errors: [`ToolFailure::Fatal`].
    ///
    /// The model sees the handler's error message: return an error whose message is fit for it,
    /// or classify with [`ToolRegistry::answer_call_with`].
    #[must_use]
    pub fn classify(self) -> ToolFailure {
        match self {
            Self::Arguments { source, .. } => {
                ToolFailure::RespondToModel(format!("failed to parse function arguments: {source}"))
            }
            Self::InvalidArguments { .. } => {
                ToolFailure::RespondToModel("function arguments must be a JSON object".into())
            }
            Self::Unknown { name } => {
                ToolFailure::RespondToModel(format!("unsupported call: {name}"))
            }
            Self::InvalidCall { reason } | Self::UnsupportedOutput { reason } => {
                ToolFailure::RespondToModel(reason)
            }
            Self::Execution { source, .. } if !source.is_fatal() => {
                ToolFailure::RespondToModel(model_message(&source))
            }
            Self::Call { call_id, source } => match source.classify() {
                ToolFailure::Fatal(source) => ToolFailure::Fatal(Self::Call {
                    call_id,
                    source: Box::new(source),
                }),
                other => other,
            },
            other => ToolFailure::Fatal(other),
        }
    }
}

/// The message of a handler error for the model: an application error's own message.
fn model_message(error: &Error) -> String {
    match error {
        Error::Application(error) => error.to_string(),
        other => other.to_string(),
    }
}

/// The identity of a registered tool: its name, and its namespace if it has one.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[non_exhaustive]
pub struct ToolKey {
    /// The namespace ([`ToolRegistry::namespace`]), if any.
    pub namespace: Option<String>,
    /// The tool name.
    pub name: String,
}

impl ToolKey {
    /// A tool outside any namespace.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            namespace: None,
            name: name.into(),
        }
    }

    /// A tool of `namespace`.
    #[must_use]
    pub fn namespaced(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: Some(namespace.into()),
            name: name.into(),
        }
    }

    /// The tool `call` names.
    #[must_use]
    pub fn of(call: &ToolCall) -> Self {
        Self {
            namespace: call.namespace.clone(),
            name: call.name.clone(),
        }
    }
}

/// What a handler produced: JSON from a [`FunctionTool`], a body from the others.
enum ErasedOutput {
    /// A function tool's output, and its details for the application.
    Json(Value, Option<Value>),
    Body(ToolOutputBody),
}

impl ErasedOutput {
    /// The details of a function tool's output.
    fn details(&self) -> Option<Value> {
        match self {
            Self::Json(_, details) => details.clone(),
            Self::Body(_) => None,
        }
    }

    fn into_body(self) -> ToolOutputBody {
        match self {
            Self::Json(value, _) => ToolOutputBody::Text(value.to_string()),
            Self::Body(body) => body,
        }
    }

    fn into_value(self) -> Value {
        match self {
            Self::Json(value, _) => value,
            Self::Body(ToolOutputBody::Text(text)) => Value::String(text),
            Self::Body(body) => serde_json::to_value(&body).unwrap_or(Value::Null),
        }
    }
}

/// Object-safe form of the tool traits: decodes the input, boxes the future and encodes the
/// output.
trait ErasedTool: MaybeSend + MaybeSync {
    fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a str,
    ) -> BoxFuture<'a, std::result::Result<ErasedOutput, ToolError>>;
}

/// Decode a JSON object of arguments; the original text is decoded so that Serde still detects
/// duplicate struct fields.
fn decode_arguments<I: DeserializeOwned>(
    name: &str,
    arguments: &str,
) -> std::result::Result<I, ToolError> {
    let argument_error = |source| ToolError::Arguments {
        name: name.into(),
        source,
    };
    let value: Value = serde_json::from_str(arguments).map_err(argument_error)?;
    if !value.is_object() {
        return Err(ToolError::InvalidArguments { name: name.into() });
    }
    serde_json::from_str(arguments).map_err(argument_error)
}

fn execution_error(name: &str) -> impl FnOnce(Error) -> ToolError + '_ {
    move |source| ToolError::Execution {
        name: name.into(),
        source,
    }
}

struct FunctionAdapter<T>(T);

impl<T: FunctionTool> ErasedTool for FunctionAdapter<T> {
    fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a str,
    ) -> BoxFuture<'a, std::result::Result<ErasedOutput, ToolError>> {
        Box::pin(async move {
            let input = decode_arguments(name, input)?;
            let output = FunctionTool::execute(&self.0, input)
                .await
                .map_err(execution_error(name))?;
            let details = FunctionTool::details(&self.0, &output);
            serde_json::to_value(output)
                .map(|value| ErasedOutput::Json(value, details))
                .map_err(|source| ToolError::Output {
                    name: name.into(),
                    source,
                })
        })
    }
}

struct RichAdapter<T>(T);

impl<T: RichTool> ErasedTool for RichAdapter<T> {
    fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a str,
    ) -> BoxFuture<'a, std::result::Result<ErasedOutput, ToolError>> {
        Box::pin(async move {
            let input = decode_arguments(name, input)?;
            RichTool::execute(&self.0, input)
                .await
                .map(ErasedOutput::Body)
                .map_err(execution_error(name))
        })
    }
}

struct CustomAdapter<T>(T);

impl<T: CustomTool> ErasedTool for CustomAdapter<T> {
    fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a str,
    ) -> BoxFuture<'a, std::result::Result<ErasedOutput, ToolError>> {
        Box::pin(async move {
            CustomTool::execute(&self.0, input.to_string())
                .await
                .map(ErasedOutput::Body)
                .map_err(execution_error(name))
        })
    }
}

struct DynamicAdapter(DynamicHandler);

impl ErasedTool for DynamicAdapter {
    fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a str,
    ) -> BoxFuture<'a, std::result::Result<ErasedOutput, ToolError>> {
        Box::pin(async move {
            let arguments: Value = decode_arguments(name, input)?;
            (self.0)(arguments)
                .await
                .map(ErasedOutput::Body)
                .map_err(execution_error(name))
        })
    }
}

#[derive(Clone)]
struct RegisteredTool {
    kind: ToolKind,
    description: String,
    /// The parameters schema of a function tool.
    parameters: Value,
    /// The input format of a custom tool.
    format: CustomFormat,
    handler: Arc<dyn ErasedTool>,
    supports_parallel: bool,
    output_budget: Option<OutputBudget>,
    exposure: Exposure,
    /// The name a dynamic tool was registered with, before sanitizing.
    raw_name: Option<String>,
    /// Whether its output holds content from outside the application.
    untrusted: bool,
}

impl RegisteredTool {
    fn function(
        description: &str,
        parameters: Value,
        handler: Arc<dyn ErasedTool>,
        supports_parallel: bool,
        output_budget: Option<OutputBudget>,
        exposure: Exposure,
    ) -> Self {
        Self {
            kind: ToolKind::Function,
            description: description.to_string(),
            parameters,
            format: CustomFormat::Text,
            handler,
            supports_parallel,
            output_budget,
            exposure,
            raw_name: None,
            untrusted: false,
        }
    }

    fn definition(&self, name: &str, defer_loading: bool) -> ToolDefinition {
        match self.kind {
            ToolKind::Custom => ToolDefinition::Custom {
                name: name.to_string(),
                description: self.description.clone(),
                format: self.format.clone(),
                defer_loading,
            },
            _ => ToolDefinition::Function {
                name: name.to_string(),
                description: self.description.clone(),
                parameters: self.parameters.clone(),
                defer_loading,
            },
        }
    }
}

/// Check a tool or namespace name: 1–64 ASCII letters, digits, underscores or hyphens.
fn check_name(name: &str) -> std::result::Result<(), ToolError> {
    if name.is_empty()
        || name.len() > TOOL_NAME_MAX
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(invalid(
            name,
            "name must contain 1–64 ASCII letters, digits, underscores or hyphens",
        ));
    }
    Ok(())
}

fn check_description(name: &str, description: &str) -> std::result::Result<(), ToolError> {
    if description.trim().is_empty() {
        return Err(invalid(name, "description must not be blank"));
    }
    Ok(())
}

fn check_schema(name: &str, schema: &Value) -> std::result::Result<(), ToolError> {
    if !schema.is_object() || schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err(invalid(
            name,
            "parameters schema must have root type object",
        ));
    }
    Ok(())
}

fn invalid(name: &str, reason: &str) -> ToolError {
    ToolError::InvalidDefinition {
        name: name.into(),
        reason: reason.into(),
    }
}

/// Heterogeneous tools, with definitions snapshotted at registration and returned
/// in name order. Execution is explicit.
///
/// A clone shares the tools' handlers with the original: changing one registry (registering,
/// [removing](Self::remove)) leaves the other as it was, so a run that holds a clone keeps the
/// tools it started with.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<ToolKey, RegisteredTool>,
    /// Namespace names and descriptions.
    namespaces: BTreeMap<String, String>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys())
            .field("namespaces", &self.namespaces.keys())
            .finish_non_exhaustive()
    }
}

impl ToolRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an instance without replacing an existing tool.
    ///
    /// Rejects duplicate names, invalid names, blank descriptions and schemas
    /// without a root `"type": "object"`. Failure leaves the registry unchanged.
    /// Metadata is captured once; later state changes do not alter definitions.
    pub fn register<T: FunctionTool + 'static>(
        &mut self,
        tool: T,
    ) -> std::result::Result<(), ToolError> {
        self.register_function_in(None, tool)
    }

    /// Register a [`RichTool`], with the same checks as [`register`](Self::register).
    pub fn register_rich<T: RichTool + 'static>(
        &mut self,
        tool: T,
    ) -> std::result::Result<(), ToolError> {
        self.register_rich_in(None, tool)
    }

    /// Register a [`CustomTool`]: checks its name and description as
    /// [`register`](Self::register) does.
    pub fn register_custom<T: CustomTool + 'static>(
        &mut self,
        tool: T,
    ) -> std::result::Result<(), ToolError> {
        self.register_custom_in(None, tool)
    }

    /// Register a tool only known at runtime, and return the name the model sees.
    ///
    /// Unlike [`register`](Self::register), it adapts the tool instead of rejecting it:
    ///
    /// - the name (with its [`prefix`](DynamicTool::prefix), as `prefix__name`) is sanitized
    ///   ([`sanitize_tool_name`]); one that is too long or already taken gets a hash suffix
    ///   ([`ToolNamer`]); [`raw_name`](Self::raw_name) gives the original back;
    /// - the schema is sanitized ([`schema::sanitize`]), compacted above
    ///   [`DYNAMIC_SCHEMA_COMPACT_BYTES`] ([`schema::compact`]), and replaced with one that
    ///   accepts any object if it is still above [`DYNAMIC_SCHEMA_MAX_BYTES`];
    /// - a blank description becomes the raw name.
    ///
    /// It fails only for a schema whose root is not an object.
    pub fn register_dynamic(
        &mut self,
        tool: DynamicTool,
    ) -> std::result::Result<String, ToolError> {
        self.register_dynamic_in(None, tool)
    }

    /// Remove a registered tool; returns whether it was registered. Its namespace stays, even
    /// when it is left without tools (it is then not advertised).
    pub fn remove(&mut self, key: &ToolKey) -> bool {
        self.tools.remove(key).is_some()
    }

    /// Remove a namespace and all its tools; returns whether it was registered.
    pub fn remove_namespace(&mut self, name: &str) -> bool {
        self.tools
            .retain(|key, _| key.namespace.as_deref() != Some(name));
        self.namespaces.remove(name).is_some()
    }

    // On WebAssembly handlers are not `Send`: the registry stays on one thread there.
    #[cfg_attr(target_family = "wasm", allow(clippy::arc_with_non_send_sync))]
    fn register_dynamic_in(
        &mut self,
        namespace: Option<&str>,
        tool: DynamicTool,
    ) -> std::result::Result<String, ToolError> {
        let mut schema = tool.schema;
        schema::sanitize(&mut schema);
        if schema.get("type").and_then(Value::as_str) != Some("object") {
            return Err(invalid(
                &tool.raw_name,
                "parameters schema must have root type object",
            ));
        }
        if schema::size(&schema) > DYNAMIC_SCHEMA_COMPACT_BYTES {
            schema::compact(&mut schema, DYNAMIC_SCHEMA_COMPACT_BYTES);
        }
        if schema::size(&schema) > DYNAMIC_SCHEMA_MAX_BYTES {
            schema = json!({"type": "object", "properties": {}, "additionalProperties": true});
        }
        let name = names::unique_name(
            tool.prefix.as_deref(),
            &tool.raw_name,
            TOOL_NAME_MAX,
            |candidate| match namespace {
                Some(namespace) => self
                    .tools
                    .contains_key(&ToolKey::namespaced(namespace, candidate)),
                None => {
                    self.tools.contains_key(&ToolKey::new(candidate))
                        || self.namespaces.contains_key(candidate)
                }
            },
        );
        let description = if tool.description.trim().is_empty() {
            tool.raw_name.clone()
        } else {
            tool.description
        };
        let mut registered = RegisteredTool::function(
            &description,
            schema,
            Arc::new(DynamicAdapter(tool.handler)),
            tool.supports_parallel,
            tool.output_budget,
            tool.exposure,
        );
        registered.raw_name = Some(tool.raw_name);
        registered.untrusted = tool.untrusted;
        self.insert(key(namespace, name.clone()), registered)?;
        Ok(name)
    }

    /// Group tools under a namespace: `name` follows the rules of tool names, `description` says
    /// what the tools are for. Register its tools with the returned [`NamespaceRegistrar`];
    /// asking again for the same namespace adds to it.
    ///
    /// The model calls them with the namespace on the call ([`ToolCall::namespace`]). Namespaced
    /// tools are advertised only to models that take namespaces
    /// ([`ModelInfo::namespaces`](crate::ModelInfo::namespaces)).
    pub fn namespace(
        &mut self,
        name: &str,
        description: &str,
    ) -> std::result::Result<NamespaceRegistrar<'_>, ToolError> {
        check_name(name)?;
        check_description(name, description)?;
        if self.tools.contains_key(&ToolKey::new(name)) {
            return Err(ToolError::Duplicate { name: name.into() });
        }
        match self.namespaces.get(name) {
            Some(existing) if existing != description => {
                return Err(invalid(
                    name,
                    "namespace already registered with another description",
                ));
            }
            Some(_) => {}
            None => {
                self.namespaces.insert(name.into(), description.into());
            }
        }
        Ok(NamespaceRegistrar {
            registry: self,
            namespace: name.into(),
        })
    }

    fn register_function_in<T: FunctionTool + 'static>(
        &mut self,
        namespace: Option<&str>,
        tool: T,
    ) -> std::result::Result<(), ToolError> {
        let name = tool.name().to_owned();
        check_name(&name)?;
        check_description(&name, tool.description())?;
        let schema = tool.parameters_schema();
        check_schema(&name, &schema)?;
        let (parallel, budget, exposure) = (
            FunctionTool::supports_parallel(&tool),
            FunctionTool::output_budget(&tool),
            FunctionTool::exposure(&tool),
        );
        let description = tool.description().to_owned();
        let untrusted = FunctionTool::untrusted_output(&tool);
        let mut registered = RegisteredTool::function(
            &description,
            schema,
            Arc::new(FunctionAdapter(tool)),
            parallel,
            budget,
            exposure,
        );
        registered.untrusted = untrusted;
        self.insert(key(namespace, name), registered)
    }

    fn register_rich_in<T: RichTool + 'static>(
        &mut self,
        namespace: Option<&str>,
        tool: T,
    ) -> std::result::Result<(), ToolError> {
        let name = tool.name().to_owned();
        check_name(&name)?;
        check_description(&name, tool.description())?;
        let schema = tool.parameters_schema();
        check_schema(&name, &schema)?;
        let (parallel, budget, exposure) = (
            tool.supports_parallel(),
            tool.output_budget(),
            tool.exposure(),
        );
        let description = tool.description().to_owned();
        let untrusted = tool.untrusted_output();
        let mut registered = RegisteredTool::function(
            &description,
            schema,
            Arc::new(RichAdapter(tool)),
            parallel,
            budget,
            exposure,
        );
        registered.untrusted = untrusted;
        self.insert(key(namespace, name), registered)
    }

    fn register_custom_in<T: CustomTool + 'static>(
        &mut self,
        namespace: Option<&str>,
        tool: T,
    ) -> std::result::Result<(), ToolError> {
        let name = tool.name().to_owned();
        check_name(&name)?;
        check_description(&name, tool.description())?;
        let registered = RegisteredTool {
            kind: ToolKind::Custom,
            description: tool.description().to_string(),
            parameters: Value::Null,
            format: tool.format(),
            supports_parallel: tool.supports_parallel(),
            output_budget: tool.output_budget(),
            exposure: tool.exposure(),
            raw_name: None,
            untrusted: tool.untrusted_output(),
            handler: Arc::new(CustomAdapter(tool)),
        };
        self.insert(key(namespace, name), registered)
    }

    fn insert(&mut self, key: ToolKey, tool: RegisteredTool) -> std::result::Result<(), ToolError> {
        if key.namespace.is_none() && self.namespaces.contains_key(&key.name) {
            return Err(ToolError::Duplicate { name: key.name });
        }
        match self.tools.entry(key) {
            Entry::Occupied(entry) => Err(ToolError::Duplicate {
                name: entry.key().name.clone(),
            }),
            Entry::Vacant(entry) => {
                entry.insert(tool);
                Ok(())
            }
        }
    }

    /// The definitions of the [`Direct`](Exposure::Direct) tools: the tools outside namespaces
    /// in name order, then one [`ToolDefinition::Namespace`] per namespace with its tools.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.definitions_where(|_, exposure| {
            (exposure == Exposure::Direct).then_some(Advertise::Plain)
        })
    }

    /// The definitions of the tools `show` advertises, plainly or for the model's own tool
    /// search ([`Advertise::Deferred`]): the tools outside namespaces in name order, then one
    /// [`ToolDefinition::Namespace`] per namespace with the tools it advertises (a namespace
    /// without any is left out).
    pub fn definitions_where(
        &self,
        show: impl Fn(&ToolKey, Exposure) -> Option<Advertise>,
    ) -> Vec<ToolDefinition> {
        let mut definitions: Vec<ToolDefinition> = self
            .tools
            .iter()
            .filter(|(key, _)| key.namespace.is_none())
            .filter_map(|(key, tool)| {
                let advertise = show(key, tool.exposure)?;
                Some(tool.definition(&key.name, advertise == Advertise::Deferred))
            })
            .collect();
        for (namespace, description) in &self.namespaces {
            let members: Vec<ToolDefinition> = self
                .tools
                .iter()
                .filter(|(key, _)| key.namespace.as_deref() == Some(namespace.as_str()))
                .filter_map(|(key, tool)| {
                    let advertise = show(key, tool.exposure)?;
                    Some(tool.definition(&key.name, advertise == Advertise::Deferred))
                })
                .collect();
            if members.is_empty() {
                continue;
            }
            definitions.push(ToolDefinition::Namespace {
                name: namespace.clone(),
                description: description.clone(),
                tools: members,
            });
        }
        definitions
    }

    /// The kind of a registered tool.
    #[must_use]
    pub fn kind(&self, key: &ToolKey) -> Option<&ToolKind> {
        self.tools.get(key).map(|tool| &tool.kind)
    }

    /// The [`Exposure`] of a registered tool.
    #[must_use]
    pub fn exposure(&self, key: &ToolKey) -> Option<Exposure> {
        self.tools.get(key).map(|tool| tool.exposure)
    }

    /// Change the [`Exposure`] of a registered tool; returns whether it is registered.
    pub fn set_exposure(&mut self, key: &ToolKey, exposure: Exposure) -> bool {
        self.tools.get_mut(key).is_some_and(|tool| {
            tool.exposure = exposure;
            true
        })
    }

    /// Search the [`Deferred`](Exposure::Deferred) tools for `query`: at most `limit` tools whose
    /// name, namespace or description share words with it, best first (ties in name order).
    ///
    /// Words are runs of letters and digits, compared without case; `snake_case` and
    /// `camelCase` names are split into words. A word of the name counts twice as much as one of
    /// the description. A [`Runner`](crate::Runner) answers its `tool_search` tool with it.
    #[must_use]
    pub fn search_deferred(&self, query: &str, limit: usize) -> Vec<ToolKey> {
        let query = search::words(query);
        let mut scored: Vec<(usize, &ToolKey)> = self
            .tools
            .iter()
            .filter(|(_, tool)| tool.exposure == Exposure::Deferred)
            .filter_map(|(key, tool)| {
                let mut names = search::words(&key.name);
                if let Some(namespace) = &key.namespace {
                    names.extend(search::words(namespace));
                }
                let description = search::words(&tool.description);
                let score = search::score(&query, &names, &description);
                (score > 0).then_some((score, key))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        scored
            .into_iter()
            .take(limit)
            .map(|(_, key)| key.clone())
            .collect()
    }

    /// The description of a registered tool.
    #[must_use]
    pub fn description(&self, key: &ToolKey) -> Option<&str> {
        self.tools.get(key).map(|tool| tool.description.as_str())
    }

    /// Whether a tool of this name is registered, in any namespace.
    #[must_use]
    pub fn has_tool_named(&self, name: &str) -> bool {
        self.keys().any(|key| key.name == name)
    }

    /// The registered tools, namespaced ones included, in order.
    pub fn keys(&self) -> impl Iterator<Item = &ToolKey> {
        self.tools.keys()
    }

    /// The number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether no tool is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The name a dynamic tool was registered with ([`register_dynamic`](Self::register_dynamic)),
    /// for the tool advertised as `name`.
    #[must_use]
    pub fn raw_name(&self, name: &str) -> Option<&str> {
        self.tools.get(&ToolKey::new(name))?.raw_name.as_deref()
    }

    /// Whether the tool `call` names allows parallel execution
    /// ([`FunctionTool::supports_parallel`]); `false` for an unknown tool.
    #[must_use]
    pub fn supports_parallel(&self, call: &ToolCall) -> bool {
        self.tools
            .get(&ToolKey::of(call))
            .is_some_and(|tool| tool.supports_parallel)
    }

    /// The output budget of the tool `call` names ([`FunctionTool::output_budget`]), if it
    /// sets one.
    #[must_use]
    pub fn output_budget(&self, call: &ToolCall) -> Option<OutputBudget> {
        self.tools.get(&ToolKey::of(call))?.output_budget
    }

    /// Dispatch by name. Arguments must be a JSON object matching `FunctionTool::Input`.
    /// Returns JSON output; does not retry, catch panics, or enforce timeouts.
    ///
    /// Only function tools outside namespaces are found here. A rich or dynamic tool's text
    /// output is returned as a JSON string, its content parts as an array.
    pub async fn execute(
        &self,
        name: &str,
        arguments: &str,
    ) -> std::result::Result<Value, ToolError> {
        let tool = self
            .tools
            .get(&ToolKey::new(name))
            .filter(|tool| tool.kind == ToolKind::Function)
            .ok_or_else(|| ToolError::Unknown { name: name.into() })?;
        tool.handler
            .call(name, arguments)
            .await
            .map(ErasedOutput::into_value)
    }

    /// Answer one approved call, so that every call gets an answer.
    ///
    /// Runs the tool the call names (with its namespace) and returns its output, or, when the
    /// call fails in a way the model can act on, the failure as the output with
    /// `success: false` (see [`ToolError::classify`]): bad arguments, an unknown tool, a call of
    /// the wrong kind, a handler error. Only a [`ToolFailure::Fatal`] failure, or a call without
    /// an identifier, is returned as an error ([`ToolError::Call`]).
    ///
    /// The output is not truncated: see [`ToolOutput::truncated`].
    pub async fn answer_call(
        &self,
        call: impl Into<ToolCall>,
    ) -> std::result::Result<ToolOutput, ToolError> {
        self.answer_call_with(call, ToolError::classify).await
    }

    /// [`answer_call`](Self::answer_call) with your own classification of failures, for
    /// example to hide handler errors from the model.
    pub async fn answer_call_with(
        &self,
        call: impl Into<ToolCall>,
        classify: impl FnOnce(ToolError) -> ToolFailure,
    ) -> std::result::Result<ToolOutput, ToolError> {
        let call = call.into();
        if call.call_id.is_empty() {
            return Err(ToolError::Call {
                call_id: String::new(),
                source: Box::new(ToolError::InvalidCall {
                    reason: "call identifier must not be empty".into(),
                }),
            });
        }
        match self.run(&call).await {
            Ok((output, details)) => {
                let untrusted = self
                    .tools
                    .get(&ToolKey::of(&call))
                    .is_some_and(|tool| tool.untrusted);
                let mut answer = ToolOutput::new(&call, output, true).with_untrusted(untrusted);
                answer.details = details;
                Ok(answer)
            }
            Err(error) => match classify(error) {
                ToolFailure::Fatal(error) => Err(ToolError::Call {
                    call_id: call.call_id.clone(),
                    source: Box::new(error),
                }),
                ToolFailure::RespondToModel(message) => Ok(ToolOutput::failure(&call, message)),
            },
        }
    }

    async fn run(
        &self,
        call: &ToolCall,
    ) -> std::result::Result<(ToolOutputBody, Option<Value>), ToolError> {
        if let ToolKind::Other(kind) = &call.kind {
            return Err(ToolError::InvalidCall {
                reason: format!("unsupported call type {kind:?}"),
            });
        }
        let Some(tool) = self.tools.get(&ToolKey::of(call)) else {
            let name = call.namespace.as_ref().map_or_else(
                || call.name.clone(),
                |namespace| format!("{namespace}.{}", call.name),
            );
            return Err(ToolError::Unknown { name });
        };
        if tool.kind != call.kind {
            let reason = match tool.kind {
                ToolKind::Custom => format!("{} is a custom tool: send text input", call.name),
                _ => format!("{} is a function tool: send JSON arguments", call.name),
            };
            return Err(ToolError::InvalidCall { reason });
        }
        tool.handler
            .call(&call.name, &call.input)
            .await
            .map(|output| {
                let details = output.details();
                (output.into_body(), details)
            })
    }
}

/// Registers tools in a namespace ([`ToolRegistry::namespace`]).
#[derive(Debug)]
pub struct NamespaceRegistrar<'a> {
    registry: &'a mut ToolRegistry,
    namespace: String,
}

impl NamespaceRegistrar<'_> {
    /// Register a [`FunctionTool`] in the namespace ([`ToolRegistry::register`]).
    pub fn register<T: FunctionTool + 'static>(
        &mut self,
        tool: T,
    ) -> std::result::Result<&mut Self, ToolError> {
        self.registry
            .register_function_in(Some(&self.namespace), tool)?;
        Ok(self)
    }

    /// Register a [`RichTool`] in the namespace ([`ToolRegistry::register_rich`]).
    pub fn register_rich<T: RichTool + 'static>(
        &mut self,
        tool: T,
    ) -> std::result::Result<&mut Self, ToolError> {
        self.registry
            .register_rich_in(Some(&self.namespace), tool)?;
        Ok(self)
    }

    /// Register a [`CustomTool`] in the namespace ([`ToolRegistry::register_custom`]).
    pub fn register_custom<T: CustomTool + 'static>(
        &mut self,
        tool: T,
    ) -> std::result::Result<&mut Self, ToolError> {
        self.registry
            .register_custom_in(Some(&self.namespace), tool)?;
        Ok(self)
    }

    /// Register a [`DynamicTool`] in the namespace ([`ToolRegistry::register_dynamic`]), and
    /// return the name the model sees, unique within the namespace.
    pub fn register_dynamic(
        &mut self,
        tool: DynamicTool,
    ) -> std::result::Result<String, ToolError> {
        self.registry
            .register_dynamic_in(Some(&self.namespace), tool)
    }
}

fn key(namespace: Option<&str>, name: String) -> ToolKey {
    ToolKey {
        namespace: namespace.map(str::to_string),
        name,
    }
}

/// How a tool is advertised ([`ToolRegistry::definitions_where`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Advertise {
    /// With its full definition.
    Plain,
    /// With `defer_loading`: the model's own tool search loads it.
    Deferred,
}
