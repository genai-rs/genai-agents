//! Recorded runs: a [`Model`] that records the requests it sends and the responses it gets
//! ([`RecordingModel`]), and one that answers from such a recording ([`ReplayModel`]).
//!
//! A test or an eval then runs without a key or a network, the same way every time. Replay
//! checks every request against the recording: one that no longer matches (because a prompt, a
//! tool or a setting changed) fails with an error that names the first difference, so the
//! recording is made again on purpose rather than drifting.
//!
//! The prompt cache key is not recorded (the runner's default one is random); everything else
//! of a request is: the items, the tools, the settings and whether calls may run in parallel.
//!
//! ```
//! use genai_agents::record::{RecordingModel, ReplayModel};
//! use genai_agents::testing::{Reply, ScriptedModel};
//! use genai_agents::{Conversation, Runner, ToolRegistry};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> genai_agents::Result<()> {
//! // Record a run (with a real model, the provider's; a scripted one here).
//! let live = ScriptedModel::replies(vec![Reply::text("Take an umbrella.")]);
//! let mut runner = Runner::new(RecordingModel::new(live), ToolRegistry::new());
//! runner.run(&mut Conversation::user("Umbrella in Ghent?")).await?;
//! let recording = runner.model().recording();
//! let jsonl = recording.to_jsonl()?; // or `save` it to a file
//!
//! // Replay it: no model behind it.
//! let replay = ReplayModel::from_jsonl(&jsonl)?;
//! let mut runner = Runner::new(replay, ToolRegistry::new());
//! let outcome = runner.run(&mut Conversation::user("Umbrella in Ghent?")).await?;
//! assert_eq!(outcome.output_text(), "Take an umbrella.");
//!
//! // Another question is not in the recording.
//! let replay = ReplayModel::from_jsonl(&jsonl)?;
//! let mut runner = Runner::new(replay, ToolRegistry::new());
//! let error = runner.run(&mut Conversation::user("Sunscreen?")).await.unwrap_err();
//! assert!(error.to_string().contains("/items/0/content/0/text"));
//! # Ok(())
//! # }
//! ```

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{ErrorKind, ModelError, RetryClass};
use crate::item::Item;
use crate::model::{
    Model, ModelEvent, ModelInfo, ModelRequest, ModelSettings, ModelStream, OutputSchema,
    RateLimits, ToolDefinition,
};
use crate::platform::MaybeSend;

/// A request as it is recorded: everything but the prompt cache key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RecordedRequest {
    /// The items sent: the projection of the conversation.
    pub items: Vec<Item>,
    /// The tools advertised.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// The model settings.
    #[serde(default)]
    pub settings: ModelSettings,
    /// Whether the model could call several tools at once.
    pub parallel_tool_calls: bool,
    /// The instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The schema of the answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputSchema>,
}

impl RecordedRequest {
    /// The recorded form of a request.
    #[must_use]
    pub fn of(request: &ModelRequest<'_>) -> Self {
        Self {
            items: request.items.to_vec(),
            tools: request.tools.to_vec(),
            settings: request.settings,
            parallel_tool_calls: request.parallel_tool_calls,
            instructions: request.instructions.map(str::to_owned),
            output: request.output.cloned(),
        }
    }
}

/// A [`ModelError`] as it is recorded: what the runner acts on, without its source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RecordedError {
    /// The message.
    pub message: String,
    /// The kind.
    pub kind: ErrorKind,
    /// How it may be retried.
    pub retry: RetryClass,
    /// The delay the server asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<Duration>,
    /// Whether it happened before the response's body.
    #[serde(default)]
    pub before_body: bool,
    /// The HTTP status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// The provider's error code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl From<&ModelError> for RecordedError {
    fn from(error: &ModelError) -> Self {
        Self {
            message: error.message().to_owned(),
            kind: error.kind(),
            retry: error.retry_class(),
            retry_after: error.retry_after(),
            before_body: error.is_before_body(),
            status: error.status(),
            code: error.code().map(str::to_owned),
        }
    }
}

impl From<&RecordedError> for ModelError {
    fn from(recorded: &RecordedError) -> Self {
        let mut error = ModelError::new(recorded.message.clone())
            .with_kind(recorded.kind)
            .with_retry_class(recorded.retry)
            .with_retry_after(recorded.retry_after)
            .with_before_body(recorded.before_body);
        if let Some(status) = recorded.status {
            error = error.with_status(status);
        }
        if let Some(code) = &recorded.code {
            error = error.with_code(code.clone());
        }
        error
    }
}

/// One event of a recorded response, or the error that ended it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RecordedEvent {
    /// An event.
    Event(ModelEvent),
    /// The error the stream failed with.
    Error(RecordedError),
}

/// A recorded response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum RecordedResponse {
    /// The request failed before a stream began.
    Failed {
        /// The error.
        error: RecordedError,
    },
    /// The stream of the response, as far as the runner read it.
    Streamed {
        /// The rate limits it reported.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rate_limits: Option<RateLimits>,
        /// Its events.
        events: Vec<RecordedEvent>,
    },
}

/// A request and its response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Exchange {
    /// The request.
    pub request: RecordedRequest,
    /// The response.
    pub response: RecordedResponse,
}

/// The requests a model got and its responses, in order, and what the model said of itself.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Recording {
    /// The model's [`info`](Model::info) when the recording was made: a replay reports it.
    pub info: Option<ModelInfo>,
    /// The exchanges, in the order the requests were sent.
    pub exchanges: Vec<Exchange>,
}

/// A line of a recording in JSON Lines.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Line {
    Model { info: ModelInfo },
    Exchange(Exchange),
}

impl Recording {
    /// The recording in [JSON Lines](https://jsonlines.org): the model first, then an exchange
    /// per line, so that a changed request shows up as a changed line in a diff.
    ///
    /// # Errors
    ///
    /// When a value cannot be encoded.
    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        let mut out = String::new();
        if let Some(info) = &self.info {
            out.push_str(&serde_json::to_string(&Line::Model { info: info.clone() })?);
            out.push('\n');
        }
        for exchange in &self.exchanges {
            out.push_str(&serde_json::to_string(&Line::Exchange(exchange.clone()))?);
            out.push('\n');
        }
        Ok(out)
    }

    /// A recording from [`to_jsonl`](Self::to_jsonl)'s text.
    ///
    /// # Errors
    ///
    /// When a line is not a recording's.
    pub fn from_jsonl(text: &str) -> Result<Self, serde_json::Error> {
        let mut recording = Self::default();
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            match serde_json::from_str(line)? {
                Line::Model { info } => recording.info = Some(info),
                Line::Exchange(exchange) => recording.exchanges.push(exchange),
            }
        }
        Ok(recording)
    }

    /// Write the recording to a file, in JSON Lines.
    ///
    /// # Errors
    ///
    /// When it cannot be encoded or written.
    #[cfg(not(target_family = "wasm"))]
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_jsonl()?)
    }

    /// Read a recording from a file.
    ///
    /// # Errors
    ///
    /// When it cannot be read or decoded.
    #[cfg(not(target_family = "wasm"))]
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        Ok(Self::from_jsonl(&std::fs::read_to_string(path)?)?)
    }
}

/// A [`Model`] that passes every request on to `M` and records it with its response.
#[derive(Debug)]
pub struct RecordingModel<M> {
    inner: M,
    recording: Arc<Mutex<Recording>>,
}

impl<M: Model> RecordingModel<M> {
    /// Record the requests to `inner` and its responses.
    pub fn new(inner: M) -> Self {
        let recording = Recording {
            info: Some(inner.info()),
            exchanges: Vec::new(),
        };
        Self {
            inner,
            recording: Arc::new(Mutex::new(recording)),
        }
    }

    /// The model it records.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// What it recorded so far.
    #[must_use]
    pub fn recording(&self) -> Recording {
        self.recording
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Write what it recorded so far to a file, in JSON Lines.
    ///
    /// # Errors
    ///
    /// When it cannot be encoded or written.
    #[cfg(not(target_family = "wasm"))]
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.recording().save(path)
    }
}

fn update(recording: &Mutex<Recording>, index: usize, change: impl FnOnce(&mut RecordedResponse)) {
    let mut recording = recording.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(exchange) = recording.exchanges.get_mut(index) {
        change(&mut exchange.response);
    }
}

impl<M: Model> Model for RecordingModel<M> {
    fn info(&self) -> ModelInfo {
        self.inner.info()
    }

    fn retries_before_body(&self) -> bool {
        self.inner.retries_before_body()
    }

    fn stream<'a>(
        &'a self,
        request: ModelRequest<'_>,
    ) -> impl std::future::Future<Output = Result<ModelStream<'a>, ModelError>> + MaybeSend {
        let recorded = RecordedRequest::of(&request);
        let index = {
            let mut recording = self
                .recording
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            recording.exchanges.push(Exchange {
                request: recorded,
                response: RecordedResponse::Streamed {
                    rate_limits: None,
                    events: Vec::new(),
                },
            });
            recording.exchanges.len() - 1
        };
        let sent = self.inner.stream(request);
        async move {
            let stream = match sent.await {
                Ok(stream) => stream,
                Err(error) => {
                    update(&self.recording, index, |response| {
                        *response = RecordedResponse::Failed {
                            error: RecordedError::from(&error),
                        };
                    });
                    return Err(error);
                }
            };
            let rate_limits = stream.rate_limits;
            update(&self.recording, index, |response| {
                *response = RecordedResponse::Streamed {
                    rate_limits,
                    events: Vec::new(),
                };
            });
            let recording = Arc::clone(&self.recording);
            let events = stream.events.map(move |event| {
                let recorded = match &event {
                    Ok(event) => RecordedEvent::Event(event.clone()),
                    Err(error) => RecordedEvent::Error(RecordedError::from(error)),
                };
                update(&recording, index, |response| {
                    if let RecordedResponse::Streamed { events, .. } = response {
                        events.push(recorded);
                    }
                });
                event
            });
            Ok(ModelStream::new(Box::pin(events)).with_rate_limits(rate_limits))
        }
    }
}

/// A [`Model`] that answers from a [`Recording`]: the `n`th request with the `n`th response, once
/// it checked that the request is the one recorded. A request that differs fails (not retried)
/// with the first difference, as a JSON pointer with both values; so does a request beyond the
/// recording.
#[derive(Debug)]
pub struct ReplayModel {
    info: ModelInfo,
    exchanges: Vec<Exchange>,
    next: Mutex<usize>,
}

impl ReplayModel {
    /// Replay this recording.
    #[must_use]
    pub fn new(recording: Recording) -> Self {
        Self {
            info: recording.info.unwrap_or_default(),
            exchanges: recording.exchanges,
            next: Mutex::new(0),
        }
    }

    /// Replay a recording in JSON Lines ([`Recording::to_jsonl`]).
    ///
    /// # Errors
    ///
    /// When the text is not a recording.
    pub fn from_jsonl(text: &str) -> Result<Self, serde_json::Error> {
        Recording::from_jsonl(text).map(Self::new)
    }

    /// Replay a recording from a file.
    ///
    /// # Errors
    ///
    /// When it cannot be read or decoded.
    #[cfg(not(target_family = "wasm"))]
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        Recording::load(path).map(Self::new)
    }

    /// How many recorded requests were not asked for yet.
    #[must_use]
    pub fn remaining(&self) -> usize {
        let next = *self.next.lock().unwrap_or_else(PoisonError::into_inner);
        self.exchanges.len().saturating_sub(next)
    }

    fn answer(&self, request: &RecordedRequest) -> Result<ModelStream<'static>, ModelError> {
        let index = {
            let mut next = self.next.lock().unwrap_or_else(PoisonError::into_inner);
            let index = *next;
            *next += 1;
            index
        };
        let Some(exchange) = self.exchanges.get(index) else {
            return Err(mismatch(format!(
                "request {} is beyond the recording, which has {}",
                index + 1,
                self.exchanges.len()
            )));
        };
        if &exchange.request != request {
            let expected = serde_json::to_value(&exchange.request).unwrap_or_default();
            let actual = serde_json::to_value(request).unwrap_or_default();
            let difference = difference("", &expected, &actual)
                .unwrap_or_else(|| "they encode the same".to_owned());
            return Err(mismatch(format!(
                "request {} differs from the recording at {difference}",
                index + 1
            )));
        }
        match &exchange.response {
            RecordedResponse::Failed { error } => Err(ModelError::from(error)),
            RecordedResponse::Streamed {
                rate_limits,
                events,
            } => {
                let events: Vec<Result<ModelEvent, ModelError>> = events
                    .iter()
                    .map(|event| match event {
                        RecordedEvent::Event(event) => Ok(event.clone()),
                        RecordedEvent::Error(error) => Err(ModelError::from(error)),
                    })
                    .collect();
                Ok(ModelStream::new(Box::pin(futures::stream::iter(events)))
                    .with_rate_limits(*rate_limits))
            }
        }
    }
}

fn mismatch(message: String) -> ModelError {
    ModelError::new(format!("replay: {message}")).with_kind(ErrorKind::Local)
}

impl Model for ReplayModel {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn stream<'a>(
        &'a self,
        request: ModelRequest<'_>,
    ) -> impl std::future::Future<Output = Result<ModelStream<'a>, ModelError>> + MaybeSend {
        let answer = self.answer(&RecordedRequest::of(&request));
        async move { answer }
    }
}

/// The first place where `actual` differs from `expected`: a JSON pointer and both values.
fn difference(path: &str, expected: &Value, actual: &Value) -> Option<String> {
    match (expected, actual) {
        (Value::Object(expected), Value::Object(actual)) => {
            let mut keys: Vec<&String> = expected.keys().chain(actual.keys()).collect();
            keys.sort();
            keys.dedup();
            keys.into_iter().find_map(|key| {
                let at = format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
                match (expected.get(key), actual.get(key)) {
                    (Some(expected), Some(actual)) => difference(&at, expected, actual),
                    (Some(expected), None) => {
                        Some(format!("{at}: recorded {}, now missing", show(expected)))
                    }
                    (None, Some(actual)) => {
                        Some(format!("{at}: not recorded, now {}", show(actual)))
                    }
                    (None, None) => None,
                }
            })
        }
        (Value::Array(expected), Value::Array(actual)) => {
            let found =
                expected
                    .iter()
                    .zip(actual)
                    .enumerate()
                    .find_map(|(index, (expected, actual))| {
                        difference(&format!("{path}/{index}"), expected, actual)
                    });
            found.or_else(|| {
                (expected.len() != actual.len()).then(|| {
                    format!(
                        "{}: recorded {} entries, now {}",
                        if path.is_empty() { "/" } else { path },
                        expected.len(),
                        actual.len()
                    )
                })
            })
        }
        _ if expected == actual => None,
        _ => Some(format!(
            "{}: recorded {}, now {}",
            if path.is_empty() { "/" } else { path },
            show(expected),
            show(actual)
        )),
    }
}

/// A value for a message, cut to 120 characters.
fn show(value: &Value) -> String {
    let text = value.to_string();
    if text.chars().count() <= 120 {
        text
    } else {
        let cut: String = text.chars().take(120).collect();
        format!("{cut}…")
    }
}
