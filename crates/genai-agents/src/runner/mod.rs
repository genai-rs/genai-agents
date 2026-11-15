//! The runner: [`Runner`] sends a [`Conversation`] to a [`Model`], runs the tools it calls, sends
//! their outputs back, and repeats until the model answers.
//!
//! It:
//!
//! - keeps the conversation as it came and sends each request a projection of it for the model
//!   ([`Conversation::for_request`]): every call answered, orphan outputs dropped, the
//!   reasoning of other models left out;
//! - answers every call ([`ToolRegistry::answer_call`]), runs the calls of tools that
//!   [allow it](crate::tools::FunctionTool::supports_parallel) at the same time and the others
//!   alone, and records the answers in call order, cut to their
//!   [budget](crate::tools::OutputBudget);
//! - starts a call as soon as the model finished writing it ([`ModelEvent::ItemDone`]), while
//!   the response goes on, but never one the response stopped in the middle of
//!   ([`StopReason::Length`], [`ItemMeta::interrupted`](crate::ItemMeta::interrupted));
//! - keeps the partial text of a cancelled or failed response, marked as cut off;
//! - stops after [`RunConfig::max_iterations`] requests, or when a [`CancellationSignal`] fires,
//!   with a history that can be sent again;
//! - retries a failed request as a whole, after the delay the server asked for or with backoff
//!   ([`RetryConfig`], [`ModelError::retry_class`]);
//! - advertises [deferred](Exposure::Deferred) tools once a tool search found them
//!   ([`ToolSearchConfig`], [`ModelInfo::tool_search`]);
//! - keeps the rate limits of every request ([`TurnRecord::rate_limits`]).
//!
//! Nothing is spawned: the calls run on the run's own task, so the runner needs no runtime of
//! its own and runs on WebAssembly with tools that are not `Send`.

mod cancel;
mod config;
mod context;
mod dispatch;
mod trace;

use std::collections::BTreeSet;
use std::future::Future;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use serde_json::{json, Value};
use tracing::field::Empty;
use tracing::Instrument;

pub use cancel::CancellationSignal;
pub use config::{PromptCacheKey, RetryConfig, RunConfig, ToolSearchConfig};
pub use context::{
    estimate_tokens, ContextConfig, PruneConfig, RunInput, COMPACTION_PROMPT, INTERRUPTED_NOTE,
    PRUNED_OUTPUT, REPEATED_FAILURE, TEXT_CALL_NOTE,
};
pub use trace::CacheBreak;

use crate::conversation::Conversation;
use crate::error::{Error, ModelError, Result, RetryClass};
use crate::item::{Compaction, Item, ItemMeta, Message, Role, ToolCall, ToolKind, ToolOutput};
use crate::model::{
    fnv1a, Model, ModelEvent, ModelInfo, ModelRequest, ModelResponse, ModelStream, RateLimits,
    ReasoningLevel, StopReason, ToolDefinition, Usage,
};
use crate::platform::{random_jitter, random_u64, sleep, Instant, MaybeSend};
use crate::session::{Session, SessionEntry, SessionStore, TurnSettings};
use crate::tools::{Advertise, Exposure, ToolError, ToolKey, ToolRegistry};
use dispatch::Dispatcher;

/// The answer to a call the model did not finish writing because its response stopped at the
/// output limit: it is not run.
pub const CUT_OFF_OUTPUT: &str =
    "the response stopped at the output limit before this call was complete; it was not run";

/// The name of the tool search tool the runner offers and answers itself.
pub const TOOL_SEARCH: &str = "tool_search";

/// Something that happened during a run, for [`Runner::run_with`]'s callback (to stream text to
/// a UI, log, or show progress).
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum RunEvent<'a> {
    /// A request is about to be sent: the `iteration`'s `attempt`, from 0.
    Request {
        /// The model request of the run, from 1.
        iteration: u32,
        /// The retry of this request, 0 for the first attempt.
        attempt: u32,
    },
    /// An event of the model's response, such as a text delta ([`ModelEvent::text_delta`]).
    Model(&'a ModelEvent),
    /// The model called a tool; the call is queued and runs as soon as it may.
    ToolCalled(&'a ToolCall),
    /// A call was answered (successfully or not); its answer goes into the history after the
    /// response, in call order.
    ToolAnswered(&'a ToolOutput),
    /// The request failed and will be sent again after `delay`.
    Retrying {
        /// The retry, from 1.
        attempt: u32,
        /// The most retries of this request ([`RetryConfig::max_retries`]).
        max: u32,
        /// The wait before it.
        delay: Duration,
        /// Why the request failed.
        error: &'a ModelError,
    },
    /// A model request and its tool calls are done.
    TurnDone(&'a TurnRecord),
    /// The conversation was compacted: a summary now stands in for what came before
    /// ([`Item::Compaction`]).
    Compacted {
        /// The estimated tokens before.
        tokens_before: u64,
        /// The estimated tokens after.
        tokens_after: u64,
    },
    /// A message of the [`RunInput`] went into the conversation: steering, or a follow-up.
    Input(&'a Item),
}

/// One model request of a run and what came of it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TurnRecord {
    /// The model request of the run, from 1.
    pub iteration: u32,
    /// The id the provider gave the response.
    pub response_id: Option<String>,
    /// The text the model wrote.
    pub text: String,
    /// The tools it called, in order.
    pub calls: Vec<ToolCall>,
    /// The rate limits the provider reported with the response.
    pub rate_limits: Option<RateLimits>,
    /// How many times the request was retried.
    pub retries: u32,
    /// The input tokens of the response.
    pub input_tokens: Option<i64>,
    /// The output tokens of the response.
    pub output_tokens: Option<i64>,
    /// The tokens of the response in detail, when the provider reported them.
    pub usage: Option<Usage>,
    /// Why the model stopped, when the provider says.
    pub stop: Option<StopReason>,
    /// Why the request could not reuse the prompt cache of the request before it, if it could
    /// not; `None` for the first request of a runner.
    pub cache_break: Option<CacheBreak>,
}

impl TurnRecord {
    fn new(iteration: u32) -> Self {
        Self {
            iteration,
            response_id: None,
            text: String::new(),
            calls: Vec::new(),
            rate_limits: None,
            retries: 0,
            input_tokens: None,
            output_tokens: None,
            usage: None,
            stop: None,
            cache_break: None,
        }
    }

    fn describe(&mut self, response: &ModelResponse) {
        self.response_id.clone_from(&response.id);
        self.input_tokens = response.usage.map(|usage| usage.input_tokens);
        self.output_tokens = response.usage.map(|usage| usage.output_tokens);
        self.usage = response.usage;
        self.stop.clone_from(&response.stop);
    }
}

/// Why a run stopped.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunStop {
    /// The model answered without calling a tool.
    Completed,
    /// The model still called tools after [`RunConfig::max_iterations`] requests. Their answers
    /// are in the history: run again to continue.
    MaxIterations,
    /// The [`CancellationSignal`] fired. Every call in the history has an answer (the calls that
    /// were running say `"aborted by user after …s"`): run again to continue.
    Cancelled,
    /// The response stopped early, such as at the output token limit or for its content filter.
    Incomplete {
        /// Why it stopped.
        reason: Option<String>,
    },
}

/// The result of a run. The history is in the [`Conversation`] it ran on.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RunOutcome {
    /// Why the run stopped.
    pub stop: RunStop,
    /// The model requests of the run, in order.
    pub turns: Vec<TurnRecord>,
}

impl RunOutcome {
    fn new(stop: RunStop, turns: Vec<TurnRecord>) -> Self {
        Self { stop, turns }
    }

    /// The text of the last response.
    #[must_use]
    pub fn output_text(&self) -> &str {
        self.turns.last().map_or("", |turn| turn.text.as_str())
    }

    /// Whether the model answered ([`RunStop::Completed`]).
    #[must_use]
    pub fn is_completed(&self) -> bool {
        self.stop == RunStop::Completed
    }

    /// The rate limits of the last response that reported them.
    #[must_use]
    pub fn rate_limits(&self) -> Option<RateLimits> {
        self.turns.iter().rev().find_map(|turn| turn.rate_limits)
    }
}

/// What a streamed attempt waits for.
// Short-lived, one at a time: boxing the event would only add an allocation per event.
#[allow(clippy::large_enum_variant)]
enum Step {
    Event(Option<Result<ModelEvent, ModelError>>),
    Answered(std::result::Result<usize, ToolError>),
    Cancelled,
}

/// How a request attempt ended.
enum AttemptEnd {
    /// A response, complete (`incomplete` is `None`) or stopped early.
    Done {
        incomplete: Option<String>,
        calls: usize,
    },
    Failed(ModelError),
    Cancelled,
}

/// Runs a model with tools: the model, the tools and the configuration. See the
/// [module documentation](self).
///
/// ```rust,no_run
/// use genai_agents::{Conversation, Model, Runner, ToolRegistry};
///
/// # async fn example(model: impl Model, tools: ToolRegistry) -> genai_agents::Result<()> {
/// let mut runner = Runner::new(model, tools);
/// let mut conversation = Conversation::user("Should I carry an umbrella in Brussels today?");
/// let outcome = runner.run(&mut conversation).await?;
/// println!("{}", outcome.output_text());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Runner<M> {
    model: M,
    state: State,
}

/// Everything of a runner but its model, so that a response (which borrows the model) can be read
/// while the state changes.
#[derive(Debug)]
struct State {
    registry: Arc<ToolRegistry>,
    config: RunConfig,
    loaded: BTreeSet<ToolKey>,
    cache_key: String,
    /// What the previous request sent, for [`TurnRecord::cache_break`].
    last_shape: Option<trace::RequestShape>,
    /// Messages for the run while it goes.
    input: RunInput,
    /// The conversation's length when the last response ended, and the tokens the server
    /// counted for it then: the base of the token estimate.
    accounted: Option<(usize, u64)>,
    /// When the last request was sent: whether the prompt cache may have expired.
    last_request_at: Option<Instant>,
    /// The tool outputs, counted in the request, that pruning applies to.
    prune_boundary: usize,
    /// The failures of each call (tool and arguments) in this run.
    failures: std::collections::BTreeMap<(Option<String>, String, u64), u32>,
}

impl<M: Model> Runner<M> {
    /// A runner of `model` with these tools.
    pub fn new(model: M, registry: impl Into<Arc<ToolRegistry>>) -> Self {
        Self {
            model,
            state: State {
                registry: registry.into(),
                config: RunConfig::default(),
                loaded: BTreeSet::new(),
                cache_key: format!("genai-agents-{:016x}", random_u64()),
                last_shape: None,
                input: RunInput::new(),
                accounted: None,
                last_request_at: None,
                prune_boundary: 0,
                failures: std::collections::BTreeMap::new(),
            },
        }
    }

    /// The same runner with this configuration.
    #[must_use]
    pub fn with_config(mut self, config: RunConfig) -> Self {
        self.state.config = config;
        self
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &RunConfig {
        &self.state.config
    }

    /// The configuration, to change it between runs.
    pub fn config_mut(&mut self) -> &mut RunConfig {
        &mut self.state.config
    }

    /// The model.
    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// The model, to change it between runs.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// The tools.
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.state.registry
    }

    /// The prompt cache key the requests carry, if any ([`RunConfig::prompt_cache_key`]).
    #[must_use]
    pub fn prompt_cache_key(&self) -> Option<&str> {
        self.state.prompt_cache_key()
    }

    /// The deferred tools loaded so far: found by a tool search (the runner's own, or the
    /// model's when it [searches itself](ModelInfo::tool_search)), or loaded with
    /// [`load_tool`](Self::load_tool). They stay loaded across runs.
    pub fn loaded_tools(&self) -> impl Iterator<Item = &ToolKey> {
        self.state.loaded.iter()
    }

    /// The input of the runs: messages the application sends while a run is going (steering,
    /// follow-ups). Clone it and push from anywhere; see [`RunInput`].
    #[must_use]
    pub fn input(&self) -> RunInput {
        self.state.input.clone()
    }

    /// The estimated tokens of the next request on `conversation`: what the server counted for
    /// the last response, plus an [estimate](estimate_tokens) of the items since; an estimate of
    /// the whole when no response was counted yet.
    #[must_use]
    pub fn context_tokens(&self, conversation: &Conversation) -> u64 {
        self.state.context_tokens(conversation)
    }

    /// Compact `conversation` now: the model summarises it, and an [`Item::Compaction`] with the
    /// summary, the user's recent messages and the injected context goes at its end. Requests
    /// are built from it from then on; the items before it stay, unsent. Returns `false` when
    /// `cancel` fired first.
    ///
    /// The runner compacts by itself ([`ContextConfig`]); this is for the application that
    /// wants to, such as before handing the conversation to a model with a smaller window.
    ///
    /// # Errors
    ///
    /// [`Error::Compaction`] when the summary is empty or was cut off at the output limit, or
    /// the error of the request.
    pub async fn compact(
        &mut self,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
    ) -> Result<bool> {
        self.compact_with(conversation, cancel, &mut |_| {}).await
    }

    /// Load a [deferred](Exposure::Deferred) tool: advertise it from the next request on, as if a
    /// tool search had found it. Returns whether the tool is registered and deferred. (A model
    /// that searches tools itself keeps getting deferred tools with `defer_loading`.)
    pub fn load_tool(&mut self, key: ToolKey) -> bool {
        let deferred = self.state.registry.exposure(&key) == Some(Exposure::Deferred);
        if deferred {
            self.state.loaded.insert(key);
        }
        deferred
    }

    /// Run until the model answers, [`RunConfig::max_iterations`] requests were sent, or a
    /// response stops early. See [`run_with`](Self::run_with).
    pub async fn run(&mut self, conversation: &mut Conversation) -> Result<RunOutcome> {
        self.run_with(conversation, &CancellationSignal::new(), |_| {})
            .await
    }

    /// Run on `conversation` until the model answers, [`RunConfig::max_iterations`] requests
    /// were sent, a response stops early, or `cancel` fires; `on_event` sees what happens
    /// ([`RunEvent`]).
    ///
    /// The model's output items and the answers to its calls are appended to `conversation`, as
    /// they came; each request is a [projection](Conversation::for_request) of it, which answers
    /// a call that has no answer yet (after an error, say) with `"aborted"`.
    ///
    /// Fails with the error of a request that cannot be retried (or still fails after
    /// [`RetryConfig::max_retries`] retries), or with [`Error::Tool`] when a tool fails fatally
    /// ([`ToolFailure::Fatal`](crate::tools::ToolFailure::Fatal)); the answers of the calls that
    /// finished before are in the history.
    pub async fn run_with<F>(
        &mut self,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
        on_event: F,
    ) -> Result<RunOutcome>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let span = self.run_span(None);
        self.run_inner(conversation, cancel, on_event, &mut NoSink, &span)
            .instrument(span.clone())
            .await
    }

    /// Run on a [`Session`]: on its [conversation](Session::conversation), as
    /// [`run_with`](Self::run_with) does, appending to the session what the run adds after each
    /// request (the model's items and the answers to its calls, the model settings when they
    /// changed, the usage), so that a run that stops or fails half-way leaves its finished turns
    /// in the session.
    ///
    /// The input goes into the session before the run, with
    /// [`append_input`](Session::append_input) or [`append_item`](Session::append_item). The
    /// spans carry the session's id as `gen_ai.conversation.id`.
    ///
    /// # Errors
    ///
    /// As [`run_with`](Self::run_with), and [`Error::Session`] when the session cannot be
    /// written, such as when another handle opened it since
    /// ([`SessionError::Superseded`](crate::session::SessionError::Superseded)).
    pub async fn run_session<S, F>(
        &mut self,
        session: &mut Session<S>,
        cancel: &CancellationSignal,
        on_event: F,
    ) -> Result<RunOutcome>
    where
        S: SessionStore,
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let mut conversation = session.conversation();
        let span = self.run_span(Some(session.id().as_str()));
        let mut sink = SessionSink {
            synced: conversation.len(),
            session,
        };
        self.run_inner(&mut conversation, cancel, on_event, &mut sink, &span)
            .instrument(span.clone())
            .await
    }

    /// The `invoke_agent` span of a run.
    fn run_span(&self, conversation: Option<&str>) -> tracing::Span {
        let info = self.model.info();
        let name = self.state.config.name.as_deref().unwrap_or_default();
        let origin = info.origin.as_ref();
        tracing::info_span!(
            "invoke_agent",
            otel.name = trace::span_name("invoke_agent", name),
            otel.kind = "internal",
            otel.status_code = Empty,
            gen_ai.operation.name = "invoke_agent",
            gen_ai.agent.name = (!name.is_empty()).then_some(name),
            gen_ai.provider.name = origin.map(|origin| origin.provider.as_str()),
            gen_ai.request.model = origin.map(|origin| origin.model.as_str()),
            gen_ai.conversation.id = conversation,
            gen_ai.usage.input_tokens = Empty,
            gen_ai.usage.output_tokens = Empty,
            error.type = Empty,
        )
    }

    /// The settings the session records for this run's requests.
    fn turn_settings(&self) -> TurnSettings {
        TurnSettings::new(self.model.info().origin, self.state.config.model_settings)
    }

    async fn run_inner<F, K>(
        &mut self,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
        mut on_event: F,
        sink: &mut K,
        span: &tracing::Span,
    ) -> Result<RunOutcome>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
        K: Sink,
    {
        let registry = Arc::clone(&self.state.registry);
        let settings = self.turn_settings();
        let mut turns = Vec::new();
        let mut iteration = 0_u32;
        let mut nudged = false;
        self.state.failures.clear();
        let result = loop {
            if cancel.is_cancelled() {
                break Ok(RunOutcome::new(RunStop::Cancelled, turns));
            }
            if self
                .state
                .config
                .max_iterations
                .is_some_and(|max| iteration >= max.get())
            {
                break Ok(RunOutcome::new(RunStop::MaxIterations, turns));
            }
            for item in self.state.input.take_steering() {
                on_event(RunEvent::Input(&item));
                conversation.push(item);
            }
            if self.state.over_threshold(conversation, &self.model.info()) {
                match self.compact_with(conversation, cancel, &mut on_event).await {
                    Ok(true) => {}
                    Ok(false) => break Ok(RunOutcome::new(RunStop::Cancelled, turns)),
                    // The request may still fit; if it does not, the model says so and the
                    // compaction is tried once more then.
                    Err(error) => tracing::warn!(error = %error, "the compaction failed"),
                }
            }
            iteration += 1;
            let mut turn = TurnRecord::new(iteration);
            let mut dispatcher =
                Dispatcher::new(&registry, span.clone(), self.state.config.trace_content);
            let end = self
                .turn(
                    conversation,
                    &mut dispatcher,
                    &mut turn,
                    cancel,
                    &mut on_event,
                )
                .await;
            let end = match end {
                Ok(end) => end,
                Err(error) => {
                    let answers = dispatcher.finished_answers();
                    self.state
                        .record_outputs(conversation, answers, &self.model.info());
                    break Err(error);
                }
            };
            match end {
                AttemptEnd::Cancelled => {
                    let answers = dispatcher.abort();
                    self.state
                        .record_outputs(conversation, answers, &self.model.info());
                    if self.state.config.context.note_interruptions {
                        conversation.push(context::note(INTERRUPTED_NOTE));
                    }
                    turns.push(turn);
                    break Ok(RunOutcome::new(RunStop::Cancelled, turns));
                }
                AttemptEnd::Failed(failure) => break Err(Error::Model(failure)),
                AttemptEnd::Done { incomplete, calls } => {
                    sink.commit(conversation, turn.usage, &settings).await?;
                    on_event(RunEvent::TurnDone(&turn));
                    let text_call = calls == 0
                        && incomplete.is_none()
                        && self.state.config.context.nudge_text_calls
                        && !nudged
                        && context::looks_like_text_call(&turn.text, |name| {
                            registry.has_tool_named(name)
                        });
                    turns.push(turn);
                    match incomplete {
                        Some(_) => {
                            break Ok(RunOutcome::new(
                                RunStop::Incomplete { reason: incomplete },
                                turns,
                            ));
                        }
                        None if text_call => {
                            // Never run from the text: one more request, to call the tool.
                            nudged = true;
                            conversation.push(context::note(TEXT_CALL_NOTE));
                        }
                        None if calls == 0 => {
                            let mut more = self.state.input.take_steering();
                            more.extend(self.state.input.take_follow_ups());
                            if more.is_empty() {
                                break Ok(RunOutcome::new(RunStop::Completed, turns));
                            }
                            for item in more {
                                on_event(RunEvent::Input(&item));
                                conversation.push(item);
                            }
                        }
                        None => {}
                    }
                }
            }
        };
        // What the last turn added when it stopped early (cancelled, failed) goes into the
        // session too; a failure to write it does not hide the run's own error.
        let committed = sink.commit(conversation, None, &settings).await;
        let result = match (result, committed) {
            (Ok(outcome), Ok(())) => Ok(outcome),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Ok(())) => Err(error),
            (Err(error), Err(lost)) => {
                tracing::warn!(error = %lost, "the session could not be written after a failed run");
                Err(error)
            }
        };
        match &result {
            Ok(outcome) => {
                let (input, output) = outcome.turns.iter().filter_map(|turn| turn.usage).fold(
                    (0_i64, 0_i64),
                    |(input, output), usage| {
                        (input + usage.input_tokens, output + usage.output_tokens)
                    },
                );
                span.record("gen_ai.usage.input_tokens", input);
                span.record("gen_ai.usage.output_tokens", output);
            }
            Err(error) => {
                span.record("otel.status_code", "ERROR");
                span.record("error.type", error_type(error));
            }
        }
        result
    }

    /// One model request, with its retries, and its tool calls.
    async fn turn<F>(
        &mut self,
        conversation: &mut Conversation,
        dispatcher: &mut Dispatcher<'_>,
        turn: &mut TurnRecord,
        cancel: &CancellationSignal,
        on_event: &mut F,
    ) -> Result<AttemptEnd>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let mut attempt = 0_u32;
        let mut compacted = false;
        loop {
            on_event(RunEvent::Request {
                iteration: turn.iteration,
                attempt,
            });
            let end = self
                .attempt(conversation, dispatcher, turn, cancel, on_event)
                .await?;
            if matches!(end, AttemptEnd::Cancelled) {
                return Ok(end);
            }
            // The calls of this attempt (a failed one included) are answered before anything
            // else: the next request carries them.
            if !self
                .drain(conversation, dispatcher, cancel, on_event)
                .await?
            {
                return Ok(AttemptEnd::Cancelled);
            }
            let AttemptEnd::Failed(failure) = end else {
                return Ok(end);
            };
            if failure.kind() == crate::error::ErrorKind::ContextOverflow
                && self.state.config.context.compact_on_overflow
                && !compacted
            {
                compacted = true;
                tracing::info!(error = %failure, "the context is too long: compacting");
                if !self.compact_with(conversation, cancel, on_event).await? {
                    return Ok(AttemptEnd::Cancelled);
                }
                continue;
            }
            let class = failure.retry_class();
            // A failure before the body was already retried by the provider's own HTTP retries,
            // when it has them: retrying it here as well would multiply the requests.
            let retried_by_provider = failure.is_before_body() && self.model.retries_before_body();
            if !class.is_retryable()
                || retried_by_provider
                || attempt >= self.state.config.retry.max_retries
            {
                return Ok(AttemptEnd::Failed(failure));
            }
            attempt += 1;
            turn.retries = attempt;
            let retry = self.state.config.retry;
            let delay = failure
                .retry_after()
                .or(match class {
                    RetryClass::ServerDirected { after } => Some(after),
                    _ => None,
                })
                .map_or_else(
                    || retry.backoff(attempt, random_jitter()),
                    |after| after.min(retry.max_delay),
                );
            on_event(RunEvent::Retrying {
                attempt,
                max: retry.max_retries,
                delay,
                error: &failure,
            });
            tracing::info!(
                attempt,
                max = retry.max_retries,
                delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                error.type = failure.kind().as_str(),
                error = %failure,
                "retrying the request"
            );
            let slept =
                futures::future::select(Box::pin(sleep(delay)), Box::pin(cancel.cancelled())).await;
            if matches!(slept, futures::future::Either::Right(_)) {
                return Ok(AttemptEnd::Cancelled);
            }
        }
    }

    /// Wait for the running calls and record their answers in call order. `false` when
    /// cancelled (the dispatcher keeps its calls, for [`Dispatcher::abort`]).
    async fn drain<F>(
        &mut self,
        conversation: &mut Conversation,
        dispatcher: &mut Dispatcher<'_>,
        cancel: &CancellationSignal,
        on_event: &mut F,
    ) -> Result<bool>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        while !dispatcher.is_idle() {
            let step = {
                let finished = dispatcher.finished();
                let cancelled = cancel.cancelled();
                futures::pin_mut!(finished, cancelled);
                match futures::future::select(finished, cancelled).await {
                    futures::future::Either::Left((finished, _)) => Some(finished),
                    futures::future::Either::Right(_) => None,
                }
            };
            match step {
                None => return Ok(false),
                Some(finished) => {
                    let index = finished.map_err(Error::from)?;
                    if let Some(answer) = dispatcher.answer(index) {
                        on_event(RunEvent::ToolAnswered(answer));
                    }
                }
            }
        }
        let answers = dispatcher
            .take()
            .into_iter()
            .filter_map(|(_, answer)| answer)
            .collect();
        self.state
            .record_outputs(conversation, answers, &self.model.info());
        Ok(true)
    }

    /// [`compact`](Self::compact), with the run's events.
    async fn compact_with<F>(
        &mut self,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
        on_event: &mut F,
    ) -> Result<bool>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let info = self.model.info();
        let tokens_before = self.state.context_tokens(conversation);
        let model = info
            .origin
            .as_ref()
            .map(|origin| origin.model.as_str())
            .unwrap_or_default();
        let span = tracing::info_span!(
            "compact",
            otel.name = trace::span_name("compact", model),
            otel.status_code = Empty,
            gen_ai.operation.name = "compact",
            gen_ai.request.model = (!model.is_empty()).then_some(model),
            gen_ai.usage.input_tokens = Empty,
            gen_ai.usage.output_tokens = Empty,
            genai_agents.tokens_before = tokens_before,
            genai_agents.tokens_after = Empty,
            error.type = Empty,
        );
        let summary = self
            .summarise(conversation, &info, cancel)
            .instrument(span.clone())
            .await;
        let summary = match summary {
            Ok(Some(summary)) => summary,
            Ok(None) => return Ok(false),
            Err(error) => {
                span.record("otel.status_code", "ERROR");
                span.record("error.type", error_type(&error));
                return Err(error);
            }
        };
        let active = conversation.active_items();
        let (kept, context) =
            context::kept_items(&active, self.state.config.context.keep_user_tokens);
        let mut meta = ItemMeta::new().with_turn(conversation.next_turn());
        meta.origin.clone_from(&info.origin);
        conversation.push(
            Compaction::new(summary)
                .with_kept(kept)
                .with_context(context)
                .with_tokens_before(tokens_before)
                .with_meta(meta),
        );
        self.state.accounted = None;
        self.state.prune_boundary = 0;
        let tokens_after = self.state.context_tokens(conversation);
        span.record("genai_agents.tokens_after", tokens_after);
        on_event(RunEvent::Compacted {
            tokens_before,
            tokens_after,
        });
        Ok(true)
    }

    /// The model's summary of the conversation; `None` when cancelled. A request too long even
    /// for that is sent again without its oldest quarter, up to three times.
    async fn summarise(
        &self,
        conversation: &Conversation,
        info: &ModelInfo,
        cancel: &CancellationSignal,
    ) -> Result<Option<String>> {
        let config = &self.state.config;
        let mut items = conversation.for_request(info);
        let prompt = context::note(&config.context.compaction_prompt);
        let settings = config
            .context
            .compaction_settings
            .unwrap_or(config.model_settings);
        let mut trimmed = 0;
        loop {
            let mut request_items = items.items().to_vec();
            request_items.push(prompt.clone());
            let request = ModelRequest::new(&request_items)
                .with_settings(settings)
                .with_prompt_cache_key(self.state.prompt_cache_key());
            let outcome = {
                let read = read_summary(&self.model, request);
                let cancelled = cancel.cancelled();
                futures::pin_mut!(read, cancelled);
                match futures::future::select(read, cancelled).await {
                    futures::future::Either::Left((outcome, _)) => outcome,
                    futures::future::Either::Right(_) => return Ok(None),
                }
            };
            match outcome {
                Ok(summary) => return Ok(Some(summary)),
                Err(Error::Model(failure))
                    if failure.kind() == crate::error::ErrorKind::ContextOverflow
                        && trimmed < 3
                        && items.len() > 1 =>
                {
                    trimmed += 1;
                    let target = items.len() - items.len() / 4;
                    while items.len() > target.max(1) {
                        items.remove(0);
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// One request, in its `chat` span.
    async fn attempt<F>(
        &mut self,
        conversation: &mut Conversation,
        dispatcher: &mut Dispatcher<'_>,
        turn: &mut TurnRecord,
        cancel: &CancellationSignal,
        on_event: &mut F,
    ) -> Result<AttemptEnd>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let info = self.model.info();
        let origin = info.origin.as_ref();
        let model = origin
            .map(|origin| origin.model.as_str())
            .unwrap_or_default();
        let settings = self.state.config.model_settings;
        let span = tracing::info_span!(
            "chat",
            otel.name = trace::span_name("chat", model),
            otel.kind = "client",
            otel.status_code = Empty,
            gen_ai.operation.name = "chat",
            gen_ai.provider.name = origin.map(|origin| origin.provider.as_str()),
            gen_ai.request.model = (!model.is_empty()).then_some(model),
            gen_ai.request.max_tokens = settings.max_output_tokens,
            gen_ai.request.reasoning = settings.reasoning.map(ReasoningLevel::as_str),
            gen_ai.response.id = Empty,
            gen_ai.response.finish_reasons = Empty,
            gen_ai.usage.input_tokens = Empty,
            gen_ai.usage.output_tokens = Empty,
            gen_ai.usage.cache_read.input_tokens = Empty,
            gen_ai.usage.cache_creation.input_tokens = Empty,
            gen_ai.usage.reasoning.output_tokens = Empty,
            gen_ai.input.messages = Empty,
            gen_ai.output.messages = Empty,
            genai_agents.iteration = turn.iteration,
            genai_agents.retry = turn.retries,
            genai_agents.time_to_first_token_ms = Empty,
            genai_agents.cache_break = Empty,
            error.type = Empty,
        );
        let before = conversation.len();
        let end = self
            .attempt_inner(conversation, dispatcher, turn, cancel, on_event)
            .instrument(span.clone())
            .await;
        if let Some(id) = &turn.response_id {
            span.record("gen_ai.response.id", id.as_str());
        }
        if let Some(stop) = &turn.stop {
            span.record("gen_ai.response.finish_reasons", stop_name(stop));
        }
        if let Some(usage) = turn.usage {
            span.record("gen_ai.usage.input_tokens", usage.input_tokens);
            span.record("gen_ai.usage.output_tokens", usage.output_tokens);
            if let Some(tokens) = usage.cache_read_tokens {
                span.record("gen_ai.usage.cache_read.input_tokens", tokens);
            }
            if let Some(tokens) = usage.cache_write_tokens {
                span.record("gen_ai.usage.cache_creation.input_tokens", tokens);
            }
            if let Some(tokens) = usage.reasoning_tokens {
                span.record("gen_ai.usage.reasoning.output_tokens", tokens);
            }
        }
        if self.state.config.trace_content {
            let output: Vec<Item> = conversation.items()[before.min(conversation.len())..]
                .iter()
                .filter(|item| !matches!(item, Item::ToolOutput(_)))
                .cloned()
                .collect();
            span.record("gen_ai.output.messages", trace::messages(&output));
        }
        let failure = match &end {
            Ok(AttemptEnd::Failed(failure)) => Some(failure.kind().as_str()),
            Ok(AttemptEnd::Cancelled) => Some("cancelled"),
            Err(error) => Some(error_type(error)),
            Ok(AttemptEnd::Done { .. }) => None,
        };
        if let Some(kind) = failure {
            span.record("otel.status_code", "ERROR");
            span.record("error.type", kind);
        }
        end
    }

    /// One request: sent, its events read, its calls started as they come.
    async fn attempt_inner<F>(
        &mut self,
        conversation: &mut Conversation,
        dispatcher: &mut Dispatcher<'_>,
        turn: &mut TurnRecord,
        cancel: &CancellationSignal,
        on_event: &mut F,
    ) -> Result<AttemptEnd>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let model = &self.model;
        let state = &mut self.state;
        let caps = model.info();
        let tools = state.definitions(&caps);
        let mut projection = conversation.for_request(&caps);
        let now = Instant::now();
        if let Some(prune) = &state.config.context.prune {
            // Pruning changes the start of the request: only once the cache expired anyway.
            if state
                .last_request_at
                .is_some_and(|at| now.duration_since(at) >= prune.cache_ttl)
            {
                state.prune_boundary =
                    context::outputs(projection.items()).saturating_sub(prune.keep_recent);
            }
            context::prune(projection.items_mut(), state.prune_boundary, prune);
        }
        state.last_request_at = Some(now);
        let shape = trace::RequestShape::new(
            caps.origin
                .as_ref()
                .map_or("", |origin| origin.model.as_str()),
            &tools,
            state.config.model_settings,
            state.prompt_cache_key(),
            projection.items(),
        );
        if turn.retries == 0 {
            turn.cache_break = state
                .last_shape
                .as_ref()
                .and_then(|previous| shape.break_from(previous));
            if let Some(cache_break) = turn.cache_break {
                tracing::Span::current().record("genai_agents.cache_break", cache_break.as_str());
            }
        }
        state.last_shape = Some(shape);
        if state.config.trace_content {
            tracing::Span::current()
                .record("gen_ai.input.messages", trace::messages(projection.items()));
        }
        let started = Instant::now();
        let mut first_token = true;
        let meta = ItemMeta {
            turn: Some(conversation.next_turn()),
            ..ItemMeta::default()
        };
        let sent = {
            let request = ModelRequest::new(projection.items())
                .with_tools(&tools)
                .with_parallel_tool_calls(state.config.parallel_tool_calls)
                .with_prompt_cache_key(state.prompt_cache_key())
                .with_settings(state.config.model_settings);
            let call = model.stream(request);
            let cancelled = cancel.cancelled();
            futures::pin_mut!(call, cancelled);
            match futures::future::select(call, cancelled).await {
                futures::future::Either::Left((sent, _)) => sent,
                futures::future::Either::Right(_) => return Ok(AttemptEnd::Cancelled),
            }
        };
        let ModelStream {
            rate_limits,
            mut events,
        } = match sent {
            Ok(stream) => stream,
            Err(failure) => return Ok(AttemptEnd::Failed(failure)),
        };
        turn.rate_limits = rate_limits.or(turn.rate_limits);
        turn.text.clear();
        let mut calls = 0;
        // The text written since the last complete assistant message: kept, cut off, when the
        // response does not finish.
        let mut partial = String::new();
        let keep_partial =
            |partial: &mut String, conversation: &mut Conversation, turn: &mut TurnRecord| {
                if partial.is_empty() {
                    return;
                }
                let text = std::mem::take(partial);
                turn.text.push_str(&text);
                let mut cut = meta.clone().with_interrupted(true);
                cut.origin.clone_from(&caps.origin);
                conversation.push(Message::text(Role::Assistant, text).with_meta(cut));
            };
        let cancelled = cancel.cancelled();
        futures::pin_mut!(cancelled);
        loop {
            // Cancellation first, then the calls that finished, then the response.
            let step = futures::future::poll_fn(|cx| {
                if cancelled.as_mut().poll(cx).is_ready() {
                    return Poll::Ready(Step::Cancelled);
                }
                if let Poll::Ready(finished) = dispatcher.poll_finished(cx) {
                    return Poll::Ready(Step::Answered(finished));
                }
                events.as_mut().poll_next(cx).map(Step::Event)
            })
            .await;
            let event = match step {
                Step::Cancelled => {
                    keep_partial(&mut partial, conversation, turn);
                    return Ok(AttemptEnd::Cancelled);
                }
                Step::Answered(finished) => {
                    let index = finished.map_err(Error::from)?;
                    if let Some(answer) = dispatcher.answer(index) {
                        on_event(RunEvent::ToolAnswered(answer));
                    }
                    continue;
                }
                Step::Event(None) => {
                    keep_partial(&mut partial, conversation, turn);
                    return Ok(AttemptEnd::Failed(ModelError::stream_ended()));
                }
                Step::Event(Some(Err(error))) => {
                    keep_partial(&mut partial, conversation, turn);
                    return Ok(AttemptEnd::Failed(error));
                }
                Step::Event(Some(Ok(event))) => event,
            };
            on_event(RunEvent::Model(&event));
            if first_token
                && matches!(
                    event,
                    ModelEvent::TextDelta(_)
                        | ModelEvent::ReasoningDelta(_)
                        | ModelEvent::ToolCallArgsDelta { .. }
                        | ModelEvent::ItemDone(_)
                )
            {
                first_token = false;
                let elapsed = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                tracing::Span::current().record("genai_agents.time_to_first_token_ms", elapsed);
            }
            match event {
                ModelEvent::TextDelta(delta) => partial.push_str(&delta),
                ModelEvent::ItemDone(mut item) => {
                    if matches!(&item, Item::Message(message) if message.role == Role::Assistant) {
                        partial.clear();
                    }
                    let item_meta = item.meta_mut();
                    if item_meta.turn.is_none() {
                        item_meta.turn = meta.turn;
                    }
                    if item_meta.origin.is_none() {
                        item_meta.origin.clone_from(&caps.origin);
                    }
                    calls += state.take_item(item, &caps, conversation, dispatcher, turn, on_event);
                }
                ModelEvent::ToolsLoaded(keys) => {
                    for key in keys {
                        if state.registry.exposure(&key) == Some(Exposure::Deferred) {
                            state.loaded.insert(key);
                        }
                    }
                }
                ModelEvent::Completed(response) => {
                    if response.stop == Some(StopReason::Length) {
                        // Calls of a response cut off at the output limit are not run, except
                        // those that started already (their arguments were complete).
                        dispatcher
                            .answer_unstarted(|call| ToolOutput::failure(call, CUT_OFF_OUTPUT));
                    }
                    turn.describe(&response);
                    if let Some(usage) = response.usage {
                        let tokens = usage.input_tokens.saturating_add(usage.output_tokens);
                        state.accounted = Some((
                            conversation.len(),
                            u64::try_from(tokens).unwrap_or_default(),
                        ));
                    }
                    return Ok(AttemptEnd::Done {
                        incomplete: response.incomplete,
                        calls,
                    });
                }
                _ => {}
            }
        }
    }
}

impl State {
    fn context_tokens(&self, conversation: &Conversation) -> u64 {
        match self.accounted {
            Some((len, tokens))
                if len <= conversation.len()
                    && conversation.latest_compaction().is_none_or(|at| at < len) =>
            {
                tokens + estimate_tokens(&conversation.items()[len..])
            }
            _ => estimate_tokens(&conversation.active_items()),
        }
    }

    /// Whether the next request passes the share of the context window that triggers a
    /// compaction, and there is something to compact since the last one.
    fn over_threshold(&self, conversation: &Conversation, info: &ModelInfo) -> bool {
        let (Some(percent), Some(window)) = (self.config.context.compact_at, info.context_window)
        else {
            return false;
        };
        let since = conversation
            .latest_compaction()
            .map_or(conversation.len(), |at| conversation.len() - at - 1);
        since > 0
            && self.context_tokens(conversation) * 100 >= u64::from(window) * u64::from(percent)
    }

    fn prompt_cache_key(&self) -> Option<&str> {
        match &self.config.prompt_cache_key {
            PromptCacheKey::PerRunner => Some(&self.cache_key),
            PromptCacheKey::Custom(key) => Some(key),
            PromptCacheKey::Omit => None,
        }
    }

    /// Append answers to the history, cut to their tools' budgets (and as a failure when they
    /// hold images or files the model does not take).
    fn record_outputs(
        &mut self,
        conversation: &mut Conversation,
        answers: Vec<ToolOutput>,
        caps: &ModelInfo,
    ) {
        if answers.is_empty() {
            return;
        }
        let truncation = self.config.truncation;
        let calls: std::collections::BTreeMap<String, ToolCall> = conversation
            .calls()
            .into_iter()
            .map(|call| (call.call_id.clone(), call))
            .collect();
        for answer in answers {
            let call = calls.get(&answer.call_id);
            if let Some(call) = call {
                let key = failure_key(call);
                if answer.success {
                    self.failures.remove(&key);
                } else {
                    *self.failures.entry(key).or_default() += 1;
                }
            }
            let budget = call.and_then(|call| self.registry.output_budget(call));
            let mut answer = answer.truncated(truncation.budget_for(budget), truncation.header);
            if let (false, Some(call)) = (caps.rich_tool_outputs, call) {
                if answer.body.to_text().is_none() {
                    let error = ToolError::UnsupportedOutput {
                        reason: "the model takes only text in tool outputs".into(),
                    };
                    answer = ToolOutput::failure(call, error.to_string());
                }
            }
            conversation.push_output(&answer);
        }
    }

    /// Queue a call of the model (or answer it, for the runner's own tool search).
    fn dispatch<F>(
        &mut self,
        call: ToolCall,
        caps: &ModelInfo,
        dispatcher: &mut Dispatcher<'_>,
        turn: &mut TurnRecord,
        on_event: &mut F,
    ) where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        on_event(RunEvent::ToolCalled(&call));
        turn.calls.push(call.clone());
        let max = self.config.context.max_repeated_failures;
        if max > 0
            && self
                .failures
                .get(&failure_key(&call))
                .is_some_and(|&failed| failed >= max)
        {
            let answer = ToolOutput::failure(&call, REPEATED_FAILURE);
            on_event(RunEvent::ToolAnswered(&answer));
            dispatcher.push_answered(call, answer);
            return;
        }
        if self.is_own_search(&call, caps) {
            let answer = self.search(&call, caps);
            on_event(RunEvent::ToolAnswered(&answer));
            dispatcher.push_answered(call, answer);
            return;
        }
        let parallel = self.config.parallel_tool_calls && self.registry.supports_parallel(&call);
        dispatcher.push(call, parallel);
    }

    /// Record an output item of a response, and dispatch it when it is a call. Returns the
    /// number of calls (0 or 1).
    fn take_item<F>(
        &mut self,
        item: Item,
        caps: &ModelInfo,
        conversation: &mut Conversation,
        dispatcher: &mut Dispatcher<'_>,
        turn: &mut TurnRecord,
        on_event: &mut F,
    ) -> usize
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        match item {
            Item::ToolCall(call) if call.meta.interrupted => {
                // The response stopped in the middle of it: not run.
                conversation.push_call(&call);
                on_event(RunEvent::ToolCalled(&call));
                turn.calls.push(call.clone());
                let answer = ToolOutput::failure(&call, CUT_OFF_OUTPUT);
                on_event(RunEvent::ToolAnswered(&answer));
                dispatcher.push_answered(call, answer);
                1
            }
            Item::ToolCall(call) => {
                conversation.push_call(&call);
                self.dispatch(call, caps, dispatcher, turn, on_event);
                1
            }
            Item::Message(message) => {
                if message.role == Role::Assistant {
                    turn.text.push_str(&message.text_content());
                }
                conversation.push(message);
                0
            }
            other => {
                conversation.push(other);
                0
            }
        }
    }

    /// The tools of a request: the direct ones and the loaded deferred ones (or every deferred
    /// one, for a model that searches tools itself), and the tool search.
    fn definitions(&self, caps: &ModelInfo) -> Vec<ToolDefinition> {
        let native = caps.tool_search;
        let mut tools = self.registry.definitions_where(|key, exposure| {
            if !self.fits(key, caps) {
                return None;
            }
            match exposure {
                Exposure::Direct => Some(Advertise::Plain),
                Exposure::Deferred if native => Some(Advertise::Deferred),
                Exposure::Deferred if self.loaded.contains(key) => Some(Advertise::Plain),
                _ => None,
            }
        });
        if native && self.searchable(caps).next().is_some() {
            tools.push(ToolDefinition::ToolSearch);
        }
        if self.offers_own_search(caps) {
            if let Some(config) = &self.config.tool_search {
                tools.push(ToolDefinition::Function {
                    name: TOOL_SEARCH.into(),
                    description: config.description.clone(),
                    parameters: search_schema(),
                    defer_loading: false,
                });
            }
        }
        tools
    }

    /// Whether the model can take the tool: namespaced and custom tools only when it takes
    /// them.
    fn fits(&self, key: &ToolKey, caps: &ModelInfo) -> bool {
        (caps.namespaces || key.namespace.is_none())
            && (caps.custom_tools || self.registry.kind(key) != Some(&ToolKind::Custom))
    }

    /// Whether the runner answers this call itself: its tool search, when no tool of that name is
    /// registered.
    fn is_own_search(&self, call: &ToolCall, caps: &ModelInfo) -> bool {
        call.name == TOOL_SEARCH
            && call.namespace.is_none()
            && call.kind == ToolKind::Function
            && self.offers_own_search(caps)
    }

    /// Whether the runner offers its own `tool_search` function.
    fn offers_own_search(&self, caps: &ModelInfo) -> bool {
        self.config.tool_search.is_some()
            && !caps.tool_search
            && self.registry.exposure(&ToolKey::new(TOOL_SEARCH)).is_none()
            && self.searchable(caps).next().is_some()
    }

    /// The deferred tools a search can find for this model.
    fn searchable<'s>(&'s self, caps: &'s ModelInfo) -> impl Iterator<Item = &'s ToolKey> + 's {
        self.registry.keys().filter(move |key| {
            self.registry.exposure(key) == Some(Exposure::Deferred) && self.fits(key, caps)
        })
    }

    /// Answer the runner's own tool search: load the deferred tools that match the query.
    fn search(&mut self, call: &ToolCall, caps: &ModelInfo) -> ToolOutput {
        let Some(config) = &self.config.tool_search else {
            return ToolOutput::failure(call, "tool search is off");
        };
        let arguments: Value = serde_json::from_str(&call.input).unwrap_or(Value::Null);
        let Some(query) = arguments.get("query").and_then(Value::as_str) else {
            return ToolOutput::failure(
                call,
                "failed to parse function arguments: a `query` string is required",
            );
        };
        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .and_then(|limit| usize::try_from(limit).ok())
            .map_or(config.limit, |limit| limit.clamp(1, config.limit.max(1)));
        let found: Vec<ToolKey> = self
            .registry
            .search_deferred(query, usize::MAX)
            .into_iter()
            .filter(|key| self.fits(key, caps))
            .take(limit)
            .collect();
        let tools: Vec<Value> = found
            .iter()
            .map(|key| {
                let mut tool = json!({
                    "name": key.name,
                    "description": self.registry.description(key).unwrap_or_default(),
                });
                if let Some(namespace) = &key.namespace {
                    tool["namespace"] = Value::String(namespace.clone());
                }
                tool
            })
            .collect();
        self.loaded.extend(found);
        let message = if tools.is_empty() {
            "No tool matched. Try other words."
        } else {
            "These tools can be called from the next step on."
        };
        ToolOutput::new(
            call,
            json!({"tools": tools, "message": message}).to_string(),
            true,
        )
    }
}

/// The schema of the runner's `tool_search` function.
fn search_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What the tools should do, in a few words"},
            "limit": {"type": "integer", "minimum": 1, "description": "The most tools to load"},
        },
        "required": ["query"],
        "additionalProperties": false,
    })
}

/// Where a run writes what it adds: nowhere ([`Runner::run_with`]) or a session.
trait Sink: MaybeSend {
    /// Write the items added since the last commit, the model settings when they changed, and
    /// the usage of a response.
    fn commit(
        &mut self,
        conversation: &Conversation,
        usage: Option<Usage>,
        settings: &TurnSettings,
    ) -> impl Future<Output = Result<()>> + MaybeSend;
}

struct NoSink;

impl Sink for NoSink {
    async fn commit(&mut self, _: &Conversation, _: Option<Usage>, _: &TurnSettings) -> Result<()> {
        Ok(())
    }
}

struct SessionSink<'s, S> {
    session: &'s mut Session<S>,
    /// The items of the conversation the session has.
    synced: usize,
}

impl<S: SessionStore> Sink for SessionSink<'_, S> {
    async fn commit(
        &mut self,
        conversation: &Conversation,
        usage: Option<Usage>,
        settings: &TurnSettings,
    ) -> Result<()> {
        let items = conversation.items().get(self.synced..).unwrap_or_default();
        if items.is_empty() && usage.is_none() {
            return Ok(());
        }
        let mut entries = Vec::with_capacity(items.len() + 2);
        if !items.is_empty() && self.session.turn_settings() != Some(settings) {
            entries.push(SessionEntry::TurnSettings(settings.clone()));
        }
        entries.extend(items.iter().cloned().map(SessionEntry::Item));
        entries.extend(usage.map(SessionEntry::Usage));
        self.session.append_all(entries).await?;
        self.synced = conversation.len();
        Ok(())
    }
}

/// The `error.type` of a run's error.
fn error_type(error: &Error) -> &'static str {
    match error {
        Error::Model(failure) => failure.kind().as_str(),
        Error::Tool(_) => "tool",
        Error::Session(_) => "session",
        Error::Compaction(_) => "compaction",
        Error::Json(_) => "json",
        _ => "other",
    }
}

/// A stop reason as `gen_ai.response.finish_reasons` names it.
fn stop_name(stop: &StopReason) -> &str {
    match stop {
        StopReason::Stop => "stop",
        StopReason::ToolUse => "tool_calls",
        StopReason::Length => "length",
        StopReason::ContentFilter => "content_filter",
        StopReason::Aborted => "aborted",
        StopReason::Other(other) => other,
    }
}

/// The key of a call for the repeated-failure guard: its tool and a hash of its arguments.
fn failure_key(call: &ToolCall) -> (Option<String>, String, u64) {
    (
        call.namespace.clone(),
        call.name.clone(),
        fnv1a(call.input.as_bytes()),
    )
}

/// Send a compaction request and read the summary from its response.
async fn read_summary<M: Model>(model: &M, request: ModelRequest<'_>) -> Result<String> {
    let ModelStream { mut events, .. } = model.stream(request).await?;
    let mut streamed = String::new();
    let mut written = String::new();
    while let Some(event) = futures::StreamExt::next(&mut events).await {
        match event? {
            ModelEvent::TextDelta(delta) => streamed.push_str(&delta),
            ModelEvent::ItemDone(Item::Message(message)) if message.role == Role::Assistant => {
                written.push_str(&message.text_content());
            }
            ModelEvent::Completed(response) => {
                if response.stop == Some(StopReason::Length) || response.incomplete.is_some() {
                    return Err(Error::Compaction(
                        "the summary was cut off before it was complete".into(),
                    ));
                }
                if let Some(usage) = response.usage {
                    let span = tracing::Span::current();
                    span.record("gen_ai.usage.input_tokens", usage.input_tokens);
                    span.record("gen_ai.usage.output_tokens", usage.output_tokens);
                }
                let summary = if written.is_empty() {
                    streamed
                } else {
                    written
                };
                let summary = summary.trim();
                if summary.is_empty() {
                    return Err(Error::Compaction("the summary is empty".into()));
                }
                return Ok(summary.to_owned());
            }
            _ => {}
        }
    }
    Err(Error::Model(ModelError::stream_ended()))
}
