//! The runner: [`Runner`] sends a [`Conversation`] to a [`Model`], runs the tools it calls, sends
//! their outputs back, and repeats until the model answers.
//!
//! It:
//!
//! - repairs the history before every request ([`Conversation::normalize`]): every call
//!   answered, orphaned outputs dropped;
//! - answers every call ([`ToolRegistry::answer_call`]), runs the calls of tools that
//!   [allow it](crate::tools::FunctionTool::supports_parallel) at the same time and the others
//!   alone, and records the answers in call order, cut to their
//!   [budget](crate::tools::OutputBudget);
//! - starts a call as soon as the model finished writing it ([`ModelEvent::ItemDone`]), while
//!   the response goes on;
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
mod dispatch;

use std::collections::BTreeSet;
use std::future::Future;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use serde_json::{json, Value};

pub use cancel::CancellationSignal;
pub use config::{PromptCacheKey, RetryConfig, RunConfig, ToolSearchConfig};

use crate::conversation::Conversation;
use crate::error::{Error, ModelError, Result, RetryClass};
use crate::item::{Item, Role, ToolCall, ToolKind, ToolOutput};
use crate::model::{
    Model, ModelEvent, ModelInfo, ModelRequest, ModelResponse, ModelStream, RateLimits, StopReason,
    ToolDefinition, Usage,
};
use crate::platform::{random_jitter, random_u64, sleep, MaybeSend};
use crate::tools::{Advertise, Exposure, ToolError, ToolKey, ToolRegistry};
use dispatch::Dispatcher;

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
    /// The model's output items and the answers to its calls are appended to `conversation`,
    /// which stays consistent whatever happens: after an error too, every call it holds has an
    /// answer or gets one ("aborted") at the next [`normalize`](Conversation::normalize).
    ///
    /// Fails with the error of a request that cannot be retried (or still fails after
    /// [`RetryConfig::max_retries`] retries), or with [`Error::Tool`] when a tool fails fatally
    /// ([`ToolFailure::Fatal`](crate::tools::ToolFailure::Fatal)); the answers of the calls that
    /// finished before are in the history.
    pub async fn run_with<F>(
        &mut self,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
        mut on_event: F,
    ) -> Result<RunOutcome>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let registry = Arc::clone(&self.state.registry);
        let mut turns = Vec::new();
        let mut iteration = 0_u32;
        loop {
            if cancel.is_cancelled() {
                conversation.normalize(self.model.info());
                return Ok(RunOutcome::new(RunStop::Cancelled, turns));
            }
            if self
                .state
                .config
                .max_iterations
                .is_some_and(|max| iteration >= max.get())
            {
                return Ok(RunOutcome::new(RunStop::MaxIterations, turns));
            }
            iteration += 1;
            let mut turn = TurnRecord::new(iteration);
            let mut dispatcher = Dispatcher::new(&registry);
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
                        .record_outputs(conversation, answers, self.model.info());
                    return Err(error);
                }
            };
            match end {
                AttemptEnd::Cancelled => {
                    let answers = dispatcher.abort();
                    self.state
                        .record_outputs(conversation, answers, self.model.info());
                    conversation.normalize(self.model.info());
                    turns.push(turn);
                    return Ok(RunOutcome::new(RunStop::Cancelled, turns));
                }
                AttemptEnd::Failed(failure) => return Err(Error::Model(failure)),
                AttemptEnd::Done { incomplete, calls } => {
                    on_event(RunEvent::TurnDone(&turn));
                    turns.push(turn);
                    match incomplete {
                        Some(_) => {
                            return Ok(RunOutcome::new(
                                RunStop::Incomplete { reason: incomplete },
                                turns,
                            ));
                        }
                        None if calls == 0 => {
                            return Ok(RunOutcome::new(RunStop::Completed, turns));
                        }
                        None => {}
                    }
                }
            }
        }
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
        loop {
            conversation.normalize(self.model.info());
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
            tracing::debug!(attempt, ?delay, error = %failure, "retrying the request");
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
        &self,
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
            .record_outputs(conversation, answers, self.model.info());
        Ok(true)
    }

    /// One request: sent, its events read, its calls started as they come.
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
        let model = &self.model;
        let state = &mut self.state;
        let caps = model.info();
        let tools = state.definitions(caps);
        let sent = {
            let request = ModelRequest::new(conversation.items())
                .with_tools(&tools)
                .with_parallel_tool_calls(state.config.parallel_tool_calls)
                .with_prompt_cache_key(state.prompt_cache_key());
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
                Step::Cancelled => return Ok(AttemptEnd::Cancelled),
                Step::Answered(finished) => {
                    let index = finished.map_err(Error::from)?;
                    if let Some(answer) = dispatcher.answer(index) {
                        on_event(RunEvent::ToolAnswered(answer));
                    }
                    continue;
                }
                Step::Event(None) => return Ok(AttemptEnd::Failed(ModelError::stream_ended())),
                Step::Event(Some(Err(error))) => return Ok(AttemptEnd::Failed(error)),
                Step::Event(Some(Ok(event))) => event,
            };
            on_event(RunEvent::Model(&event));
            match event {
                ModelEvent::ItemDone(item) => {
                    calls += state.take_item(item, caps, conversation, dispatcher, turn, on_event);
                }
                ModelEvent::ToolsLoaded(keys) => {
                    for key in keys {
                        if state.registry.exposure(&key) == Some(Exposure::Deferred) {
                            state.loaded.insert(key);
                        }
                    }
                }
                ModelEvent::Completed(response) => {
                    turn.describe(&response);
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
        &self,
        conversation: &mut Conversation,
        answers: Vec<ToolOutput>,
        caps: ModelInfo,
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
        caps: ModelInfo,
        dispatcher: &mut Dispatcher<'_>,
        turn: &mut TurnRecord,
        on_event: &mut F,
    ) where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        on_event(RunEvent::ToolCalled(&call));
        turn.calls.push(call.clone());
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
        caps: ModelInfo,
        conversation: &mut Conversation,
        dispatcher: &mut Dispatcher<'_>,
        turn: &mut TurnRecord,
        on_event: &mut F,
    ) -> usize
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        match item {
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
    fn definitions(&self, caps: ModelInfo) -> Vec<ToolDefinition> {
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
    fn fits(&self, key: &ToolKey, caps: ModelInfo) -> bool {
        (caps.namespaces || key.namespace.is_none())
            && (caps.custom_tools || self.registry.kind(key) != Some(&ToolKind::Custom))
    }

    /// Whether the runner answers this call itself: its tool search, when no tool of that name is
    /// registered.
    fn is_own_search(&self, call: &ToolCall, caps: ModelInfo) -> bool {
        call.name == TOOL_SEARCH
            && call.namespace.is_none()
            && call.kind == ToolKind::Function
            && self.offers_own_search(caps)
    }

    /// Whether the runner offers its own `tool_search` function.
    fn offers_own_search(&self, caps: ModelInfo) -> bool {
        self.config.tool_search.is_some()
            && !caps.tool_search
            && self.registry.exposure(&ToolKey::new(TOOL_SEARCH)).is_none()
            && self.searchable(caps).next().is_some()
    }

    /// The deferred tools a search can find for this model.
    fn searchable(&self, caps: ModelInfo) -> impl Iterator<Item = &ToolKey> {
        self.registry.keys().filter(move |key| {
            self.registry.exposure(key) == Some(Exposure::Deferred) && self.fits(key, caps)
        })
    }

    /// Answer the runner's own tool search: load the deferred tools that match the query.
    fn search(&mut self, call: &ToolCall, caps: ModelInfo) -> ToolOutput {
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
