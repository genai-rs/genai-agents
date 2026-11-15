//! What an eval runs: a target, and what one run of it gave (a [`Trial`]).

use std::future::Future;
use std::time::Duration;

use genai_agents::agent::AgentOutcome;
use genai_agents::{
    BoxFuture, Conversation, Item, MaybeSend, MaybeSync, RunOutcome, RunStop, ToolCall, Usage,
};
use serde::{Deserialize, Serialize};

use crate::case::Case;

/// What a target's run on a case gave.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Run {
    /// The conversation after the run: the case's input and what the run added.
    pub conversation: Conversation,
    /// How the run went.
    pub outcome: RunOutcome,
    /// The answer: the text of a completed run, or the JSON of an agent's structured answer.
    pub output: Option<String>,
}

impl Run {
    /// A run of a [`Runner`](genai_agents::Runner): the answer is the last response's text
    /// when the model answered.
    #[must_use]
    pub fn new(conversation: Conversation, outcome: RunOutcome) -> Self {
        let output = (outcome.stop == RunStop::Completed).then(|| outcome.output_text().to_owned());
        Self {
            conversation,
            outcome,
            output,
        }
    }

    /// A run of an [`Agent`](genai_agents::agent::Agent): the answer is its output, as JSON
    /// unless it is a text.
    #[must_use]
    pub fn from_agent<O: Serialize>(conversation: Conversation, outcome: AgentOutcome<O>) -> Self {
        let output = outcome
            .output
            .as_ref()
            .map(|output| match serde_json::to_value(output) {
                Ok(serde_json::Value::String(text)) => text,
                Ok(value) => value.to_string(),
                Err(error) => format!("<the output did not encode: {error}>"),
            });
        Self {
            conversation,
            outcome: outcome.run,
            output,
        }
    }
}

/// What an eval runs on each case: an agent, a runner, a pipeline of the application's.
///
/// [`target_fn`] makes one from an async closure. Each call starts from nothing (a new agent,
/// a new runner), so that the trials of a case are independent.
pub trait Target: MaybeSend + MaybeSync {
    /// Run on a case.
    fn run<'a>(&'a self, case: &'a Case) -> BoxFuture<'a, genai_agents::Result<Run>>;
}

/// A [`Target`] from a closure ([`target_fn`]).
#[derive(Debug, Clone, Copy)]
pub struct FnTarget<F>(F);

/// A [`Target`] from an async closure that takes the case by value. The future owns what it
/// uses: clone what it needs (an `Arc` of shared state) into it.
///
/// ```
/// use genai_agents::testing::{Reply, ScriptedModel};
/// use genai_agents::{Runner, ToolRegistry};
/// use genai_agents_eval::{target_fn, Case, Run};
///
/// let target = target_fn(|case: Case| async move {
///     let model = ScriptedModel::replies(vec![Reply::text("Paris")]);
///     let mut runner = Runner::new(model, ToolRegistry::new());
///     let mut conversation = case.input;
///     let outcome = runner.run(&mut conversation).await?;
///     Ok(Run::new(conversation, outcome))
/// });
/// # let _ = target;
/// ```
pub fn target_fn<F, Fut>(f: F) -> FnTarget<F>
where
    F: Fn(Case) -> Fut + MaybeSend + MaybeSync,
    Fut: Future<Output = genai_agents::Result<Run>> + MaybeSend + 'static,
{
    FnTarget(f)
}

impl<F, Fut> Target for FnTarget<F>
where
    F: Fn(Case) -> Fut + MaybeSend + MaybeSync,
    Fut: Future<Output = genai_agents::Result<Run>> + MaybeSend + 'static,
{
    fn run<'a>(&'a self, case: &'a Case) -> BoxFuture<'a, genai_agents::Result<Run>> {
        Box::pin((self.0)(case.clone()))
    }
}

/// One run of a target on a case: what the scorers score.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Trial {
    /// The case's id.
    pub case_id: String,
    /// Which of the case's runs it is, from 0.
    pub attempt: u32,
    /// The run, when the target did not fail.
    pub run: Option<Run>,
    /// The target's error, when it failed.
    pub error: Option<String>,
    /// How long it took.
    pub elapsed: Duration,
}

impl Trial {
    /// The answer, if there is one.
    #[must_use]
    pub fn output(&self) -> Option<&str> {
        self.run.as_ref()?.output.as_deref()
    }

    /// The tool calls the run made, in order (those of the case's input left out).
    #[must_use]
    pub fn calls(&self, case: &Case) -> Vec<&ToolCall> {
        let Some(run) = &self.run else {
            return Vec::new();
        };
        let skip = case.input.len().min(run.conversation.len());
        run.conversation.items()[skip..]
            .iter()
            .filter_map(|item| match item {
                Item::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }

    /// How many model requests the run took.
    #[must_use]
    pub fn requests(&self) -> usize {
        self.run.as_ref().map_or(0, |run| run.outcome.turns.len())
    }

    /// The tokens the run used, added up over its requests.
    #[must_use]
    pub fn usage(&self) -> Usage {
        let mut total = Usage::default();
        for usage in self
            .run
            .iter()
            .flat_map(|run| &run.outcome.turns)
            .filter_map(|turn| turn.usage)
        {
            total.input_tokens += usage.input_tokens;
            total.output_tokens += usage.output_tokens;
        }
        total
    }

    /// Why the run stopped, if it ran.
    #[must_use]
    pub fn stop(&self) -> Option<&RunStop> {
        self.run.as_ref().map(|run| &run.outcome.stop)
    }
}

/// The parts of a trial a report keeps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrialSummary {
    /// The answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// The target's error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Why the run stopped, in words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    /// The model requests.
    pub requests: usize,
    /// The input and output tokens.
    pub tokens: i64,
    /// The time, in milliseconds.
    pub millis: u64,
}

impl From<&Trial> for TrialSummary {
    fn from(trial: &Trial) -> Self {
        let usage = trial.usage();
        Self {
            output: trial.output().map(str::to_owned),
            error: trial.error.clone(),
            stop: trial.stop().map(|stop| format!("{stop:?}")),
            requests: trial.requests(),
            tokens: usage.input_tokens + usage.output_tokens,
            millis: u64::try_from(trial.elapsed.as_millis()).unwrap_or(u64::MAX),
        }
    }
}
