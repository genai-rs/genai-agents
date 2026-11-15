//! Scorers: each gives a trial a number from 0 to 1, with a reason.

use std::marker::PhantomData;
use std::time::Duration;

use genai_agents::{
    BoxFuture, Conversation, Item, MaybeSend, MaybeSync, Model, OutputSchema, RunConfig, RunStop,
    Runner, ToolRegistry,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::case::Case;
use crate::target::Trial;

/// A scorer's verdict on a trial: a number from 0 (wrong) to 1 (right), and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Score {
    /// From 0 to 1.
    pub value: f64,
    /// Why, when the scorer says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl Score {
    /// This value, kept between 0 and 1.
    #[must_use]
    pub fn new(value: f64) -> Self {
        let value = if value.is_nan() {
            0.0
        } else {
            value.clamp(0.0, 1.0)
        };
        Self {
            value,
            reason: None,
        }
    }

    /// 1 when `pass`, else 0.
    #[must_use]
    pub fn pass(pass: bool) -> Self {
        Self::new(if pass { 1.0 } else { 0.0 })
    }

    /// The same score, with a reason.
    #[must_use]
    pub fn because(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// Something that scores trials. Its name keys its scores in a report: an eval numbers the
/// names that repeat (`name #2`). A scorer's [`threshold`](Self::threshold) decides whether a
/// trial passes it; a trial passes when it passes every scorer.
pub trait Scorer: MaybeSend + MaybeSync {
    /// Its name, in reports.
    fn name(&self) -> &str;

    /// Score one trial of a case.
    fn score<'a>(
        &'a self,
        case: &'a Case,
        trial: &'a Trial,
    ) -> BoxFuture<'a, genai_agents::Result<Score>>;

    /// The score from which a trial passes. Default 1.
    fn threshold(&self) -> f64 {
        1.0
    }
}

/// A scorer that decides at once, from a function.
macro_rules! sync_scorer {
    ($type:ty, name($n:ident) => $name:expr, |$self:ident, $case:ident, $trial:ident| $body:expr) => {
        impl Scorer for $type {
            fn name(&self) -> &str {
                let $n = self;
                $name
            }

            fn score<'a>(
                &'a self,
                case: &'a Case,
                trial: &'a Trial,
            ) -> BoxFuture<'a, genai_agents::Result<Score>> {
                let $self = self;
                let $case = case;
                let $trial = trial;
                let score = $body;
                Box::pin(async move { Ok(score) })
            }
        }
    };
}

fn no_output() -> Score {
    Score::pass(false).because("no answer")
}

/// The answer equals the case's expected answer: as text (trimmed), or as JSON when the
/// expected answer is not a string.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactMatch;

sync_scorer!(ExactMatch, name(_s) => "exact_match", |_s, case, trial| {
    match (trial.output(), &case.expected) {
        (None, _) => no_output(),
        (Some(_), None) => Score::pass(false).because("the case expects nothing"),
        (Some(output), Some(Value::String(expected))) => {
            Score::pass(output.trim() == expected.trim())
        }
        (Some(output), Some(expected)) => match serde_json::from_str::<Value>(output) {
            Ok(value) => Score::pass(&value == expected),
            Err(error) => Score::pass(false).because(format!("not JSON: {error}")),
        },
    }
});

/// The answer contains a text.
#[derive(Debug, Clone)]
pub struct Contains {
    text: String,
    ignore_case: bool,
    name: String,
}

impl Contains {
    /// The answer contains `text`.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            name: format!("contains {text:?}"),
            text,
            ignore_case: false,
        }
    }

    /// The same, regardless of case.
    #[must_use]
    pub fn ignore_case(mut self) -> Self {
        self.ignore_case = true;
        self
    }
}

sync_scorer!(Contains, name(n) => n.name.as_str(),
    |s, _case, trial| {
        match trial.output() {
            None => no_output(),
            Some(output) if s.ignore_case => {
                Score::pass(output.to_lowercase().contains(&s.text.to_lowercase()))
            }
            Some(output) => Score::pass(output.contains(&s.text)),
        }
    }
);

/// The answer matches a regular expression.
#[derive(Debug, Clone)]
pub struct Matches {
    pattern: regex::Regex,
    name: String,
}

impl Matches {
    /// The answer matches `pattern`.
    ///
    /// # Errors
    ///
    /// When `pattern` is not a regular expression.
    pub fn new(pattern: &str) -> Result<Self, regex::Error> {
        Ok(Self {
            pattern: regex::Regex::new(pattern)?,
            name: format!("matches /{pattern}/"),
        })
    }
}

sync_scorer!(Matches, name(n) => n.name.as_str(), |s, _case, trial| {
    trial
        .output()
        .map_or_else(no_output, |output| Score::pass(s.pattern.is_match(output)))
});

/// The answer parses as JSON of `T`: the agent's output type, say.
#[derive(Debug)]
pub struct ParsesAs<T>(PhantomData<fn() -> T>);

impl<T> ParsesAs<T> {
    /// The answer parses as `T`.
    #[must_use]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Default for ParsesAs<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: serde::de::DeserializeOwned> Scorer for ParsesAs<T> {
    fn name(&self) -> &str {
        "parses"
    }

    fn score<'a>(
        &'a self,
        _: &'a Case,
        trial: &'a Trial,
    ) -> BoxFuture<'a, genai_agents::Result<Score>> {
        let score = match trial.output() {
            None => no_output(),
            Some(output) => match serde_json::from_str::<T>(output) {
                Ok(_) => Score::pass(true),
                Err(error) => Score::pass(false).because(error.to_string()),
            },
        };
        Box::pin(async move { Ok(score) })
    }
}

/// Whether `actual` holds everything `expected` holds: equal values, and for objects every key
/// of `expected` with a value it holds.
fn holds(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => expected
            .iter()
            .all(|(key, value)| actual.get(key).is_some_and(|actual| holds(actual, value))),
        _ => actual == expected,
    }
}

/// The run called a tool, optionally with arguments that hold these (every key given, with
/// its value; other keys may be there too).
#[derive(Debug, Clone)]
pub struct ToolCalled {
    tool: String,
    arguments: Option<Value>,
    name: String,
}

impl ToolCalled {
    /// The run called `tool`.
    #[must_use]
    pub fn new(tool: impl Into<String>) -> Self {
        let tool = tool.into();
        Self {
            name: format!("called {tool}"),
            tool,
            arguments: None,
        }
    }

    /// The same, with arguments that hold these.
    #[must_use]
    pub fn with_arguments(mut self, arguments: Value) -> Self {
        self.name = format!("called {} with {arguments}", self.tool);
        self.arguments = Some(arguments);
        self
    }
}

sync_scorer!(ToolCalled, name(n) => n.name.as_str(),
    |s, case, trial| {
        let calls = trial.calls(case);
        let named: Vec<_> = calls.iter().filter(|call| call.name == s.tool).collect();
        match (&s.arguments, named.is_empty()) {
            (_, true) => Score::pass(false).because(format!("{} was not called", s.tool)),
            (None, false) => Score::pass(true),
            (Some(expected), false) => {
                let found = named.iter().any(|call| {
                    serde_json::from_str::<Value>(&call.input)
                        .is_ok_and(|actual| holds(&actual, expected))
                });
                if found {
                    Score::pass(true)
                } else {
                    Score::pass(false)
                        .because(format!("{} was called, never with {expected}", s.tool))
                }
            }
        }
    }
);

/// The run did not call a tool.
#[derive(Debug, Clone)]
pub struct ToolNotCalled {
    tool: String,
    name: String,
}

impl ToolNotCalled {
    /// The run did not call `tool`.
    #[must_use]
    pub fn new(tool: impl Into<String>) -> Self {
        let tool = tool.into();
        Self {
            name: format!("did not call {tool}"),
            tool,
        }
    }
}

sync_scorer!(ToolNotCalled, name(n) => n.name.as_str(),
    |s, case, trial| { Score::pass(!trial.calls(case).iter().any(|call| call.name == s.tool)) }
);

/// The run called these tools in this order (others may come between). The score is the share
/// of them it got through.
#[derive(Debug, Clone)]
pub struct ToolOrder {
    tools: Vec<String>,
    name: String,
}

impl ToolOrder {
    /// The run called `tools` in this order.
    #[must_use]
    pub fn new(tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let tools: Vec<String> = tools.into_iter().map(Into::into).collect();
        Self {
            name: format!("tool order {}", tools.join(" → ")),
            tools,
        }
    }
}

sync_scorer!(ToolOrder, name(n) => n.name.as_str(), |s, case, trial| {
    let mut next = 0;
    for call in trial.calls(case) {
        if s.tools.get(next).is_some_and(|tool| *tool == call.name) {
            next += 1;
        }
    }
    if s.tools.is_empty() {
        Score::pass(true)
    } else {
        #[allow(clippy::cast_precision_loss)] // a handful of tools
        let share = next as f64 / s.tools.len() as f64;
        let score = Score::new(share);
        if next == s.tools.len() {
            score
        } else {
            score.because(format!("stopped before {}", s.tools[next]))
        }
    }
});

/// At most this many model requests: 1 within it, less the more it took.
#[derive(Debug, Clone, Copy)]
pub struct MaxRequests(pub usize);

sync_scorer!(MaxRequests, name(_s) => "requests", |s, _case, trial| {
    ratio(trial.requests() as f64, s.0 as f64, "requests")
});

/// At most this many tokens (input and output): 1 within it, less the more it took.
#[derive(Debug, Clone, Copy)]
pub struct MaxTokens(pub i64);

sync_scorer!(MaxTokens, name(_s) => "tokens", |s, _case, trial| {
    let usage = trial.usage();
    #[allow(clippy::cast_precision_loss)]
    let used = (usage.input_tokens + usage.output_tokens) as f64;
    #[allow(clippy::cast_precision_loss)]
    let limit = s.0 as f64;
    ratio(used, limit, "tokens")
});

/// At most this long: 1 within it, less the longer it took.
#[derive(Debug, Clone, Copy)]
pub struct MaxTime(pub Duration);

sync_scorer!(MaxTime, name(_s) => "time", |s, _case, trial| {
    ratio(trial.elapsed.as_secs_f64(), s.0.as_secs_f64(), "seconds")
});

#[allow(clippy::cast_precision_loss)]
fn ratio(used: f64, limit: f64, unit: &str) -> Score {
    if used <= limit {
        Score::pass(true)
    } else {
        Score::new(if used > 0.0 { limit / used } else { 0.0 })
            .because(format!("{used} {unit}, more than {limit}"))
    }
}

/// The run stopped as expected: the model answered, or a guardrail tripped.
#[derive(Debug, Clone)]
pub struct Stopped {
    guardrail: Option<String>,
    name: String,
}

impl Stopped {
    /// The model answered ([`RunStop::Completed`]).
    #[must_use]
    pub fn completed() -> Self {
        Self {
            guardrail: None,
            name: "completed".into(),
        }
    }

    /// The guardrail `name` tripped.
    #[must_use]
    pub fn by_guardrail(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name: format!("stopped by {name}"),
            guardrail: Some(name),
        }
    }
}

sync_scorer!(Stopped, name(n) => n.name.as_str(), |s, _case, trial| {
    match (trial.stop(), &s.guardrail) {
        (None, _) => Score::pass(false).because(trial.error.clone().unwrap_or_default()),
        (Some(RunStop::Completed), None) => Score::pass(true),
        (Some(RunStop::Guardrail { name, .. }), Some(expected)) if name == expected => {
            Score::pass(true)
        }
        (Some(stop), _) => Score::pass(false).because(format!("stopped: {stop:?}")),
    }
});

#[cfg(not(target_family = "wasm"))]
type ScoreFn = dyn Fn(&Case, &Trial) -> Score + Send + Sync;
#[cfg(target_family = "wasm")]
type ScoreFn = dyn Fn(&Case, &Trial) -> Score;

/// A scorer from a name and a function ([`scorer_fn`]).
pub struct FnScorer {
    name: String,
    score: Box<ScoreFn>,
    threshold: f64,
}

impl std::fmt::Debug for FnScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnScorer")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl FnScorer {
    /// The same scorer, passing from this score.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

/// A scorer of the application's own.
#[cfg(not(target_family = "wasm"))]
pub fn scorer_fn(
    name: impl Into<String>,
    score: impl Fn(&Case, &Trial) -> Score + Send + Sync + 'static,
) -> FnScorer {
    FnScorer {
        name: name.into(),
        score: Box::new(score),
        threshold: 1.0,
    }
}

/// A scorer of the application's own.
#[cfg(target_family = "wasm")]
pub fn scorer_fn(
    name: impl Into<String>,
    score: impl Fn(&Case, &Trial) -> Score + 'static,
) -> FnScorer {
    FnScorer {
        name: name.into(),
        score: Box::new(score),
        threshold: 1.0,
    }
}

impl Scorer for FnScorer {
    fn name(&self) -> &str {
        &self.name
    }

    fn score<'a>(
        &'a self,
        case: &'a Case,
        trial: &'a Trial,
    ) -> BoxFuture<'a, genai_agents::Result<Score>> {
        let score = (self.score)(case, trial);
        Box::pin(async move { Ok(score) })
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// What the judge prompt asks for.
const JUDGE_INSTRUCTIONS: &str = "You grade the answer of an AI assistant against a rubric. \
Read the conversation, the answer and the rubric, then give a score from 0 (fails the rubric) \
to 1 (meets it fully) and a short reason. Judge only by the rubric.";

/// What the judge answers.
#[derive(Deserialize)]
struct Verdict {
    score: f64,
    reason: String,
}

/// An LLM judge: another model scores the answer against a rubric. Use a model other than the
/// one judged, and keep the rubric concrete ("names the city and the temperature"); a judge's
/// score is an opinion, so compare it across runs rather than reading one alone.
pub struct Judge<M> {
    name: String,
    rubric: String,
    model: M,
    threshold: f64,
}

impl<M> std::fmt::Debug for Judge<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Judge")
            .field("name", &self.name)
            .field("rubric", &self.rubric)
            .finish_non_exhaustive()
    }
}

impl<M: Model + Clone> Judge<M> {
    /// A judge on `model`, scoring by `rubric`; it passes from 0.7.
    #[must_use]
    pub fn new(name: impl Into<String>, model: M, rubric: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rubric: rubric.into(),
            model,
            threshold: 0.7,
        }
    }

    /// The same judge, passing from this score.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    async fn judge(&self, case: &Case, trial: &Trial) -> genai_agents::Result<Score> {
        let Some(output) = trial.output() else {
            return Ok(no_output());
        };
        let transcript = serde_json::to_string_pretty(&case.input)?;
        let expected = case
            .expected
            .as_ref()
            .map(|expected| format!("\n\nA reference answer:\n{expected}"))
            .unwrap_or_default();
        let prompt = format!(
            "The conversation:\n{transcript}\n\nThe answer:\n{output}{expected}\n\nThe rubric:\n{}",
            self.rubric
        );
        let schema = OutputSchema::new(
            "verdict",
            json!({
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["score", "reason"],
                "additionalProperties": false,
            }),
        )
        .with_strict(true);
        let mut runner = Runner::new(self.model.clone(), ToolRegistry::new()).with_config(
            RunConfig::new()
                .with_instructions(JUDGE_INSTRUCTIONS)
                .with_output(schema)
                .with_name(format!("judge {}", self.name)),
        );
        let mut conversation = Conversation::from_items([Item::user(prompt)]);
        let outcome = runner.run(&mut conversation).await?;
        let verdict: Verdict = serde_json::from_str(outcome.output_text()).map_err(|error| {
            genai_agents::Error::Output(format!("the judge's answer is not a verdict: {error}"))
        })?;
        Ok(Score::new(verdict.score).because(verdict.reason))
    }
}

impl<M: Model + Clone> Scorer for Judge<M> {
    fn name(&self) -> &str {
        &self.name
    }

    fn score<'a>(
        &'a self,
        case: &'a Case,
        trial: &'a Trial,
    ) -> BoxFuture<'a, genai_agents::Result<Score>> {
        Box::pin(self.judge(case, trial))
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }
}
