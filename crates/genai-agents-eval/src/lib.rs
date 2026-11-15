//! Evals for [genai-agents](https://docs.rs/genai-agents): whether a change to a prompt, a tool,
//! a model or a setting made an agent better.
//!
//! - A [`Dataset`] of [`Case`]s: an input, optionally the expected answer.
//! - A [`Target`]: what runs on each case (an agent, a runner, the application's pipeline);
//!   [`target_fn`] makes one from an async closure.
//! - [Scorers](Scorer), each a number from 0 to 1 with a reason: of the answer
//!   ([`ExactMatch`], [`Contains`], [`Matches`], [`ParsesAs`]), of the trajectory
//!   ([`ToolCalled`], [`ToolNotCalled`], [`ToolOrder`], [`MaxRequests`], [`Stopped`]), of the
//!   cost ([`MaxTokens`], [`MaxTime`]), an LLM [`Judge`], and the application's own
//!   ([`scorer_fn`]).
//! - An [`Eval`] runs every case `k` times, a few at once, within a token budget, and gives a
//!   [`Report`]: the mean and spread of each scorer, pass@k (a case passed at least once in
//!   `k`) and pass^k (it passed every time). [`Report::compare`] against a baseline lists what
//!   got worse, for CI to fail on.
//! - With the `langfuse` feature, a Langfuse dataset runs as a dataset run, its scores on the
//!   traces (`langfuse::run_dataset`).
//!
//! Tests and evals in CI run without a key on [recorded runs](genai_agents::record).
//!
//! ```
//! use genai_agents::testing::{Reply, ScriptedModel};
//! use genai_agents::{Runner, ToolRegistry};
//! use genai_agents_eval::{target_fn, Case, Contains, Dataset, Eval, ExactMatch, Run};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let dataset = Dataset::new(
//!     "capitals",
//!     vec![
//!         Case::user("fr", "Capital of France?").with_expected("Paris"),
//!         Case::user("be", "Capital of Belgium?").with_expected("Brussels"),
//!     ],
//! );
//! // A model that knows one capital (a real agent in practice, or a recorded run).
//! let target = target_fn(|case: Case| async move {
//!     let model = ScriptedModel::replies(vec![Reply::text("Paris")]);
//!     let mut runner = Runner::new(model, ToolRegistry::new());
//!     let mut conversation = case.input;
//!     let outcome = runner.run(&mut conversation).await?;
//!     Ok(Run::new(conversation, outcome))
//! });
//! let exact = ExactMatch;
//! let report = Eval::new("baseline", &dataset, &target)
//!     .scorer(&exact)
//!     .trials(3)
//!     .run()
//!     .await;
//! assert_eq!(report.summary.pass_at_k, 0.5);
//! println!("{}", report.to_markdown());
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

mod case;
mod eval;
#[cfg(feature = "langfuse")]
#[cfg_attr(docsrs, doc(cfg(feature = "langfuse")))]
pub mod langfuse;
mod scorer;
mod target;

pub use case::{Case, Dataset};
pub use eval::{CaseReport, Change, Comparison, Eval, Report, ScorerSummary, Summary, TrialReport};
pub use scorer::{
    scorer_fn, Contains, ExactMatch, FnScorer, Judge, Matches, MaxRequests, MaxTime, MaxTokens,
    ParsesAs, Score, Scorer, Stopped, ToolCalled, ToolNotCalled, ToolOrder,
};
pub use target::{target_fn, FnTarget, Run, Target, Trial, TrialSummary};

/// Why an eval could not be set up.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EvalError {
    /// JSON that could not be read.
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A case's input of an unknown shape.
    #[error("case input: {0}")]
    Input(String),
    /// A file that could not be read.
    #[error("I/O: {0}")]
    Io(#[source] std::io::Error),
    /// Langfuse failed.
    #[error("Langfuse: {0}")]
    Langfuse(String),
}

impl EvalError {
    pub(crate) fn io(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
