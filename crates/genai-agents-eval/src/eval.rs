//! Running an eval, and its report.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicI64, Ordering};

use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::case::{Case, Dataset};
use crate::scorer::{Score, Scorer};
use crate::target::{Target, Trial, TrialSummary};

#[cfg(not(target_family = "wasm"))]
use std::time::Instant;
#[cfg(target_family = "wasm")]
use web_time::Instant;

/// An eval: a dataset, a target to run on each case, and scorers. See the
/// [crate documentation](crate).
pub struct Eval<'a> {
    name: String,
    dataset: &'a Dataset,
    target: &'a dyn Target,
    scorers: Vec<&'a dyn Scorer>,
    trials: u32,
    concurrency: usize,
    max_tokens: Option<i64>,
}

impl std::fmt::Debug for Eval<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Eval")
            .field("name", &self.name)
            .field("dataset", &self.dataset.name)
            .field(
                "scorers",
                &self.scorers.iter().map(|s| s.name()).collect::<Vec<_>>(),
            )
            .field("trials", &self.trials)
            .field("concurrency", &self.concurrency)
            .field("max_tokens", &self.max_tokens)
            .finish_non_exhaustive()
    }
}

impl<'a> Eval<'a> {
    /// An eval named `name` of `target` on `dataset`: one trial per case, four at once.
    #[must_use]
    pub fn new(name: impl Into<String>, dataset: &'a Dataset, target: &'a dyn Target) -> Self {
        Self {
            name: name.into(),
            dataset,
            target,
            scorers: Vec::new(),
            trials: 1,
            concurrency: 4,
            max_tokens: None,
        }
    }

    /// The same eval, with this scorer.
    #[must_use]
    pub fn scorer(mut self, scorer: &'a dyn Scorer) -> Self {
        self.scorers.push(scorer);
        self
    }

    /// The same eval, running each case `k` times (at least 1): for pass@k and pass^k.
    #[must_use]
    pub fn trials(mut self, k: u32) -> Self {
        self.trials = k.max(1);
        self
    }

    /// The same eval, with at most this many trials running at once (at least 1). They run on
    /// the calling task; nothing is spawned.
    #[must_use]
    pub fn concurrency(mut self, trials: usize) -> Self {
        self.concurrency = trials.max(1);
        self
    }

    /// The same eval, starting no trial once the trials used this many tokens: the rest are
    /// reported as skipped.
    #[must_use]
    pub fn max_tokens(mut self, tokens: i64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Run every trial and score it.
    pub async fn run(&self) -> Report {
        let used = AtomicI64::new(0);
        let jobs: Vec<(usize, u32)> = (0..self.dataset.cases.len())
            .flat_map(|case| (0..self.trials).map(move |attempt| (case, attempt)))
            .collect();
        let used = &used;
        let mut results: Vec<(usize, u32, TrialReport)> = futures::stream::iter(jobs)
            .map(|(index, attempt)| async move {
                let case = &self.dataset.cases[index];
                let report = self.trial(case, attempt, used).await;
                (index, attempt, report)
            })
            .buffer_unordered(self.concurrency)
            .collect()
            .await;
        results.sort_by_key(|(index, attempt, _)| (*index, *attempt));
        let mut cases: Vec<CaseReport> = self
            .dataset
            .cases
            .iter()
            .map(|case| CaseReport::new(case.id.clone()))
            .collect();
        for (index, _, trial) in results {
            cases[index].trials.push(trial);
        }
        for case in &mut cases {
            case.conclude();
        }
        let summary = Summary::of(&cases);
        Report {
            name: self.name.clone(),
            dataset: self.dataset.name.clone(),
            trials_per_case: self.trials,
            cases,
            summary,
        }
    }

    async fn trial(&self, case: &Case, attempt: u32, used: &AtomicI64) -> TrialReport {
        if self
            .max_tokens
            .is_some_and(|max| used.load(Ordering::SeqCst) >= max)
        {
            return TrialReport::skipped(attempt);
        }
        let (trial, report) = run_trial(self.target, &self.scorers, case, attempt).await;
        let usage = trial.usage();
        used.fetch_add(usage.input_tokens + usage.output_tokens, Ordering::SeqCst);
        report
    }
}

/// Run the target on a case once, and score the trial.
pub(crate) async fn run_trial(
    target: &dyn Target,
    scorers: &[&dyn Scorer],
    case: &Case,
    attempt: u32,
) -> (Trial, TrialReport) {
    let started = Instant::now();
    let run = target.run(case).await;
    let elapsed = started.elapsed();
    let (run, error) = match run {
        Ok(run) => (Some(run), None),
        Err(error) => (None, Some(error.to_string())),
    };
    let trial = Trial {
        case_id: case.id.clone(),
        attempt,
        run,
        error,
        elapsed,
    };
    let mut scores = BTreeMap::new();
    let mut passed = trial.error.is_none();
    for scorer in scorers {
        let score = match scorer.score(case, &trial).await {
            Ok(score) => score,
            Err(error) => Score::pass(false).because(format!("the scorer failed: {error}")),
        };
        passed &= score.value >= scorer.threshold();
        // Two scorers of one name would share a key: the later ones are numbered.
        let mut name = scorer.name().to_owned();
        let mut number = 1;
        while scores.contains_key(&name) {
            number += 1;
            name = format!("{} #{number}", scorer.name());
        }
        scores.insert(name, score);
    }
    let report = TrialReport {
        attempt,
        skipped: false,
        passed,
        scores,
        summary: Some(TrialSummary::from(&trial)),
    };
    (trial, report)
}

/// One trial in a report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrialReport {
    /// Which of the case's trials, from 0.
    pub attempt: u32,
    /// Whether it was not run, for the token budget.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub skipped: bool,
    /// Whether it passed every scorer (and the target did not fail).
    pub passed: bool,
    /// Each scorer's score.
    pub scores: BTreeMap<String, Score>,
    /// What the run gave.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<TrialSummary>,
}

impl TrialReport {
    fn skipped(attempt: u32) -> Self {
        Self {
            attempt,
            skipped: true,
            passed: false,
            scores: BTreeMap::new(),
            summary: None,
        }
    }
}

/// One case in a report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CaseReport {
    /// The case's id.
    pub id: String,
    /// Its trials.
    pub trials: Vec<TrialReport>,
    /// How many trials passed.
    pub passed: usize,
    /// pass@k: at least one of its `k` trials passed.
    pub pass_at_k: bool,
    /// pass^k: all of its `k` trials passed.
    pub pass_all_k: bool,
}

impl CaseReport {
    pub(crate) fn new(id: String) -> Self {
        Self {
            id,
            trials: Vec::new(),
            passed: 0,
            pass_at_k: false,
            pass_all_k: false,
        }
    }

    pub(crate) fn conclude(&mut self) {
        let run: Vec<&TrialReport> = self.trials.iter().filter(|trial| !trial.skipped).collect();
        self.passed = run.iter().filter(|trial| trial.passed).count();
        self.pass_at_k = self.passed > 0;
        self.pass_all_k =
            !run.is_empty() && self.passed == run.len() && run.len() == self.trials.len();
    }
}

/// A scorer's results over every trial that ran.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScorerSummary {
    /// The mean score.
    pub mean: f64,
    /// The standard deviation of the scores.
    pub stddev: f64,
    /// The share of trials it scored 1.
    pub perfect: f64,
}

/// What a report adds up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Summary {
    /// The cases.
    pub cases: usize,
    /// The trials that ran.
    pub trials: usize,
    /// The trials skipped for the token budget.
    pub skipped: usize,
    /// The trials whose target failed.
    pub errors: usize,
    /// The share of trials that passed.
    pub pass_rate: f64,
    /// pass@k: the share of cases with at least one passing trial.
    pub pass_at_k: f64,
    /// pass^k: the share of cases whose trials all passed. Much lower than pass@k means an agent
    /// that works only some of the time.
    pub pass_all_k: f64,
    /// Each scorer's mean and spread.
    pub scorers: BTreeMap<String, ScorerSummary>,
    /// The tokens of all trials.
    pub tokens: i64,
    /// The mean time of a trial, in milliseconds.
    pub mean_millis: f64,
}

#[allow(clippy::cast_precision_loss)] // counts far below 2^52
impl Summary {
    pub(crate) fn of(cases: &[CaseReport]) -> Self {
        let ran: Vec<&TrialReport> = cases
            .iter()
            .flat_map(|case| &case.trials)
            .filter(|trial| !trial.skipped)
            .collect();
        let share = |count: usize, of: usize| {
            if of == 0 {
                0.0
            } else {
                count as f64 / of as f64
            }
        };
        let mut by_scorer: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for trial in &ran {
            for (name, score) in &trial.scores {
                by_scorer.entry(name.clone()).or_default().push(score.value);
            }
        }
        let scorers = by_scorer
            .into_iter()
            .map(|(name, values)| {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / values.len() as f64;
                let perfect = share(
                    values.iter().filter(|value| **value >= 1.0).count(),
                    values.len(),
                );
                (
                    name,
                    ScorerSummary {
                        mean,
                        stddev: variance.sqrt(),
                        perfect,
                    },
                )
            })
            .collect();
        let millis: Vec<u64> = ran
            .iter()
            .filter_map(|trial| trial.summary.as_ref().map(|summary| summary.millis))
            .collect();
        Self {
            cases: cases.len(),
            trials: ran.len(),
            skipped: cases
                .iter()
                .flat_map(|case| &case.trials)
                .filter(|trial| trial.skipped)
                .count(),
            errors: ran
                .iter()
                .filter(|trial| {
                    trial
                        .summary
                        .as_ref()
                        .is_some_and(|summary| summary.error.is_some())
                })
                .count(),
            pass_rate: share(ran.iter().filter(|trial| trial.passed).count(), ran.len()),
            pass_at_k: share(
                cases.iter().filter(|case| case.pass_at_k).count(),
                cases.len(),
            ),
            pass_all_k: share(
                cases.iter().filter(|case| case.pass_all_k).count(),
                cases.len(),
            ),
            scorers,
            tokens: ran
                .iter()
                .filter_map(|trial| trial.summary.as_ref().map(|summary| summary.tokens))
                .sum(),
            mean_millis: if millis.is_empty() {
                0.0
            } else {
                millis.iter().sum::<u64>() as f64 / millis.len() as f64
            },
        }
    }
}

/// The result of an eval: every trial with its scores, and what they add up to. `Serialize`
/// and `Deserialize`: keep it as a baseline and [compare](Self::compare) later runs with it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Report {
    /// The eval's name.
    pub name: String,
    /// The dataset's name.
    pub dataset: String,
    /// The trials of each case (`k`).
    pub trials_per_case: u32,
    /// Each case.
    pub cases: Vec<CaseReport>,
    /// The totals.
    pub summary: Summary,
}

impl Report {
    /// The report as pretty JSON.
    ///
    /// # Errors
    ///
    /// When it cannot be encoded.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// A report from [`to_json`](Self::to_json)'s text.
    ///
    /// # Errors
    ///
    /// When the text is not a report.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// The report in Markdown: the totals, each scorer, and the cases that did not always pass.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let s = &self.summary;
        let mut out = format!("# {} ({})\n\n", self.name, self.dataset);
        let _ = writeln!(out, "| | |\n| --- | --- |");
        let _ = writeln!(out, "| Cases | {} |", s.cases);
        let _ = writeln!(out, "| Trials per case (k) | {} |", self.trials_per_case);
        let _ = writeln!(out, "| Trials passed | {} |", percent(s.pass_rate));
        let _ = writeln!(out, "| pass@k | {} |", percent(s.pass_at_k));
        let _ = writeln!(out, "| pass^k | {} |", percent(s.pass_all_k));
        let _ = writeln!(out, "| Failed runs | {} |", s.errors);
        if s.skipped > 0 {
            let _ = writeln!(out, "| Skipped (budget) | {} |", s.skipped);
        }
        let _ = writeln!(out, "| Tokens | {} |", s.tokens);
        let _ = writeln!(out, "| Mean time | {:.0} ms |", s.mean_millis);
        if !s.scorers.is_empty() {
            out.push_str("\n| Scorer | Mean | Spread | Perfect |\n| --- | --- | --- | --- |\n");
            for (name, scorer) in &s.scorers {
                let _ = writeln!(
                    out,
                    "| {} | {:.2} | {:.2} | {} |",
                    escape(name),
                    scorer.mean,
                    scorer.stddev,
                    percent(scorer.perfect)
                );
            }
        }
        let failing: Vec<&CaseReport> = self.cases.iter().filter(|case| !case.pass_all_k).collect();
        if !failing.is_empty() {
            out.push_str("\n## Cases that did not always pass\n\n");
            for case in failing {
                let _ = writeln!(
                    out,
                    "- `{}`: {} of {} trials passed",
                    case.id,
                    case.passed,
                    case.trials.len()
                );
                for trial in case
                    .trials
                    .iter()
                    .filter(|trial| !trial.passed && !trial.skipped)
                {
                    for (name, score) in trial.scores.iter().filter(|(_, score)| score.value < 1.0)
                    {
                        let _ = writeln!(
                            out,
                            "  - trial {}: {} {:.2}{}",
                            trial.attempt,
                            name,
                            score.value,
                            score
                                .reason
                                .as_ref()
                                .map(|reason| format!(" ({reason})"))
                                .unwrap_or_default()
                        );
                    }
                    if let Some(error) = trial.summary.as_ref().and_then(|s| s.error.as_ref()) {
                        let _ = writeln!(out, "  - trial {}: failed: {error}", trial.attempt);
                    }
                }
            }
        }
        out
    }

    /// What changed since `baseline`: totals and scorer means that moved by more than
    /// `tolerance` (0.05 for five points), and cases that always passed then and no longer do
    /// (or the other way).
    #[must_use]
    pub fn compare(&self, baseline: &Self, tolerance: f64) -> Comparison {
        let mut comparison = Comparison::default();
        let mut check = |what: String, before: f64, after: f64| {
            let change = Change {
                what,
                before,
                after,
            };
            if after < before - tolerance {
                comparison.regressions.push(change);
            } else if after > before + tolerance {
                comparison.improvements.push(change);
            }
        };
        let (old, new) = (&baseline.summary, &self.summary);
        check("pass@k".into(), old.pass_at_k, new.pass_at_k);
        check("pass^k".into(), old.pass_all_k, new.pass_all_k);
        check("trials passed".into(), old.pass_rate, new.pass_rate);
        for (name, before) in &old.scorers {
            if let Some(after) = new.scorers.get(name) {
                check(format!("{name} (mean)"), before.mean, after.mean);
            }
        }
        for case in &self.cases {
            let Some(before) = baseline.cases.iter().find(|old| old.id == case.id) else {
                continue;
            };
            let (before, after) = (before.pass_all_k, case.pass_all_k);
            let change = Change {
                what: format!("case {}", case.id),
                before: f64::from(u8::from(before)),
                after: f64::from(u8::from(after)),
            };
            match (before, after) {
                (true, false) => comparison.regressions.push(change),
                (false, true) => comparison.improvements.push(change),
                _ => {}
            }
        }
        comparison
    }
}

fn percent(share: f64) -> String {
    format!("{:.0} %", share * 100.0)
}

fn escape(text: &str) -> String {
    text.replace('|', "\\|")
}

/// One measure that moved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Change {
    /// What: a total, a scorer's mean, a case.
    pub what: String,
    /// In the baseline.
    pub before: f64,
    /// Now.
    pub after: f64,
}

/// What a report changed against a baseline ([`Report::compare`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Comparison {
    /// What got worse.
    pub regressions: Vec<Change>,
    /// What got better.
    pub improvements: Vec<Change>,
}

impl Comparison {
    /// Whether anything got worse: for CI to fail on.
    #[must_use]
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }

    /// The comparison in Markdown.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        for (title, changes) in [
            ("Regressions", &self.regressions),
            ("Improvements", &self.improvements),
        ] {
            if changes.is_empty() {
                continue;
            }
            let _ = writeln!(
                out,
                "## {title}\n\n| | Before | After |\n| --- | --- | --- |"
            );
            for change in changes {
                let _ = writeln!(
                    out,
                    "| {} | {:.2} | {:.2} |",
                    escape(&change.what),
                    change.before,
                    change.after
                );
            }
            out.push('\n');
        }
        if out.is_empty() {
            out.push_str("No change beyond the tolerance.\n");
        }
        out
    }
}
