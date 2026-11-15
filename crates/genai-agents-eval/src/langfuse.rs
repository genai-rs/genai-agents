//! Langfuse dataset runs (feature `langfuse`): each case of a Langfuse dataset runs as a trace,
//! linked to its dataset item in a dataset run, with the scorers' scores on it.
//!
//! [`run_dataset`] runs on langfuse-ergonomic's
//! [experiments](https://docs.rs/langfuse-ergonomic/latest/langfuse_ergonomic/experiments/):
//! a trace per item (the item's input in, the answer out), the scores as Langfuse scores (the
//! reason as the comment), and the dataset run, which Langfuse compares with other runs of the
//! same dataset. It also gives the eval's own [`Report`], to keep as a baseline.
//!
//! The trace is the experiment's: the runner's `tracing` spans go wherever the application
//! exports them (to Langfuse too, with opentelemetry-langfuse), as traces of their own.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use langfuse_ergonomic::experiments::{
    evaluator_fn, Evaluation, EvaluatorContext, ExperimentResult,
};
use langfuse_ergonomic::LangfuseClient;
use serde_json::Value;

use crate::case::Case;
use crate::eval::{run_trial, CaseReport, Report, Summary, TrialReport};
use crate::scorer::Scorer;
use crate::target::Target;
use crate::EvalError;

/// The result of a dataset run: Langfuse's, and the eval's report.
#[derive(Debug)]
#[non_exhaustive]
pub struct DatasetRun {
    /// The experiment: the dataset run's id and URL, each item's trace.
    pub experiment: ExperimentResult,
    /// The report, as [`Eval::run`](crate::Eval::run) gives it (one trial per case).
    pub report: Report,
}

/// Run `target` on every active item of the Langfuse dataset `dataset`, as the dataset run
/// `run_name`, `concurrency` items at once, and score each with `scorers`.
///
/// An item's input becomes the case's conversation ([`Case::input_from_json`]: a string, an
/// array of items, or `{"messages": […]}`); its expected output, the case's expected answer.
///
/// # Errors
///
/// When the dataset cannot be read or the run cannot be recorded.
pub async fn run_dataset(
    client: &LangfuseClient,
    dataset: &str,
    run_name: &str,
    target: &dyn Target,
    scorers: &[&dyn Scorer],
    concurrency: usize,
) -> Result<DatasetRun, EvalError> {
    let trials: Mutex<HashMap<String, TrialReport>> = Mutex::new(HashMap::new());
    let trials = &trials;
    let task = |item: langfuse_ergonomic::DatasetItem| async move {
        let input = item.input.clone().unwrap_or(Value::Null);
        let mut case = Case::new(item.id.clone(), Case::input_from_json(&input)?);
        case.expected = item.expected_output.clone();
        case.metadata = item.metadata.clone().unwrap_or(Value::Null);
        let (trial, report) = run_trial(target, scorers, &case, 0).await;
        trials
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(item.id.clone(), report);
        match (trial.output(), trial.error.clone()) {
            (_, Some(error)) => Err(EvalError::Langfuse(error)),
            (Some(output), None) => Ok(serde_json::from_str::<Value>(output)
                .ok()
                .filter(|value| value.is_object() || value.is_array())
                .unwrap_or_else(|| Value::String(output.to_owned()))),
            (None, None) => Ok(Value::Null),
        }
    };
    let scores = evaluator_fn(move |context: EvaluatorContext| async move {
        let report = trials
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&context.item.id)
            .cloned();
        let evaluations = report
            .map(|report| {
                report
                    .scores
                    .into_iter()
                    .map(|(name, score)| {
                        let evaluation = Evaluation::new(name, score.value);
                        match score.reason {
                            Some(reason) => evaluation.with_comment(reason),
                            None => evaluation,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(evaluations)
    });
    let experiment = client
        .run_experiment(dataset, task)
        .run_name(run_name)
        .evaluator(scores)
        .concurrency(concurrency)
        .run()
        .await
        .map_err(|error| EvalError::Langfuse(error.to_string()))?;

    let mut by_item = trials.lock().unwrap_or_else(PoisonError::into_inner);
    let mut cases: Vec<CaseReport> = experiment
        .items
        .iter()
        .map(|item| {
            let mut case = CaseReport::new(item.item.id.clone());
            if let Some(trial) = by_item.remove(&item.item.id) {
                case.trials.push(trial);
            }
            case.conclude();
            case
        })
        .collect();
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    let summary = Summary::of(&cases);
    let report = Report {
        name: run_name.to_owned(),
        dataset: dataset.to_owned(),
        trials_per_case: 1,
        cases,
        summary,
    };
    Ok(DatasetRun { experiment, report })
}
