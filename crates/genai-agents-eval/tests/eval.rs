//! Evals: scorers of the answer, the trajectory and the cost; pass@k and pass^k; the token
//! budget; reports, their comparison with a baseline; a judge; recorded runs as targets;
//! datasets from JSON Lines.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use genai_agents::record::{RecordingModel, ReplayModel};
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{Item, Result, Runner, ToolCall};
use genai_agents_eval::{
    scorer_fn, target_fn, Case, Contains, Dataset, Eval, ExactMatch, Judge, Matches, MaxRequests,
    MaxTokens, Report, Run, Score, Stopped, ToolCalled, ToolNotCalled, ToolOrder,
};
use serde_json::{json, Value};

struct Lookup(&'static str);

impl FunctionTool for Lookup {
    type Input = Value;
    type Output = String;
    fn name(&self) -> &str {
        self.0
    }
    fn description(&self) -> &str {
        "Look something up"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _: Value) -> Result<String> {
        Ok("found".into())
    }
}

fn tools() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(Lookup("search")).unwrap();
    tools.register(Lookup("fetch")).unwrap();
    tools
}

/// Searches, fetches, then answers.
fn researcher() -> ScriptedModel {
    ScriptedModel::new(|index, _| match index {
        0 => Reply::items(vec![Item::ToolCall(ToolCall::function(
            "c1",
            "search",
            r#"{"query":"capital of France","limit":3}"#,
        ))]),
        1 => Reply::items(vec![Item::ToolCall(ToolCall::function(
            "c2", "fetch", "{}",
        ))]),
        _ => Reply::text("The capital is Paris."),
    })
}

#[tokio::test]
async fn trajectory_and_cost_scorers() {
    let dataset = Dataset::new(
        "research",
        vec![Case::user("fr", "Capital of France?").with_expected("The capital is Paris.")],
    );
    let target = target_fn(|case: Case| async move {
        let mut runner = Runner::new(researcher(), tools());
        let mut conversation = case.input;
        let outcome = runner.run(&mut conversation).await?;
        Ok(Run::new(conversation, outcome))
    });
    let exact = ExactMatch;
    let contains = Contains::new("paris").ignore_case();
    let pattern = Matches::new(r"capital is \w+").unwrap();
    let searched = ToolCalled::new("search").with_arguments(json!({"query": "capital of France"}));
    let wrong_arguments = ToolCalled::new("search").with_arguments(json!({"query": "Belgium"}));
    let no_delete = ToolNotCalled::new("delete");
    let order = ToolOrder::new(["search", "fetch"]);
    let reversed = ToolOrder::new(["fetch", "search"]);
    let requests = MaxRequests(2);
    let tokens = MaxTokens(1_000);
    let completed = Stopped::completed();
    let report = Eval::new("trajectory", &dataset, &target)
        .scorer(&exact)
        .scorer(&contains)
        .scorer(&pattern)
        .scorer(&searched)
        .scorer(&wrong_arguments)
        .scorer(&no_delete)
        .scorer(&order)
        .scorer(&reversed)
        .scorer(&requests)
        .scorer(&tokens)
        .scorer(&completed)
        .run()
        .await;
    let scores = &report.cases[0].trials[0].scores;
    let value = |name: &str| scores[name].value;
    assert_eq!(value("exact_match"), 1.0);
    assert_eq!(value("contains \"paris\""), 1.0);
    assert_eq!(value("matches /capital is \\w+/"), 1.0);
    assert_eq!(
        value("called search with {\"query\":\"capital of France\"}"),
        1.0
    );
    assert_eq!(value("called search with {\"query\":\"Belgium\"}"), 0.0);
    assert_eq!(value("did not call delete"), 1.0);
    assert_eq!(value("completed"), 1.0);
    assert_eq!(value("tokens"), 1.0);
    // Three requests for a budget of two.
    assert!((value("requests") - 2.0 / 3.0).abs() < 1e-9);
    assert_eq!(value("tool order search → fetch"), 1.0);
    // "fetch" before "search" got half-way.
    assert_eq!(value("tool order fetch → search"), 0.5);
    assert!(!report.cases[0].trials[0].passed);
    assert_eq!(
        report.cases[0].trials[0].summary.as_ref().unwrap().requests,
        3
    );
}

#[tokio::test]
async fn pass_at_k_and_pass_all_k() {
    // Right every other time.
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let target = target_fn(move |case: Case| {
        let answer = if counter.fetch_add(1, Ordering::SeqCst).is_multiple_of(2) {
            "Paris"
        } else {
            "Lyon"
        };
        async move {
            let mut runner = Runner::new(
                ScriptedModel::replies(vec![Reply::text(answer)]),
                ToolRegistry::new(),
            );
            let mut conversation = case.input;
            let outcome = runner.run(&mut conversation).await?;
            Ok(Run::new(conversation, outcome))
        }
    });
    let dataset = Dataset::new(
        "flaky",
        vec![Case::user("fr", "Capital?").with_expected("Paris")],
    );
    let exact = ExactMatch;
    let report = Eval::new("flaky", &dataset, &target)
        .scorer(&exact)
        .trials(4)
        .concurrency(1)
        .run()
        .await;
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    let case = &report.cases[0];
    assert_eq!(case.trials.len(), 4);
    assert_eq!(case.passed, 2);
    assert!(case.pass_at_k && !case.pass_all_k);
    assert_eq!(report.summary.pass_at_k, 1.0);
    assert_eq!(report.summary.pass_all_k, 0.0);
    assert_eq!(report.summary.pass_rate, 0.5);
    let exact = &report.summary.scorers["exact_match"];
    assert_eq!(exact.mean, 0.5);
    assert_eq!(exact.stddev, 0.5);

    // The budget: each trial costs 15 tokens; after two, the rest are skipped.
    let report = Eval::new("budget", &dataset, &target)
        .scorer(&ExactMatch)
        .trials(4)
        .concurrency(1)
        .max_tokens(30)
        .run()
        .await;
    assert_eq!(report.summary.trials, 2);
    assert_eq!(report.summary.skipped, 2);
    assert!(!report.cases[0].pass_all_k);
}

#[tokio::test]
async fn reports_compare_with_a_baseline() {
    let dataset = Dataset::new(
        "capitals",
        vec![
            Case::user("fr", "France?").with_expected("Paris"),
            Case::user("be", "Belgium?").with_expected("Brussels"),
        ],
    );
    let answering = |answer: &'static str| {
        target_fn(move |case: Case| async move {
            let model = ScriptedModel::replies(vec![Reply::text(answer)]);
            let mut runner = Runner::new(model, ToolRegistry::new());
            let mut conversation = case.input;
            let outcome = runner.run(&mut conversation).await?;
            Ok(Run::new(conversation, outcome))
        })
    };
    let exact = ExactMatch;
    let brussels = answering("Brussels");
    let baseline = Eval::new("v1", &dataset, &brussels)
        .scorer(&exact)
        .run()
        .await;
    let paris = answering("Paris");
    let now = Eval::new("v2", &dataset, &paris).scorer(&exact).run().await;

    // A baseline survives JSON.
    let baseline = Report::from_json(&baseline.to_json().unwrap()).unwrap();
    // As many cases pass, but not the same ones: one regressed, one improved.
    let comparison = now.compare(&baseline, 0.05);
    assert!(comparison.has_regressions());
    let what = |changes: &[genai_agents_eval::Change]| {
        changes
            .iter()
            .map(|change| change.what.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(what(&comparison.regressions), ["case be"]);
    assert_eq!(what(&comparison.improvements), ["case fr"]);
    let markdown = comparison.to_markdown();
    assert!(markdown.contains("## Regressions") && markdown.contains("## Improvements"));
    assert!(now.to_markdown().contains("`be`: 0 of 1 trials passed"));

    // Same scores: nothing beyond the tolerance.
    assert!(!now.compare(&now, 0.0).has_regressions());
    assert_eq!(
        now.compare(&now, 0.0).to_markdown(),
        "No change beyond the tolerance.\n"
    );
}

#[tokio::test]
async fn a_judge_scores_against_a_rubric() {
    let judge_model = ScriptedModel::replies(vec![Reply::text(
        r#"{"score": 0.8, "reason": "names the city, not the country"}"#,
    )]);
    let judge = Judge::new("helpful", judge_model.clone(), "Names the capital city.");
    let target = target_fn(|case: Case| async move {
        let model = ScriptedModel::replies(vec![Reply::text("Paris")]);
        let mut runner = Runner::new(model, ToolRegistry::new());
        let mut conversation = case.input;
        let outcome = runner.run(&mut conversation).await?;
        Ok(Run::new(conversation, outcome))
    });
    let dataset = Dataset::new("judged", vec![Case::user("fr", "Capital of France?")]);
    let report = Eval::new("judged", &dataset, &target)
        .scorer(&judge)
        .run()
        .await;
    let score = &report.cases[0].trials[0].scores["helpful"];
    assert_eq!(score.value, 0.8);
    assert_eq!(
        score.reason.as_deref(),
        Some("names the city, not the country")
    );
    // 0.8 passes the judge's default threshold of 0.7.
    assert!(report.cases[0].trials[0].passed);
    let request = &judge_model.requests()[0];
    assert!(request.output.as_ref().unwrap().strict);
    let prompt = serde_json::to_string(&request.items).unwrap();
    assert!(prompt.contains("Names the capital city.") && prompt.contains("Paris"));
}

#[tokio::test]
async fn a_recorded_run_is_a_target_without_a_model() {
    // Record once (with a real model in practice)…
    let mut runner = Runner::new(RecordingModel::new(researcher()), tools());
    let mut conversation = Case::user("fr", "Capital of France?").input;
    runner.run(&mut conversation).await.unwrap();
    let recording = Arc::new(runner.model().recording().to_jsonl().unwrap());

    // …and evaluate on the recording, the same way every time.
    let target = target_fn(move |case: Case| {
        let recording = Arc::clone(&recording);
        async move {
            let replay = ReplayModel::from_jsonl(&recording)?;
            let mut runner = Runner::new(replay, tools());
            let mut conversation = case.input;
            let outcome = runner.run(&mut conversation).await?;
            Ok(Run::new(conversation, outcome))
        }
    });
    let dataset = Dataset::new("recorded", vec![Case::user("fr", "Capital of France?")]);
    let order = ToolOrder::new(["search", "fetch"]);
    let report = Eval::new("recorded", &dataset, &target)
        .scorer(&order)
        .trials(3)
        .run()
        .await;
    assert_eq!(report.summary.pass_all_k, 1.0);

    // A case the recording does not hold fails as an error, not a score.
    let other = Dataset::new("other", vec![Case::user("be", "Capital of Belgium?")]);
    let report = Eval::new("other", &other, &target)
        .scorer(&order)
        .run()
        .await;
    assert_eq!(report.summary.errors, 1);
    let error = report.cases[0].trials[0]
        .summary
        .as_ref()
        .unwrap()
        .error
        .clone()
        .unwrap();
    assert!(error.contains("differs from the recording"), "{error}");
}

#[tokio::test]
async fn own_scorers_and_datasets_from_json_lines() {
    let dataset = Dataset::from_jsonl(
        "mixed",
        r#"{"id": "a", "input": "Hello", "expected": "Hi"}
{"input": [{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hey"}]}]}

{"id": 7, "input": {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Yo"}]}, "metadata": {"lang": "en"}}"#,
    )
    .unwrap();
    assert_eq!(dataset.cases.len(), 3);
    assert_eq!(dataset.cases[0].expected_text(), Some("Hi"));
    assert_eq!(dataset.cases[1].id, "2");
    assert_eq!(dataset.cases[2].id, "7");
    assert_eq!(dataset.cases[2].input.len(), 2);
    assert_eq!(dataset.cases[2].metadata["lang"], "en");
    assert!(Dataset::from_jsonl("bad", r#"{"input": 3}"#).is_err());

    let short = scorer_fn("short", |_, trial| {
        let length = trial.output().map_or(usize::MAX, str::len);
        Score::pass(length <= 5).because(format!("{length} characters"))
    });
    let target = target_fn(|case: Case| async move {
        let model = ScriptedModel::replies(vec![Reply::text("Hi!")]);
        let mut runner = Runner::new(model, ToolRegistry::new());
        let mut conversation = case.input;
        let outcome = runner.run(&mut conversation).await?;
        Ok(Run::new(conversation, outcome))
    });
    let report = Eval::new("own", &dataset, &target)
        .scorer(&short)
        .run()
        .await;
    assert_eq!(report.summary.pass_rate, 1.0);
    assert_eq!(
        report.cases[0].trials[0].scores["short"].reason.as_deref(),
        Some("3 characters")
    );
}
