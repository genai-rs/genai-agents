# Evals

Every change to an agent (a prompt, a tool, a model, a setting) can make it better at one thing and worse at another.
`genai-agents-eval` measures it: run the agent on a dataset, score every run, and compare with the last time.

## Cases and datasets

A `Case` is an input conversation, optionally the expected answer (a text or JSON), and metadata of its own. A
`Dataset` is a named list of them, in code or in JSON Lines (`Dataset::from_jsonl`, `Dataset::load`), a case per line:

```json
{"id": "fr", "input": "Capital of France?", "expected": "Paris"}
{"id": "chat", "input": {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]}}
```

An input is a string (a user message), an array of items (a `Conversation`'s JSON), or an object with `messages`.

## Targets

A `Target` runs on one case and returns a `Run`: the conversation, the run's outcome and the answer. `target_fn`
makes one from an async closure; each call starts from nothing, so the trials of a case are independent.
`Run::new` takes a runner's outcome, `Run::from_agent` an agent's (its structured answer as JSON).

In CI, a target runs on a [recorded run](recording.md) (`ReplayModel`): no key, the same answers every time, and a
case whose requests changed fails as an error naming the difference.

## Scorers

Each scorer gives a trial a score from 0 to 1 with a reason; a trial passes when every scorer's score reaches its
threshold (1 unless the scorer says otherwise) and the target did not fail.

| Scorer | Scores |
| --- | --- |
| `ExactMatch` | the answer equals the expected answer (as text, or as JSON) |
| `Contains`, `Matches` | the answer contains a text, matches a regular expression |
| `ParsesAs::<T>` | the answer parses as `T` (the agent's output type) |
| `ToolCalled`, `ToolNotCalled` | a tool was called (with arguments that hold these), or was not |
| `ToolOrder` | tools were called in this order; the score is the share it got through |
| `MaxRequests`, `MaxTokens`, `MaxTime` | within a budget: 1, less the more it went over |
| `Stopped` | the model answered, or a given guardrail tripped |
| `Judge` | another model scores the answer against a rubric (passes from 0.7) |
| `scorer_fn` | the application's own |

## Runs and reports

```rust,ignore
let report = Eval::new("v2 prompt", &dataset, &target)
    .scorer(&ExactMatch)
    .scorer(&ToolNotCalled::new("delete_file"))
    .trials(5)          // k: every case five times
    .concurrency(4)     // on the calling task; nothing is spawned
    .max_tokens(200_000) // the trials after that are skipped
    .run()
    .await;
std::fs::write("eval.md", report.to_markdown())?;
```

The `Report` has every trial with its scores, and per case and in total:

- **pass@k**: the share of cases that passed at least once in `k` trials;
- **pass^k**: the share that passed every time. An agent with a high pass@k and a low pass^k works only some of the
  time, which a single run hides;
- each scorer's mean, spread and share of perfect scores; tokens and time.

## Baselines

A report is JSON (`to_json`, `from_json`). Keep one as the baseline, and compare each new run with it:

```rust,ignore
let baseline = Report::from_json(&std::fs::read_to_string("baseline.json")?)?;
let comparison = report.compare(&baseline, 0.05);
println!("{}", comparison.to_markdown());
if comparison.has_regressions() {
    std::process::exit(1);
}
```

`compare` lists the totals and scorer means that moved by more than the tolerance, and the cases that always passed
before and no longer do (and the other way round).

## Langfuse

With the `langfuse` feature, `langfuse::run_dataset` runs a Langfuse dataset: every item is a case (its input and
expected output), every run a trace linked to its item in a dataset run, with the scores on it (the reason as the
comment). Langfuse then compares the runs of a dataset side by side. It returns the experiment (the run's id and URL)
and the eval's own `Report`.
