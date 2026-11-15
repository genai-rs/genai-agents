# Recorded runs

A test or an eval that calls a real model needs a key, costs money, and answers differently each time. A recorded
run calls the model once and replays its answers after that: without a key or a network, the same way every time.

## Record

`RecordingModel` wraps a model, passes every request on, and records it with the response: its events, its rate
limits, or the error it failed with.

```rust,no_run
use genai_agents::record::RecordingModel;
use genai_agents::{Conversation, Model, Runner, ToolRegistry};

# async fn example(live: impl Model) -> Result<(), Box<dyn std::error::Error>> {
let mut runner = Runner::new(RecordingModel::new(live), ToolRegistry::new());
runner.run(&mut Conversation::user("Umbrella in Ghent today?")).await?;
runner.model().save("tests/recordings/umbrella.jsonl")?;
# Ok(())
# }
```

The file is [JSON Lines](https://jsonlines.org): the model's `ModelInfo` on the first line, then a request and its
response per line, so a review shows a changed request as a changed line.

## Replay

`ReplayModel` answers the `n`th request with the `n`th recorded response, after checking that the request is the one
that was recorded: its items, tools, settings and parallel calls (the prompt cache key is left out; the runner's
default one is random).

```rust,no_run
use genai_agents::record::ReplayModel;
use genai_agents::{Conversation, Runner, ToolRegistry};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let replay = ReplayModel::load("tests/recordings/umbrella.jsonl")?;
let mut runner = Runner::new(replay, ToolRegistry::new());
let outcome = runner.run(&mut Conversation::user("Umbrella in Ghent today?")).await?;
assert_eq!(runner.model().remaining(), 0); // every recorded request was asked for
# let _ = outcome;
# Ok(())
# }
```

A request that differs fails the replay, not retried, with the first difference as a JSON pointer and both values:

```text
replay: request 1 differs from the recording at /tools/0/description: recorded "Current weather of a city", now "The weather, now"
```

A changed prompt, tool or setting then shows up as a failing test, and the recording is made again on purpose rather
than drifting. A request beyond the recording fails the same way.

Failures replay too: a request that failed with a `503` and was retried replays as a failure and a retry, so the
runner's retries can be tested against what a provider really sent.
