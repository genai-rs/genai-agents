# The runner

A `Runner` sends a `Conversation` to a `Model`, runs the tools it calls, sends their outputs back, and repeats until
the model answers. It is a copy of openai-ergonomic's `ToolLoop`, on a provider-independent `Model`:

- every call gets an answer, failures included, and the history is repaired before each request;
- the calls of tools that allow it run at the same time, the others alone, and the answers go into the history in
  call order, cut to their budget;
- a call starts as soon as the model finished writing it, while the response is still streaming;
- a run stops after a number of requests, or when you cancel it, with a history you can send again;
- a failed request is sent again, after the delay the server asked for or with backoff, without retrying what the
  provider's HTTP layer already retries;
- tools can wait until a tool search finds them, so that large tool sets do not fill the context;
- every request keeps its rate limits and token usage.

Nothing is spawned: the calls run on the run's own task, so the runner needs no runtime of its own and runs on
WebAssembly with tools that are not `Send`.

## A run

```rust,no_run
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{Conversation, Result, Runner};
use genai_agents_openai::OpenAiResponses;
use openai_ergonomic::{Client, ResponsesBuilder};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct City {
    city: String,
}

struct Weather;

impl FunctionTool for Weather {
    type Input = City;
    type Output = Value;
    fn name(&self) -> &str {
        "get_weather"
    }
    fn description(&self) -> &str {
        "Current weather of a city"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
    }
    async fn execute(&self, input: City) -> Result<Value> {
        Ok(json!({"city": input.city, "sky": "light rain"}))
    }
    fn supports_parallel(&self) -> bool {
        true
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?.build();
    let model = OpenAiResponses::new(
        client,
        ResponsesBuilder::new("gpt-5-mini").instructions("You are a household assistant."),
    );
    let mut tools = ToolRegistry::new();
    tools.register(Weather)?;
    let mut runner = Runner::new(model, tools);

    let mut conversation = Conversation::user("Umbrella in Brussels or in Ghent today?");
    let outcome = runner.run(&mut conversation).await?;
    println!("{}", outcome.output_text());

    // The conversation holds the whole exchange: ask a follow-up and run again.
    conversation.push_user("And tomorrow?");
    let outcome = runner.run(&mut conversation).await?;
    println!("{} ({:?})", outcome.output_text(), outcome.stop);
    Ok(())
}
```

`OpenAiChat::new(client, ChatCompletionBuilder)` is the same on Chat Completions, which also serves the providers
that speak it. Chat Completions takes function tools only, without namespaces, and text in tool messages; the
runner advertises only such tools to it and sends an output with images as a failure.

`run` returns a `RunOutcome`: why it stopped (`RunStop::Completed`, `MaxIterations`, `Cancelled`,
`Incomplete { reason }`) and a `TurnRecord` per request: the response id, its text, the calls, the retries, the
tokens (`usage`), why the model stopped (`stop`) and the rate limits.

## Models

A provider implements `Model`: it turns a `ModelRequest` (the conversation, the tools, a few settings) into its own
request, and returns the response as a stream of `ModelEvent`s: text and reasoning deltas, each output item once
it is done (`ItemDone`, so the runner can start a call at once), and `Completed` with the usage and why the model
stopped. Its `info()` (`ModelInfo`) says what it takes: images, rich tool outputs, custom tools, namespaces, a tool
search of its own, reasoning, its context window. A failure is a `ModelError`, with a `RetryClass`, the delay the
server asked for, whether it happened before the body, and an `ErrorKind` (context overflow, unknown model,
authentication, rate limit, …).

With the `testing` feature, `genai_agents::testing::ScriptedModel` answers each request as a script says and
records the requests, for tests of your own.

## Events

`run_with(&mut conversation, &cancel, on_event)` calls `on_event` with a `RunEvent`: a request about to be sent,
each event of the response (`RunEvent::Model`, for text deltas), a call queued and a call answered, a retry, and a
finished request.

```rust,no_run
use genai_agents::{CancellationSignal, Conversation, Model, ModelEvent, RunEvent, Runner};

# async fn example(runner: &mut Runner<impl Model>) -> genai_agents::Result<()> {
let mut conversation = Conversation::user("Plan my day.");
let outcome = runner
    .run_with(&mut conversation, &CancellationSignal::new(), |event| match event {
        RunEvent::Model(ModelEvent::TextDelta(text)) => print!("{text}"),
        RunEvent::ToolCalled(call) => println!("[{}]", call.name),
        RunEvent::Retrying { attempt, max, delay, error, .. } => {
            println!("[retry {attempt} of {max} in {delay:?}: {error}]");
        }
        _ => {}
    })
    .await?;
# let _ = outcome;
# Ok(())
# }
```

## The history

A `Conversation` holds `Item`s, independent of the provider: messages (`push_user`, `push_system`,
`push_developer`, `push_assistant`), tool calls and their outputs (`push_call`, `push_output`), the model's
reasoning, and items only their provider understands (hosted tools). It is `Serialize` and `Deserialize`: store it,
and run on it again later.

Before every request, retries included, the runner calls `normalize`, which makes any history one a model accepts:

- a call without an output is answered `"aborted"` (after the call and the calls and outputs that directly follow
  it);
- an output whose call is not earlier in the conversation is dropped;
- when the model takes no images, images in messages and tool outputs become a short text.

```rust
use genai_agents::{Conversation, ModelInfo, ToolCall, ToolOutput};

let mut conversation = Conversation::user("Weather in Ghent and Paris?");
let ghent = ToolCall::function("call_1", "get_weather", r#"{"city":"Ghent"}"#);
let paris = ToolCall::function("call_2", "get_weather", r#"{"city":"Paris"}"#);
conversation.push_call(&ghent);
conversation.push_call(&paris);
conversation.push_output(&ToolOutput::new(&paris, r#"{"sky":"sun"}"#, true));
// An answer to a call the conversation does not have.
let stray = ToolCall::function("call_9", "get_weather", "{}");
conversation.push_output(&ToolOutput::new(&stray, "{}", true));

let report = conversation.normalize(ModelInfo::default());
assert_eq!(report.aborted, ["call_1"]);
assert_eq!(report.orphans, ["call_9"]);
assert!(conversation.unanswered_calls().is_empty());
// Idempotent: a second pass changes nothing.
assert!(conversation.normalize(ModelInfo::default()).is_clean());
```

`remove(index)` removes an item with its partner (a call with its outputs, an output with its call).

## OpenAI: stateless requests

`OpenAiResponses` sends stateless requests by default:

- `store: false`, and every request carries the whole conversation (the items without their ids, which the server
  did not keep); requests are never chained with `previous_response_id`;
- with a reasoning model (`ModelInfo::with_reasoning(true)`, or reasoning options on the template), it asks for the
  encrypted reasoning (`include: ["reasoning.encrypted_content"]`) and sends it back, so the model keeps its
  reasoning between requests;
- one `prompt_cache_key` per runner (`PromptCacheKey::PerRunner`), the same for all its requests; `Custom(key)` or
  `Omit` instead;
- `parallel_tool_calls` when there are tools.

`with_store(true)` stores the responses (and sends the items with their ids); `with_stream(false)` waits for whole
responses. The requests go through the client's HTTP stack (its retries, provider, credentials and layers); the
client's interceptors are not called.

## Running the calls

Each call is answered with `ToolRegistry::answer_call`: failures the model can act on become its answer, a fatal
one ends the run with `Error::Tool` (the answers of the calls that finished are kept; the others are answered
`"aborted"` at the next `normalize`).

The calls go through a first-in, first-out read/write gate: a call of a tool whose `supports_parallel()` is true
starts while only such calls run; any other call waits until everything before it finished, and everything after
it waits for it. `on_event` sees the answers as they come; the history gets them in call order.
`RunConfig::with_parallel_tool_calls(false)` asks the model for one call at a time and runs every call alone.

## Limits and cancellation

`RunConfig::with_max_iterations(n)` (default 16; 0 for no limit) stops a run after `n` requests when the model
still calls tools: `RunStop::MaxIterations`, with every call answered. Run again to continue.

A `CancellationSignal` stops a run at its next await: the request or stream is dropped, the running calls are
dropped and answered `"aborted by user after 1.2s"`, the calls not started yet `"aborted"`, and the run returns
`RunStop::Cancelled`.

## Retries

A request that fails as a whole (a connection error, a stream cut off or failed midway, a failed response, a rate
limit, a server error) is sent again when its `RetryClass` allows it, up to `RetryConfig::max_retries` times
(default 5). The delay is the one the server asked for (a `retry-after-ms` or `retry-after` header, or a "try again
in 1.5s" message), else `base_delay · 2^(n − 1)` with ±10 % jitter (default from 200 ms), at most `max_delay`
(60 s). A context that is too long, an exhausted quota or a rejected request is not retried; neither is a response
that stopped early, which ends the run.

The calls the failed attempt started are not run again: the runner waits for them and records their answers, and
the retry carries them.

A failure before any of the body (a refused connection, a rate limit or server error status) is what the
provider's HTTP retries cover when it has them (`Model::retries_before_body`; for the OpenAI models, the default
client of openai-ergonomic with `max_retries` above 0). The runner then leaves those to it and retries only what it
cannot, so one request is never retried by both.

## Deferred tools and tool search

When there are [deferred](tools.md#when-the-model-sees-a-tool) tools, the runner offers a `tool_search` function
(`ToolSearchConfig`: at most 8 tools per search, its description). It answers it itself with
`ToolRegistry::search_deferred`, records the tools found (`Runner::loaded_tools`), and advertises them from the
next request on. `Runner::load_tool(key)` loads one by hand; `RunConfig::with_tool_search(None)` turns the search
off.

A model that searches tools itself (`ModelInfo::with_tool_search(true)`, OpenAI's Responses API) gets the deferred
tools with `defer_loading` and its hosted tool search instead; the tools it loads are recorded the same way.
