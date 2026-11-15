# Agents

An `Agent` is a [runner](runner.md) with what an application adds around it: a name, instructions, a context its
runs get, hooks, guardrails, a tool policy and an output type.

```rust
use genai_agents::agent::Agent;
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::Conversation;

struct Household {
    city: String,
}

# #[tokio::main(flavor = "current_thread")]
# async fn main() -> genai_agents::Result<()> {
let model = ScriptedModel::replies(vec![Reply::text("Take an umbrella.")]);
let mut agent = Agent::new("household", model)
    .dynamic_instructions(|home: &Household| format!("You run a household in {}.", home.city));
let home = Household { city: "Ghent".into() };
let outcome = agent.run(&home, &mut Conversation::user("Umbrella today?")).await?;
assert_eq!(outcome.output().map(String::as_str), Some("Take an umbrella."));
# Ok(())
# }
```

- **Instructions** go with every request as the system prompt (the Responses API's `instructions`; a first
  developer or system message for Chat Completions), not into the conversation. `instructions` takes a text,
  `dynamic_instructions` a function of the context.
- **The context** (`C`) is the application's: the user, a database handle, settings. Instructions, hooks and
  guardrails get `&C`. Tools take what they need when they are made.
- `with_tools`, `with_config` and `tool_policy` set the runner's tools and configuration; `runner()` and
  `runner_mut()` reach it.
- `run`, `run_with` (cancellation, events) and `run_session` (a [session](sessions.md)) return an `AgentOutcome`:
  the run's `RunOutcome`, and the answer as the output type when the model answered.

## Hooks

A hook implements `Hooks<C>`; every method has a default that does nothing.

| Method | When | Can |
| --- | --- | --- |
| `before_request` | before each request | change its settings, tools, instructions (`RequestPlan`) |
| `before_call` | before a call runs | change its arguments, or block it: `CallDecision::Block(answer)` is the call's answer, and later hooks are not asked |
| `after_call` | after a call ran | change its answer before the model sees it and the conversation keeps it |
| `on_stop` | when the model answered and the run would stop | return items to go on with |
| `before_compaction` | before a [compaction](context.md) | save what should outlive it; it cannot stop it |

Hooks run in the order they were added, each seeing what the earlier ones changed. A hook that fails ends the run with
its error.

## Guardrails

- **Input guardrails** (`InputGuardrail<C>`, or `input_guardrail_fn`) check the conversation as the run starts,
  at the same time as the first request. The first that trips cancels the run.
- **Output guardrails** (`OutputGuardrail<C>`, or `output_guardrail_fn`) check the answer's text (the JSON of a
  structured answer) before it is returned.

A tripped guardrail is not an error: the run ends with `RunStop::Guardrail { name, reason }` and no output.

## Structured output

With `output::<T>()` (feature `schemars`; `output_with_schema` takes a schema of your own), every request asks for
JSON that matches `T`'s schema, and the answer is parsed into `T`. An answer that does not parse is asked for again,
with the parse error, `with_output_retries` times (default once); after that the run fails with `Error::Output`.

```rust
# #[cfg(feature = "schemars")]
# {
use genai_agents::agent::Agent;
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::Conversation;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Forecast {
    /// Whether to take an umbrella.
    umbrella: bool,
}

# #[tokio::main(flavor = "current_thread")]
# async fn main() -> genai_agents::Result<()> {
let model = ScriptedModel::replies(vec![Reply::text(r#"{"umbrella": true}"#)]);
let mut agent = Agent::new("forecast", model).output::<Forecast>();
let forecast = agent.run(&(), &mut Conversation::user("Umbrella?")).await?.into_output()?;
assert!(forecast.umbrella);
# Ok(())
# }
# }
```

## Tool policy

`ToolPolicy` decides which registered tools a run may use, by name pattern (`*` for any run of characters, and
`namespace.name` for a tool in a namespace): a tool that matches a `deny` pattern is left out, and when there are
`allow` patterns, only the tools that match one are kept. A tool left out is not advertised, a tool search does not
find it, and a call of it is answered "not allowed" without running.

## Untrusted content

`genai_agents::untrusted` helps with content from outside (web pages, emails): `wrap` puts it between
`<untrusted source="…">` markers that it cannot close itself, and `strip_template_tokens` removes the special tokens
of chat templates (`<|im_start|>`, `[INST]`, …) with which content could pretend to start a message of its own. Tell
the model, in the instructions, to treat what is inside the markers as data.
