# Agents

An `Agent` is a [runner](runner.md) with what an application adds around it: a name, instructions, a context its
runs get, hooks, guardrails, a tool policy, an output type, approvals, handoffs and sub-agents.

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

## Approvals

An `ApprovalPolicy` decides before a call runs: approve, deny (the reason is the call's answer), or ask a person.
`AskFor::tools(["delete_*", "send_*"])` asks for the tools that match and approves the rest; a policy of your own can
look at the arguments, keep a record, or ask another model.

When the policy asks, the other calls of the response run, and the run stops with `RunStop::Approval { pending }`.
A `PendingCall` is data (`Serialize`): show it, store it, and decide later, in another process if need be. Each
decision is bound to a hash of the call's arguments:

- `approve()`: run it;
- `approve_for_session()`: run it, and the later calls with the same key (by default the tool; a policy's `key`
  can narrow it to a file or a command) without asking. `Runner::approved_keys` and `approve_key` carry the keys
  over to another runner;
- `deny(reason)`: do not run it; the model gets the reason;
- `abort()`: stop the run; the pending calls are answered "aborted".

`Runner::resume` (and `Agent::resume`, `Agent::resume_session`) takes the decisions, runs what was approved, and
goes on. A decision for a call that is no longer waiting, or whose arguments changed, fails with `Error::Approval`.

```rust,ignore
let mut agent = Agent::new("ops", model).with_tools(tools).approvals(AskFor::tools(["delete_*"]));
let outcome = agent.run_session(&ctx, &mut session, &cancel, |_| {}).await?;
if let RunStop::Approval { pending } = &outcome.run.stop {
    // … ask, maybe hours later, in another process that opens the same session …
    let decisions: Vec<_> = pending.iter().map(PendingCall::approve).collect();
    agent.resume_session(&ctx, &mut session, &decisions, &cancel, |_| {}).await?;
}
```

## Handoffs

`handoff(other)` gives the model a `transfer_to_<name>` tool (with the other agent's `description`). When the model
calls it, the run ends with `RunStop::Handoff`, and the agent runs `other` on the conversation; `other`'s answer is
the run's, parsed as this agent's output type and checked by its output guardrails. `handoff_filtered` gives the
other agent a filtered conversation (`without_tool_calls`: only the messages); what it adds goes into the whole one.

## Agents as tools

`AgentTool::new(name, description, agent, context)` is a tool that runs `agent` on a conversation of its own (the
call's `input` as the user's message) and answers with its answer: an expert the model consults and gets back from.

## Sub-agents

`sub_agent(name, description, make)` gives the model a `start_<name>` tool that starts a fresh agent (`make()`) on a
task and answers at once. The run goes on; the sub-agent's answer comes back as a user message when it finishes, and
a run that would stop waits for its sub-agents first. Nothing is spawned: the runner polls the sub-agents with the
response and the calls, on its own task. When the run ends (answered, cancelled, failed), its sub-agents stop.

- `SubAgentLimits`: how many run at once (default 4), and how deep sub-agents may start sub-agents (default 2);
  a sub-agent keeps the stricter of its own limit and its parent's.
- A sub-agent uses only the tools its parent's policies allow, besides its own policy.
