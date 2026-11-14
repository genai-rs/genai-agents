# Design

Status: proposed. Nothing on this page is implemented yet.

genai-agents runs agents: a model, instructions and tools, called in a loop until the model answers. It takes the
feature list of OpenAI's [Agents SDK](https://openai.github.io/openai-agents-python) (Python and TypeScript; there
is no Rust version) as its checklist, with an API of its own:

- independent of the model provider, with OpenAI first (`genai-agents-openai`);
- typed: tools, their arguments and the final output are Rust types, checked when compiling;
- traced with [OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/)'s GenAI conventions, so a run
  shows up in Langfuse (through [opentelemetry-langfuse](https://github.com/genai-rs/opentelemetry-langfuse)) or
  any other OpenTelemetry backend;
- one code base natively and on WebAssembly: the core spawns nothing and needs no runtime.

## Where it starts

openai-ergonomic 0.24 has a tool loop (`ToolLoop`) and a tool framework (`ToolRegistry`, `FunctionTool`, …). They
already do the hard parts:

- every call gets an answer, failures and cancellation included, and the history is repaired before each request;
- calls run concurrently where the tool allows it, through a read/write gate, and on the Responses API as soon as
  the model finished writing them;
- retries, with the delay the server asked for, without retrying what the client's HTTP layer already retries;
- deferred tools and tool search; output budgets; rate limits per request.

That code is copied here and made independent of OpenAI's types. openai-ergonomic keeps its own (see
[openai-ergonomic](#openai-ergonomic)).

## Compared with the Agents SDK

| Agents SDK | Here | Slice |
| --- | --- | --- |
| Agent and Runner | `Agent` and `Runner`, from `ToolLoop` | 1, 3 |
| Function tools | `FunctionTool`, `RichTool`, `CustomTool`, dynamic tools; schemas from `schemars`, then a `#[tool]` macro | 1, 3 |
| Hosted tools (web search, file search, code interpreter, hosted MCP) | passed to the provider as they are | 1 |
| Structured output (`output_type`) | `Agent<C, O>` with `O: JsonSchema + DeserializeOwned` | 3 |
| Sessions | `Session` trait; in memory, then SQLite | 2 |
| Tracing | `tracing` spans with the GenAI conventions | 2 |
| Lifecycle hooks | `Hooks`, which can stop a run or change a request, not only watch it | 3 |
| Guardrails | input guardrails run next to the first request and stop the run when they trip; output and tool guardrails | 3 |
| Context | `Agent<C>`: tools, hooks and guardrails get `&C` | 3 |
| Handoffs | a tool that hands the conversation to another agent | 4 |
| Agents as tools | a tool that runs an agent of its own | 4 |
| Human in the loop | tools that need approval; a run stops with the pending calls and resumes with the decisions | 4 |
| MCP | `genai-agents-mcp`, on [rmcp](https://docs.rs/rmcp) | 5 |
| — | Skills (`SKILL.md`), loaded when the model asks for them | 6 |
| Realtime and voice | later, if wanted | — |

## The core, `genai-agents`

### Items and the conversation

A run works on a `Conversation`: a list of `Item`s, the provider-independent form of what the Responses API calls
input items.

```rust
pub enum Item {
    Message(Message),                // role, content parts (text, image, file)
    ToolCall(ToolCall),              // call id, tool name and namespace, arguments (JSON or free text)
    ToolOutput(ToolOutput),          // call id, content parts, success
    Reasoning(Reasoning),            // summary text and the provider's opaque state (encrypted content)
    Provider(ProviderItem),          // anything else, kept as the provider's JSON (hosted tool calls, …)
}
```

- `Conversation` keeps `normalize` (unanswered calls answered `"aborted"`, orphan outputs dropped, images replaced
  for models without vision) and `remove` from openai-ergonomic.
- Items are `Serialize` and `Deserialize`, so a conversation can be stored and sent again later, also to another
  provider: a provider sends the `Provider` and `Reasoning` items it recognises and leaves out the others.

### The model

```rust
pub trait Model: MaybeSend + MaybeSync {
    /// One request: the conversation, the tools and the settings; the response as a stream of events.
    fn stream(&self, request: ModelRequest<'_>)
        -> impl Future<Output = Result<ModelStream, ModelError>> + MaybeSend;
    fn caps(&self) -> ModelCaps;
}

pub enum ModelEvent {
    TextDelta(String),
    ItemDone(Item),                  // a finished item: the loop can start a call before the response ends
    Completed(ModelResponse),        // usage, rate limits, response id, stop reason
}
```

- `ModelError` has a `RetryClass` and the delay the server asked for, and says whether it happened before the
  body (the HTTP layer's retries cover those). The runner retries from that alone.
- `ModelCaps`: images, reasoning, server-side tool search, parallel calls.
- A provider that does not stream sends one `ItemDone` per item and `Completed`.

### Tools

The tool framework comes over as it is, without the OpenAI types:

- `FunctionTool` (typed input and output), `RichTool` (images and files in the output), `CustomTool` (free text),
  dynamic tools for tools known only at run time, namespaces;
- `ToolRegistry`: names checked when registering, `answer_call` (every failure the model can act on becomes the
  answer; `ToolFailure::Fatal` ends the run), output budgets;
- `Exposure` (`Direct`, `Deferred`, `Hidden`) and tool search.

New: with the `schemars` feature, `parameters_schema` defaults to the schema of `Input`. The definitions a provider
sends (`ToolDefinition`) are neutral; the provider turns them into its own.

### The runner

`Runner` is `ToolLoop` on a `Model`: the request loop, the call dispatcher, cancellation (`CancellationSignal`),
retries (`RetryConfig`), iteration limits, deferred tools, events (`RunEvent`) and the outcome (`RunOutcome`,
`RunStop`, a `TurnRecord` per request). Its behaviour does not change; its tests come with it.

### The agent (slice 3)

```rust
let agent = Agent::new("household")
    .instructions("You are a household assistant.")
    .model(OpenAiResponses::new(client, "gpt-5"))
    .tool(Weather)
    .output::<Answer>();
let outcome = Runner::new().run(&agent, &mut session, &context, "Umbrella in Ghent today?").await?;
let answer: Answer = outcome.output()?;
```

- `Agent<C, O = String>`: a name, instructions (a string, or a function of `&C`), a model and its settings, tools,
  guardrails, hooks, handoffs and the output type.
- `Hooks<C>`: before and after a request, a tool call, a handoff and the run. A hook can change the request, deny a
  call (the denial is the call's answer) or stop the run.
- Guardrails: an input guardrail runs concurrently with the first request and cancels the run when it trips (the
  request is dropped). Output guardrails check the final output; tool guardrails a call's arguments and its answer.
  A tripped guardrail is `RunStop::Guardrail { name, reason }`, not an error.

### Sessions (slice 2)

```rust
pub trait Session {
    fn load(&self) -> impl Future<Output = Result<Vec<Item>, SessionError>>;
    fn append(&self, items: &[Item]) -> impl Future<Output = Result<(), SessionError>>;
    fn clear(&self) -> impl Future<Output = Result<(), SessionError>>;
}
```

A run loads the session, runs on it and appends the items it added, the answers included. In memory first, then
SQLite behind a feature (`sqlx`, not on WebAssembly). Other stores implement the trait.

### Tracing (slice 2)

The runner opens [`tracing`](https://docs.rs/tracing) spans with the attributes of OpenTelemetry's GenAI
conventions:

- `invoke_agent {agent}` around a run;
- `chat {model}` around a request, with the tokens and the response id;
- `execute_tool {tool}` around a call.

Message and tool content is recorded only when the application turns it on, as the conventions advise.
`tracing-opentelemetry` exports the spans, opentelemetry-langfuse sends them to Langfuse. The core does not depend
on OpenTelemetry.

### Handoffs, agents as tools, approvals (slice 4)

- A handoff is a tool (`transfer_to_<agent>`) that makes another agent the active one; the conversation goes with
  it, filtered if the handoff says so. The outcome names the agent that answered.
- An agent as a tool runs its own loop on its own conversation and answers with its output.
- A tool that needs approval is not run: the run stops with `RunStop::Approval(pending)`, the conversation keeps
  the calls, and `Runner::resume` runs the approved ones and answers the others with the denial. A stopped run is
  data (serde), so the decision can come much later, from another process.

### MCP (slice 5), `genai-agents-mcp`

The tools of an MCP server (stdio or streamable HTTP, through rmcp) as dynamic tools, in a namespace per server,
`Deferred` by default so that a large server does not fill the context. Hosted MCP (the provider calls the server)
goes through `genai-agents-openai`.

### Skills (slice 6)

A skill is a directory with a `SKILL.md` (a name and a description in its front matter, then instructions) and the
files it refers to. The runner lists the skills' names and descriptions in the instructions, and a `load_skill`
tool hands the model a skill's instructions when it asks; a `read_skill_file` tool its files. Skills are read from
directories when the agent runs, so they can change without a rebuild.

## The OpenAI provider, `genai-agents-openai`

On [openai-ergonomic](https://github.com/genai-rs/openai-ergonomic):

- `OpenAiResponses`: the Responses API, streamed, stateless by default (`store: false`, encrypted reasoning, one
  `prompt_cache_key` per run), as `ToolLoop` today;
- `OpenAiChat`: Chat Completions, which also covers the compatible providers (Azure OpenAI, OpenRouter, Groq,
  Ollama, gateways);
- the conversion between items and OpenAI's input items, tool definitions and hosted tools.

## openai-ergonomic

openai-ergonomic keeps `ToolLoop` and its tool framework: they suit an application that only talks to OpenAI and
wants no further dependency. genai-agents starts from a copy of that code and goes its own way; a fix to the shared
behaviour (history repair, the dispatcher, retries) is made in both. openai-ergonomic does not depend on
genai-agents: the dependency goes the other way.

## Slices

1. **The core and the OpenAI provider, as `ToolLoop` does today:** items and `Conversation`, `Model`, the tools,
   the runner; `OpenAiResponses` and `OpenAiChat`. The tests of `ToolLoop` and the tool framework come along and
   pass. Released as 0.1.
2. **Sessions and tracing:** the `Session` trait, in memory and SQLite; the spans, with an example that sends a run
   to Langfuse.
3. **Agents:** `Agent`, structured output, context, hooks, guardrails, and the `#[tool]` macro
   (`genai-agents-macros`).
4. **Handoffs, agents as tools, approvals.**
5. **MCP.**
6. **Skills.**

## Open questions

- **The item types:** the OpenAI conversion must not lose anything a request needs back (reasoning, hosted tool
  calls, their ids with `store: true`). `Item::Provider` is the escape hatch; slice 1 tests the round trip on the
  recorded responses of openai-ergonomic's tests.
- **`Send`:** natively a tool's future must be `Send`, as in openai-ergonomic; on WebAssembly nothing is
  (`MaybeSend`). Handoffs and agents as tools need boxed futures for the recursion.
- **Other providers:** Anthropic and Gemini are next in line for a provider crate, once the `Model` trait has been
  used by two providers (Responses and Chat Completions) without changing.
