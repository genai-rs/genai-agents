# Design

Status: implemented, slices 1 to 9. The guides describe each part as it is; where the implementation differs from a slice below, its pull request says so.

genai-agents runs agents: a model, instructions and tools, called in a loop until the model answers. It takes the
feature list of OpenAI's [Agents SDK](https://openai.github.io/openai-agents-python) (Python and TypeScript; there
is no Rust version) as its checklist, with an API of its own:

- independent of the model provider, with OpenAI first (`genai-agents-openai`);
- typed: tools, their arguments and the final output are Rust types, checked when compiling;
- traced with [OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/)'s GenAI conventions, so a run
  shows up in Langfuse (through [opentelemetry-langfuse](https://github.com/genai-rs/opentelemetry-langfuse)) or
  any other OpenTelemetry backend;
- one code base natively and on WebAssembly: the core spawns nothing and needs no runtime.

It is a library: channels, schedules, memory files, a skill registry and sandboxes belong to the applications built
on it. Where the design borrows from [other agents](#prior-art), the section says so.

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
| Agent and Runner | `Agent` and `Runner`, from `ToolLoop` | 1, 4 |
| Function tools | `FunctionTool`, `RichTool`, `CustomTool`, dynamic tools; schemas from `schemars`, then a `#[tool]` macro | 1, 4 |
| Hosted tools (web search, file search, code interpreter, hosted MCP) | passed to the provider as they are | 1 |
| Structured output (`output_type`) | `Agent<C, O>` with `O: JsonSchema + DeserializeOwned` | 4 |
| Sessions | an append-only log of entries, with fork and rollback | 2 |
| Tracing | `tracing` spans with the GenAI conventions | 2 |
| — | Context management: token accounting, compaction, pruning | 3 |
| — | Input while a run is going (steering, follow-ups) | 3 |
| Lifecycle hooks | `Hooks`, which can change a request, block a call or keep a run going | 4 |
| Guardrails | on top of hooks; an input guardrail runs next to the first request | 4 |
| Context | `Agent<C>`: tools, hooks and guardrails get `&C` | 4 |
| — | Evals: datasets, scorers, reports, Langfuse dataset runs; recorded runs for CI | 2, 5 |
| Human in the loop | an approval policy; a run stops with the pending calls and resumes with the decisions | 6 |
| Handoffs, agents as tools | a tool that hands over, a tool that runs an agent; sub-agents with limits | 6 |
| MCP | `genai-agents-mcp`, on [rmcp](https://docs.rs/rmcp) | 7 |
| — | Skills (`SKILL.md`) and instruction files (`AGENTS.md`) | 8 |
| — | Fallback models and credentials | 9 |
| Realtime and voice | later, if wanted | — |

## The core, `genai-agents`

### Items and the conversation

A run works on a `Conversation`: a list of `Item`s, the provider-independent form of what the Responses API calls
input items.

```rust
#[non_exhaustive]
pub enum Item {
    Message(Message),                // role, content parts (text, image, file)
    ToolCall(ToolCall),              // call id, tool name and namespace, arguments (JSON or free text)
    ToolOutput(ToolOutput),          // call id, content parts, success
    Reasoning(Reasoning),            // summary text and the provider's opaque state (encrypted content)
    Provider(ProviderItem),          // anything else, kept as the provider's JSON (hosted tool calls, …)
}
```

- **Origin.** Reasoning, tool calls and provider items record the API, provider and model that produced them, and a
  tool call keeps the provider's opaque state for it (a Gemini thought signature). Messages record whether they are
  real input or injected context (instructions, a skill, a "run was interrupted" note), so that compaction can strip
  injected context and put it back (from pi and Codex).
- **Projection.** The stored conversation keeps the items as they came. Each request is built from a projection for
  the target model (`for_request`), which:
  - answers unanswered calls `"aborted"`, with an id derived from the call's, so that two projections of the same
    history are byte for byte the same and the prompt cache holds (Codex);
  - drops outputs whose call is missing;
  - replaces images for a model without vision;
  - drops reasoning and provider state from another model, never turning it into text (pi);
  - rewrites call ids the target refuses (Anthropic takes at most 64 characters of `[a-zA-Z0-9_-]`; Responses ids
    are longer).
- Items are `Serialize` and `Deserialize`, so a conversation can be stored and sent again later, also to another
  provider.

### The model

```rust
pub trait Model: MaybeSend + MaybeSync {
    /// One request: the conversation, the tools and the settings; the response as a stream of events.
    fn stream(&self, request: ModelRequest<'_>)
        -> impl Future<Output = Result<ModelStream, ModelError>> + MaybeSend;
    fn info(&self) -> &ModelInfo;
}

#[non_exhaustive]
pub enum ModelEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallArgsDelta { call_id: String, delta: String },
    ItemDone(Item),                  // a finished item: the loop can start a call before the response ends
    Completed(ModelResponse),        // stop reason, usage, rate limits, response id
}
```

- `ModelInfo`: the context window, the maximum output, the input kinds (text, images), reasoning, server-side tool
  search, parallel calls (Codex's and pi's model metadata).
- Model settings carry a reasoning level (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`) that each provider
  maps to its own values (pi).
- `Usage` counts input, output, cached and cache-write tokens and reasoning tokens.
- A stop reason: `Stop`, `ToolUse`, `Length`, `ContentFilter`, `Aborted`. When a reply stopped at the output limit,
  its tool calls are answered with an error, not run from arguments that may be cut off (pi).
- `ModelError` has a class the runner acts on:
  - retryable, with the delay the server asked for, and whether it happened before the body (the HTTP layer's
    retries cover those);
  - context overflow: not retried, compacted ([slice 3](#context-management-slice-3));
  - unknown model or refused credentials: not retried, a [fallback](#fallback-models-and-credentials-slice-9)
    may take over;
  - terminal (content filter, incomplete response).
- A provider that does not stream sends one `ItemDone` per item and `Completed`.

### Tools

The tool framework comes over as it is, without the OpenAI types:

- `FunctionTool` (typed input and output), `RichTool` (images and files in the output), `CustomTool` (free text),
  dynamic tools for tools known only at run time, namespaces;
- `ToolRegistry`: names checked when registering, `answer_call` (every failure the model can act on becomes the
  answer; `ToolFailure::Fatal` ends the run), output budgets;
- `Exposure` (`Direct`, `Deferred`, `Hidden`) and tool search. Codex and pi have the same three; tool search ranks
  with BM25 there, here with words of the query until that is not enough.

Additions:

- with the `schemars` feature, `parameters_schema` defaults to the schema of `Input`;
- a tool output has `content` for the model and `details` for the application only, which the model never sees (pi);
- a tool can mark its output as untrusted (content from the network). The runner records it on the turn; guardrails
  and approvals can use it, for instance to ask before a tool with side effects runs in a turn that read a web page
  (OpenClaw's turn taint);
- an output cut to its budget keeps its head and its tail, with a line that says how much was left out (Codex). A
  session stores the whole output;
- a call cancelled before it started is answered "aborted"; one cancelled while running "aborted after 1.2s".

### The runner

`Runner` is `ToolLoop` on a `Model`: the request loop, the call dispatcher, cancellation (`CancellationSignal`),
retries (`RetryConfig`, with an idle timeout for a stream that stops sending), iteration limits, deferred tools,
events (`RunEvent`, `#[non_exhaustive]`, with `Retrying { attempt, max, delay }`) and the outcome (`RunOutcome`,
`RunStop`, a `TurnRecord` per request). Its behaviour does not change; its tests come with it. A retry is built
again from the conversation, so the answers of calls the failed attempt started are kept.

Guards (slice 3, from OpenClaw): the same call with the same arguments failing three times is no longer run; a tool
call the model wrote as text gets one request to make it again as a call, and is never run from the text.

### Sessions (slice 2)

A session is an append-only log of entries, not a list of items (Codex's rollout files, pi's session tree):

```rust
#[non_exhaustive]
pub enum SessionEntry {
    Item(Item),
    TurnSettings(TurnSettings),      // the model and its settings from here on
    Usage(Usage),                    // what the server reported
    Compaction(Compaction),          // a summary and the first entry it keeps
    Label(String),
    Custom(serde_json::Value),       // the application's own, never sent to the model
}
```

- The conversation is rebuilt from the log, from the latest compaction on. Nothing is deleted: compaction and
  pruning change what is sent, not what is stored.
- Entries have ids and parents, so a session can fork (from the `n`th user message) and roll back.
- `append` takes the run that writes and a key per input: a run that was superseded cannot write, and the same
  input sent twice is stored once (OpenClaw).
- A session has an explicit id; there is no shared default (OpenClaw's shared default session leaked one user's
  conversation to another).
- Stores: in memory, JSONL files, SQLite (`genai-agents-sqlite`, on rusqlite with SQLite built in; not on
  WebAssembly). Other stores implement the trait.

### Tracing (slice 2)

The runner opens [`tracing`](https://docs.rs/tracing) spans with the attributes of OpenTelemetry's GenAI
conventions:

- `invoke_agent {agent}` around a run;
- `chat {model}` around a request: the tokens (cached ones included), the response id, the time to the first
  token, and why the prompt cache could not be used when the request no longer extends the previous one (the tools,
  the reasoning or the cache key changed; Codex);
- `execute_tool {tool}` around a call;
- a span for a compaction, and an event for a retry.

Message and tool content is recorded only when the application turns it on, as the conventions advise.
`tracing-opentelemetry` exports the spans, opentelemetry-langfuse sends them to Langfuse. The core does not depend
on OpenTelemetry. MCP calls pass the trace context on to the server ([slice 7](#mcp-slice-7-genai-agents-mcp)).

### Context management (slice 3)

None of the Agents SDK, but each of Codex, pi and OpenClaw needed it: a long run outgrows the context window.

- **Token accounting:** the usage the server reported last, plus an estimate for the items added since (Codex).
- **A `ContextStrategy`** between the conversation and the request, with the default:
  - **compaction** when the estimate passes a share of the window (90 %), or when the model answers with a context
    overflow: the model summarises the history (on top of the previous summary), the summary replaces it, and the
    recent user messages are kept up to a budget. It never cuts between a call and its answer, and it rejects a
    summary that was itself cut off at the output limit. Injected context is put back after it;
  - **pruning:** old tool outputs are cut, first to their head and tail, later to a placeholder, only once the
    prompt cache for them has expired (OpenClaw);
  - **switching models:** when the next model has a smaller window, or cannot read the previous model's reasoning,
    compact with the previous model first (Codex).
- **Input while a run is going:** the application pushes user messages into the run. Steering messages go in after
  the current calls are answered; follow-ups when the run would otherwise stop. The runner asks for them, so it
  still spawns nothing (pi, OpenClaw). A cancelled run leaves a note in the conversation that tools may have run
  partly (Codex).

### Agents, hooks and guardrails (slice 4)

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
  a tool policy, guardrails, hooks, handoffs and the output type.
- `Hooks<C>`, around a request, a call, a compaction, a handoff and the run:
  - before a request: change the model, its settings or the tools for that request;
  - before a call: change its arguments, or block it (the block is the call's answer; a later hook cannot undo
    it);
  - after a call: change what is stored, apart from what the model sees;
  - when the run would stop: keep it going with more input (Codex's stop hook);
  - around a compaction: watch, not veto.
- Guardrails are hooks with a name and a verdict. An input guardrail runs concurrently with the first request and
  cancels the run when it trips. A tripped guardrail is `RunStop::Guardrail { name, reason }`, not an error.
- A tool policy filters the tools before their exposure: deny lists win, a non-empty allow list leaves out the rest,
  groups by name pattern (OpenClaw).
- Helpers for untrusted content: wrap it in markers, strip chat-template tokens from it (OpenClaw).

### Evals (slices 2 and 5), `genai-agents-eval`

An agent changes with every prompt, tool, model or setting; evals say whether the change made it better. The Agents
SDK leaves them to OpenAI's platform; here they are a crate that runs anywhere, CI included.

- **Recorded runs (slice 2).** A `Model` that records every request and response to a file, and one that replays
  them. Tests and evals then run without a key, the same way every time; a request that no longer matches the
  recording fails the replay and names the difference. Live runs stay possible.
- **Cases and datasets.** A case is an input (a conversation, a session to start from, a context), optionally an
  expected output, and what to check. A dataset is a list of cases in JSONL, or a Langfuse dataset.
- **Scorers**, each a number from 0 to 1 with a reason:
  - the output: equal, contains, a regular expression, matches a JSON schema or the agent's output type;
  - the trajectory: which tools were called, with which arguments, in which order, which were not, how many
    requests it took, whether a guardrail tripped;
  - the cost: tokens, money (from `ModelInfo`), time to the answer;
  - an LLM judge with a rubric, on another model than the one judged;
  - a function of the application's own.
- **Runs.** Every case runs `n` times, a few at once, within a budget. Reports give the mean and spread per scorer,
  and pass@k (it worked at least once in `k`) next to pass^k (it worked all `k` times), since an agent that works
  one time in three is not reliable.
- **Reports and comparisons.** A run writes JSON and Markdown; compared with a baseline run it lists what got worse,
  so CI can fail on a regression beyond a threshold.
- **Langfuse.** Each case's run is a trace (slice 2's spans), linked to the dataset item as a dataset run, with the
  scores on it, through [langfuse-ergonomic](https://github.com/genai-rs/langfuse-ergonomic). Runs can then be
  compared in Langfuse, and scores given by people there come back as data.

### Approvals and sub-agents (slice 6)

- **Approvals.** An `ApprovalPolicy` decides first (never ask, ask, a cache of earlier decisions, an LLM reviewer).
  When it asks, the run stops with `RunStop::Approval(pending)`; `Runner::resume` takes the decisions:
  - approve, or approve for the session: a tool names the key a decision covers (a file, a command), so the next
    call with the same key is not asked again (Codex);
  - deny and continue: the denial is the call's answer;
  - abort the run.

  A pending call is bound to a hash of its arguments, checked again on resume (OpenClaw). A tool says whether a call
  whose result is unknown, after a crash, may run again. A stopped run is data (serde), so the decision can come much
  later, from another process.
- **Handoffs:** a tool (`transfer_to_<agent>`) that makes another agent the active one; the conversation goes with
  it, filtered if the handoff says so.
- **Agents as tools:** a tool that runs an agent on a conversation of its own and answers with its output.
- **Sub-agents:** an agent started without waiting, whose result comes back as input when it finishes; a limit on
  depth and on children per agent; a child gets its parent's tool policy at most, and stops when its parent stops
  (OpenClaw).

### MCP (slice 7), `genai-agents-mcp`

The tools of an MCP server (stdio or streamable HTTP, through rmcp) as dynamic tools:

- in a namespace per server, `Deferred` by default so that a large server does not fill the context; the server's
  description helps tool search (pi);
- schemas cleaned up for the models (a missing `type`, `const`, unreachable `$defs`), names kept short with a hash
  when two collide (Codex);
- a changed tool list takes effect for new calls; running calls keep the list they started with (Codex);
- the server's `readOnlyHint` makes a tool parallel in the dispatcher; `destructiveHint` is there for approvals;
- the trace context goes to the server.

Hosted MCP (the provider calls the server) goes through `genai-agents-openai`.

### Skills and instruction files (slice 8)

- **Skills** follow the [Agent Skills](https://agentskills.io) format: a directory with a `SKILL.md` (a name and a
  description in its front matter, then instructions) and the files it refers to.
  - The instructions list each skill's name and description, within a budget (a share of the context window) that
    drops descriptions before names (Codex, OpenClaw).
  - With a file system, the model reads `SKILL.md` with the agent's own read tool (pi); without one (WebAssembly), a
    `load_skill` tool hands it over.
  - A skill can require binaries, variables or an operating system, and be hidden from the model
    (`disable-model-invocation`).
  - Skills are untrusted: they are read only from the directories the application lists, symbolic links out of a
    skill's directory are refused, and nothing in a skill is run when it is loaded (OpenClaw's malicious skills).
- **Instruction files:** `AGENTS.md` from the project root down to the working directory, within a budget (Codex).

### Fallback models and credentials (slice 9)

A `Model` that wraps others: on refused or expired credentials, another credential of the same provider; on an
unknown model or a provider that keeps failing, the next model. A session keeps the credential it started with, for
the provider's cache; a credential waits only as long as the provider asked; local faults never fall back
(OpenClaw).

## The OpenAI provider, `genai-agents-openai`

On [openai-ergonomic](https://github.com/genai-rs/openai-ergonomic):

- `OpenAiResponses`: the Responses API, streamed, stateless by default (`store: false`, encrypted reasoning, one
  `prompt_cache_key` per run), as `ToolLoop` today;
- `OpenAiChat`: Chat Completions, which also covers the compatible providers (Azure OpenAI, OpenRouter, Groq,
  Ollama, gateways). "Compatible" differs per server, so a `Compat` setting says what the server takes: the
  developer role, a reasoning effort, usage in streams, `max_tokens` or `max_completion_tokens`, a name on tool
  results, `store` (pi);
- the conversion between items and OpenAI's input items, tool definitions and hosted tools.

## openai-ergonomic

openai-ergonomic keeps `ToolLoop` and its tool framework: they suit an application that only talks to OpenAI and
wants no further dependency. genai-agents starts from a copy of that code and goes its own way; a fix to the shared
behaviour (history repair, the dispatcher, retries) is made in both. openai-ergonomic does not depend on
genai-agents: the dependency goes the other way.

## Left to applications

From OpenClaw, which is an assistant built on an agent core: channels and their session keys, scheduled runs,
memory files (`MEMORY.md`, daily notes), a skill registry with its scanning, container sandboxes. The library gives
them the hooks (sessions with explicit ids, input during a run, a pre-compaction hook to save memories, a `Memory`
trait if two applications need the same one).

## Slices

1. **The core and the OpenAI provider, as `ToolLoop` does today**, with the item origins, the projection, the model
   metadata, the stop reasons and the error classes above, so that later slices do not break 0.1. Released as 0.1.
2. **Sessions, tracing and recorded runs:** the entry log with fork and rollback, in memory, JSONL and SQLite; the
   spans, with an example that sends a run to Langfuse; the recording and replaying models.
3. **Context management:** token accounting, compaction, pruning, input while a run is going, the guards.
4. **Agents:** `Agent`, structured output, context, hooks, guardrails, the tool policy, and the `#[tool]` macro
   (`genai-agents-macros`).
5. **Evals:** cases, datasets, scorers, repeated runs, reports with a baseline, Langfuse dataset runs and scores.
   From here on, every slice comes with evals of its own.
6. **Approvals, handoffs, agents as tools, sub-agents.**
7. **MCP.**
8. **Skills and instruction files.**
9. **Fallback models and credentials.**

## Prior art

| Project | Language, license | What it showed |
| --- | --- | --- |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | Python, MIT | the feature list: agents, handoffs, guardrails, sessions, tracing |
| [Codex CLI](https://github.com/openai/codex) | Rust, Apache-2.0 | the projection with stable ids, token accounting, compaction, the session log, approvals with keys, cache diagnostics, MCP schema clean-up |
| [pi](https://github.com/earendil-works/pi) | TypeScript, MIT | items that know their model, the reasoning level, `Compat` for compatible servers, steering, the session tree, how little an agent core needs |
| [OpenClaw](https://github.com/openclaw/openclaw) | TypeScript, MIT | turn taint, session writers, pruning by cache lifetime, failover, sub-agent limits, skill gating, what goes wrong with skills |

Where code is borrowed rather than ideas, the file says where it comes from, with the license's notice.

## Open questions

- **Handoffs and guardrails:** pi has neither and lets hooks do it. Here they stay, as thin layers on hooks, since
  they are what users of the Agents SDK look for.
- **`Send`:** natively a tool's future must be `Send`, as in openai-ergonomic; on WebAssembly nothing is
  (`MaybeSend`). Handoffs and agents as tools need boxed futures for the recursion.
- **Other providers:** Anthropic and Gemini are next in line for a provider crate, once the `Model` trait has been
  used by two providers (Responses and Chat Completions) without changing.
