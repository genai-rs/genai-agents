# Tracing

The runner opens [`tracing`](https://docs.rs/tracing) spans named and attributed after OpenTelemetry's
[GenAI conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). The core does not depend on OpenTelemetry:
[`tracing-opentelemetry`](https://docs.rs/tracing-opentelemetry) exports the spans, to Langfuse through
[opentelemetry-langfuse](https://docs.rs/opentelemetry-langfuse) or to any OpenTelemetry backend.

## Spans

| Span | Around | Attributes |
| --- | --- | --- |
| `invoke_agent {name}` | a run | `gen_ai.agent.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.conversation.id` (the session), the tokens of the run |
| `chat {model}` | a model request, each retry its own | the request's model, output limit and reasoning level; `gen_ai.response.id`, `gen_ai.response.finish_reasons`; the tokens: input, output, cached (`gen_ai.usage.cache_read.input_tokens`), written to the cache, reasoning; `genai_agents.time_to_first_token_ms`; `genai_agents.cache_break` |
| `execute_tool {tool}` | a tool call, a child of the run | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type` |
| `compact {model}` | a [compaction](context.md) | the tokens of the summary request; `genai_agents.tokens_before` and `genai_agents.tokens_after` |

`otel.name` carries the names with their model or tool, and `otel.kind` the span kind. A failure sets
`error.type` (the `ErrorKind`, such as `rate_limit` or `context_overflow`) and `otel.status_code`. A retry is an
event in the request's span, with its attempt, delay and error.

`RunConfig::with_name` names the agent.

## Content

The messages of each request and response (`gen_ai.input.messages`, `gen_ai.output.messages`, as
`[{role, parts}]`), and the arguments and results of tool calls (`gen_ai.tool.call.arguments`,
`gen_ai.tool.call.result`) are recorded only with `RunConfig::with_trace_content(true)`. They can hold personal or
confidential data, and the conventions advise leaving them out unless asked for.

## Why the prompt cache broke

A provider caches the start of a request; the next request reuses it when it starts with the same tokens. When a
request does not extend the one before it, `TurnRecord::cache_break` and the span's `genai_agents.cache_break` say
why: another `model`, other `tools` (a tool search loaded one), other `settings`, another cache `key`, or a
`history` that no longer starts with what was sent (a compaction, an edited message). A cache that keeps breaking
costs money and time; this tells which change did it.

## To Langfuse

`crates/genai-agents-openai/examples/langfuse.rs` runs an agent on a session with OpenAI and sends its trace to
Langfuse. With [opentelemetry-langfuse](https://docs.rs/opentelemetry-langfuse) (Rust 1.93.1 or later), the set-up
is:

```rust,ignore
let exporter = opentelemetry_langfuse::ExporterBuilder::from_env()?.build()?;
let provider = SdkTracerProvider::builder().with_batch_exporter(exporter).build();
tracing_subscriber::registry()
    .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("genai-agents")))
    .init();
// … run …
provider.shutdown()?; // send the last batch
```

The example builds the same exporter with opentelemetry-otlp (Langfuse's endpoint, its keys as basic
authentication), to stay within this workspace's minimum Rust version:

```bash
cargo run -p genai-agents-openai --example langfuse
```

with `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` set.
