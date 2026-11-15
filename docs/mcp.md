# MCP

`genai-agents-mcp` gives the model the tools of [MCP](https://modelcontextprotocol.io) servers. It connects to a
server with [rmcp](https://docs.rs/rmcp), the official Rust SDK, reads the server's tools and registers them in a
`ToolRegistry` as dynamic tools; a call of one goes to the server.

```bash
cargo add genai-agents genai-agents-mcp
```

Natively only: the crate is empty on WebAssembly.

## A server's tools

```rust,no_run
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::{Conversation, Runner, ToolRegistry};
use genai_agents_mcp::{McpOptions, McpServer};
use tokio::process::Command;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let mut command = Command::new("npx");
command.args(["-y", "@modelcontextprotocol/server-filesystem", "/srv/notes"]);
let notes = McpServer::stdio("notes", command, McpOptions::new()).await?;

let model = ScriptedModel::replies(vec![Reply::text("You have three notes on the garden.")]);
let mut runner = Runner::new(model, ToolRegistry::new());
for tool in notes.register(runner.registry_mut()).await? {
    println!("{} (read-only: {})", tool.key.name, tool.read_only());
}
let outcome = runner.run(&mut Conversation::user("What did I write about the garden?")).await?;
println!("{}", outcome.output_text());
# Ok(())
# }
```

A server is reached in one of three ways:

- `McpServer::stdio(name, command, options)` starts a command and talks to it over its standard input and output
  (the `stdio` feature);
- `McpServer::http(name, url, options)` connects over streamable HTTP (the `http` feature, with rustls by default
  or `native-tls`);
- `McpServer::connect(name, transport, options)` takes any rmcp transport: a `StreamableHttpClientTransport` built
  with an authorization header, or an in-process pipe (`tokio::io::duplex`) to a server in the same program.

`name` names the server in the registry. `rmcp` is re-exported, and `McpServer::peer` gives the connection for what
this crate does not cover (resources, prompts).

## How the tools are registered

`register(&mut registry)` reads the server's tool list and registers each tool with `register_dynamic`, which adapts
what a server sends to what the model APIs accept:

- **Names.** A name is cleaned up (`github.create-issue` becomes `github_create_issue`), and one that is too long
  or already taken gets a hash of the original: `look.up` and `look-up` become `look_up` and `look_up_<hash>`.
  `McpTool::name` is the server's name, `McpTool::key` the one the model calls.
- **Schemas.** A missing `type` is filled in, `const` becomes a one-value `enum`, definitions that nothing refers
  to are dropped, and a large schema is compacted (see [tools](tools.md#rich-custom-and-dynamic-tools)). A tool
  whose schema is not an object is left out, with a warning.
- **Grouping.** The tools go into a namespace named after the server, described with the server's description (its
  title or its instructions when it has none, or `McpOptions::description`). Only models that take namespaces see
  namespaced tools; for the others (Chat Completions), `Grouping::Prefix` registers them outside namespaces as
  `server__tool`.
- **Exposure.** The tools are `Deferred`: the model finds them with a tool search (the runner's own, or the
  model's), so that a server with many tools does not fill the context. `McpOptions::exposure(Exposure::Direct)`
  advertises them in every request.
- **Parallel calls.** A tool the server marks read-only (`readOnlyHint`) may run at the same time as other tools
  that allow it.
- **Untrusted output.** A server's output is content from outside the application: the answers are marked
  `untrusted`. `McpOptions::untrusted_output(false)` is for a server you trust.

The server's instructions are not sent to the model: `McpServer::instructions` returns them, for the agent's
instructions if the model needs them.

## What a call returns

| The server sends | The model gets |
| --- | --- |
| text | text (several text blocks on separate lines) |
| an image | an image (a `data:` URL) |
| an embedded resource | its text, or a file (an image for an image type) |
| a resource link | a JSON description of the link, as text |
| audio | a note that it was left out |
| only structured content | its JSON, as text |
| `isError: true` | a failure: the text, with `success: false` |

A request that fails (the server is gone, it does not know the tool) is a failure the model sees too. A model that
takes only text in tool outputs gets an output with images or files as a failure.

## Hints for approvals

`McpServer::tools()` lists the registered tools with the server's annotations, and `McpServer::tool(&key)` finds
the one a call names. `read_only()`, `destructive()`, `idempotent()` and `open_world()` read them with the
protocol's defaults: a tool without hints may destroy something. They are hints from the server: a policy should not
trust them more than it trusts the server.

## A changed tool list

A server can say its tool list changed (`notifications/tools/list_changed`). `tools_changed()` is then true, and
`changed().await` waits for it. `register` again replaces the server's tools: the namespace (or, with prefixes, the
tools of the last registration) is removed and the new list is registered.

A run keeps the tools it started with: change a runner's tools between runs, with `Runner::registry_mut`, which
copies the registry first when a run or the application still shares it.

```rust,no_run
# use genai_agents::{Conversation, Model, Runner};
# use genai_agents_mcp::McpServer;
# async fn example(mut runner: Runner<impl Model>, notes: McpServer) -> Result<(), Box<dyn std::error::Error>> {
if notes.tools_changed() {
    notes.register(runner.registry_mut()).await?;
}
runner.run(&mut Conversation::user("And about the shed?")).await?;
# Ok(())
# }
```

A call that is running when the list changes finishes with the tool it started with: the registry it was found in
is not changed, only a copy is. A new call of a tool that is gone is answered `unsupported call`.

## The trace context

A call can take the W3C trace context to the server, in the request's `_meta` (`traceparent` and `tracestate`), so
that the server's spans join the run's trace. The crate does not depend on OpenTelemetry, and a `tracing` span has
no trace id of its own: `McpOptions::trace_context` says where the context comes from. It runs as the call starts,
inside the call's span. With `tracing-opentelemetry` and OpenTelemetry's propagator:

```rust,ignore
use genai_agents_mcp::{McpOptions, TraceContext};
use opentelemetry::propagation::TextMapPropagator;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use tracing_opentelemetry::OpenTelemetrySpanExt;

let options = McpOptions::new().trace_context(|| {
    let mut headers = std::collections::HashMap::new();
    TraceContextPropagator::new().inject_context(&tracing::Span::current().context(), &mut headers);
    let context = TraceContext::new(headers.remove("traceparent")?);
    Some(match headers.remove("tracestate") {
        Some(state) if !state.is_empty() => context.with_tracestate(state),
        _ => context,
    })
});
```

## Closing

The registered tools hold the connection: it closes when the `McpServer` handles and the tools are dropped (a child
process is then ended), or with `close()`. Calls of its tools fail after that.
