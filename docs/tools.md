# Tools

A tool is a Rust type: implement `FunctionTool` (typed JSON arguments and output), `RichTool` (an output with
images or files) or `CustomTool` (free text input, optionally constrained by a grammar), or build a `DynamicTool`
for a tool only known at run time. Register tools in a `ToolRegistry`; a `Runner` advertises them to the model and
answers its calls. The registry works on its own too: `answer_call` answers one call.

The code of this page is a copy of openai-ergonomic's tool framework, without OpenAI's types.

## A function tool

```rust
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{Result, ToolCall};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchInput {
    query: String,
    limit: Option<u32>,
}

struct Search;

impl FunctionTool for Search {
    type Input = SearchInput;
    type Output = Vec<String>;

    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search the documentation and return matching page titles."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for"},
                "limit": {"type": ["integer", "null"], "description": "At most this many results"}
            },
            "required": ["query"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: SearchInput) -> Result<Vec<String>> {
        let limit = input.limit.unwrap_or(3) as usize;
        Ok(vec![format!("Results for {}", input.query)].into_iter().take(limit).collect())
    }

    fn supports_parallel(&self) -> bool {
        true
    }
}

# #[tokio::main(flavor = "current_thread")]
# async fn main() -> Result<()> {
let mut tools = ToolRegistry::new();
tools.register(Search)?;
let answer = tools
    .answer_call(ToolCall::function("call_1", "search", r#"{"query":"Rust"}"#))
    .await?;
assert!(answer.success);
assert_eq!(answer.body.as_text(), Some(r#"["Results for Rust"]"#));
# Ok(())
# }
```

`execute` is a plain `async fn`. Natively its future must be `Send`; on WebAssembly neither the tool nor its future
needs to be. Keep state (clients, `Arc`s) in the tool and read it through `&self`.

`register` checks the tool: a name of 1 to 64 ASCII letters, digits, underscores or hyphens, not taken; a
description that is not blank; a schema whose root is an object. It fails without changing the registry. The
definitions are read once, when registering.

### The schema

Write the schema for the model, with a description for each property, and keep it in line with `Input`: list the
fields without a default in `required`, and pair `additionalProperties: false` with
`#[serde(deny_unknown_fields)]`. The registry does not validate arguments against the schema: Serde decodes them,
and arguments that are not a JSON object or do not decode are answered with the error, without running the tool.

With the `schemars` feature, `tools::schema::for_type::<Input>()` derives the schema from the input type (doc
comments become descriptions, definitions are inlined):

```rust,ignore
fn parameters_schema(&self) -> Value {
    genai_agents::tools::schema::for_type::<SearchInput>()
}
```

## Answers and failures

`answer_call` gives an answer to every call it can: the output, or a failure the model can act on, as the output
with `success: false`:

- arguments that are not JSON, not an object, or not the tool's input: `failed to parse function arguments: …`;
- an unknown tool: `unsupported call: <name>`;
- a call of the wrong kind (text for a function tool, JSON for a custom one);
- an error of the tool's own: its message. Return `Error::application(error)` to keep the error's source.

A tool that must stop the run instead returns `Error::fatal(error)`: `answer_call` returns it as an error, and the
runner ends the run with `Error::Tool`. `answer_call_with` takes a classification of your own, for example to hide
the tools' error messages from the model.

How much of an output the model sees is cut to a budget: the tool's own (`output_budget`), else the runner's
(`RunConfig::truncation`, 10 000 bytes by default). The middle of the text is left out, with a line saying how
long it was.

A tool whose output holds content from outside the application (a web page, an email) says so with
`untrusted_output`: its answers are marked `untrusted`, which the runner records.

What only the application should see goes into the answer's `details`, which the model never gets: a function tool
fills it from its output with `details` (the full data behind a short answer, a diff to show), and the application
can set it on any answer (`ToolOutput::with_details`).

## Rich, custom and dynamic tools

- `RichTool` returns a `ToolOutputBody`: text, or content parts (`Content::text`, `Content::image_url`,
  `Content::file_id`, …). A model that takes only text in tool outputs (Chat Completions) gets an output with images
  or files as a failure.
- `CustomTool` takes the model's text as it is, for patches, queries or programs; `format` can constrain it with a
  Lark grammar or a regular expression. Only models that take custom tools see them.
- `DynamicTool::new(name, description, schema, handler)` is a tool known at run time, such as one of an MCP server.
  `register_dynamic` adapts it instead of rejecting it: the name is cleaned up and made unique (`raw_name` gives
  the original back), the schema is rewritten into one models accept and compacted when it is large.

`ToolRegistry::namespace(name, description)` groups tools; the model calls them with the namespace on the call.
Only models that take namespaces see them.

## When the model sees a tool

A tool's `Exposure` says when the model sees it:

- `Direct` (the default): in every request;
- `Deferred`: once a tool search found it, for large tool sets whose definitions would fill the context;
- `Hidden`: never advertised, but a call of it is answered.

Set it with `exposure` on the tool, or `ToolRegistry::set_exposure`. `search_deferred` searches the deferred tools
by the words of their names and descriptions; the runner answers its `tool_search` tool with it (see
[the runner](runner.md#deferred-tools-and-tool-search)).
