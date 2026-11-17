# genai-agents demo: a household assistant

A small application on [genai-agents](../README.md): an assistant that answers questions about the notes in a
directory (`workspace/`). It shows most of the library at work:

- an `Agent` with instructions, and the workspace's `AGENTS.md` added to them (`skills::instruction_files`);
- tools: `list_files`, `read_file`, `search` (kept inside the workspace) and `write_note`, as `FunctionTool`s, and
  `current_time` from `#[tool]`;
- an approval: `write_note` asks before it writes, on the terminal or on the web page (`AskFor`, `resume_session`);
- the conversation in a SQLite session (`genai-agents-sqlite`), so a chat goes on where it stopped;
- models through `Fallback`: OpenAI when a key is set, with a local Ollama behind it, or Ollama alone;
- spans to Langfuse, when its keys are set;
- an eval (`genai-agents-eval`) of five questions, run `k` times, compared with a baseline;
- a web page (`serve`): type or talk (GPT-Live hands the household's questions to the assistant), with the tool calls.

## Run it

With [Ollama](https://ollama.com) running and `ollama pull qwen2.5:3b` done, nothing else is needed:

```bash
cargo run -p genai-agents-demo -- ask "When are we painting the kitchen?"
cargo run -p genai-agents-demo -- chat            # a conversation; an empty line ends it
cargo run -p genai-agents-demo -- chat holidays   # another session
cargo run -p genai-agents-demo -- eval 3          # each question three times
cargo run -p genai-agents-demo -- serve           # the web page, on http://127.0.0.1:8080
```

The answer goes to stdout; the tool calls, the number of requests and tokens, and the trace id to stderr. Sessions,
eval reports and the baseline are in `workspace/.demo/`.

| Variable | What it does |
| --- | --- |
| `OPENAI_API_KEY` | use OpenAI's Responses API, with Ollama as the fallback |
| `DEMO_MODEL` | the OpenAI model (default `gpt-5-mini`) |
| `DEMO_OLLAMA_MODEL`, `OLLAMA_BASE_URL` | the Ollama model (default `qwen2.5:3b`) and server (default `http://localhost:11434/v1`) |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | send the spans to Langfuse (default host: Langfuse Cloud) |
| `DEMO_WORKSPACE` | another directory to work on |
| `DEMO_ADDR` | where `serve` listens (default `127.0.0.1:8080`) |
| `DEMO_LIVE_MODEL` | the GPT-Live model of the voice call (default `gpt-live-1`; needs `OPENAI_API_KEY`) |
| `LANGFUSE_PROJECT_ID` | link the page's traces to the Langfuse UI |

## The eval

`eval [k]` asks each question of `evals/cases.jsonl` `k` times, on a new assistant each time, and scores every answer:

- **answer**: it contains the expected text;
- **looked it up**: it called the tool the case names (`search` or `read_file`), or any of the workspace's tools;
- **requests**: it took at most six requests.

It prints a report (also in `workspace/.demo/eval-report.md` and `.json`) with pass@k (a question answered right
at least once) and pass^k (every time). `eval k --save-baseline` keeps the run as the baseline; later runs list
what got better or worse against it, and exit with a failure on a regression.

With `qwen2.5:3b`, the eval showed what a small model gets wrong: it often answered "I have no information" without
calling a tool, or searched for an exact phrase that is not in the notes and gave up. Two changes, measured
against the baseline, took the trials that pass from 20 % to 50 %: a search that matches each word (and its stem)
instead of the whole phrase, and instructions that say to search, read, and search again before saying something
is not there. What still fails is mostly the date format `AGENTS.md` asks for.

With OpenAI's `gpt-5-mini` (and three trials a case), every answer was right and 80 % of the trials passed; the one
miss was the eval's: it wanted `list_files` for "which notes do we have?", and the model found the names with `search`.
That case now takes any lookup.

## The web page

`serve` puts the assistant on one page (`/`): a conversation you can type or speak, with each tool call the
assistant made as a card that opens into its input, output and duration.

- **Typed**: a session per browser (kept in the SQLite store, so a reload shows it again), and Approve and Deny
  buttons when the assistant wants to write a note.
- **Spoken**: the microphone button starts a call with
  [GPT-Live](https://developers.openai.com/api/docs/guides/live) over WebRTC. GPT-Live does the talking; with
  client delegation it hands the household's questions to the application, and the page asks the assistant
  (`POST /api/live/delegations`, with the transcript so far) and gives GPT-Live the answer to say
  (`session.commentary.append`). Notes are not written by voice: the assistant hears that the chat does that.

The settings (the gear) pick GPT-Live's voice and start a new session.

The OpenAI key stays on the server: it creates the Live session for the browser's WebRTC offer
(`create_live` of [openai-client-base](https://github.com/genai-rs/openai-client-base)) and returns the answer.

A browser only gives a page the microphone on https or on `localhost`: open the page as `http://localhost:8080/` on
the machine that runs the demo, or through an SSH tunnel (`ssh -L 8080:localhost:8080 <host>`).
`DEMO_ADDR=0.0.0.0:8080` serves the page to the network; the chat works there, but anyone who reaches the server
uses your OpenAI key.

### The API, generated

The page talks to an API described in [`api/openapi.yaml`](api/openapi.yaml). Its server, the `genai-agents-demo-api`
crate in `api/`, is generated from that file by
[openapi-generator-timvw-rust](https://github.com/timvw/openapi-generator-timvw-rust) (`targets: [server]`): a trait
per tag (`AssistantApi`, `LiveApi`) with a method per operation, typed requests and responses, and an axum router.
`src/web.rs` implements the traits. After changing the spec, regenerate (Docker), and commit the result:

```bash
demo/scripts/generate-api.sh
```

The crate's `Cargo.toml` is kept by hand (it is in `api/.openapi-generator-ignore`): the generator cannot set
`publish = false` yet.
