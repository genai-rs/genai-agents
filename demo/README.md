# genai-agents demo: a household assistant

A small application on [genai-agents](../README.md): an assistant that answers questions about the notes in a
directory (`workspace/`). It shows most of the library at work:

- an `Agent` with instructions, and the workspace's `AGENTS.md` added to them (`skills::instruction_files`);
- tools: `list_files`, `read_file`, `search` (kept inside the workspace) and `write_note`, as `FunctionTool`s, and
  `current_time` from `#[tool]`;
- an approval: `write_note` asks on the terminal before it writes (`AskFor`, `resume_session`);
- the conversation in a SQLite session (`genai-agents-sqlite`), so a chat goes on where it stopped;
- models through `Fallback`: OpenAI when a key is set, with a local Ollama behind it, or Ollama alone;
- spans to Langfuse, when its keys are set;
- an eval (`genai-agents-eval`) of five questions, run `k` times, compared with a baseline.

## Run it

With [Ollama](https://ollama.com) running and `ollama pull qwen2.5:3b` done, nothing else is needed:

```bash
cargo run -p genai-agents-demo -- ask "When are we painting the kitchen?"
cargo run -p genai-agents-demo -- chat            # a conversation; an empty line ends it
cargo run -p genai-agents-demo -- chat holidays   # another session
cargo run -p genai-agents-demo -- eval 3          # each question three times
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

## The eval

`eval [k]` asks each question of `evals/cases.jsonl` `k` times, on a new assistant each time, and scores every answer:

- **answer**: it contains the expected text;
- **looked it up**: it called the tool the case names (`search` or `read_file`, or `list_files`);
- **requests**: it took at most six requests.

It prints a report (also in `workspace/.demo/eval-report.md` and `.json`) with pass@k (a question answered right
at least once) and pass^k (every time). `eval k --save-baseline` keeps the run as the baseline; later runs list
what got better or worse against it, and exit with a failure on a regression.

With `qwen2.5:3b`, the eval showed what a small model gets wrong: it often answered "I have no information" without
calling a tool, or searched for an exact phrase that is not in the notes and gave up. Two changes, measured
against the baseline, took the trials that pass from 20 % to 50 %: a search that matches each word (and its stem)
instead of the whole phrase, and instructions that say to search, read, and search again before saying something
is not there. What still fails is mostly the date format `AGENTS.md` asks for.
