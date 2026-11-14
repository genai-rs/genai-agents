# genai-agents

[![CI](https://github.com/genai-rs/genai-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/genai-rs/genai-agents/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

Agents for Rust, on top of the [genai-rs](https://github.com/genai-rs) clients: a model, typed tools, and a runner
that calls the model in a loop, runs the tools it asks for and sends their outputs back until it answers.

| Crate | What it holds |
| --- | --- |
| [`genai-agents`](crates/genai-agents) | the runner, tools, the conversation and the `Model` trait, independent of the model provider |
| [`genai-agents-openai`](crates/genai-agents-openai) | OpenAI's Responses API and Chat Completions as models, on [openai-ergonomic](https://github.com/genai-rs/openai-ergonomic) |

What works today:

- typed tools (function, rich, custom and runtime tools, namespaces) and when the model sees them (deferred tools
  found by a tool search);
- a runner that starts calls while the response streams, runs them concurrently where tools allow it, retries
  failed requests, stops after a number of requests or when cancelled, and keeps a history that can be sent again;
- a provider-independent, serde-serializable conversation;
- natively and on WebAssembly.

Sessions, tracing, agents with guardrails and handoffs, MCP and skills come next: see the [design](docs/design.md).

- [The runner](docs/runner.md)
- [Tools](docs/tools.md)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your
option.
