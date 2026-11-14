# genai-agents

[![Crates.io](https://img.shields.io/crates/v/genai-agents)](https://crates.io/crates/genai-agents)
[![docs.rs](https://img.shields.io/docsrs/genai-agents)](https://docs.rs/genai-agents)

Agents for Rust, independent of the model provider: typed tools, a provider-independent conversation, and a
`Runner` that calls a `Model` in a loop, runs the tools it asks for and sends their outputs back until it answers.

```bash
cargo add genai-agents genai-agents-openai
```

A provider crate gives the model: [genai-agents-openai](https://docs.rs/genai-agents-openai) for OpenAI and the
APIs that speak it. The [API reference](https://docs.rs/genai-agents) has a complete example; the guides cover
[the runner](https://github.com/genai-rs/genai-agents/blob/main/docs/runner.md) and
[tools](https://github.com/genai-rs/genai-agents/blob/main/docs/tools.md).

Features:

- `schemars`: a tool's parameters schema derived from its input type (`tools::schema::for_type`);
- `testing`: `testing::ScriptedModel`, a model that answers as a script says, for tests.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-MIT) at your option.
