# genai-agents-openai

[![Crates.io](https://img.shields.io/crates/v/genai-agents-openai)](https://crates.io/crates/genai-agents-openai)
[![docs.rs](https://img.shields.io/docsrs/genai-agents-openai)](https://docs.rs/genai-agents-openai)

OpenAI's APIs as models for [genai-agents](https://docs.rs/genai-agents), on
[openai-ergonomic](https://docs.rs/openai-ergonomic):

- `OpenAiResponses`: the Responses API, streamed and stateless by default, with hosted tools and the encrypted
  reasoning of reasoning models;
- `OpenAiChat`: Chat Completions, which also serves the providers that speak it (Azure OpenAI, OpenRouter, Groq,
  Ollama, gateways).

```bash
cargo add genai-agents genai-agents-openai openai-ergonomic
```

The [API reference](https://docs.rs/genai-agents-openai) has an example; the
[runner guide](https://github.com/genai-rs/genai-agents/blob/main/docs/runner.md) goes further.

Features: `rustls-tls` (default) or `native-tls`, passed on to openai-ergonomic.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-MIT) at your option.
