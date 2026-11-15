# genai-agents-eval

[![Crates.io](https://img.shields.io/crates/v/genai-agents-eval)](https://crates.io/crates/genai-agents-eval)
[![docs.rs](https://img.shields.io/docsrs/genai-agents-eval)](https://docs.rs/genai-agents-eval)

Evals for [genai-agents](https://docs.rs/genai-agents): datasets of cases, scorers of the answer, the trajectory and
the cost (an LLM judge too), every case run several times for pass@k and pass^k, reports in JSON and Markdown,
comparison with a baseline for CI, and Langfuse dataset runs.

```bash
cargo add --dev genai-agents-eval
```

The [API reference](https://docs.rs/genai-agents-eval) has an example; the
[evals guide](https://github.com/genai-rs/genai-agents/blob/main/docs/evals.md) goes further.

Features: `langfuse`, Langfuse dataset runs through [langfuse-ergonomic](https://docs.rs/langfuse-ergonomic).

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-MIT) at your option.
