# genai-agents-macros

[![Crates.io](https://img.shields.io/crates/v/genai-agents-macros)](https://crates.io/crates/genai-agents-macros)
[![docs.rs](https://img.shields.io/docsrs/genai-agents-macros)](https://docs.rs/genai-agents-macros)

The `#[tool]` attribute of [genai-agents](https://docs.rs/genai-agents): a function tool from an async function, with
its name, description and parameters schema taken from the function and its doc comments.

Use it through genai-agents, with the `macros` feature, rather than on its own:

```bash
cargo add genai-agents --features macros
```

```rust,ignore
use genai_agents::{tool, Result};

/// Current weather of a city.
#[tool]
async fn get_weather(
    /// The city, such as "Ghent".
    city: String,
) -> Result<String> {
    Ok(format!("light rain in {city}"))
}

tools.register(GetWeather)?;
```

The [API reference](https://docs.rs/genai-agents-macros) has the details; the
[tools guide](https://github.com/genai-rs/genai-agents/blob/main/docs/tools.md#a-tool-from-a-function) goes further.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-MIT) at your option.
