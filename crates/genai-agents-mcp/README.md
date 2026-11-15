# genai-agents-mcp

[![Crates.io](https://img.shields.io/crates/v/genai-agents-mcp)](https://crates.io/crates/genai-agents-mcp)
[![docs.rs](https://img.shields.io/docsrs/genai-agents-mcp)](https://docs.rs/genai-agents-mcp)

The tools of [MCP](https://modelcontextprotocol.io) servers as [genai-agents](https://docs.rs/genai-agents) tools, on
the official Rust SDK [rmcp](https://docs.rs/rmcp): a server over stdio or streamable HTTP, its tools in a namespace,
deferred until a tool search finds them, with names and schemas cleaned up for the model APIs, read-only tools run
in parallel, and a changed tool list registered again between runs.

```bash
cargo add genai-agents genai-agents-mcp
```

The [API reference](https://docs.rs/genai-agents-mcp) has an example; the
[MCP guide](https://github.com/genai-rs/genai-agents/blob/main/docs/mcp.md) goes further.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-MIT) at your option.
