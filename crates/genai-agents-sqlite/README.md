# genai-agents-sqlite

[![Crates.io](https://img.shields.io/crates/v/genai-agents-sqlite)](https://crates.io/crates/genai-agents-sqlite)
[![docs.rs](https://img.shields.io/docsrs/genai-agents-sqlite)](https://docs.rs/genai-agents-sqlite)

[genai-agents](https://docs.rs/genai-agents) sessions in SQLite, on [sqlx](https://docs.rs/sqlx): every append is a
transaction, so several processes can share one database and a superseded writer is refused wherever it runs.

```bash
cargo add genai-agents genai-agents-sqlite
```

The [API reference](https://docs.rs/genai-agents-sqlite) has an example; the
[sessions guide](https://github.com/genai-rs/genai-agents/blob/main/docs/sessions.md) goes further.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/genai-rs/genai-agents/blob/main/LICENSE-MIT) at your option.
