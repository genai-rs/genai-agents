# Contributing

Changes go through pull requests against `main`, with [Conventional Commits](https://www.conventionalcommits.org)
titles: [release-plz](https://release-plz.ieni.dev) derives the versions and the changelogs from them.

Before opening a pull request:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

[local-ci](https://github.com/genai-rs/.github/blob/main/docs/local-ci.md) runs the same checks as CI and reports
them on the pull request.
