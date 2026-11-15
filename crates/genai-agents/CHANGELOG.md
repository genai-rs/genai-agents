# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/genai-rs/genai-agents/compare/genai-agents-v0.2.0...genai-agents-v0.3.0) - 2026-11-15

### Added

- genai-agents-mcp: MCP servers' tools as genai-agents tools
- approvals, handoffs, agents as tools, sub-agents
- skills (SKILL.md) and instruction files (AGENTS.md)
- fallback models and credentials
- #[tool], a tool from an async function
- agents with instructions, context, hooks, guardrails and structured output
- [**breaking**] context management: compaction, pruning, input during a run, guards
- sessions, tracing spans and recorded runs

### Other

- release

## [0.2.0](https://github.com/genai-rs/genai-agents/compare/genai-agents-v0.1.0...genai-agents-v0.2.0) - 2026-11-14

### Added

- [**breaking**] item origins, request projections, Compat and the client's interceptors

### Other

- release

## [0.1.0](https://github.com/genai-rs/genai-agents/compare/genai-agents-v0.0.1...genai-agents-v0.1.0) - 2026-11-14

### Added

- *(genai-agents)* the provider-independent core

### Other

- release

## [0.0.1](https://github.com/genai-rs/genai-agents/releases/tag/genai-agents-v0.0.1) - 2026-11-14

### Other

- scaffold the genai-agents workspace
