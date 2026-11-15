# Context management

A long run outgrows the model's context window: every tool output stays in the history and is sent again with each
request. The runner keeps the request within the window, and guards against a model that loops. `RunConfig::context`
(`ContextConfig`) sets all of it; the defaults are on.

## Token accounting

`Runner::context_tokens` is what the next request will cost: the tokens the server counted for the last response,
plus an estimate of the items added since (their JSON at four bytes a token, a flat 1 000 per image or file). When no
response was counted yet, the whole conversation is estimated (`estimate_tokens`).

## Compaction

When the next request passes `compact_at` percent of the model's context window (`ModelInfo::context_window`;
default 90), or when the model answers that the context is too long (`ErrorKind::ContextOverflow`, once per request),
the runner compacts:

1. It asks the model for a summary of the conversation as it would send it (the previous summary included), without
   tools, with `compaction_prompt`. A request that is too long even for that is sent again without its oldest
   quarter, up to three times.
2. A summary cut off at the output limit, or empty, is rejected (`Error::Compaction`): a run goes on without it
   when the request may still fit.
3. It appends an `Item::Compaction` to the conversation with the summary, the user's most recent messages up to
   `keep_user_tokens` (default 20 000), and the injected context (instructions and the like, `ItemMeta::injected`).

From then on, requests start at the compaction: the kept messages, the summary (as a user message that starts with
`SUMMARY_PREFIX`), the context, then what came after. The items before it stay in the conversation (and in a
session), unsent. Since a compaction happens before a request, never in the middle of one, no call is separated
from its answer.

`Runner::compact` compacts by hand, for instance before handing a conversation to a model with a smaller window.
`RunEvent::Compacted` and a `compact` span report it.

```rust
use genai_agents::{ContextConfig, RunConfig};

let config = RunConfig::new().with_context(
    ContextConfig::new()
        .with_compact_at(Some(80))
        .with_keep_user_tokens(10_000),
);
# let _ = config;
```

## Pruning

Old tool outputs are cut in the request, not in the conversation:

- the outputs before the last `keep_recent` (default 4) keep their head and tail, `soft_bytes` (default 2 000) in
  all;
- the outputs before the last `hard_after` (default 16) are replaced with `PRUNED_OUTPUT`.

Changing the start of a request breaks the provider's prompt cache, so pruning waits until the cache has expired
anyway: `cache_ttl` (default 5 minutes) after the previous request. It then applies up to that point and stays the
same for the requests after it, which keeps their start stable.

## Input while a run is going

`Runner::input` hands out a `RunInput`: clone it, and push from another task or a UI handler.

- `steer(item)`: the next request carries it, once the current request's calls are answered.
- `follow_up(item)`: when the model answers and the run would stop, the follow-ups go in and the run continues.

The runner takes the messages at those points (`RunEvent::Input`); it spawns nothing.

## Guards

- **Repeated failures.** A call that failed `max_repeated_failures` times (default 3) with the same arguments in a
  run is not run again: it is answered with `REPEATED_FAILURE`, which tells the model to change course.
- **Calls written as text.** An answer that is only a tool call written as text (a JSON object naming a tool, a
  `<tool_call>` block) is never run. The model is told so (`TEXT_CALL_NOTE`) and gets one more request, once per run.
- **Interrupted runs.** A cancelled run ends its conversation with `INTERRUPTED_NOTE`, so that the next run knows
  tools may have done part of their work.
