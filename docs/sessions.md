# Sessions

A session keeps a conversation between runs: a user's chat, a ticket, a channel. It is an append-only log, not a list
of items, so that nothing a run did is lost: a rollback, a fork or (later) a compaction changes what is sent to the
model, never what is stored.

## A run on a session

```rust
use genai_agents::session::{MemoryStore, Session, SessionId};
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::{CancellationSignal, Item, Runner, ToolRegistry};

# #[tokio::main(flavor = "current_thread")]
# async fn main() -> genai_agents::Result<()> {
let store = MemoryStore::new();
let mut session = Session::open(&store, SessionId::new("user-42")).await?;

// The input goes into the session first, with a key of its own.
session.append_input("msg-1", Item::user("Umbrella in Ghent today?")).await?;

let model = ScriptedModel::replies(vec![Reply::text("Yes, light rain.")]);
let mut runner = Runner::new(model, ToolRegistry::new());
let outcome = runner.run_session(&mut session, &CancellationSignal::new(), |_| {}).await?;
assert_eq!(outcome.output_text(), "Yes, light rain.");
assert_eq!(session.conversation().len(), 2);
# Ok(())
# }
```

`Runner::run_session` runs on `session.conversation()` and appends to the session after every model request:

- the model's items and the answers to its calls;
- the model and its settings, when they differ from the last ones the session holds (`SessionEntry::TurnSettings`);
- the usage the server reported (`SessionEntry::Usage`; `Session::usage` adds it up).

A run that is cancelled or fails half-way leaves its finished requests, and what the last one got to, in the
session: the next run goes on from there.

## The log

Each `Record` has an id, the id of the record before it, an optional input key, and a `SessionEntry`:

| Entry | Holds |
| --- | --- |
| `Item` | an item of the conversation |
| `TurnSettings` | the model and its settings from here on |
| `Usage` | what the server reported a response used |
| `Compaction` | a summary that stands in for the conversation before it, and the first record it keeps |
| `Label` | a name for this point, to roll back or fork to |
| `Custom` | the application's own data, never sent to the model |
| `Rollback` | nothing: the head moved back to its parent |
| `Claim` | a writer opened the session |

The records form a tree. The conversation is the path from the first record to the head (the latest record), from
the latest compaction on: its summary first, as an injected user message, then the records it keeps, then the rest.

- `rollback(Some(&id))` moves the head back to a record; `user_inputs()` lists the user's messages on the path, the
  usual places to go back to. The records after it stay in the log.
- `fork(new_id, Some(&id))` copies the path up to a record into a new session of the same store.

## Guarded writes

- **One writer.** `Session::open` claims the session. A handle that opened it earlier can no longer append: it gets
  `SessionError::Superseded` (a run on it, `Error::Session`). A run that a newer one replaced, say because the user
  sent another message, cannot write behind its back.
- **Inputs stored once.** `append_input(key, item)` stores an input under a key unique to it, such as the id of the
  message in the channel it came from. Sent again with the same key, by a client's retry or a redelivered webhook, it
  is not stored again (`None`).
- **No default session.** Every session has an explicit `SessionId`; a session shared by default would mix
  conversations.

## Stores

| Store | Where | Guards across |
| --- | --- | --- |
| `MemoryStore` | memory; natively and on WebAssembly | the process |
| `JsonlStore` (feature `jsonl`) | a JSON Lines file per session in a directory | the process |
| [`SqliteStore`](https://docs.rs/genai-agents-sqlite) | a SQLite file, every append a transaction | processes sharing the file |

Another store implements `SessionStore`: `load` returns a session's records in order, and `append` stores records
after calling `session::admit` under the session's lock (or in a transaction), which applies the guards above.
