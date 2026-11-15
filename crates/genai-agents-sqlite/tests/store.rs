//! The SQLite store: the log across stores on one file, the guards, and a run on it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use genai_agents::session::{Session, SessionEntry, SessionError, SessionId};
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::{CancellationSignal, Item, Runner, ToolRegistry};
use genai_agents_sqlite::SqliteStore;

struct Database(std::path::PathBuf);

impl Database {
    /// A file of its own per test: tests run at once, and the clock may give two of them the same
    /// time.
    fn new(test: &str) -> Self {
        Self(std::env::temp_dir().join(format!(
            "genai-agents-sqlite-{test}-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )))
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", self.0.display()));
        }
    }
}

fn texts(conversation: &genai_agents::Conversation) -> Vec<String> {
    conversation
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::Message(message) => Some(message.text_content()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn sessions_are_shared_and_guarded_across_stores() {
    let database = Database::new("shared");
    let first_store = SqliteStore::open(&database.0).await.unwrap();
    // A second store on the same file: as another process would have.
    let second_store = SqliteStore::open(&database.0).await.unwrap();
    let id = SessionId::new("user-42");

    let mut first = Session::open(&first_store, id.clone()).await.unwrap();
    assert!(first
        .append_input("m1", Item::user("Umbrella?"))
        .await
        .unwrap()
        .is_some());
    let mut runner = Runner::new(
        ScriptedModel::replies(vec![Reply::text("Yes.")]),
        ToolRegistry::new(),
    );
    runner
        .run_session(&mut first, &CancellationSignal::new(), |_| {})
        .await
        .unwrap();

    let mut second = Session::open(&second_store, id.clone()).await.unwrap();
    assert_eq!(texts(&second.conversation()), ["Umbrella?", "Yes."]);
    // The key holds in the other store.
    assert_eq!(
        second
            .append_input("m1", Item::user("Umbrella?"))
            .await
            .unwrap(),
        None
    );
    // The first writer is superseded.
    let error = first.append_item(Item::user("late")).await.unwrap_err();
    assert!(matches!(error, SessionError::Superseded { .. }));
    second
        .append_input("m2", Item::user("And tomorrow?"))
        .await
        .unwrap();

    // Rollback and fork, as in any store.
    let inputs = second.user_inputs();
    second.rollback(Some(&inputs[0])).await.unwrap();
    let fork = second.fork(SessionId::new("fork"), None).await.unwrap();
    assert_eq!(texts(&fork.conversation()), ["Umbrella?"]);

    let reopened = Session::open(&first_store, id).await.unwrap();
    assert_eq!(texts(&reopened.conversation()), ["Umbrella?"]);
    assert!(reopened
        .records()
        .iter()
        .any(|record| matches!(record.entry, SessionEntry::Rollback)));
    assert_eq!(
        first_store.sessions().await.unwrap(),
        [SessionId::new("fork"), SessionId::new("user-42")]
    );
}

#[tokio::test]
async fn concurrent_appends_keep_one_order() {
    let database = Database::new("concurrent");
    let store = SqliteStore::open(&database.0).await.unwrap();
    let id = SessionId::new("busy");
    let session = Session::open(store.clone(), id.clone()).await.unwrap();
    let writer = session.writer().clone();
    // Many appends of one writer at once, through the store: each gets its own place.
    let tasks: Vec<_> = (0..20)
        .map(|index| {
            let store = store.clone();
            let id = id.clone();
            let writer = writer.clone();
            tokio::spawn(async move {
                let record = genai_agents::session::Record::new(
                    None,
                    SessionEntry::Label(format!("label {index}")),
                )
                .with_key(format!("k{index}"));
                genai_agents::session::SessionStore::append(&store, &id, &writer, vec![record])
                    .await
            })
        })
        .collect();
    for task in tasks {
        assert_eq!(task.await.unwrap().unwrap().len(), 1);
    }
    let records = genai_agents::session::SessionStore::load(&store, &id)
        .await
        .unwrap();
    assert_eq!(records.len(), 21); // the claim and 20 labels
}
