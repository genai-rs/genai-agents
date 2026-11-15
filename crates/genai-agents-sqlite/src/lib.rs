//! [genai-agents](https://docs.rs/genai-agents) sessions in SQLite: [`SqliteStore`] is a
//! [`SessionStore`] on a [rusqlite](https://docs.rs/rusqlite) connection, with SQLite built in
//! (rusqlite's `bundled` feature).
//!
//! Every append runs in a write transaction (`BEGIN IMMEDIATE`), so the guards of a session
//! (a superseded writer is refused, an input key is stored once) hold across processes that
//! share the database file.
//!
//! The connection is blocking: each operation runs on Tokio's blocking threads
//! ([`spawn_blocking`](tokio::task::spawn_blocking)), so the store is used from a Tokio runtime.
//!
//! ```no_run
//! use genai_agents::session::{Session, SessionId};
//! use genai_agents::Item;
//! use genai_agents_sqlite::SqliteStore;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let store = SqliteStore::open("sessions.db").await?;
//! let mut session = Session::open(&store, SessionId::new("user-42")).await?;
//! session.append_input("msg-1", Item::user("Umbrella in Ghent today?")).await?;
//! # Ok(())
//! # }
//! ```
//!
//! The records live in one table, `genai_agents_session_records`, created when the store is
//! opened: a row per record, with its session, its position in the log and its JSON.

#![cfg_attr(docsrs, feature(doc_cfg))]

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use genai_agents::session::{
    admit, AppendGuard, Record, SessionEntry, SessionError, SessionId, SessionStore, WriterId,
};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS genai_agents_session_records (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    record TEXT NOT NULL,
    claim TEXT,
    input_key TEXT,
    PRIMARY KEY (session_id, seq)
);
CREATE UNIQUE INDEX IF NOT EXISTS genai_agents_session_records_input_key
    ON genai_agents_session_records (session_id, input_key) WHERE input_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS genai_agents_session_records_claim
    ON genai_agents_session_records (session_id, seq) WHERE claim IS NOT NULL;
";

/// How long a write waits for another process's transaction before it fails.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// A [`SessionStore`] in SQLite. Cloning it shares its connection.
#[derive(Debug, Clone)]
pub struct SqliteStore {
    connection: Arc<Mutex<Connection>>,
}

impl SqliteStore {
    /// A store in the database file at `path`, created when it does not exist, in
    /// write-ahead-log mode so that readers do not wait for writers.
    ///
    /// # Errors
    ///
    /// When the database cannot be opened or its table created.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let path = path.as_ref().to_owned();
        blocking(move || {
            let connection = Connection::open(path).map_err(SessionError::store)?;
            connection
                .pragma_update(None, "journal_mode", "WAL")
                .map_err(SessionError::store)?;
            Self::with_connection(connection)
        })
        .await
    }

    /// A store in a database in memory, gone when the store is: for tests.
    ///
    /// # Errors
    ///
    /// When the database cannot be made.
    pub fn in_memory() -> Result<Self, SessionError> {
        Self::with_connection(Connection::open_in_memory().map_err(SessionError::store)?)
    }

    /// A store on this connection; its table is created when it does not exist.
    ///
    /// # Errors
    ///
    /// When the table cannot be created.
    pub fn with_connection(connection: Connection) -> Result<Self, SessionError> {
        connection
            .busy_timeout(BUSY_TIMEOUT)
            .map_err(SessionError::store)?;
        connection
            .execute_batch(SCHEMA)
            .map_err(SessionError::store)?;
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    /// The ids of the sessions it holds.
    ///
    /// # Errors
    ///
    /// When the query fails.
    pub async fn sessions(&self) -> Result<Vec<SessionId>, SessionError> {
        self.with(|connection| {
            let mut statement = connection
                .prepare(
                    "SELECT DISTINCT session_id FROM genai_agents_session_records
                     ORDER BY session_id",
                )
                .map_err(SessionError::store)?;
            let ids = statement
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(SessionError::store)?
                .map(|id| id.map(SessionId::new).map_err(SessionError::store))
                .collect();
            ids
        })
        .await
    }

    /// Run `work` on the connection, on a blocking thread.
    async fn with<T: Send + 'static>(
        &self,
        work: impl FnOnce(&mut Connection) -> Result<T, SessionError> + Send + 'static,
    ) -> Result<T, SessionError> {
        let connection = Arc::clone(&self.connection);
        blocking(move || {
            let mut connection = connection.lock().unwrap_or_else(PoisonError::into_inner);
            work(&mut connection)
        })
        .await
    }
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, SessionError> + Send + 'static,
) -> Result<T, SessionError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(SessionError::store)?
}

impl SessionStore for SqliteStore {
    async fn load(&self, session: &SessionId) -> Result<Vec<Record>, SessionError> {
        let session = session.clone();
        self.with(move |connection| {
            let mut statement = connection
                .prepare(
                    "SELECT record FROM genai_agents_session_records
                     WHERE session_id = ?1 ORDER BY seq",
                )
                .map_err(SessionError::store)?;
            let rows = statement
                .query_map([session.as_str()], |row| row.get::<_, String>(0))
                .map_err(SessionError::store)?;
            let mut records = Vec::new();
            for row in rows {
                let text = row.map_err(SessionError::store)?;
                records.push(serde_json::from_str(&text)?);
            }
            Ok(records)
        })
        .await
    }

    async fn append(
        &self,
        session: &SessionId,
        writer: &WriterId,
        records: Vec<Record>,
    ) -> Result<Vec<Record>, SessionError> {
        let session = session.clone();
        let writer = writer.clone();
        self.with(move |connection| {
            let transaction = connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(SessionError::store)?;
            let admitted = append_in(&transaction, &session, &writer, records)?;
            transaction.commit().map_err(SessionError::store)?;
            Ok(admitted)
        })
        .await
    }
}

/// The append, inside the transaction (rolled back when it is dropped without a commit).
fn append_in(
    connection: &Connection,
    session: &SessionId,
    writer: &WriterId,
    records: Vec<Record>,
) -> Result<Vec<Record>, SessionError> {
    let writer_of_claim = connection
        .query_row(
            "SELECT claim FROM genai_agents_session_records
             WHERE session_id = ?1 AND claim IS NOT NULL ORDER BY seq DESC LIMIT 1",
            [session.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(SessionError::store)?
        .map(WriterId::new);
    let mut guard = AppendGuard {
        writer: writer_of_claim,
        ..AppendGuard::default()
    };
    for key in records.iter().filter_map(|record| record.key.as_deref()) {
        let stored = connection
            .query_row(
                "SELECT 1 FROM genai_agents_session_records
                 WHERE session_id = ?1 AND input_key = ?2",
                params![session.as_str(), key],
                |_| Ok(()),
            )
            .optional()
            .map_err(SessionError::store)?;
        if stored.is_some() {
            guard.keys.insert(key.to_owned());
        }
    }
    let admitted = admit(session, &mut guard, writer, records)?;
    let last: i64 = connection
        .query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM genai_agents_session_records WHERE session_id = ?1",
            [session.as_str()],
            |row| row.get(0),
        )
        .map_err(SessionError::store)?;
    let mut insert = connection
        .prepare(
            "INSERT INTO genai_agents_session_records (session_id, seq, record, claim, input_key)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )
        .map_err(SessionError::store)?;
    for (seq, record) in (last + 1..).zip(&admitted) {
        let claim = match &record.entry {
            SessionEntry::Claim(writer) => Some(writer.as_str()),
            _ => None,
        };
        insert
            .execute(params![
                session.as_str(),
                seq,
                serde_json::to_string(record)?,
                claim,
                record.key.as_deref(),
            ])
            .map_err(SessionError::store)?;
    }
    Ok(admitted)
}
