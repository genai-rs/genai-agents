//! Sessions kept in memory.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use super::{admit, AppendGuard, Record, SessionError, SessionId, SessionStore, WriterId};

/// A [`SessionStore`] in memory: for tests, and for sessions that need not outlive the process.
/// Share it between handles with a reference or an `Arc`.
#[derive(Debug, Default)]
pub struct MemoryStore {
    sessions: Mutex<HashMap<SessionId, Stored>>,
}

#[derive(Debug, Default)]
struct Stored {
    records: Vec<Record>,
    guard: AppendGuard,
}

impl MemoryStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The ids of the sessions it holds.
    #[must_use]
    pub fn sessions(&self) -> Vec<SessionId> {
        let sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        let mut ids: Vec<SessionId> = sessions.keys().cloned().collect();
        ids.sort();
        ids
    }
}

impl SessionStore for MemoryStore {
    async fn load(&self, session: &SessionId) -> Result<Vec<Record>, SessionError> {
        let sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(sessions
            .get(session)
            .map(|stored| stored.records.clone())
            .unwrap_or_default())
    }

    async fn append(
        &self,
        session: &SessionId,
        writer: &WriterId,
        records: Vec<Record>,
    ) -> Result<Vec<Record>, SessionError> {
        let mut sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        let stored = sessions.entry(session.clone()).or_default();
        let admitted = admit(session, &mut stored.guard, writer, records)?;
        stored.records.extend(admitted.iter().cloned());
        Ok(admitted)
    }
}
