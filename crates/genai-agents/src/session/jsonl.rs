//! Sessions in JSON Lines files.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use super::{admit, AppendGuard, Record, SessionError, SessionId, SessionStore, WriterId};

/// A [`SessionStore`] that keeps each session in a [JSON Lines](https://jsonlines.org) file of
/// its own, one record per line, in a directory: `<directory>/<session id>.jsonl`.
///
/// A file is only ever appended to. The store guards the writes of one process (it keeps what
/// it needs of each session it wrote to); two processes writing the same session need a store
/// with transactions, such as SQLite. The files are read and written with blocking I/O; the
/// records are small.
///
/// A session id becomes a file name: only letters, digits, `-`, `_` and `.` are kept as they
/// are, any other byte is written as `%XX`.
#[derive(Debug)]
pub struct JsonlStore {
    directory: PathBuf,
    guards: Mutex<HashMap<SessionId, AppendGuard>>,
}

impl JsonlStore {
    /// A store in `directory`, which is created when it does not exist.
    ///
    /// # Errors
    ///
    /// When the directory cannot be created.
    pub fn new(directory: impl Into<PathBuf>) -> Result<Self, SessionError> {
        let directory = directory.into();
        std::fs::create_dir_all(&directory).map_err(SessionError::store)?;
        Ok(Self {
            directory,
            guards: Mutex::new(HashMap::new()),
        })
    }

    /// The directory.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// The file of a session.
    #[must_use]
    pub fn path(&self, session: &SessionId) -> PathBuf {
        let mut name = String::new();
        for byte in session.as_str().bytes() {
            if byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_')
                || (byte == b'.' && !name.is_empty())
            {
                name.push(char::from(byte));
            } else {
                name.push_str(&format!("%{byte:02X}"));
            }
        }
        self.directory.join(format!("{name}.jsonl"))
    }

    fn read(&self, session: &SessionId) -> Result<Vec<Record>, SessionError> {
        let file = match File::open(self.path(session)) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(SessionError::store(error)),
        };
        let mut records = Vec::new();
        for line in BufReader::new(file).lines() {
            let line = line.map_err(SessionError::store)?;
            if line.trim().is_empty() {
                continue;
            }
            records.push(serde_json::from_str(&line)?);
        }
        Ok(records)
    }
}

impl SessionStore for JsonlStore {
    async fn load(&self, session: &SessionId) -> Result<Vec<Record>, SessionError> {
        self.read(session)
    }

    async fn append(
        &self,
        session: &SessionId,
        writer: &WriterId,
        records: Vec<Record>,
    ) -> Result<Vec<Record>, SessionError> {
        let mut guards = self.guards.lock().unwrap_or_else(PoisonError::into_inner);
        if !guards.contains_key(session) {
            let guard = AppendGuard::of(&self.read(session)?);
            guards.insert(session.clone(), guard);
        }
        let Some(guard) = guards.get_mut(session) else {
            return Ok(Vec::new());
        };
        // Admitted on a copy, so that a failed write leaves the guard as the file is.
        let mut next = guard.clone();
        let admitted = admit(session, &mut next, writer, records)?;
        if admitted.is_empty() {
            return Ok(admitted);
        }
        let mut lines = Vec::new();
        for record in &admitted {
            serde_json::to_writer(&mut lines, record)?;
            lines.push(b'\n');
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.path(session))
            .map_err(SessionError::store)?;
        file.write_all(&lines).map_err(SessionError::store)?;
        file.flush().map_err(SessionError::store)?;
        *guard = next;
        Ok(admitted)
    }
}
