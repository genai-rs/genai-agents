//! Sessions: a conversation stored as an append-only log of entries.
//!
//! A [`Session`] is a log of [`Record`]s in a [`SessionStore`]. Each record holds one
//! [`SessionEntry`] (an item of the conversation, the model settings from then on, the usage the
//! server reported, a compaction, a label, or an entry of the application's own) and points at
//! the record before it, so that the log is a tree:
//!
//! - the conversation is the path from the first record to the *head*, the latest record
//!   ([`Session::conversation`]), from its latest [compaction](Compaction) on;
//! - [`rollback`](Session::rollback) moves the head back to an earlier record; the records after
//!   it stay in the log, off the path;
//! - [`fork`](Session::fork) copies the path up to a record into a new session.
//!
//! Nothing is ever removed: compaction and rollback change what is sent, not what is stored.
//!
//! Writes are guarded:
//!
//! - opening a session claims it: a writer that opened it earlier can no longer append
//!   ([`SessionError::Superseded`]), so a run that was replaced by a newer one cannot write into
//!   the session behind its back;
//! - an input appended with a key ([`Session::append_input`]) is stored once, however often it is
//!   sent.
//!
//! There is no default session: every session has an explicit [`SessionId`].
//!
//! ```
//! use genai_agents::session::{MemoryStore, Session, SessionId};
//! use genai_agents::Item;
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), genai_agents::session::SessionError> {
//! let store = MemoryStore::new();
//! let mut session = Session::open(&store, SessionId::new("user-42")).await?;
//! session.append_input("msg-1", Item::user("Umbrella in Ghent today?")).await?;
//! // The same message again (a retry of the client, say) is stored once.
//! session.append_input("msg-1", Item::user("Umbrella in Ghent today?")).await?;
//! session.append_item(Item::assistant("Yes, light rain.")).await?;
//! assert_eq!(session.conversation().len(), 2);
//!
//! // Back to the question: the answer stays in the log, off the conversation.
//! let question = session.user_inputs()[0].clone();
//! session.rollback(Some(&question)).await?;
//! assert_eq!(session.conversation().len(), 1);
//! assert_eq!(session.records().len(), 4); // question, answer, a claim and the rollback
//! # Ok(())
//! # }
//! ```

#[cfg(all(feature = "jsonl", not(target_family = "wasm")))]
mod jsonl;
mod memory;

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::conversation::Conversation;
use crate::item::{Item, ItemMeta, Message, Origin, Role};
use crate::model::{ModelSettings, Usage};
use crate::platform::{random_u64, MaybeSend, MaybeSync};

#[cfg(all(feature = "jsonl", not(target_family = "wasm")))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "jsonl", not(target_family = "wasm")))))]
pub use jsonl::JsonlStore;
pub use memory::MemoryStore;

/// The id of a session: chosen by the application (a user, a channel, a ticket), or
/// [random](Self::random).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    /// This id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// A new random id.
    #[must_use]
    pub fn random() -> Self {
        Self(format!(
            "session-{:016x}{:016x}",
            random_u64(),
            random_u64()
        ))
    }

    /// The id as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The id of a record, unique within its session (and kept when a session is forked).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EntryId(String);

impl EntryId {
    /// This id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// A new random id.
    #[must_use]
    pub fn random() -> Self {
        Self(format!("{:016x}", random_u64()))
    }

    /// The id as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EntryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Who writes into a session: the handle that [opened](Session::open) it last. A store refuses
/// the records of an earlier writer ([`SessionError::Superseded`]).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WriterId(String);

impl WriterId {
    /// This id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// A new random id.
    #[must_use]
    pub fn random() -> Self {
        Self(format!("writer-{:016x}", random_u64()))
    }

    /// The id as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The model and its settings from a record on, as a run used them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TurnSettings {
    /// The model, when it says ([`ModelInfo::origin`](crate::ModelInfo::origin)).
    pub model: Option<Origin>,
    /// Its settings.
    pub settings: ModelSettings,
}

impl TurnSettings {
    /// These settings for this model.
    #[must_use]
    pub fn new(model: Option<Origin>, settings: ModelSettings) -> Self {
        Self { model, settings }
    }
}

/// A summary that stands in for the conversation before it.
///
/// The conversation of a session starts at its latest compaction: the summary, as an injected
/// user message ([`ItemMeta::injected`]), then the items from [`first_kept`](Self::first_kept)
/// on, then those after the compaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Compaction {
    /// The summary of what came before.
    pub summary: String,
    /// The first record before the compaction whose item is still sent after the summary (the
    /// recent user messages, say); `None` when the summary replaces everything before it.
    pub first_kept: Option<EntryId>,
    /// The input tokens of the conversation it replaced, when known.
    pub tokens_before: Option<u64>,
}

impl Compaction {
    /// A summary that replaces everything before it.
    #[must_use]
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            first_kept: None,
            tokens_before: None,
        }
    }

    /// The same compaction, keeping the items from this record on.
    #[must_use]
    pub fn with_first_kept(mut self, first_kept: EntryId) -> Self {
        self.first_kept = Some(first_kept);
        self
    }

    /// The same compaction, with the tokens of what it replaced.
    #[must_use]
    pub fn with_tokens_before(mut self, tokens: u64) -> Self {
        self.tokens_before = Some(tokens);
        self
    }
}

/// What a record holds.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SessionEntry {
    /// An item of the conversation.
    Item(Item),
    /// The model and settings of the requests from here on.
    TurnSettings(TurnSettings),
    /// What the server reported a response used.
    Usage(Usage),
    /// A summary that stands in for the conversation before it.
    Compaction(Compaction),
    /// A name for this point of the conversation, to roll back or fork to.
    Label(String),
    /// An entry of the application's own, never sent to the model.
    Custom {
        /// What kind of entry it is, for the application.
        kind: String,
        /// Its data.
        data: Value,
    },
    /// The head moved back to the record's parent ([`Session::rollback`]): nothing of its own.
    Rollback,
    /// A writer opened the session; earlier writers can no longer append. Not part of the
    /// conversation tree.
    Claim(WriterId),
}

/// One record of a session's log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Record {
    /// Its id.
    pub id: EntryId,
    /// The record before it on its path; `None` for the first one (and for claims).
    pub parent: Option<EntryId>,
    /// The key of an input, so that it is stored once ([`Session::append_input`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// What it holds.
    pub entry: SessionEntry,
}

impl Record {
    /// A record with a new random id.
    #[must_use]
    pub fn new(parent: Option<EntryId>, entry: SessionEntry) -> Self {
        Self {
            id: EntryId::random(),
            parent,
            key: None,
            entry,
        }
    }

    /// The same record, with an input key.
    #[must_use]
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    fn is_claim(&self) -> bool {
        matches!(self.entry, SessionEntry::Claim(_))
    }
}

/// Why a session operation failed.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SessionError {
    /// Another writer opened the session after this one: it can no longer append.
    #[error("session {session} was opened by another writer since")]
    Superseded {
        /// The session.
        session: SessionId,
    },
    /// The record is not in the session.
    #[error("record {0} is not in the session")]
    UnknownEntry(EntryId),
    /// A record of the store could not be encoded or decoded.
    #[error("session record: {0}")]
    Json(#[from] serde_json::Error),
    /// The store failed.
    #[error("session store: {0}")]
    Store(#[source] crate::error::BoxError),
}

impl SessionError {
    /// A failure of a store, with its source kept.
    pub fn store(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Store(Box::new(error))
    }
}

/// Where the records of sessions are kept.
///
/// [`MemoryStore`] keeps them in memory; with the `jsonl` feature, [`JsonlStore`] in a JSON Lines
/// file per session; [genai-agents-sqlite](https://docs.rs/genai-agents-sqlite) in SQLite. Another
/// store implements the two methods below, and calls [`admit`] while it holds the session's lock
/// (or in a transaction) so that its appends are guarded as every store's are.
pub trait SessionStore: MaybeSend + MaybeSync {
    /// The records of a session, in the order they were appended; empty for a new session.
    fn load(
        &self,
        session: &SessionId,
    ) -> impl Future<Output = Result<Vec<Record>, SessionError>> + MaybeSend;

    /// Append records written by `writer`, as [`admit`] decides: all of them, the inputs already
    /// stored left out, or none when another writer claimed the session since. Returns the
    /// records appended.
    fn append(
        &self,
        session: &SessionId,
        writer: &WriterId,
        records: Vec<Record>,
    ) -> impl Future<Output = Result<Vec<Record>, SessionError>> + MaybeSend;
}

impl<S: SessionStore + ?Sized> SessionStore for &S {
    fn load(
        &self,
        session: &SessionId,
    ) -> impl Future<Output = Result<Vec<Record>, SessionError>> + MaybeSend {
        (**self).load(session)
    }

    fn append(
        &self,
        session: &SessionId,
        writer: &WriterId,
        records: Vec<Record>,
    ) -> impl Future<Output = Result<Vec<Record>, SessionError>> + MaybeSend {
        (**self).append(session, writer, records)
    }
}

impl<S: SessionStore + ?Sized> SessionStore for Arc<S> {
    fn load(
        &self,
        session: &SessionId,
    ) -> impl Future<Output = Result<Vec<Record>, SessionError>> + MaybeSend {
        (**self).load(session)
    }

    fn append(
        &self,
        session: &SessionId,
        writer: &WriterId,
        records: Vec<Record>,
    ) -> impl Future<Output = Result<Vec<Record>, SessionError>> + MaybeSend {
        (**self).append(session, writer, records)
    }
}

/// What a store needs to know of a session to guard an append: the writer of its latest claim,
/// and whether an input key is stored already.
#[derive(Debug, Clone, Default)]
pub struct AppendGuard {
    /// The writer that claimed the session last.
    pub writer: Option<WriterId>,
    /// The input keys stored so far.
    pub keys: BTreeSet<String>,
}

impl AppendGuard {
    /// The guard of these records.
    #[must_use]
    pub fn of(records: &[Record]) -> Self {
        let mut guard = Self::default();
        for record in records {
            guard.observe(record);
        }
        guard
    }

    /// Take a stored record into account.
    pub fn observe(&mut self, record: &Record) {
        if let SessionEntry::Claim(writer) = &record.entry {
            self.writer = Some(writer.clone());
        }
        if let Some(key) = &record.key {
            self.keys.insert(key.clone());
        }
    }
}

/// Decide which of `records`, written by `writer`, a store appends to a session whose state is
/// `guard`; `guard` is updated with them.
///
/// - A batch that starts with `writer`'s own claim is a newly opened session: it is admitted,
///   and every writer before it is superseded.
/// - Otherwise the session's latest claim must be `writer`'s (or the session has none), or the
///   batch fails with [`SessionError::Superseded`].
/// - Records whose input key is stored already, or comes earlier in the batch, are left out.
///
/// # Errors
///
/// [`SessionError::Superseded`] when another writer claimed the session after `writer`.
pub fn admit(
    session: &SessionId,
    guard: &mut AppendGuard,
    writer: &WriterId,
    records: Vec<Record>,
) -> Result<Vec<Record>, SessionError> {
    let claims = matches!(records.first(), Some(Record { entry: SessionEntry::Claim(claimer), .. }) if claimer == writer);
    if !claims && guard.writer.as_ref().is_some_and(|last| last != writer) {
        return Err(SessionError::Superseded {
            session: session.clone(),
        });
    }
    let mut admitted = Vec::with_capacity(records.len());
    for record in records {
        if record
            .key
            .as_ref()
            .is_some_and(|key| guard.keys.contains(key))
        {
            continue;
        }
        guard.observe(&record);
        admitted.push(record);
    }
    Ok(admitted)
}

/// A session: its records, loaded from a store, and the head of its conversation.
///
/// `S` is the store, or a reference to it (`&MemoryStore`, `Arc<JsonlStore>`). See the
/// [module documentation](self).
#[derive(Debug)]
pub struct Session<S> {
    id: SessionId,
    store: S,
    writer: WriterId,
    records: Vec<Record>,
    index: BTreeMap<EntryId, usize>,
    head: Option<EntryId>,
}

impl<S: SessionStore> Session<S> {
    /// Open a session (a new one when the store has no records of it) and claim it: from now on,
    /// handles that opened it earlier can no longer append.
    ///
    /// # Errors
    ///
    /// When the store fails.
    pub async fn open(store: S, id: SessionId) -> Result<Self, SessionError> {
        let records = store.load(&id).await?;
        let mut session = Self {
            id,
            store,
            writer: WriterId::random(),
            records: Vec::new(),
            index: BTreeMap::new(),
            head: None,
        };
        for record in records {
            session.remember(record);
        }
        let claim = Record::new(None, SessionEntry::Claim(session.writer.clone()));
        session.store_records(vec![claim]).await?;
        Ok(session)
    }

    /// The session's id.
    #[must_use]
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    /// The store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// The writer this handle appends as.
    #[must_use]
    pub fn writer(&self) -> &WriterId {
        &self.writer
    }

    /// Every record of the log, in the order they were appended: those off the conversation's
    /// path (rolled back) and the claims included.
    #[must_use]
    pub fn records(&self) -> &[Record] {
        &self.records
    }

    /// A record by id.
    #[must_use]
    pub fn record(&self, id: &EntryId) -> Option<&Record> {
        self.index.get(id).map(|&index| &self.records[index])
    }

    /// The head: the latest record of the conversation; `None` while it is empty.
    #[must_use]
    pub fn head(&self) -> Option<&EntryId> {
        self.head.as_ref()
    }

    /// The records of the conversation, from the first to the head.
    #[must_use]
    pub fn path(&self) -> Vec<&Record> {
        let mut path = Vec::new();
        let mut next = self.head.as_ref();
        while let Some(id) = next {
            let Some(record) = self.record(id) else { break };
            path.push(record);
            next = record.parent.as_ref();
        }
        path.reverse();
        path
    }

    /// The conversation to run on: the items on the path, from the latest compaction on (its
    /// summary first, as an injected user message).
    #[must_use]
    pub fn conversation(&self) -> Conversation {
        let path = self.path();
        let compaction = path
            .iter()
            .rposition(|record| matches!(record.entry, SessionEntry::Compaction(_)));
        let Some(at) = compaction else {
            return Conversation::from_items(path.iter().filter_map(|record| item_of(record)));
        };
        let SessionEntry::Compaction(compaction) = &path[at].entry else {
            return Conversation::new();
        };
        let mut items = vec![Item::Message(
            Message::text(Role::User, compaction.summary.clone())
                .with_meta(ItemMeta::new().with_injected(true)),
        )];
        if let Some(first) = &compaction.first_kept {
            if let Some(from) = path[..at].iter().position(|record| &record.id == first) {
                items.extend(path[from..at].iter().filter_map(|record| item_of(record)));
            }
        }
        items.extend(path[at + 1..].iter().filter_map(|record| item_of(record)));
        Conversation::from_items(items)
    }

    /// The records of the user's messages on the path, in order: the points to roll back or
    /// fork to ("from the second question on").
    #[must_use]
    pub fn user_inputs(&self) -> Vec<EntryId> {
        self.path()
            .into_iter()
            .filter(|record| {
                matches!(&record.entry, SessionEntry::Item(Item::Message(message))
                    if message.role == Role::User && !message.meta.injected)
            })
            .map(|record| record.id.clone())
            .collect()
    }

    /// The model settings in effect at the head: the latest [`SessionEntry::TurnSettings`] on
    /// the path.
    #[must_use]
    pub fn turn_settings(&self) -> Option<&TurnSettings> {
        self.path()
            .into_iter()
            .rev()
            .find_map(|record| match &record.entry {
                SessionEntry::TurnSettings(settings) => Some(settings),
                _ => None,
            })
    }

    /// The usage the server reported for the responses on the path, added up.
    #[must_use]
    pub fn usage(&self) -> Usage {
        let mut total = Usage::default();
        for record in self.path() {
            if let SessionEntry::Usage(usage) = &record.entry {
                total.input_tokens += usage.input_tokens;
                total.output_tokens += usage.output_tokens;
                total.cache_read_tokens = add(total.cache_read_tokens, usage.cache_read_tokens);
                total.cache_write_tokens = add(total.cache_write_tokens, usage.cache_write_tokens);
                total.reasoning_tokens = add(total.reasoning_tokens, usage.reasoning_tokens);
            }
        }
        total
    }

    /// Append an entry after the head; it becomes the head.
    ///
    /// # Errors
    ///
    /// [`SessionError::Superseded`] when another handle opened the session since, or the
    /// store's failure.
    pub async fn append(&mut self, entry: SessionEntry) -> Result<EntryId, SessionError> {
        let record = Record::new(self.head.clone(), entry);
        let id = record.id.clone();
        self.store_records(vec![record]).await?;
        Ok(id)
    }

    /// Append entries after the head, in order, in one write; the last becomes the head.
    ///
    /// # Errors
    ///
    /// As [`append`](Self::append).
    pub async fn append_all(
        &mut self,
        entries: impl IntoIterator<Item = SessionEntry>,
    ) -> Result<Vec<EntryId>, SessionError> {
        let mut parent = self.head.clone();
        let records: Vec<Record> = entries
            .into_iter()
            .map(|entry| {
                let record = Record::new(parent.take(), entry);
                parent = Some(record.id.clone());
                record
            })
            .collect();
        let ids = records.iter().map(|record| record.id.clone()).collect();
        self.store_records(records).await?;
        Ok(ids)
    }

    /// Append an item of the conversation.
    ///
    /// # Errors
    ///
    /// As [`append`](Self::append).
    pub async fn append_item(&mut self, item: impl Into<Item>) -> Result<EntryId, SessionError> {
        self.append(SessionEntry::Item(item.into())).await
    }

    /// Append an input (a user message, say) with a key that is unique to it, such as the id of
    /// the message in the channel it came from: sent again with the same key, it is not stored
    /// again. Returns the record's id, or `None` when the key was stored already.
    ///
    /// # Errors
    ///
    /// As [`append`](Self::append).
    pub async fn append_input(
        &mut self,
        key: impl Into<String>,
        item: impl Into<Item>,
    ) -> Result<Option<EntryId>, SessionError> {
        let record = Record::new(self.head.clone(), SessionEntry::Item(item.into())).with_key(key);
        let id = record.id.clone();
        let stored = self.store_records(vec![record]).await?;
        Ok((stored > 0).then_some(id))
    }

    /// Move the head back to `to` (`None`: before the first record). The records after it stay
    /// in the log, off the path; what is appended next follows `to`.
    ///
    /// # Errors
    ///
    /// [`SessionError::UnknownEntry`] when `to` is not in the session; as
    /// [`append`](Self::append).
    pub async fn rollback(&mut self, to: Option<&EntryId>) -> Result<EntryId, SessionError> {
        if let Some(to) = to {
            if !self.index.contains_key(to) {
                return Err(SessionError::UnknownEntry(to.clone()));
            }
        }
        let record = Record::new(to.cloned(), SessionEntry::Rollback);
        let id = record.id.clone();
        self.store_records(vec![record]).await?;
        Ok(id)
    }

    /// A new session, `id`, in the same store, holding the path of this one up to `at` (`None`:
    /// up to the head). The records keep their ids.
    ///
    /// # Errors
    ///
    /// [`SessionError::UnknownEntry`] when `at` is not on this session's path; the store's
    /// failure.
    pub async fn fork(
        &self,
        id: SessionId,
        at: Option<&EntryId>,
    ) -> Result<Session<S>, SessionError>
    where
        S: Clone,
    {
        let path = self.path();
        let end = match at {
            None => path.len(),
            Some(at) => {
                path.iter()
                    .position(|record| &record.id == at)
                    .ok_or_else(|| SessionError::UnknownEntry(at.clone()))?
                    + 1
            }
        };
        let copied: Vec<Record> = path[..end]
            .iter()
            .filter(|record| !matches!(record.entry, SessionEntry::Rollback))
            .map(|&record| record.clone())
            .collect();
        let mut fork = Session::open(self.store.clone(), id).await?;
        // A path without its rollbacks: each record follows the one copied before it.
        let mut parent = fork.head.clone();
        let records = copied
            .into_iter()
            .map(|mut record| {
                record.parent = parent.take();
                parent = Some(record.id.clone());
                record
            })
            .collect();
        fork.store_records(records).await?;
        Ok(fork)
    }

    /// Store records and take in those the store kept. Returns how many it kept.
    async fn store_records(&mut self, records: Vec<Record>) -> Result<usize, SessionError> {
        let stored = self.store.append(&self.id, &self.writer, records).await?;
        let count = stored.len();
        for record in stored {
            self.remember(record);
        }
        Ok(count)
    }

    fn remember(&mut self, record: Record) {
        // Claims stay off the tree; a rollback is the head, and its parent is where the path
        // goes on.
        if !record.is_claim() {
            self.head = Some(record.id.clone());
        }
        self.index.insert(record.id.clone(), self.records.len());
        self.records.push(record);
    }
}

fn item_of(record: &Record) -> Option<Item> {
    match &record.entry {
        SessionEntry::Item(item) => Some(item.clone()),
        _ => None,
    }
}

fn add(total: Option<i64>, more: Option<i64>) -> Option<i64> {
    match (total, more) {
        (None, None) => None,
        (total, more) => Some(total.unwrap_or(0) + more.unwrap_or(0)),
    }
}
