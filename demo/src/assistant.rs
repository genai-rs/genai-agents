//! The household assistant, and one turn of it, for the terminal and the web pages alike.

use genai_agents::agent::Agent;
use genai_agents::fallback::Fallback;
use genai_agents::runner::{Approval, PendingCall};
use genai_agents::session::Session;
use genai_agents::skills::instruction_files;
use genai_agents::{AskFor, CancellationSignal, Item, ModelEvent, Role, RunEvent, RunStop};
use genai_agents_sqlite::SqliteStore;
use tracing::Instrument as _;

use crate::model::{self, Tracing};
use crate::tools::{self, Workspace};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

const INSTRUCTIONS: &str = "You are the assistant of a household. You answer questions about the \
household's notes, which are files in its workspace.

Before you answer a question, ALWAYS look it up with the tools: call `search` with a few plain \
words from the question (such as `paint kitchen`), then `read_file` on the file it points to. \
If a search finds nothing, try other words or call `list_files` and read the notes. Never answer \
from memory and never say the information is missing before you have searched and read the \
notes. Write a note only when you are asked to.";

/// What the assistant's answers are for: the terminal and the chat page show them, a voice says
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Text,
    Voice,
}

/// The assistant: the model, the workspace's tools, its AGENTS.md, and approvals for writes.
/// Returns it with a line that says which model it runs on.
pub fn assistant(
    workspace: &Workspace,
    channel: Channel,
) -> Result<(Agent<Fallback>, String), BoxError> {
    let (model, label) = model::from_env()?;
    let mut instructions = INSTRUCTIONS.to_owned();
    if channel == Channel::Voice {
        instructions.push_str(
            "\n\nA voice assistant speaks your answer to the caller: answer in one to three short \
             sentences of plain text, without lists, markdown or file names.",
        );
    }
    let agents_md = instruction_files(workspace.root(), workspace.root(), 4_000)?;
    if !agents_md.is_empty() {
        instructions.push_str("\n\n");
        instructions.push_str(&agents_md);
    }
    let agent = Agent::new("household-assistant", model)
        .description("Answers questions about the household's notes.")
        .with_tools(tools::registry(workspace)?)
        .instructions(instructions)
        .approvals(AskFor::tools(["write_note"]));
    Ok((agent, label))
}

/// The sessions of a workspace, in `<workspace>/.demo/sessions.db`.
pub async fn store(workspace: &Workspace) -> Result<SqliteStore, BoxError> {
    let dir = workspace.root().join(".demo");
    std::fs::create_dir_all(&dir)?;
    Ok(SqliteStore::open(dir.join("sessions.db")).await?)
}

/// What starts or continues a turn.
pub enum Input<'a> {
    /// A question, with the key that makes appending it idempotent.
    Question { key: String, text: &'a str },
    /// Decisions about the calls the turn waits for.
    Decisions(&'a [Approval]),
}

/// One run of a turn: how it stopped, and what it did.
#[derive(Debug)]
pub struct TurnReport {
    pub stop: RunStop,
    pub answer: Option<String>,
    /// The tools called, with their arguments, in order.
    pub calls: Vec<(String, String)>,
    pub requests: usize,
    pub tokens: i64,
    /// The Langfuse trace, when tracing is on.
    pub trace_id: Option<String>,
}

impl TurnReport {
    /// The calls the turn waits for, when it waits for approvals.
    pub fn pending(&self) -> &[PendingCall] {
        match &self.stop {
            RunStop::Approval { pending } => pending,
            _ => &[],
        }
    }
}

/// Run a turn of the assistant in a session, up to its answer or the approvals it waits for.
pub async fn turn<F>(
    agent: &mut Agent<Fallback>,
    session: &mut Session<&SqliteStore>,
    tracing: &Tracing,
    input: Input<'_>,
    on_event: F,
) -> Result<TurnReport, BoxError>
where
    F: FnMut(RunEvent<'_>) + Send,
{
    let cancel = CancellationSignal::new();
    let outcome = match input {
        Input::Question { key, text } => {
            session.append_input(key, Item::user(text)).await?;
            let span = tracing::info_span!("question", question = text);
            let outcome = agent
                .run_session(&(), session, &cancel, on_event)
                .instrument(span.clone())
                .await?;
            (outcome, span)
        }
        Input::Decisions(decisions) => {
            let span = tracing::info_span!("approvals", decisions = decisions.len());
            let outcome = agent
                .resume_session(&(), session, decisions, &cancel, on_event)
                .instrument(span.clone())
                .await?;
            (outcome, span)
        }
    };
    let (outcome, span) = outcome;
    let calls = outcome
        .run
        .turns
        .iter()
        .flat_map(|turn| turn.calls.iter())
        .map(|call| (call.name.clone(), call.input.clone()))
        .collect();
    let tokens = outcome
        .run
        .turns
        .iter()
        .filter_map(|turn| turn.usage)
        .fold(0, |total, usage| {
            total + usage.input_tokens + usage.output_tokens
        });
    Ok(TurnReport {
        answer: match outcome.run.stop {
            RunStop::Completed => outcome.output().cloned(),
            _ => None,
        },
        stop: outcome.run.stop,
        calls,
        requests: outcome.run.turns.len(),
        tokens,
        trace_id: tracing.enabled().then(|| Tracing::trace_id(&span)),
    })
}

/// The messages of a conversation a person reads: their questions and the assistant's answers,
/// without the calls and their outputs.
pub fn transcript(items: &[Item]) -> Vec<(Role, String)> {
    items
        .iter()
        .filter_map(|item| match item {
            Item::Message(message) if matches!(message.role, Role::User | Role::Assistant) => {
                let text = message.text_content();
                (!text.trim().is_empty()).then_some((message.role, text))
            }
            _ => None,
        })
        .collect()
}

/// Show a run on stderr as it goes: its calls, retries and compactions.
pub fn print_events(event: RunEvent<'_>) {
    match event {
        RunEvent::ToolCalled(call) => eprintln!("  → {}({})", call.name, call.input),
        RunEvent::Retrying { attempt, error, .. } => eprintln!("  ↻ retry {attempt}: {error}"),
        RunEvent::Compacted {
            tokens_before,
            tokens_after,
        } => {
            eprintln!("  ⇢ compacted {tokens_before} → {tokens_after} tokens");
        }
        RunEvent::Model(ModelEvent::TextDelta(_)) | _ => {}
    }
}
