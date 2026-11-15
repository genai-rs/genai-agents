//! A workspace assistant built on genai-agents: it answers questions about the files of a
//! directory, keeps the conversation in a SQLite session, asks before it writes, traces to
//! Langfuse, and has an eval of its own.
//!
//! ```text
//! genai-agents-demo ask "When are we painting the kitchen?"
//! genai-agents-demo chat [session]
//! genai-agents-demo eval [k] [--save-baseline]
//! ```
//!
//! See the README for the environment variables.

mod model;
mod tools;

use std::env;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use genai_agents::agent::Agent;
use genai_agents::fallback::Fallback;
use genai_agents::session::{Session, SessionId};
use genai_agents::skills::instruction_files;
use genai_agents::{AskFor, CancellationSignal, Conversation, Item, ModelEvent, RunEvent, RunStop};
use genai_agents_eval::{
    scorer_fn, target_fn, Case, Dataset, Eval, MaxRequests, Report, Run, Score,
};
use genai_agents_sqlite::SqliteStore;
use tokio::io::{AsyncBufReadExt as _, BufReader};
use tracing::Instrument as _;

use crate::model::Tracing;
use crate::tools::Workspace;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const INSTRUCTIONS: &str = "You are the assistant of a household. You answer questions about the \
household's notes, which are files in its workspace.

Before you answer a question, ALWAYS look it up with the tools: call `search` with a few plain \
words from the question (such as `paint kitchen`), then `read_file` on the file it points to. \
If a search finds nothing, try other words or call `list_files` and read the notes. Never answer \
from memory and never say the information is missing before you have searched and read the \
notes. Write a note only when you are asked to.";

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<ExitCode, BoxError> {
    let args: Vec<String> = env::args().skip(1).collect();
    let workspace_dir = env::var("DEMO_WORKSPACE").map_or_else(
        |_| Path::new(env!("CARGO_MANIFEST_DIR")).join("workspace"),
        PathBuf::from,
    );
    let workspace = Workspace::new(&workspace_dir)?;
    let tracing = Tracing::init()?;
    let code = match args.first().map(String::as_str) {
        Some("ask") => {
            let question = args[1..].join(" ");
            if question.is_empty() {
                return Ok(usage());
            }
            ask(&workspace, &tracing, "default", &question).await?;
            ExitCode::SUCCESS
        }
        Some("chat") | None => {
            let session = args.get(1).map_or("default", String::as_str);
            chat(&workspace, &tracing, session).await?;
            ExitCode::SUCCESS
        }
        Some("eval") => {
            let k = args.get(1).and_then(|k| k.parse().ok()).unwrap_or(2);
            let save = args.iter().any(|arg| arg == "--save-baseline");
            eval(&workspace, k, save).await?
        }
        _ => usage(),
    };
    tracing.shutdown();
    Ok(code)
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: genai-agents-demo ask <question>\n       genai-agents-demo chat [session]\n       \
         genai-agents-demo eval [k] [--save-baseline]"
    );
    ExitCode::FAILURE
}

/// The assistant: the model, the workspace's tools, its AGENTS.md, and approvals for writes.
fn assistant(workspace: &Workspace) -> Result<(Agent<Fallback>, String), BoxError> {
    let (model, label) = model::from_env()?;
    let mut instructions = INSTRUCTIONS.to_owned();
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
async fn store(workspace: &Workspace) -> Result<SqliteStore, BoxError> {
    let dir = workspace.root().join(".demo");
    std::fs::create_dir_all(&dir)?;
    Ok(SqliteStore::open(dir.join("sessions.db")).await?)
}

/// Answer one question, in the session `session`.
async fn ask(
    workspace: &Workspace,
    tracing: &Tracing,
    session: &str,
    question: &str,
) -> Result<(), BoxError> {
    let (mut agent, label) = assistant(workspace)?;
    eprintln!("[{label}]");
    let store = store(workspace).await?;
    let mut session = Session::open(&store, SessionId::new(session)).await?;
    turn(&mut agent, &mut session, tracing, question).await
}

/// A conversation on the terminal, one question per line, until an empty line.
async fn chat(workspace: &Workspace, tracing: &Tracing, session: &str) -> Result<(), BoxError> {
    let (mut agent, label) = assistant(workspace)?;
    let store = store(workspace).await?;
    let mut session = Session::open(&store, SessionId::new(session)).await?;
    eprintln!(
        "[{label}; session {} with {} earlier items; an empty line ends]",
        session.id(),
        session.conversation().len()
    );
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        eprint!("> ");
        std::io::stderr().flush()?;
        let Some(line) = lines.next_line().await? else {
            break;
        };
        if line.trim().is_empty() {
            break;
        }
        turn(&mut agent, &mut session, tracing, line.trim()).await?;
    }
    Ok(())
}

/// One question: the run, its approvals, and the answer printed as it comes.
async fn turn(
    agent: &mut Agent<Fallback>,
    session: &mut Session<&SqliteStore>,
    tracing: &Tracing,
    question: &str,
) -> Result<(), BoxError> {
    let key = format!(
        "terminal-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    session.append_input(key, Item::user(question)).await?;
    let span = tracing::info_span!("question", question);
    let cancel = CancellationSignal::new();
    let mut outcome = agent
        .run_session(&(), session, &cancel, print_events)
        .instrument(span.clone())
        .await?;
    // Ask on the terminal for every call that needs it, until the run goes through.
    while let RunStop::Approval { pending } = &outcome.run.stop {
        let mut decisions = Vec::new();
        for call in pending {
            eprint!("\nRun {}({})? [y/N] ", call.name, call.arguments);
            std::io::stderr().flush()?;
            let mut answer = String::new();
            BufReader::new(tokio::io::stdin())
                .read_line(&mut answer)
                .await?;
            decisions.push(if answer.trim().eq_ignore_ascii_case("y") {
                call.approve()
            } else {
                call.deny("the user said no")
            });
        }
        outcome = agent
            .resume_session(&(), session, &decisions, &cancel, print_events)
            .instrument(span.clone())
            .await?;
    }
    match (&outcome.run.stop, outcome.output()) {
        (RunStop::Completed, Some(answer)) => println!("\n{answer}"),
        (stop, _) => println!("\n[the run stopped: {stop:?}]"),
    }
    let usage = outcome
        .run
        .turns
        .iter()
        .filter_map(|turn| turn.usage)
        .fold(0, |total, usage| {
            total + usage.input_tokens + usage.output_tokens
        });
    if tracing.enabled() {
        eprintln!(
            "[{} requests, {usage} tokens; trace {}]",
            outcome.run.turns.len(),
            Tracing::trace_id(&span)
        );
    } else {
        eprintln!("[{} requests, {usage} tokens]", outcome.run.turns.len());
    }
    Ok(())
}

/// Show the run as it goes: the calls on stderr.
fn print_events(event: RunEvent<'_>) {
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

/// The eval: every case of `evals/cases.jsonl`, `k` times, on a fresh assistant each time.
async fn eval(workspace: &Workspace, k: u32, save_baseline: bool) -> Result<ExitCode, BoxError> {
    let dataset = Dataset::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("evals/cases.jsonl"))?;
    let shared = Arc::new(workspace.clone());
    let target = target_fn(move |case: Case| {
        let workspace = Arc::clone(&shared);
        async move {
            let (mut agent, _) = assistant(&workspace)
                .map_err(|error| genai_agents::Error::Output(error.to_string()))?;
            let mut conversation: Conversation = case.input;
            let outcome = agent.run(&(), &mut conversation).await?;
            Ok(Run::from_agent(conversation, outcome))
        }
    });
    let answer = scorer_fn("answer", |case, trial| {
        let expected = case.expected_text().unwrap_or_default().to_lowercase();
        match trial.output() {
            Some(output) if output.to_lowercase().contains(&expected) => Score::pass(true),
            Some(output) => {
                Score::pass(false).because(format!("expected {expected:?} in {output:?}"))
            }
            None => Score::pass(false).because("no answer"),
        }
    });
    let looked_up = scorer_fn("looked it up", |case, trial| {
        let calls: Vec<String> = trial
            .calls(case)
            .iter()
            .map(|call| call.name.clone())
            .collect();
        let wanted = case.metadata["tool"].as_str().unwrap_or_default();
        let ok = match wanted {
            "search_or_read" => calls
                .iter()
                .any(|name| name == "search" || name == "read_file"),
            other => calls.iter().any(|name| name == other),
        };
        Score::pass(ok).because(format!("called {calls:?}"))
    });
    let requests = MaxRequests(6);
    let report = Eval::new("household assistant", &dataset, &target)
        .scorer(&answer)
        .scorer(&looked_up)
        .scorer(&requests)
        .trials(k)
        .concurrency(1)
        .run()
        .await;
    let dir = workspace.root().join(".demo");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("eval-report.md"), report.to_markdown())?;
    std::fs::write(dir.join("eval-report.json"), report.to_json()?)?;
    println!("{}", report.to_markdown());
    let baseline_path = dir.join("eval-baseline.json");
    if save_baseline {
        std::fs::write(&baseline_path, report.to_json()?)?;
        eprintln!("saved as the baseline");
        return Ok(ExitCode::SUCCESS);
    }
    if let Ok(text) = std::fs::read_to_string(&baseline_path) {
        let baseline = Report::from_json(&text)?;
        let comparison = report.compare(&baseline, 0.1);
        println!("## Against the baseline\n\n{}", comparison.to_markdown());
        if comparison.has_regressions() {
            return Ok(ExitCode::FAILURE);
        }
    }
    Ok(ExitCode::SUCCESS)
}
