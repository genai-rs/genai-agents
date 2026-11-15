//! A workspace assistant built on genai-agents: it answers questions about the files of a
//! directory, keeps the conversation in a SQLite session, asks before it writes, traces to
//! Langfuse, and has an eval of its own. `serve` puts it on web pages: a chat, and a voice page
//! on GPT-Live that hands the questions to the assistant.
//!
//! ```text
//! genai-agents-demo ask "When are we painting the kitchen?"
//! genai-agents-demo chat [session]
//! genai-agents-demo eval [k] [--save-baseline]
//! genai-agents-demo serve
//! ```
//!
//! See the README for the environment variables.

mod assistant;
mod model;
mod tools;
mod web;

use std::env;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use genai_agents::session::{Session, SessionId};
use genai_agents::{Conversation, RunStop};
use genai_agents_eval::{
    scorer_fn, target_fn, Case, Dataset, Eval, MaxRequests, Report, Run, Score,
};
use tokio::io::{AsyncBufReadExt as _, BufReader};

use crate::assistant::{assistant, print_events, store, turn, BoxError, Channel, Input};
use crate::model::Tracing;
use crate::tools::Workspace;

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
        Some("serve") => {
            let tracing = Arc::new(tracing);
            web::serve(workspace, Arc::clone(&tracing)).await?;
            if let Ok(tracing) = Arc::try_unwrap(tracing) {
                tracing.shutdown();
            }
            return Ok(ExitCode::SUCCESS);
        }
        _ => usage(),
    };
    tracing.shutdown();
    Ok(code)
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: genai-agents-demo ask <question>\n       genai-agents-demo chat [session]\n       \
         genai-agents-demo eval [k] [--save-baseline]\n       genai-agents-demo serve"
    );
    ExitCode::FAILURE
}

/// Answer one question, in the session `session`.
async fn ask(
    workspace: &Workspace,
    tracing: &Tracing,
    session: &str,
    question: &str,
) -> Result<(), BoxError> {
    let (mut agent, label) = assistant(workspace, Channel::Text)?;
    eprintln!("[{label}]");
    let store = store(workspace).await?;
    let mut session = Session::open(&store, SessionId::new(session)).await?;
    terminal_turn(&mut agent, &mut session, tracing, question).await
}

/// A conversation on the terminal, one question per line, until an empty line.
async fn chat(workspace: &Workspace, tracing: &Tracing, session: &str) -> Result<(), BoxError> {
    let (mut agent, label) = assistant(workspace, Channel::Text)?;
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
        terminal_turn(&mut agent, &mut session, tracing, line.trim()).await?;
    }
    Ok(())
}

/// One question on the terminal: the run, its approvals asked on the terminal, and the answer.
async fn terminal_turn(
    agent: &mut genai_agents::agent::Agent<genai_agents::fallback::Fallback>,
    session: &mut Session<&genai_agents_sqlite::SqliteStore>,
    tracing: &Tracing,
    question: &str,
) -> Result<(), BoxError> {
    let key = format!(
        "terminal-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let input = Input::Question {
        key,
        text: question,
    };
    let mut report = turn(agent, session, tracing, input, print_events).await?;
    let (mut requests, mut tokens) = (report.requests, report.tokens);
    // Ask on the terminal for every call that needs it, until the run goes through.
    while !report.pending().is_empty() {
        let mut decisions = Vec::new();
        for call in report.pending() {
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
        report = turn(
            agent,
            session,
            tracing,
            Input::Decisions(&decisions),
            print_events,
        )
        .await?;
        requests += report.requests;
        tokens += report.tokens;
    }
    match (&report.stop, &report.answer) {
        (RunStop::Completed, Some(answer)) => println!("\n{answer}"),
        (stop, _) => println!("\n[the run stopped: {stop:?}]"),
    }
    match &report.trace_id {
        Some(trace) => eprintln!("[{requests} requests, {tokens} tokens; trace {trace}]"),
        None => eprintln!("[{requests} requests, {tokens} tokens]"),
    }
    Ok(())
}

/// The eval: every case of `evals/cases.jsonl`, `k` times, on a fresh assistant each time.
async fn eval(workspace: &Workspace, k: u32, save_baseline: bool) -> Result<ExitCode, BoxError> {
    let dataset = Dataset::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("evals/cases.jsonl"))?;
    let shared = Arc::new(workspace.clone());
    let target = target_fn(move |case: Case| {
        let workspace = Arc::clone(&shared);
        async move {
            let (mut agent, _) = assistant(&workspace, Channel::Text)
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
            // Any way of looking at the workspace: `search` also lists file names.
            "any" => calls
                .iter()
                .any(|name| matches!(name.as_str(), "search" | "read_file" | "list_files")),
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
