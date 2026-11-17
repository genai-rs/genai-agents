//! Approvals (a run that stops for a person's decision and resumes, also from a session in
//! another process), handoffs, agents as tools, and sub-agents in the background.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(not(target_family = "wasm"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use genai_agents::agent::{without_tool_calls, Agent, AgentTool, SubAgentLimits};
use genai_agents::session::{MemoryStore, Session, SessionId};
use genai_agents::testing::{Reply, ScriptedModel, SeenRequest};
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{
    AskFor, CancellationSignal, Conversation, Error, Item, PendingCall, Result, RunEvent, RunStop,
    Runner, ToolCall, ToolPolicy,
};
use serde_json::{json, Value};

/// Counts its runs.
struct Counted {
    name: &'static str,
    runs: Arc<AtomicUsize>,
}

impl FunctionTool for Counted {
    type Input = Value;
    type Output = String;
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "A file operation"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _: Value) -> Result<String> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        Ok(format!("{} done", self.name))
    }
}

fn files() -> (ToolRegistry, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let reads = Arc::new(AtomicUsize::new(0));
    let deletes = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools
        .register(Counted {
            name: "read_file",
            runs: Arc::clone(&reads),
        })
        .unwrap();
    tools
        .register(Counted {
            name: "delete_file",
            runs: Arc::clone(&deletes),
        })
        .unwrap();
    (tools, reads, deletes)
}

fn calls(calls: &[(&str, &str, Value)]) -> Reply {
    Reply::items(
        calls
            .iter()
            .map(|(id, name, arguments)| {
                Item::ToolCall(ToolCall::function(*id, *name, arguments.to_string()))
            })
            .collect(),
    )
}

/// Reads and deletes in one response, then answers.
fn cleaner() -> ScriptedModel {
    ScriptedModel::new(|index, _| match index {
        0 => calls(&[
            ("c1", "read_file", json!({"path": "a.txt"})),
            ("c2", "delete_file", json!({"path": "a.txt"})),
        ]),
        _ => Reply::text("Cleaned up."),
    })
}

fn output_of(request: &SeenRequest, call_id: &str) -> String {
    request
        .outputs()
        .into_iter()
        .find(|output| output.call_id == call_id)
        .and_then(|output| output.body.to_text())
        .unwrap()
}

fn pending(stop: &RunStop) -> Vec<PendingCall> {
    match stop {
        RunStop::Approval { pending } => pending.clone(),
        other => panic!("an approval stop expected, got {other:?}"),
    }
}

#[tokio::test]
async fn a_run_stops_for_approval_and_resumes() {
    let (tools, reads, deletes) = files();
    let model = cleaner();
    let mut runner = Runner::new(model.clone(), tools).with_approvals(AskFor::tools(["delete_*"]));
    let mut conversation = Conversation::user("Clean up a.txt");
    let outcome = runner.run(&mut conversation).await.unwrap();

    // The read ran; the delete waits.
    let waiting = pending(&outcome.stop);
    assert_eq!(waiting.len(), 1);
    assert_eq!(waiting[0].name, "delete_file");
    assert_eq!(waiting[0].key, "delete_file");
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    assert_eq!(deletes.load(Ordering::SeqCst), 0);
    assert_eq!(conversation.unanswered_calls().len(), 1);

    // The decision can come later, from elsewhere: the pending call is data.
    let stored = serde_json::to_string(&waiting).unwrap();
    let waiting: Vec<PendingCall> = serde_json::from_str(&stored).unwrap();
    let outcome = runner
        .resume(
            &mut conversation,
            &[waiting[0].approve()],
            &CancellationSignal::new(),
            |_| {},
        )
        .await
        .unwrap();
    assert_eq!(outcome.output_text(), "Cleaned up.");
    assert_eq!(deletes.load(Ordering::SeqCst), 1);
    let last = model.requests().pop().unwrap();
    assert_eq!(output_of(&last, "c1"), "\"read_file done\"");
    assert_eq!(output_of(&last, "c2"), "\"delete_file done\"");
}

#[tokio::test]
async fn resume_reports_the_answers_of_the_decided_calls() {
    let (tools, _, deletes) = files();
    let model = ScriptedModel::new(|index, _| match index {
        0 => calls(&[
            ("c1", "delete_file", json!({"path": "a"})),
            ("c2", "delete_file", json!({"path": "b"})),
        ]),
        _ => Reply::text("Deleted a, kept b."),
    });
    let mut runner = Runner::new(model, tools).with_approvals(AskFor::tools(["delete_*"]));
    let mut conversation = Conversation::user("Delete a and b");
    let waiting = pending(&runner.run(&mut conversation).await.unwrap().stop);
    let mut answered = Vec::new();
    runner
        .resume(
            &mut conversation,
            &[waiting[0].approve(), waiting[1].deny("keep b")],
            &CancellationSignal::new(),
            |event| {
                if let RunEvent::ToolAnswered(answer) = event {
                    answered.push((answer.call_id.clone(), answer.success));
                }
            },
        )
        .await
        .unwrap();
    answered.sort();
    assert_eq!(
        answered,
        [("c1".to_owned(), true), ("c2".to_owned(), false)]
    );
    assert_eq!(deletes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn denials_session_approvals_changed_arguments_and_aborts() {
    // Denied: the model gets the reason; nothing ran.
    let (tools, _, deletes) = files();
    let model = cleaner();
    let mut runner = Runner::new(model.clone(), tools).with_approvals(AskFor::tools(["delete_*"]));
    let mut conversation = Conversation::user("Clean up");
    let waiting = pending(&runner.run(&mut conversation).await.unwrap().stop);
    runner
        .resume(
            &mut conversation,
            &[waiting[0].deny("keep it")],
            &CancellationSignal::new(),
            |_| {},
        )
        .await
        .unwrap();
    assert_eq!(deletes.load(Ordering::SeqCst), 0);
    assert_eq!(
        output_of(&model.requests().pop().unwrap(), "c2"),
        "denied: keep it"
    );

    // Approved for the session: the next delete is not asked about.
    let (tools, _, deletes) = files();
    let model = ScriptedModel::new(|index, _| match index {
        0 => calls(&[("c1", "delete_file", json!({"path": "a"}))]),
        1 => calls(&[("c2", "delete_file", json!({"path": "b"}))]),
        _ => Reply::text("Both gone."),
    });
    let mut runner = Runner::new(model, tools).with_approvals(AskFor::tools(["delete_*"]));
    let mut conversation = Conversation::user("Delete a and b");
    let waiting = pending(&runner.run(&mut conversation).await.unwrap().stop);
    let outcome = runner
        .resume(
            &mut conversation,
            &[waiting[0].approve_for_session()],
            &CancellationSignal::new(),
            |_| {},
        )
        .await
        .unwrap();
    assert_eq!(outcome.stop, RunStop::Completed);
    assert_eq!(deletes.load(Ordering::SeqCst), 2);
    assert_eq!(runner.approved_keys().collect::<Vec<_>>(), ["delete_file"]);

    // A decision for arguments that changed since does not apply.
    let (tools, _, deletes) = files();
    let mut runner = Runner::new(cleaner(), tools).with_approvals(AskFor::tools(["delete_*"]));
    let mut conversation = Conversation::user("Clean up");
    let waiting = pending(&runner.run(&mut conversation).await.unwrap().stop);
    let mut tampered = waiting[0].clone();
    tampered.hash = "0000000000000000".into();
    let error = runner
        .resume(
            &mut conversation,
            &[tampered.approve()],
            &CancellationSignal::new(),
            |_| {},
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Approval(_)), "{error}");
    assert_eq!(deletes.load(Ordering::SeqCst), 0);

    // An abort ends the run, every pending call answered "aborted".
    let outcome = runner
        .resume(
            &mut conversation,
            &[waiting[0].abort()],
            &CancellationSignal::new(),
            |_| {},
        )
        .await
        .unwrap();
    assert_eq!(outcome.stop, RunStop::Cancelled);
    assert!(conversation.unanswered_calls().is_empty());
    assert_eq!(deletes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_agent_resumes_from_a_session_in_another_process() {
    let store = MemoryStore::new();
    let id = SessionId::new("ops");
    let waiting = {
        let (tools, _, _) = files();
        let mut agent = Agent::new("ops", cleaner())
            .with_tools(tools)
            .approvals(AskFor::tools(["delete_*"]));
        let mut session = Session::open(&store, id.clone()).await.unwrap();
        session
            .append_input("m1", Item::user("Clean up"))
            .await
            .unwrap();
        let outcome = agent
            .run_session(&(), &mut session, &CancellationSignal::new(), |_| {})
            .await
            .unwrap();
        pending(&outcome.run.stop)
    };
    // Later: a new agent, the session opened again.
    let (tools, _, deletes) = files();
    let model = ScriptedModel::replies(vec![Reply::text("Cleaned up.")]);
    let mut agent = Agent::new("ops", model)
        .with_tools(tools)
        .approvals(AskFor::tools(["delete_*"]));
    let mut session = Session::open(&store, id.clone()).await.unwrap();
    let outcome = agent
        .resume_session(
            &(),
            &mut session,
            &[waiting[0].approve()],
            &CancellationSignal::new(),
            |_| {},
        )
        .await
        .unwrap();
    assert_eq!(outcome.output().unwrap(), "Cleaned up.");
    assert_eq!(deletes.load(Ordering::SeqCst), 1);
    let reopened = Session::open(&store, id).await.unwrap();
    assert!(reopened.conversation().unanswered_calls().is_empty());
}

#[tokio::test]
async fn a_handoff_runs_the_other_agent_on_the_conversation() {
    let triage_model = ScriptedModel::replies(vec![
        calls(&[("h1", "get_order", json!({}))]),
        calls(&[("h2", "transfer_to_billing", json!({}))]),
    ]);
    let billing_model = ScriptedModel::replies(vec![Reply::text("Refund issued.")]);
    let mut tools = ToolRegistry::new();
    tools
        .register(Counted {
            name: "get_order",
            runs: Arc::default(),
        })
        .unwrap();
    let billing = Agent::new("billing", billing_model.clone()).description("Refunds and invoices.");
    let mut triage = Agent::new("triage", triage_model.clone())
        .with_tools(tools)
        .handoff_filtered(billing, Some(without_tool_calls));
    let mut conversation = Conversation::user("I want my money back");
    let outcome = triage.run(&(), &mut conversation).await.unwrap();
    assert_eq!(outcome.output().unwrap(), "Refund issued.");
    // Both agents' requests are in the outcome.
    assert_eq!(outcome.run.turns.len(), 3);

    // Triage saw the handoff tool, described.
    let definitions = &triage_model.requests()[0].tools;
    let handoff = definitions
        .iter()
        .find(|tool| tool.name() == Some("transfer_to_billing"))
        .unwrap();
    assert!(format!("{handoff:?}").contains("Refunds and invoices."));
    // Billing saw only the messages.
    let seen = &billing_model.requests()[0].items;
    assert!(seen.iter().all(|item| matches!(item, Item::Message(_))));
    // The conversation has everything, billing's answer last.
    assert!(conversation
        .items()
        .iter()
        .any(|item| matches!(item, Item::ToolCall(call) if call.name == "transfer_to_billing")));
    assert_eq!(
        conversation.last_assistant_text().as_deref(),
        Some("Refund issued.")
    );
}

#[tokio::test]
async fn an_agent_answers_as_a_tool() {
    let expert_model = ScriptedModel::replies(vec![Reply::text("Use the 13 mm wrench.")]);
    let expert = Agent::new("mechanic", expert_model.clone());
    let mut tools = ToolRegistry::new();
    tools
        .register(AgentTool::new(
            "ask_mechanic",
            "Ask the mechanic",
            expert,
            Arc::new(()),
        ))
        .unwrap();
    let model = ScriptedModel::new(|index, _| match index {
        0 => calls(&[(
            "t1",
            "ask_mechanic",
            json!({"input": "Which wrench for the bike?"}),
        )]),
        _ => Reply::text("The mechanic says 13 mm."),
    });
    let mut runner = Runner::new(model.clone(), tools);
    let outcome = runner
        .run(&mut Conversation::user("Fix my bike"))
        .await
        .unwrap();
    assert_eq!(outcome.output_text(), "The mechanic says 13 mm.");
    assert_eq!(
        output_of(&model.requests()[1], "t1"),
        "\"Use the 13 mm wrench.\""
    );
    // The expert got the question on a conversation of its own.
    assert_eq!(
        expert_model.requests()[0].items,
        [Item::user("Which wrench for the bike?")]
    );
}

#[tokio::test]
async fn sub_agents_run_in_the_background_and_report_back() {
    // The child takes a while; the parent goes on, then waits for it before stopping.
    let child = || {
        let mut slow = Reply::text("Three sources agree.");
        slow.events_mut().unwrap()[0].0 = Duration::from_millis(100);
        Agent::new("research", ScriptedModel::replies(vec![slow]))
    };
    let parent_model = ScriptedModel::new(|index, request| match index {
        0 => calls(&[("s1", "start_research", json!({"task": "Check the claim"}))]),
        1 => Reply::text("Research started; I will report."),
        _ => {
            let last = request.items.last().unwrap();
            assert!(matches!(last, Item::Message(message)
                if message.text_content().starts_with("The sub-agent research finished task 1")));
            Reply::text("The claim holds.")
        }
    });
    let mut parent =
        Agent::new("lead", parent_model.clone()).sub_agent("research", "Looks things up.", child);
    let started = std::time::Instant::now();
    let outcome = parent
        .run(&(), &mut Conversation::user("Is the claim true?"))
        .await
        .unwrap();
    assert_eq!(outcome.output().unwrap(), "The claim holds.");
    assert!(started.elapsed() >= Duration::from_millis(100));
    let requests = parent_model.requests();
    assert_eq!(requests.len(), 3);
    // Started at once: the tool's answer did not wait for the child.
    assert!(output_of(&requests[1], "s1").contains("Started research on task 1"));
}

#[tokio::test]
async fn sub_agent_limits_and_policy() {
    // At most one at a time.
    let parent_model = ScriptedModel::new(|index, _| match index {
        0 => calls(&[
            ("s1", "start_helper", json!({"task": "one"})),
            ("s2", "start_helper", json!({"task": "two"})),
        ]),
        _ => Reply::text("Done."),
    });
    let mut parent = Agent::new("lead", parent_model.clone())
        .sub_agent("helper", "Helps.", || {
            // Still running when the second call comes.
            let mut slow = Reply::text("ok");
            slow.events_mut().unwrap()[0].0 = Duration::from_millis(100);
            Agent::new("helper", ScriptedModel::replies(vec![slow]))
        })
        .sub_agent_limits(SubAgentLimits::new(1, 1));
    parent
        .run(&(), &mut Conversation::user("Go"))
        .await
        .unwrap();
    let second = output_of(&parent_model.requests()[1], "s2");
    assert!(second.contains("the most at once"), "{second}");

    // A sub-agent at the depth limit cannot start its own; it uses only tools its parent
    // allows.
    let child_model = ScriptedModel::new(|index, _| match index {
        0 => calls(&[
            ("g1", "start_grandchild", json!({"task": "deeper"})),
            ("g2", "delete_file", json!({})),
        ]),
        _ => Reply::text("child done"),
    });
    let make_child = {
        let child_model = child_model.clone();
        move || {
            let (tools, _, _) = files();
            Agent::new("child", child_model.clone())
                .with_tools(tools)
                .sub_agent("grandchild", "Deeper.", || {
                    Agent::new("grandchild", ScriptedModel::replies(vec![Reply::text("x")]))
                })
        }
    };
    let parent_model = ScriptedModel::new(|index, _| match index {
        0 => calls(&[("s1", "start_child", json!({"task": "work"}))]),
        _ => Reply::text("Done."),
    });
    let mut parent = Agent::new("lead", parent_model)
        .sub_agent("child", "Works.", make_child)
        .sub_agent_limits(SubAgentLimits::new(4, 1))
        .tool_policy(ToolPolicy::new().deny("delete_*"));
    parent
        .run(&(), &mut Conversation::user("Go"))
        .await
        .unwrap();
    let child_second = &child_model.requests()[1];
    assert!(output_of(child_second, "g1").contains("this deep"));
    assert!(output_of(child_second, "g2").contains("not allowed"));
    assert!(!child_model.requests()[0]
        .tool_names()
        .contains(&"delete_file".to_owned()));
}
