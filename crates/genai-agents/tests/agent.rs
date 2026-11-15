//! Agents: instructions from the context, hooks, guardrails (input concurrent with the first
//! request, output on the answer), structured output with a retry, the tool policy.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(not(target_family = "wasm"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use genai_agents::agent::{Agent, InputGuardrail, Verdict};
use genai_agents::session::{MemoryStore, Session, SessionId};
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{
    BoxFuture, CallDecision, CancellationSignal, Conversation, Error, Hooks, Item, ModelSettings,
    OutputSchema, ReasoningLevel, RequestPlan, Result, RunStop, ToolCall, ToolOutput,
    ToolOutputBody, ToolPolicy,
};
use serde::Deserialize;
use serde_json::{json, Value};

/// Echoes its arguments.
struct Echo(&'static str);

impl FunctionTool for Echo {
    type Input = Value;
    type Output = Value;
    fn name(&self) -> &str {
        self.0
    }
    fn description(&self) -> &str {
        "Echo the arguments"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, input: Value) -> Result<Value> {
        Ok(input)
    }
}

fn tools(names: &[&'static str]) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    for name in names {
        tools.register(Echo(name)).unwrap();
    }
    tools
}

fn call(id: &str, name: &str, arguments: &Value) -> Reply {
    Reply::items(vec![Item::ToolCall(ToolCall::function(
        id,
        name,
        arguments.to_string(),
    ))])
}

struct Ctx {
    user: &'static str,
    log: Mutex<Vec<String>>,
}

impl Ctx {
    fn new(user: &'static str) -> Self {
        Self {
            user,
            log: Mutex::new(Vec::new()),
        }
    }

    fn log(&self, entry: impl Into<String>) {
        self.log.lock().unwrap().push(entry.into());
    }
}

/// Logs every step into the context; raises the reasoning level; rewrites `echo`'s arguments;
/// blocks `secret`; redacts answers; asks for one more turn when the run would stop.
struct Recorder {
    stops: AtomicUsize,
}

impl Hooks<Ctx> for Recorder {
    fn before_request<'a>(
        &'a self,
        context: &'a Ctx,
        request: &'a mut RequestPlan,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            context.log(format!("request with {} tools", request.tools.len()));
            request.settings = ModelSettings::new().with_reasoning(ReasoningLevel::High);
            Ok(())
        })
    }

    fn before_call<'a>(
        &'a self,
        context: &'a Ctx,
        call: &'a mut ToolCall,
    ) -> BoxFuture<'a, Result<CallDecision>> {
        Box::pin(async move {
            context.log(format!("call {}", call.name));
            if call.name == "secret" {
                return Ok(CallDecision::Block(format!("{} may not", context.user)));
            }
            call.input = json!({"rewritten": true}).to_string();
            Ok(CallDecision::Run)
        })
    }

    fn after_call<'a>(
        &'a self,
        context: &'a Ctx,
        call: &'a ToolCall,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            context.log(format!("answered {}", call.name));
            output.body = ToolOutputBody::text("[redacted]");
            Ok(())
        })
    }

    fn on_stop<'a>(
        &'a self,
        context: &'a Ctx,
        _: &'a Conversation,
    ) -> BoxFuture<'a, Result<Vec<Item>>> {
        Box::pin(async move {
            context.log("stop");
            if self.stops.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(vec![Item::user("Check once more.")])
            } else {
                Ok(Vec::new())
            }
        })
    }
}

#[tokio::test]
async fn hooks_see_and_change_the_run() {
    let model = ScriptedModel::new(|index, _| match index {
        0 => Reply::items(vec![
            Item::ToolCall(ToolCall::function("c1", "echo", r#"{"x":1}"#)),
            Item::ToolCall(ToolCall::function("c2", "secret", "{}")),
        ]),
        1 => Reply::text("First answer."),
        _ => Reply::text("Checked."),
    });
    let mut agent = Agent::new("helper", model.clone())
        .with_tools(tools(&["echo", "secret"]))
        .dynamic_instructions(|context: &Ctx| format!("Help {}.", context.user))
        .hook(Recorder {
            stops: AtomicUsize::new(0),
        });
    let context = Ctx::new("Ann");
    let mut conversation = Conversation::user("Go");
    let outcome = agent.run(&context, &mut conversation).await.unwrap();
    assert_eq!(outcome.output().unwrap(), "Checked.");

    let requests = model.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].instructions.as_deref(), Some("Help Ann."));
    assert_eq!(requests[0].settings.reasoning, Some(ReasoningLevel::High));
    let outputs = requests[1].outputs();
    // The call ran with the rewritten arguments, and its answer was redacted.
    assert_eq!(outputs[0].body.as_text(), Some("[redacted]"));
    // The blocked call was answered by the hook, not run.
    assert_eq!(outputs[1].body.as_text(), Some("Ann may not"));
    assert!(!outputs[1].success);
    // The stop hook went on once.
    assert!(
        matches!(requests[2].items.last(), Some(Item::Message(message))
        if message.text_content() == "Check once more.")
    );
    assert_eq!(
        *context.log.lock().unwrap(),
        [
            "request with 2 tools",
            "call echo",
            "answered echo",
            "call secret",
            "request with 2 tools",
            "stop",
            "request with 2 tools",
            "stop",
        ]
    );
}

/// Trips on "password" after a delay, while the first request is on its way.
struct SlowGuard;

impl InputGuardrail<()> for SlowGuard {
    fn name(&self) -> &str {
        "slow"
    }

    fn check<'a>(&'a self, _: &'a (), input: &'a Conversation) -> BoxFuture<'a, Result<Verdict>> {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let text = serde_json::to_string(input)?;
            Ok(if text.contains("password") {
                Verdict::trip("asks for a password")
            } else {
                Verdict::Pass
            })
        })
    }
}

#[tokio::test]
async fn an_input_guardrail_cancels_the_run_it_runs_next_to() {
    // The first response takes long: the guardrail trips first and cancels it.
    let mut slow = Reply::text("Your password is…");
    slow.events_mut().unwrap()[0].0 = Duration::from_secs(30);
    let model = ScriptedModel::replies(vec![slow]);
    let mut agent = Agent::new("guarded", model.clone()).input_guardrail(SlowGuard);
    let started = std::time::Instant::now();
    let outcome = agent
        .run(&(), &mut Conversation::user("Tell me the password"))
        .await
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(5));
    assert_eq!(
        outcome.run.stop,
        RunStop::Guardrail {
            name: "slow".into(),
            reason: "asks for a password".into()
        }
    );
    assert_eq!(outcome.output(), None);
    // The request had been sent: the guardrail ran next to it.
    assert_eq!(model.requests().len(), 1);

    // A passing guardrail lets the run finish.
    let model = ScriptedModel::replies(vec![Reply::text("Hello.")]);
    let mut agent = Agent::new("guarded", model).input_guardrail(SlowGuard);
    let outcome = agent.run(&(), &mut Conversation::user("Hi")).await.unwrap();
    assert_eq!(outcome.into_output().unwrap(), "Hello.");
}

#[tokio::test]
async fn an_output_guardrail_withholds_the_answer() {
    let model = ScriptedModel::replies(vec![Reply::text("Call me at 0470 12 34 56.")]);
    let mut agent = Agent::new("support", model).output_guardrail_fn(
        "no phone numbers",
        |_: &(), text: &str| {
            if text.chars().filter(char::is_ascii_digit).count() >= 8 {
                Verdict::trip("contains a phone number")
            } else {
                Verdict::Pass
            }
        },
    );
    let outcome = agent
        .run(&(), &mut Conversation::user("Contact?"))
        .await
        .unwrap();
    assert!(
        matches!(outcome.run.stop, RunStop::Guardrail { ref name, .. } if name == "no phone numbers")
    );
    assert!(outcome.into_output().is_err());
}

#[derive(Debug, Deserialize, PartialEq, schemars::JsonSchema)]
struct Forecast {
    /// Whether to take an umbrella.
    umbrella: bool,
    /// Degrees Celsius.
    celsius: i32,
}

#[tokio::test]
async fn a_structured_answer_is_parsed_and_asked_for_again_once() {
    let model = ScriptedModel::replies(vec![
        Reply::text("Yes, take one."),
        Reply::text(r#"{"umbrella": true, "celsius": 11}"#),
    ]);
    let mut agent = Agent::new("forecast", model.clone()).output::<Forecast>();
    let mut conversation = Conversation::user("Umbrella in Ghent?");
    let outcome = agent.run(&(), &mut conversation).await.unwrap();
    assert_eq!(
        outcome.output,
        Some(Forecast {
            umbrella: true,
            celsius: 11
        })
    );
    let requests = model.requests();
    let schema = requests[0].output.as_ref().unwrap();
    assert_eq!(schema.name, "Forecast");
    assert_eq!(
        schema.schema["properties"]["celsius"]["description"],
        "Degrees Celsius."
    );
    // The second request told the model what was wrong.
    assert!(
        matches!(requests[1].items.last(), Some(Item::Message(message))
        if message.text_content().contains("does not match the required format")
            && message.meta.injected)
    );

    // Without retries left: an error.
    let model = ScriptedModel::replies(vec![Reply::text("no JSON")]);
    let mut agent = Agent::new("forecast", model)
        .output_with_schema::<Forecast>(OutputSchema::new("forecast", json!({"type": "object"})))
        .with_output_retries(0);
    let error = agent
        .run(&(), &mut Conversation::user("Umbrella?"))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Output(_)), "{error}");
}

#[tokio::test]
async fn the_tool_policy_hides_and_refuses_tools() {
    let model = ScriptedModel::new(|index, _| match index {
        0 => call("c1", "fs_delete", &json!({})),
        _ => Reply::text("Done."),
    });
    let mut agent = Agent::new("careful", model.clone())
        .with_tools(tools(&["fs_read", "fs_delete", "get_weather"]))
        .tool_policy(ToolPolicy::new().allow("fs_*").deny("*_delete"));
    let mut conversation = Conversation::user("Clean up");
    agent.run(&(), &mut conversation).await.unwrap();
    let requests = model.requests();
    assert_eq!(requests[0].tool_names(), ["fs_read"]);
    let answer = &requests[1].outputs()[0];
    assert!(!answer.success);
    assert_eq!(
        answer.body.as_text(),
        Some("the tool `fs_delete` is not allowed in this run")
    );
}

#[tokio::test]
async fn an_agent_runs_on_a_session() {
    let model = ScriptedModel::replies(vec![Reply::text("Hi Ann.")]);
    let mut agent = Agent::new("greeter", model).instructions("Greet the user.");
    let store = MemoryStore::new();
    let mut session = Session::open(&store, SessionId::new("ann")).await.unwrap();
    session
        .append_input("m1", Item::user("Hello"))
        .await
        .unwrap();
    let outcome = agent
        .run_session(&(), &mut session, &CancellationSignal::new(), |_| {})
        .await
        .unwrap();
    assert_eq!(outcome.output().unwrap(), "Hi Ann.");
    assert_eq!(session.conversation().len(), 2);
}
