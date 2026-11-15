//! The runner against a scripted model: tool calls started before the response ends, ordered
//! parallel dispatch, history repair, max iterations, cancellation, whole-request retries (with
//! `retry-after`, and without retrying what the provider retries), deferred tools and tool
//! search, rate limits, incomplete responses.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(
    unknown_lints,
    clippy::unused_async_trait_impl,
    clippy::significant_drop_tightening,
    clippy::type_complexity,
    clippy::unnecessary_literal_bound
)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::tools::{Exposure, FunctionTool, RichTool, ToolKey, ToolRegistry};
use genai_agents::{
    CancellationSignal, Content, Conversation, Error, Item, Message, ModelError, ModelEvent,
    ModelInfo, ModelResponse, PromptCacheKey, Result, RetryClass, RetryConfig, Role, RunConfig,
    RunEvent, RunStop, Runner, ToolCall, ToolDefinition, ToolOutput, ToolOutputBody, TurnRecord,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------------------------
// The scripted model (`genai_agents::testing`).

/// A response of these items, with this id.
fn reply(id: &str, items: Vec<Item>) -> Reply {
    Reply::items(items).with_id(id)
}

fn call(call_id: &str, name: &str, arguments: &Value) -> Item {
    Item::ToolCall(
        ToolCall::function(call_id, name, arguments.to_string()).with_id(format!("fc_{call_id}")),
    )
}

fn message(text: &str) -> Item {
    Item::Message(Message::text(Role::Assistant, text).with_id("msg_1"))
}

fn failure(status: u16, retry: RetryClass) -> ModelError {
    ModelError::new(format!("status {status}"))
        .with_status(status)
        .with_retry_class(retry)
}

// ---------------------------------------------------------------------------------------------
// Tools.

type Log = Arc<Mutex<Vec<(String, &'static str, Instant)>>>;

/// Sleeps, then returns its name; logs its start and end.
struct Sleepy {
    name: &'static str,
    millis: u64,
    parallel: bool,
    exposure: Exposure,
    log: Log,
}

impl Sleepy {
    fn new(name: &'static str, millis: u64, parallel: bool, log: &Log) -> Self {
        Self {
            name,
            millis,
            parallel,
            exposure: Exposure::Direct,
            log: log.clone(),
        }
    }
}

impl FunctionTool for Sleepy {
    type Input = Value;
    type Output = String;
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "Sleep for a while, then say your name."
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    async fn execute(&self, _input: Value) -> Result<String> {
        self.log
            .lock()
            .unwrap()
            .push((self.name.into(), "start", Instant::now()));
        tokio::time::sleep(Duration::from_millis(self.millis)).await;
        self.log
            .lock()
            .unwrap()
            .push((self.name.into(), "end", Instant::now()));
        Ok(self.name.to_string())
    }
    fn supports_parallel(&self) -> bool {
        self.parallel
    }
    fn exposure(&self) -> Exposure {
        self.exposure
    }
}

struct Weather;

impl FunctionTool for Weather {
    type Input = Value;
    type Output = Value;
    fn name(&self) -> &str {
        "get_weather"
    }
    fn description(&self) -> &str {
        "Get the current weather of a city."
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
    }
    async fn execute(&self, input: Value) -> Result<Value> {
        Ok(json!({"city": input["city"], "temperature": 12}))
    }
}

fn weather_tools() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(Weather).unwrap();
    tools
}

fn when(log: &Log, name: &str, what: &str) -> Instant {
    log.lock()
        .unwrap()
        .iter()
        .find(|(tool, kind, _)| tool == name && *kind == what)
        .map(|(_, _, at)| *at)
        .unwrap_or_else(|| panic!("{name} has no {what}"))
}

// ---------------------------------------------------------------------------------------------
// Tests.

#[tokio::test]
async fn the_runner_runs_tools_until_the_model_answers() {
    let model = ScriptedModel::new(|index, _| {
        let items = match index {
            0 => vec![call("call_1", "get_weather", &json!({"city": "Ghent"}))],
            _ => vec![message("12 degrees in Ghent.")],
        };
        reply(&format!("resp_{index}"), items).with_remaining_requests(99 - index as u64)
    });
    let mut runner = Runner::new(model.clone(), weather_tools());
    let mut conversation = Conversation::user("Weather in Ghent?");
    let outcome = runner.run(&mut conversation).await.unwrap();

    assert_eq!(outcome.stop, RunStop::Completed);
    assert_eq!(outcome.output_text(), "12 degrees in Ghent.");
    assert_eq!(outcome.turns.len(), 2);
    assert_eq!(outcome.turns[0].calls[0].name, "get_weather");
    assert_eq!(outcome.turns[0].response_id.as_deref(), Some("resp_0"));
    assert_eq!(outcome.turns[1].input_tokens, Some(10));
    assert_eq!(outcome.turns[1].output_tokens, Some(5));
    // The rate limits of each request.
    let first = outcome.turns[0].rate_limits.unwrap().requests.unwrap();
    assert_eq!(first.remaining, Some(99));
    let second = outcome.rate_limits().unwrap().requests.unwrap();
    assert_eq!(second.remaining, Some(98));

    let seen = model.requests();
    assert_eq!(seen.len(), 2);
    for request in &seen {
        assert!(request.parallel_tool_calls);
        assert_eq!(
            request.prompt_cache_key.as_deref(),
            runner.prompt_cache_key()
        );
        assert_eq!(request.tool_names(), ["get_weather"]);
    }
    assert!(runner
        .prompt_cache_key()
        .unwrap()
        .starts_with("genai-agents-"));
    // The second request carries the call and its output.
    let outputs = seen[1].outputs();
    assert_eq!(outputs[0].call_id, "call_1");
    assert_eq!(
        outputs[0].body.as_text(),
        Some(r#"{"city":"Ghent","temperature":12}"#)
    );
    // The history holds everything, in order.
    let kinds: Vec<&str> = conversation
        .items()
        .iter()
        .map(|item| match item {
            Item::Message(_) => "message",
            Item::ToolCall(_) => "call",
            Item::ToolOutput(_) => "output",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["message", "call", "output", "message"]);
}

#[tokio::test]
async fn parallel_calls_overlap_and_answers_keep_the_call_order() {
    let log: Log = Arc::default();
    let mut tools = ToolRegistry::new();
    tools
        .register(Sleepy::new("slow", 150, true, &log))
        .unwrap();
    tools.register(Sleepy::new("fast", 20, true, &log)).unwrap();
    tools
        .register(Sleepy::new("alone", 20, false, &log))
        .unwrap();
    tools
        .register(Sleepy::new("after", 20, true, &log))
        .unwrap();
    let model = ScriptedModel::new(|index, _| {
        let items = match index {
            0 => vec![
                call("a", "slow", &json!({})),
                call("b", "fast", &json!({})),
                call("c", "alone", &json!({})),
                call("d", "after", &json!({})),
            ],
            _ => vec![message("done")],
        };
        reply("resp", items)
    });
    let mut runner = Runner::new(model.clone(), tools);
    let mut answered = Vec::new();
    let outcome = runner
        .run_with(
            &mut Conversation::user("go"),
            &CancellationSignal::new(),
            |event| {
                if let RunEvent::ToolAnswered(answer) = event {
                    answered.push(answer.call_id.clone());
                }
            },
        )
        .await
        .unwrap();
    assert!(outcome.is_completed());

    // slow and fast ran together; alone after both; after once alone was done.
    assert!(when(&log, "fast", "start") < when(&log, "slow", "end"));
    assert!(when(&log, "alone", "start") >= when(&log, "slow", "end"));
    assert!(when(&log, "alone", "start") >= when(&log, "fast", "end"));
    assert!(when(&log, "after", "start") >= when(&log, "alone", "end"));
    // Answered as they finished, recorded in call order.
    assert_eq!(answered, ["b", "a", "c", "d"]);
    let ids: Vec<String> = model.requests()[1]
        .outputs()
        .into_iter()
        .map(|output| output.call_id)
        .collect();
    assert_eq!(ids, ["a", "b", "c", "d"]);
}

#[tokio::test]
async fn without_parallel_tool_calls_every_call_runs_alone() {
    let log: Log = Arc::default();
    let mut tools = ToolRegistry::new();
    tools.register(Sleepy::new("one", 40, true, &log)).unwrap();
    tools.register(Sleepy::new("two", 40, true, &log)).unwrap();
    let model = ScriptedModel::new(|index, _| {
        let items = match index {
            0 => vec![call("a", "one", &json!({})), call("b", "two", &json!({}))],
            _ => vec![message("done")],
        };
        reply("resp", items)
    });
    let mut runner = Runner::new(model.clone(), tools)
        .with_config(RunConfig::new().with_parallel_tool_calls(false));
    runner.run(&mut Conversation::user("go")).await.unwrap();
    assert!(when(&log, "two", "start") >= when(&log, "one", "end"));
    assert!(!model.requests()[0].parallel_tool_calls);
}

#[tokio::test]
async fn calls_start_before_the_response_ends() {
    let log: Log = Arc::default();
    let mut tools = ToolRegistry::new();
    tools
        .register(Sleepy::new("probe", 10, true, &log))
        .unwrap();
    let model = ScriptedModel::new(|index, _| {
        if index > 0 {
            return reply("resp_1", vec![message("ok")]);
        }
        let mut reply = reply(
            "resp_0",
            vec![call("a", "probe", &json!({})), message("still writing")],
        );
        // The response goes on for a while after the call is done.
        let events = reply.events_mut().unwrap();
        let last = events.len() - 1;
        events[last].0 = Duration::from_millis(300);
        reply
    });
    let mut runner = Runner::new(model, tools);
    let mut completed: Option<Instant> = None;
    let outcome = runner
        .run_with(
            &mut Conversation::user("go"),
            &CancellationSignal::new(),
            |event| {
                if let RunEvent::Model(ModelEvent::Completed(_)) = event {
                    completed.get_or_insert_with(Instant::now);
                }
            },
        )
        .await
        .unwrap();
    assert!(outcome.is_completed());
    // The tool finished well before the response did.
    assert!(when(&log, "probe", "end") + Duration::from_millis(150) < completed.unwrap());
}

#[tokio::test]
async fn max_iterations_stops_with_every_call_answered() {
    let model = ScriptedModel::new(|index, _| {
        reply(
            "resp",
            vec![call(
                &format!("call_{index}"),
                "get_weather",
                &json!({"city": "Ghent"}),
            )],
        )
    });
    let mut runner = Runner::new(model.clone(), weather_tools())
        .with_config(RunConfig::new().with_max_iterations(2));
    let mut conversation = Conversation::user("Loop forever");
    let outcome = runner.run(&mut conversation).await.unwrap();
    assert_eq!(outcome.stop, RunStop::MaxIterations);
    assert_eq!(outcome.turns.len(), 2);
    assert_eq!(model.requests().len(), 2);
    assert_eq!(conversation.unanswered_calls(), Vec::<ToolCall>::new());
    assert_eq!(conversation.calls().len(), 2);
    // A new run continues where it stopped.
    let outcome = runner.run(&mut conversation).await.unwrap();
    assert_eq!(outcome.stop, RunStop::MaxIterations);
    assert_eq!(conversation.calls().len(), 4);
}

#[tokio::test]
async fn cancellation_aborts_running_calls_and_leaves_a_valid_history() {
    let log: Log = Arc::default();
    let mut tools = ToolRegistry::new();
    tools.register(Sleepy::new("quick", 5, true, &log)).unwrap();
    tools
        .register(Sleepy::new("forever", 60_000, true, &log))
        .unwrap();
    tools
        .register(Sleepy::new("waiting", 5, false, &log))
        .unwrap();
    let model = ScriptedModel::new(|_, _| {
        reply(
            "resp",
            vec![
                call("a", "quick", &json!({})),
                call("b", "forever", &json!({})),
                call("c", "waiting", &json!({})),
            ],
        )
    });
    let mut runner = Runner::new(model.clone(), tools);
    let cancel = CancellationSignal::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        trigger.cancel();
    });
    let mut conversation = Conversation::user("go");
    let started = Instant::now();
    let outcome = runner
        .run_with(&mut conversation, &cancel, |_| {})
        .await
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(5));
    assert_eq!(outcome.stop, RunStop::Cancelled);
    assert_eq!(model.requests().len(), 1);
    assert_eq!(conversation.unanswered_calls(), Vec::<ToolCall>::new());
    let outputs: Vec<ToolOutputBody> = conversation
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::ToolOutput(output) => Some(output.body.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].as_text(), Some("\"quick\""));
    assert!(outputs[1]
        .as_text()
        .unwrap()
        .starts_with("aborted by user after 0."));
    assert_eq!(outputs[2].as_text(), Some("aborted"));
    // Cancelled before the start: no request at all.
    let outcome = runner
        .run_with(&mut conversation, &cancel, |_| {})
        .await
        .unwrap();
    assert_eq!(outcome.stop, RunStop::Cancelled);
    assert_eq!(outcome.turns, Vec::<TurnRecord>::new());
    assert_eq!(model.requests().len(), 1);
}

#[tokio::test]
async fn the_history_is_repaired_before_the_request() {
    let model = ScriptedModel::new(|_, _| reply("resp", vec![message("ok")]));
    let mut conversation = Conversation::user("hi");
    conversation.push_call(&ToolCall::function("lost", "get_weather", "{}"));
    conversation.push_output(&ToolOutput::new(
        &ToolCall::function("ghost", "get_weather", "{}"),
        "orphan",
        true,
    ));
    let mut runner = Runner::new(model.clone(), weather_tools());
    runner.run(&mut conversation).await.unwrap();
    let outputs = model.requests()[0].outputs();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].call_id, "lost");
    assert_eq!(outputs[0].body.as_text(), Some("aborted"));
}

#[tokio::test]
async fn failures_before_the_body_are_left_to_the_providers_retries() {
    let rate_limited =
        || failure(429, RetryClass::Backoff).with_retry_after(Some(Duration::from_millis(1)));
    // The provider retries them itself: the runner does not retry them again.
    let model =
        ScriptedModel::new(move |_, _| Reply::Fail(rate_limited())).with_retries_before_body(true);
    let mut runner = Runner::new(model.clone(), weather_tools())
        .with_config(RunConfig::new().with_retry(RetryConfig::new(5)));
    let mut retries = 0;
    let error = runner
        .run_with(
            &mut Conversation::user("hi"),
            &CancellationSignal::new(),
            |event| {
                if matches!(event, RunEvent::Retrying { .. }) {
                    retries += 1;
                }
            },
        )
        .await
        .unwrap_err();
    assert!(error.as_model().unwrap().is_rate_limit(), "{error}");
    assert_eq!(retries, 0);
    assert_eq!(model.requests().len(), 1);

    // It does not: the runner retries them.
    let model =
        ScriptedModel::new(move |_, _| Reply::Fail(rate_limited())).with_retries_before_body(false);
    let mut runner = Runner::new(model.clone(), weather_tools())
        .with_config(RunConfig::new().with_retry(RetryConfig::new(2)));
    runner.run(&mut Conversation::user("hi")).await.unwrap_err();
    assert_eq!(model.requests().len(), 3);
}

#[tokio::test]
async fn a_rate_limited_request_is_retried_after_retry_after() {
    let model = ScriptedModel::new(|index, _| match index {
        0 => Reply::Fail(
            failure(429, RetryClass::Backoff).with_retry_after(Some(Duration::from_millis(120))),
        ),
        _ => reply("resp", vec![message("ok")]).with_remaining_requests(7),
    });
    let mut runner = Runner::new(model.clone(), weather_tools());
    let mut delays = Vec::new();
    let started = Instant::now();
    let outcome = runner
        .run_with(
            &mut Conversation::user("hi"),
            &CancellationSignal::new(),
            |event| {
                if let RunEvent::Retrying {
                    delay,
                    error,
                    attempt,
                    max,
                } = event
                {
                    assert!(error.is_rate_limit());
                    assert_eq!(max, 5);
                    delays.push((attempt, delay));
                }
            },
        )
        .await
        .unwrap();
    assert!(started.elapsed() >= Duration::from_millis(120));
    assert_eq!(delays, [(1, Duration::from_millis(120))]);
    assert!(outcome.is_completed());
    assert_eq!(outcome.turns[0].retries, 1);
    assert_eq!(
        outcome.turns[0]
            .rate_limits
            .unwrap()
            .requests
            .unwrap()
            .remaining,
        Some(7)
    );
    assert_eq!(model.requests().len(), 2);

    // A delay in the error itself.
    let model = ScriptedModel::new(|index, _| match index {
        0 => Reply::Fail(failure(
            429,
            RetryClass::ServerDirected {
                after: Duration::from_millis(30),
            },
        )),
        _ => reply("resp", vec![message("ok")]),
    });
    let mut runner = Runner::new(model, weather_tools());
    let mut delays = Vec::new();
    runner
        .run_with(
            &mut Conversation::user("hi"),
            &CancellationSignal::new(),
            |event| {
                if let RunEvent::Retrying { delay, .. } = event {
                    delays.push(delay);
                }
            },
        )
        .await
        .unwrap();
    assert_eq!(delays, [Duration::from_millis(30)]);
}

#[tokio::test]
async fn a_cut_off_stream_is_retried_with_backoff_without_running_calls_twice() {
    let log: Log = Arc::default();
    let mut tools = ToolRegistry::new();
    tools.register(Sleepy::new("once", 5, true, &log)).unwrap();
    let model = ScriptedModel::new(|index, _| match index {
        // The call is done, then the stream stops without its end.
        0 => {
            let mut reply = reply("resp_0", vec![call("a", "once", &json!({}))]);
            reply.events_mut().unwrap().pop();
            reply
        }
        _ => reply("resp_1", vec![message("done")]),
    });
    let retry = RetryConfig::new(2).with_base_delay(Duration::from_millis(10));
    let mut runner =
        Runner::new(model.clone(), tools).with_config(RunConfig::new().with_retry(retry));
    let mut delays = Vec::new();
    let outcome = runner
        .run_with(
            &mut Conversation::user("hi"),
            &CancellationSignal::new(),
            |event| {
                if let RunEvent::Retrying { delay, error, .. } = event {
                    assert!(!error.is_before_body());
                    assert_eq!(error.retry_class(), RetryClass::Backoff);
                    delays.push(delay);
                }
            },
        )
        .await
        .unwrap();
    assert!(outcome.is_completed());
    assert_eq!(delays.len(), 1);
    assert!(delays[0] >= Duration::from_millis(9) && delays[0] <= Duration::from_millis(11));
    // The call ran once; the retry carries it with its output.
    let starts = log
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, kind, _)| *kind == "start")
        .count();
    assert_eq!(starts, 1);
    assert_eq!(
        model.requests()[1].outputs()[0].body.as_text(),
        Some("\"once\"")
    );

    // An error event midway is retried the same way.
    let model = ScriptedModel::new(|index, _| match index {
        0 => Reply::Events {
            rate_limits: None,
            events: vec![(
                Duration::ZERO,
                Err(ModelError::new("boom").with_retry_class(RetryClass::Backoff)),
            )],
        },
        _ => reply("resp", vec![message("ok")]),
    });
    let retry = RetryConfig::new(1).with_base_delay(Duration::from_millis(1));
    let mut runner =
        Runner::new(model.clone(), weather_tools()).with_config(RunConfig::new().with_retry(retry));
    assert!(runner
        .run(&mut Conversation::user("hi"))
        .await
        .unwrap()
        .is_completed());
    assert_eq!(model.requests().len(), 2);
}

#[tokio::test]
async fn retries_stop_at_errors_that_cannot_be_retried_and_at_the_limit() {
    let model = ScriptedModel::new(|_, _| Reply::Fail(failure(400, RetryClass::Never)));
    let mut runner = Runner::new(model.clone(), weather_tools());
    let error = runner.run(&mut Conversation::user("hi")).await.unwrap_err();
    assert!(matches!(error, Error::Model(ref error) if error.status() == Some(400)));
    assert_eq!(model.requests().len(), 1);

    let model = ScriptedModel::new(|_, _| Reply::Fail(failure(500, RetryClass::Backoff)));
    let retry = RetryConfig::new(2).with_base_delay(Duration::from_millis(1));
    let mut runner =
        Runner::new(model.clone(), weather_tools()).with_config(RunConfig::new().with_retry(retry));
    let error = runner.run(&mut Conversation::user("hi")).await.unwrap_err();
    assert!(error.as_model().unwrap().is_server_error());
    assert_eq!(model.requests().len(), 3);
}

#[tokio::test]
async fn a_fatal_tool_error_ends_the_run() {
    struct Fatal;
    impl FunctionTool for Fatal {
        type Input = Value;
        type Output = String;
        fn name(&self) -> &str {
            "explode"
        }
        fn description(&self) -> &str {
            "Fails fatally."
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn execute(&self, _: Value) -> Result<String> {
            Err(Error::fatal(std::io::Error::other("disk on fire")))
        }
    }
    let mut tools = weather_tools();
    tools.register(Fatal).unwrap();
    let model = ScriptedModel::new(|_, _| {
        reply(
            "resp",
            vec![
                call("a", "get_weather", &json!({"city": "Ghent"})),
                call("b", "explode", &json!({})),
            ],
        )
    });
    let mut runner =
        Runner::new(model, tools).with_config(RunConfig::new().with_parallel_tool_calls(false));
    let mut conversation = Conversation::user("hi");
    let error = runner.run(&mut conversation).await.unwrap_err();
    assert!(matches!(error, Error::Tool(_)), "{error:?}");
    // The answer of the call that finished is kept; the other is answered by normalize.
    assert_eq!(conversation.unanswered_calls().len(), 1);
    let report = conversation.normalize(&ModelInfo::default());
    assert_eq!(report.aborted, ["b"]);
}

#[tokio::test]
async fn deferred_tools_are_loaded_by_the_runners_tool_search() {
    let log: Log = Arc::default();
    let mut tools = weather_tools();
    let mut hidden = Sleepy::new("internal_audit", 1, true, &log);
    hidden.exposure = Exposure::Hidden;
    tools.register(hidden).unwrap();
    let mut deferred = Sleepy::new("convert_currency", 1, true, &log);
    deferred.exposure = Exposure::Deferred;
    tools.register(deferred).unwrap();
    let mut other = Sleepy::new("translate_text", 1, true, &log);
    other.exposure = Exposure::Deferred;
    tools.register(other).unwrap();
    // A namespaced deferred tool: only for models that take namespaces.
    let mut files = Sleepy::new("convert_file", 1, true, &log);
    files.exposure = Exposure::Deferred;
    tools
        .namespace("files", "File tools")
        .unwrap()
        .register(files)
        .unwrap();
    let registry = Arc::new(tools);

    for namespaces in [true, false] {
        let model = ScriptedModel::new(|index, _| {
            let items = match index {
                0 => vec![call(
                    "s",
                    "tool_search",
                    &json!({"query": "convert currency"}),
                )],
                1 => vec![call("c", "convert_currency", &json!({}))],
                2 => vec![call("h", "internal_audit", &json!({}))],
                _ => vec![message("done")],
            };
            reply("resp", items)
        })
        .with_info(ModelInfo::new().with_namespaces(namespaces));
        let mut runner = Runner::new(model.clone(), registry.clone());
        let outcome = runner
            .run(&mut Conversation::user("Convert 10 EUR"))
            .await
            .unwrap();
        assert!(outcome.is_completed());
        let mut loaded = vec![ToolKey::new("convert_currency")];
        if namespaces {
            loaded.push(ToolKey::namespaced("files", "convert_file"));
        }
        assert_eq!(runner.loaded_tools().cloned().collect::<Vec<_>>(), loaded);
        let seen = model.requests();
        assert_eq!(seen[0].tool_names(), ["get_weather", "tool_search"]);
        let mut second = vec!["convert_currency", "get_weather"];
        if namespaces {
            second.push("files");
        }
        second.push("tool_search");
        assert_eq!(seen[1].tool_names(), second);
        // The search's answer lists what it loaded.
        let search_output: Value =
            serde_json::from_str(seen[1].outputs()[0].body.as_text().unwrap()).unwrap();
        assert_eq!(search_output["tools"][0]["name"], "convert_currency");
        assert_eq!(
            search_output["tools"].as_array().unwrap().len(),
            if namespaces { 2 } else { 1 }
        );
        // The hidden tool is never advertised, but its call is answered.
        assert!(!seen[3].tool_names().contains(&"internal_audit".to_string()));
        assert_eq!(
            seen[3].outputs()[2].body.as_text(),
            Some("\"internal_audit\"")
        );
    }

    // Loaded by hand.
    let model = ScriptedModel::new(|_, _| reply("resp", vec![message("ok")]));
    let mut runner = Runner::new(model.clone(), registry.clone())
        .with_config(RunConfig::new().with_tool_search(None));
    assert!(runner.load_tool(ToolKey::new("translate_text")));
    assert!(
        !runner.load_tool(ToolKey::new("get_weather")),
        "not deferred"
    );
    runner.run(&mut Conversation::user("hi")).await.unwrap();
    assert_eq!(
        model.requests()[0].tool_names(),
        ["get_weather", "translate_text"]
    );
}

#[tokio::test]
async fn a_model_that_searches_tools_gets_them_deferred() {
    let log: Log = Arc::default();
    let mut tools = weather_tools();
    let mut deferred = Sleepy::new("convert_currency", 1, true, &log);
    deferred.exposure = Exposure::Deferred;
    tools.register(deferred).unwrap();
    let model = ScriptedModel::new(|index, _| match index {
        0 => {
            let mut reply = reply("resp", vec![call("c", "convert_currency", &json!({}))]);
            reply.events_mut().unwrap().insert(
                0,
                (
                    Duration::ZERO,
                    Ok(ModelEvent::ToolsLoaded(vec![ToolKey::new(
                        "convert_currency",
                    )])),
                ),
            );
            reply
        }
        _ => reply("resp", vec![message("done")]),
    })
    .with_info(ModelInfo::new().with_tool_search(true));
    let mut runner = Runner::new(model.clone(), tools);
    let outcome = runner
        .run(&mut Conversation::user("Convert"))
        .await
        .unwrap();
    assert!(outcome.is_completed());
    assert_eq!(
        runner.loaded_tools().cloned().collect::<Vec<_>>(),
        [ToolKey::new("convert_currency")]
    );
    let seen = model.requests();
    assert_eq!(seen[0].tools.len(), 3);
    assert!(seen[0].tools.contains(&ToolDefinition::ToolSearch));
    assert!(seen[0].tools.iter().any(|tool| matches!(
        tool,
        ToolDefinition::Function { name, defer_loading: true, .. } if name == "convert_currency"
    )));
    assert_eq!(
        seen[1].outputs()[0].body.as_text(),
        Some("\"convert_currency\"")
    );
}

#[test]
fn the_run_future_is_send() {
    fn assert_send<T: Send>(_: &T) {}
    let model = ScriptedModel::new(|_, _| reply("resp", vec![]));
    let mut runner = Runner::new(model, ToolRegistry::new());
    let mut conversation = Conversation::new();
    let cancel = CancellationSignal::new();
    let future = runner.run_with(&mut conversation, &cancel, |_| {});
    assert_send(&future);
}

#[tokio::test]
async fn an_incomplete_response_ends_the_run() {
    let model = ScriptedModel::new(|index, _| match index {
        0 => reply(
            "resp_0",
            vec![call("a", "get_weather", &json!({"city": "Ghent"}))],
        ),
        _ => {
            let mut reply = reply("resp_1", vec![message("Half an ans")]);
            let events = reply.events_mut().unwrap();
            let last = events.len() - 1;
            events[last].1 = Ok(ModelEvent::Completed(
                ModelResponse::new()
                    .with_id("resp_1")
                    .with_incomplete("max_output_tokens"),
            ));
            reply
        }
    });
    let mut runner = Runner::new(model.clone(), weather_tools());
    let mut conversation = Conversation::user("Write a long story");
    let outcome = runner.run(&mut conversation).await.unwrap();
    assert_eq!(
        outcome.stop,
        RunStop::Incomplete {
            reason: Some("max_output_tokens".into())
        }
    );
    assert_eq!(outcome.turns.len(), 2);
    assert_eq!(outcome.output_text(), "Half an ans");
    assert_eq!(
        conversation.last_assistant_text().as_deref(),
        Some("Half an ans")
    );
    assert_eq!(model.requests().len(), 2);
}

#[tokio::test]
async fn a_text_only_model_gets_rich_outputs_as_failures_and_no_images() {
    struct Chart;
    impl RichTool for Chart {
        type Input = Value;
        fn name(&self) -> &str {
            "chart"
        }
        fn description(&self) -> &str {
            "Draw a chart"
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn execute(&self, _: Value) -> Result<ToolOutputBody> {
            Ok(vec![
                Content::text("chart"),
                Content::image_url("https://x/c.png"),
            ]
            .into())
        }
    }
    let mut tools = ToolRegistry::new();
    tools.register_rich(Chart).unwrap();
    let model = ScriptedModel::new(|index, _| match index {
        0 => reply("resp", vec![call("a", "chart", &json!({}))]),
        _ => reply("resp", vec![message("ok")]),
    })
    .with_info(
        ModelInfo::new()
            .with_rich_tool_outputs(false)
            .with_images(false),
    );
    let mut runner = Runner::new(model.clone(), tools);
    let mut conversation = Conversation::new();
    conversation.push(Message::new(
        Role::User,
        vec![Content::text("look"), Content::image_url("https://x/a.png")],
    ));
    runner.run(&mut conversation).await.unwrap();
    let seen = model.requests();
    let Item::Message(question) = &seen[0].items[0] else {
        panic!("a message expected");
    };
    assert_eq!(
        question.content[1],
        Content::text(genai_agents::IMAGE_OMITTED)
    );
    let output = &seen[1].outputs()[0];
    assert!(!output.success);
    assert_eq!(
        output.body.as_text(),
        Some("Unsupported tool output: the model takes only text in tool outputs")
    );
}

#[test]
fn exposure_filters_the_registrys_definitions() {
    let log: Log = Arc::default();
    let mut tools = weather_tools();
    tools
        .register(Sleepy::new("convert_currency", 1, true, &log))
        .unwrap();
    tools
        .namespace("files", "File tools")
        .unwrap()
        .register(Sleepy::new("read_file", 1, true, &log))
        .unwrap();
    let key = ToolKey::new("convert_currency");
    assert_eq!(tools.exposure(&key), Some(Exposure::Direct));
    assert!(tools.set_exposure(&key, Exposure::Deferred));
    assert!(tools.set_exposure(&ToolKey::namespaced("files", "read_file"), Exposure::Hidden));
    assert!(!tools.set_exposure(&ToolKey::new("nope"), Exposure::Hidden));
    // The namespace has no direct tool left, so it is not advertised either.
    let definitions = tools.definitions();
    let names: Vec<&str> = definitions
        .iter()
        .filter_map(ToolDefinition::name)
        .collect();
    assert_eq!(names, ["get_weather"]);
    assert_eq!(
        tools.search_deferred("currency", 8),
        std::slice::from_ref(&key)
    );
    assert!(
        tools.search_deferred("weather", 8).is_empty(),
        "direct tools are not searched"
    );
    assert_eq!(
        tools.description(&key),
        Some("Sleep for a while, then say your name.")
    );
}

#[tokio::test]
async fn custom_prompt_cache_keys_and_none() {
    let model = ScriptedModel::new(|_, _| reply("resp", vec![message("ok")]));
    let mut runner = Runner::new(model.clone(), weather_tools())
        .with_config(RunConfig::new().with_prompt_cache_key(PromptCacheKey::Custom("k".into())));
    runner.run(&mut Conversation::user("hi")).await.unwrap();
    runner.config_mut().prompt_cache_key = PromptCacheKey::Omit;
    runner.run(&mut Conversation::user("hi")).await.unwrap();
    let seen = model.requests();
    assert_eq!(seen[0].prompt_cache_key.as_deref(), Some("k"));
    assert_eq!(seen[1].prompt_cache_key, None);
}

#[tokio::test]
async fn the_tools_change_between_runs() {
    let model = ScriptedModel::new(|_, _| reply("resp", vec![message("ok")]));
    let shared = Arc::new(weather_tools());
    let mut runner = Runner::new(model.clone(), Arc::clone(&shared));
    runner.run(&mut Conversation::user("hi")).await.unwrap();

    // The registry is shared with the application: the runner changes its own copy.
    assert!(runner.registry_mut().remove(&ToolKey::new("get_weather")));
    runner.run(&mut Conversation::user("hi")).await.unwrap();
    assert_eq!(shared.len(), 1);
    assert!(runner.registry().is_empty());

    let seen = model.requests();
    assert_eq!(seen[0].tool_names(), ["get_weather"]);
    assert!(seen[1].tool_names().is_empty());
}
