//! The runner on the OpenAI models against a scripted HTTP service: the requests both APIs get
//! (stateless defaults, tools, cache key, chat messages), streaming with tool calls started
//! before the response ends, whole-request retries (with `retry-after`, and without retrying what
//! the client retries), deferred tools and tool search, encrypted reasoning, and the rate limits
//! of every request.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(unknown_lints, clippy::unused_async_trait_impl)]
#![allow(
    clippy::significant_drop_tightening,
    clippy::type_complexity,
    clippy::unnecessary_literal_bound
)]

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::StreamExt;
use genai_agents::tools::{Exposure, FunctionTool, ToolKey, ToolRegistry};
use genai_agents::{
    CancellationSignal, ContextConfig, Conversation, Error, Item, ItemMeta, Message, Model,
    ModelEvent, ModelInfo, ModelSettings, Origin, OutputSchema, PromptCacheKey, Reasoning,
    ReasoningLevel, Result, RetryConfig, Role, RunConfig, RunEvent, RunStop, Runner, StopReason,
    ToolCall, ToolOutput, ToolOutputBody, Usage, CUT_OFF_OUTPUT,
};
use genai_agents_openai::{Compat, OpenAiChat, OpenAiResponses, PROVIDER, RESPONSES_API};
use openai_ergonomic::{ChatCompletionBuilder, Client, Config, ResponsesBuilder};
use serde_json::{json, Value};
use timvw_rust_runtime::{Body, BoxError};

// ---------------------------------------------------------------------------------------------
// A scripted HTTP service.

/// A reply: status, headers and body chunks, each sent after its delay.
struct Reply {
    status: u16,
    headers: Vec<(&'static str, String)>,
    chunks: Vec<(Duration, String)>,
}

impl Reply {
    fn json(body: &Value) -> Self {
        Self {
            status: 200,
            headers: vec![("content-type", "application/json".into())],
            chunks: vec![(Duration::ZERO, body.to_string())],
        }
    }

    fn sse(events: &[Value]) -> Self {
        Self::slow_sse(
            events
                .iter()
                .map(|event| (Duration::ZERO, event.clone()))
                .collect(),
        )
    }

    /// Server-sent events, each after its delay.
    fn slow_sse(events: Vec<(Duration, Value)>) -> Self {
        Self {
            status: 200,
            headers: vec![("content-type", "text/event-stream".into())],
            chunks: events
                .into_iter()
                .map(|(delay, event)| {
                    (
                        delay,
                        format!(
                            "event: {}\ndata: {event}\n\n",
                            event["type"].as_str().unwrap()
                        ),
                    )
                })
                .collect(),
        }
    }

    fn error(status: u16, code: &str) -> Self {
        Self {
            status,
            headers: vec![("content-type", "application/json".into())],
            chunks: vec![(
                Duration::ZERO,
                json!({"error": {"message": "nope", "type": code, "code": code}}).to_string(),
            )],
        }
    }

    fn header(mut self, name: &'static str, value: &str) -> Self {
        self.headers.push((name, value.to_string()));
        self
    }
}

type Seen = Arc<Mutex<Vec<Value>>>;

/// A client whose HTTP service answers request `n` with `script(n, body)` and records the JSON
/// bodies. The client does not retry: the runner does.
fn scripted(script: impl Fn(usize, &Value) -> Reply + Send + Sync + 'static) -> (Client, Seen) {
    let (config, seen) = scripted_config(script);
    (Client::builder(config).unwrap().build(), seen)
}

/// The configuration of [`scripted`]'s client, to build a client of one's own (with an
/// interceptor, say).
fn scripted_config(
    script: impl Fn(usize, &Value) -> Reply + Send + Sync + 'static,
) -> (Config, Seen) {
    let seen: Seen = Arc::default();
    let log = seen.clone();
    let script = Arc::new(script);
    let service = tower::service_fn(move |request: http::Request<Body>| {
        let log = log.clone();
        let script = script.clone();
        async move {
            let body = request.into_body().collect_bytes().await?;
            let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
            let index = {
                let mut log = log.lock().unwrap();
                log.push(body.clone());
                log.len() - 1
            };
            let reply = script(index, &body);
            let chunks = futures::stream::iter(reply.chunks).then(|(delay, chunk)| async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                Ok::<_, Infallible>(Bytes::from(chunk))
            });
            let mut response = http::Response::builder().status(reply.status);
            for (name, value) in reply.headers {
                response = response.header(name, value);
            }
            Ok::<_, BoxError>(response.body(Body::wrap_stream(chunks))?)
        }
    });
    let config = Config::builder()
        .api_key("test-key")
        .api_base("https://api.example.test/v1")
        .default_model("gpt-5.4-mini")
        .max_retries(0)
        .http_client(timvw_rust_runtime::Client::from_service(service))
        .build();
    (config, seen)
}

// ---------------------------------------------------------------------------------------------
// Responses API payloads.

fn response(id: &str, output: &[Value]) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": 1_741_476_542,
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "instructions": null,
        "max_output_tokens": null,
        "model": "gpt-5.4-mini",
        "output": output,
        "parallel_tool_calls": true,
        "previous_response_id": null,
        "reasoning": {"effort": null, "summary": null},
        "store": false,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15
        },
        "user": null,
        "metadata": {}
    })
}

fn call(call_id: &str, name: &str, arguments: &Value) -> Value {
    json!({
        "type": "function_call",
        "id": format!("fc_{call_id}"),
        "call_id": call_id,
        "name": name,
        "arguments": arguments.to_string(),
        "status": "completed"
    })
}

fn message(text: &str) -> Value {
    json!({
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}]
    })
}

/// The events of a streamed response: `response.created`, one `response.output_item.done` per
/// item (with a text delta before a message) and `response.completed`.
fn events(id: &str, output: &[Value]) -> Vec<Value> {
    let mut created = response(id, &[]);
    created["status"] = json!("in_progress");
    let mut events =
        vec![json!({"type": "response.created", "sequence_number": 0, "response": created})];
    for (index, item) in output.iter().enumerate() {
        if item["type"] == "message" {
            events.push(json!({
                "type": "response.output_text.delta", "sequence_number": 1, "item_id": "msg_1",
                "output_index": index, "content_index": 0,
                "delta": item["content"][0]["text"]
            }));
        }
        events.push(json!({
            "type": "response.output_item.done", "sequence_number": 2, "output_index": index,
            "item": item
        }));
    }
    events.push(json!({"type": "response.completed", "sequence_number": 3, "response": response(id, output)}));
    events
}

fn chat_completion(content: Option<&str>, calls: &[(&str, &str, Value)]) -> Value {
    let tool_calls: Vec<Value> = calls
        .iter()
        .map(|(id, name, arguments)| {
            json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments.to_string()}})
        })
        .collect();
    let mut message = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-5.4-mini",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": if calls.is_empty() { "stop" } else { "tool_calls" },
            "logprobs": null
        }],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
    })
}

/// The `input` items of a recorded Responses request of this type.
fn input_of(body: &Value, kind: &str) -> Vec<Value> {
    body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == kind)
        .cloned()
        .collect()
}

fn tool_names(body: &Value) -> Vec<String> {
    body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| {
                    tool["name"]
                        .as_str()
                        .or_else(|| tool["function"]["name"].as_str())
                        .unwrap_or_else(|| tool["type"].as_str().unwrap())
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// Tools.

type Log = Arc<Mutex<Vec<(String, &'static str, Instant)>>>;

/// Sleeps, then returns its name; logs its start and end.
struct Sleepy {
    name: &'static str,
    millis: u64,
    exposure: Exposure,
    log: Log,
}

impl Sleepy {
    fn new(name: &'static str, millis: u64, log: &Log) -> Self {
        Self {
            name,
            millis,
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
        true
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

fn template() -> ResponsesBuilder {
    ResponsesBuilder::new("gpt-5.4-mini").instructions("Be brief.")
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
async fn the_responses_api_runs_tools_until_the_model_answers() {
    for stream in [true, false] {
        let (client, seen) = scripted(move |index, _| {
            let output = match index {
                0 => vec![call("call_1", "get_weather", &json!({"city": "Ghent"}))],
                _ => vec![message("12 degrees in Ghent.")],
            };
            let reply = if stream {
                Reply::sse(&events(&format!("resp_{index}"), &output))
            } else {
                Reply::json(&response(&format!("resp_{index}"), &output))
            };
            reply
                .header("x-ratelimit-remaining-requests", &(99 - index).to_string())
                .header("x-ratelimit-limit-requests", "100")
        });
        let model = OpenAiResponses::new(client.clone(), template()).with_stream(stream);
        let mut runner = Runner::new(model, weather_tools());
        let mut conversation = Conversation::user("Weather in Ghent?");
        let outcome = runner.run(&mut conversation).await.unwrap();

        assert_eq!(outcome.stop, RunStop::Completed);
        assert_eq!(outcome.output_text(), "12 degrees in Ghent.");
        assert_eq!(outcome.turns.len(), 2);
        assert_eq!(outcome.turns[0].calls[0].name, "get_weather");
        assert_eq!(outcome.turns[0].response_id.as_deref(), Some("resp_0"));
        assert_eq!(outcome.turns[1].input_tokens, Some(10));
        assert_eq!(outcome.turns[0].stop, Some(StopReason::ToolUse));
        assert_eq!(outcome.turns[1].stop, Some(StopReason::Stop));
        assert_eq!(
            outcome.turns[1].usage,
            Some(
                Usage::new(10, 5)
                    .with_cache_read_tokens(0)
                    .with_reasoning_tokens(0)
            )
        );
        // The rate limits of each request, and the client's latest.
        let first = outcome.turns[0].rate_limits.unwrap().requests.unwrap();
        assert_eq!(first.remaining, Some(99));
        let second = outcome.rate_limits().unwrap().requests.unwrap();
        assert_eq!(second.remaining, Some(98));
        assert_eq!(
            client.rate_limits().unwrap().requests.unwrap().remaining,
            Some(98)
        );

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        // Stateless defaults, a stable cache key, the template's options.
        for body in seen.iter() {
            assert_eq!(body["store"], false);
            assert_eq!(body["model"], "gpt-5.4-mini");
            assert_eq!(body["instructions"], "Be brief.");
            assert_eq!(body["parallel_tool_calls"], true);
            assert_eq!(body["prompt_cache_key"], runner.prompt_cache_key().unwrap());
            assert!(
                body.get("include").is_none(),
                "no reasoning: nothing to include"
            );
            assert!(body.get("previous_response_id").is_none());
            assert_eq!(tool_names(body), ["get_weather"]);
            // The placeholder that lets the template build is not sent.
            assert_eq!(
                body["input"][0],
                json!({"type": "message", "role": "user", "content": "Weather in Ghent?"})
            );
        }
        assert_eq!(seen[0]["stream"] == json!(true), stream);
        // The second request replays the call (without its id) and its output.
        let calls = input_of(&seen[1], "function_call");
        assert_eq!(calls.len(), 1);
        assert!(calls[0].get("id").is_none());
        let outputs = input_of(&seen[1], "function_call_output");
        assert_eq!(outputs[0]["call_id"], "call_1");
        assert_eq!(outputs[0]["output"], r#"{"city":"Ghent","temperature":12}"#);
        assert_eq!(seen[1]["input"].as_array().unwrap().len(), 3);
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
}

#[tokio::test]
async fn chat_completions_run_tools_until_the_model_answers() {
    let (client, seen) = scripted(|index, _| {
        let body = match index {
            0 => chat_completion(
                Some("Checking."),
                &[("call_1", "get_weather", json!({"city": "Ghent"}))],
            ),
            _ => chat_completion(Some("12 degrees."), &[]),
        };
        Reply::json(&body).header("x-ratelimit-remaining-tokens", "5000")
    });
    let template = ChatCompletionBuilder::new("gpt-5.4-mini").system("Be brief.");
    let mut runner = Runner::new(OpenAiChat::new(client, template), weather_tools())
        .with_config(RunConfig::new().with_prompt_cache_key(PromptCacheKey::Custom("k".into())));
    let mut conversation = Conversation::user("Weather in Ghent?");
    let outcome = runner.run(&mut conversation).await.unwrap();

    assert_eq!(outcome.stop, RunStop::Completed);
    assert_eq!(outcome.output_text(), "12 degrees.");
    assert_eq!(outcome.turns[0].text, "Checking.");
    assert_eq!(
        outcome.turns[1]
            .rate_limits
            .unwrap()
            .tokens
            .unwrap()
            .remaining,
        Some(5000)
    );
    assert_eq!(outcome.turns[1].input_tokens, Some(9));
    assert_eq!(outcome.turns[1].response_id.as_deref(), Some("chatcmpl-1"));

    let seen = seen.lock().unwrap();
    let messages = seen[1]["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 4);
    assert_eq!(
        messages[0],
        json!({"role": "system", "content": "Be brief."})
    );
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["content"], "Checking.");
    assert_eq!(messages[2]["tool_calls"][0]["id"], "call_1");
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "call_1");
    assert_eq!(seen[1]["prompt_cache_key"], "k");
    assert_eq!(seen[1]["parallel_tool_calls"], true);
    assert_eq!(seen[1]["tools"][0]["function"]["name"], "get_weather");
    assert!(seen[1].get("stream").is_none());
}

#[tokio::test]
async fn streamed_calls_start_before_the_response_ends() {
    let log: Log = Arc::default();
    let mut tools = ToolRegistry::new();
    tools.register(Sleepy::new("probe", 10, &log)).unwrap();
    let (client, _) = scripted(|index, _| {
        if index > 0 {
            return Reply::sse(&events("resp_1", &[message("ok")]));
        }
        let output = [call("a", "probe", &json!({})), message("still writing")];
        let mut timed: Vec<(Duration, Value)> = events("resp_0", &output)
            .into_iter()
            .map(|event| (Duration::ZERO, event))
            .collect();
        // The response goes on for a while after the call is done.
        let last = timed.len() - 1;
        timed[last].0 = Duration::from_millis(300);
        Reply::slow_sse(timed)
    });
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), tools);
    let mut completed: Option<Instant> = None;
    let mut created = 0;
    let outcome = runner
        .run_with(
            &mut Conversation::user("go"),
            &CancellationSignal::new(),
            |event| match event {
                RunEvent::Model(ModelEvent::Completed(_)) => {
                    completed.get_or_insert_with(Instant::now);
                }
                RunEvent::Model(ModelEvent::Other { kind, .. }) if kind == "response.created" => {
                    created += 1;
                }
                _ => {}
            },
        )
        .await
        .unwrap();
    assert!(outcome.is_completed());
    assert_eq!(created, 2, "the provider's other events are passed on");
    // The tool finished well before the response did.
    assert!(when(&log, "probe", "end") + Duration::from_millis(150) < completed.unwrap());
}

/// With the default HTTP client, which retries a rate limit itself, the runner does not retry it
/// again: `max_retries(1)` gives two requests in all, not two for each of the runner's attempts.
#[tokio::test]
async fn failures_before_the_body_are_left_to_the_clients_retries() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/responses")
        .with_status(429)
        .with_header("content-type", "application/json")
        .with_header("retry-after-ms", "1")
        .with_body(
            json!({"error": {"message": "slow down", "type": "requests", "code": "rate_limit_exceeded"}})
                .to_string(),
        )
        .expect(2)
        .create_async()
        .await;
    let config = Config::builder()
        .api_key("test-key")
        .api_base(server.url())
        .default_model("gpt-5.4-mini")
        .max_retries(1)
        .build();
    let client = Client::builder(config).unwrap().build();
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), weather_tools())
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
    let error = error.as_model().unwrap();
    assert!(error.is_rate_limit(), "{error}");
    assert!(error.is_before_body());
    assert_eq!(error.status(), Some(429));
    assert_eq!(error.code(), Some("rate_limit_exceeded"));
    assert!(error.provider_error::<openai_ergonomic::Error>().is_some());
    assert_eq!(retries, 0);
    mock.assert_async().await;
    drop(server);
}

#[tokio::test]
async fn a_rate_limited_request_is_retried_after_retry_after() {
    let (client, seen) = scripted(|index, _| match index {
        0 => Reply::error(429, "rate_limit_exceeded").header("retry-after-ms", "120"),
        _ => Reply::sse(&events("resp", &[message("ok")]))
            .header("x-ratelimit-remaining-requests", "7"),
    });
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), weather_tools());
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
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn a_cut_off_stream_is_retried_with_backoff_without_running_calls_twice() {
    let log: Log = Arc::default();
    let mut tools = ToolRegistry::new();
    tools.register(Sleepy::new("once", 5, &log)).unwrap();
    let (client, seen) = scripted(|index, _| match index {
        // The call is done, then the stream stops without a terminal event.
        0 => {
            let mut cut = events("resp_0", &[call("a", "once", &json!({}))]);
            cut.pop();
            Reply::sse(&cut)
        }
        _ => Reply::sse(&events("resp_1", &[message("done")])),
    });
    let retry = RetryConfig::new(2).with_base_delay(Duration::from_millis(10));
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), tools)
        .with_config(RunConfig::new().with_retry(retry));
    let mut delays = Vec::new();
    let outcome = runner
        .run_with(
            &mut Conversation::user("hi"),
            &CancellationSignal::new(),
            |event| {
                if let RunEvent::Retrying { delay, error, .. } = event {
                    assert!(!error.is_before_body());
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
    let seen = seen.lock().unwrap();
    assert_eq!(
        input_of(&seen[1], "function_call_output")[0]["output"],
        "\"once\""
    );
}

#[tokio::test]
async fn retries_stop_at_errors_that_cannot_be_retried_and_at_the_limit() {
    let (client, seen) = scripted(|_, _| Reply::error(400, "invalid_request_error"));
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), weather_tools());
    let error = runner.run(&mut Conversation::user("hi")).await.unwrap_err();
    assert_eq!(error.as_model().unwrap().status(), Some(400));
    assert_eq!(seen.lock().unwrap().len(), 1);

    let (client, seen) = scripted(|_, _| Reply::error(500, "server_error"));
    let retry = RetryConfig::new(2).with_base_delay(Duration::from_millis(1));
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), weather_tools())
        .with_config(RunConfig::new().with_retry(retry));
    let error = runner.run(&mut Conversation::user("hi")).await.unwrap_err();
    assert!(error.as_model().unwrap().is_server_error());
    assert_eq!(seen.lock().unwrap().len(), 3);

    // A failed response (response.failed) is retried by its code.
    let (client, seen) = scripted(|index, _| {
        if index == 0 {
            let mut failed = response("resp_0", &[]);
            failed["status"] = json!("failed");
            failed["error"] = json!({"code": "server_error", "message": "boom"});
            Reply::sse(&[
                json!({"type": "response.failed", "sequence_number": 0, "response": failed}),
            ])
        } else {
            Reply::sse(&events("resp_1", &[message("ok")]))
        }
    });
    let retry = RetryConfig::new(1).with_base_delay(Duration::from_millis(1));
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), weather_tools())
        .with_config(RunConfig::new().with_retry(retry));
    assert!(runner
        .run(&mut Conversation::user("hi"))
        .await
        .unwrap()
        .is_completed());
    assert_eq!(seen.lock().unwrap().len(), 2);

    // So is an `error` event, unless its code says it will not help: a context that is too long
    // is not retried, but compacted (once), and the run fails when that is too long as well.
    let too_long = |_: usize, _: &_| {
        Reply::sse(&[json!({"type": "error", "sequence_number": 0,
                             "code": "context_length_exceeded", "message": "too long"})])
    };
    let (client, seen) = scripted(too_long);
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), weather_tools());
    let error = runner.run(&mut Conversation::user("hi")).await.unwrap_err();
    assert_eq!(
        error.as_model().unwrap().code(),
        Some("context_length_exceeded")
    );
    assert_eq!(seen.lock().unwrap().len(), 2); // the request, then the compaction's
    let (client, seen) = scripted(too_long);
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), weather_tools())
        .with_config(
            RunConfig::new().with_context(ContextConfig::new().with_compact_on_overflow(false)),
        );
    runner.run(&mut Conversation::user("hi")).await.unwrap_err();
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn deferred_tools_are_loaded_by_the_runners_tool_search() {
    let log: Log = Arc::default();
    let mut tools = weather_tools();
    let mut hidden = Sleepy::new("internal_audit", 1, &log);
    hidden.exposure = Exposure::Hidden;
    tools.register(hidden).unwrap();
    let mut deferred = Sleepy::new("convert_currency", 1, &log);
    deferred.exposure = Exposure::Deferred;
    tools.register(deferred).unwrap();
    let mut other = Sleepy::new("translate_text", 1, &log);
    other.exposure = Exposure::Deferred;
    tools.register(other).unwrap();
    let registry = Arc::new(tools);

    for chat in [false, true] {
        let (client, seen) = scripted(move |index, _| {
            let calls: Vec<(&str, &str, Value)> = match index {
                0 => vec![("s", "tool_search", json!({"query": "currency conversion"}))],
                1 => vec![("c", "convert_currency", json!({}))],
                2 => vec![("h", "internal_audit", json!({}))],
                _ => vec![],
            };
            if chat {
                return Reply::json(&chat_completion(calls.is_empty().then_some("done"), &calls));
            }
            let mut output: Vec<Value> = calls
                .iter()
                .map(|(id, name, args)| call(id, name, args))
                .collect();
            if output.is_empty() {
                output.push(message("done"));
            }
            Reply::sse(&events("resp", &output))
        });
        let outcome;
        let loaded: Vec<ToolKey>;
        if chat {
            let model = OpenAiChat::new(client, ChatCompletionBuilder::new("gpt-5.4-mini"));
            let mut runner = Runner::new(model, registry.clone());
            outcome = runner
                .run(&mut Conversation::user("Convert 10 EUR"))
                .await
                .unwrap();
            loaded = runner.loaded_tools().cloned().collect();
        } else {
            let mut runner =
                Runner::new(OpenAiResponses::new(client, template()), registry.clone());
            outcome = runner
                .run(&mut Conversation::user("Convert 10 EUR"))
                .await
                .unwrap();
            loaded = runner.loaded_tools().cloned().collect();
        }
        assert!(outcome.is_completed());
        assert_eq!(loaded, [ToolKey::new("convert_currency")]);
        let seen = seen.lock().unwrap();
        assert_eq!(tool_names(&seen[0]), ["get_weather", "tool_search"]);
        assert_eq!(
            tool_names(&seen[1]),
            ["convert_currency", "get_weather", "tool_search"]
        );
        // The search's answer lists what it loaded.
        let search_output = if chat {
            seen[1]["messages"].as_array().unwrap().last().unwrap()["content"].clone()
        } else {
            input_of(&seen[1], "function_call_output")[0]["output"].clone()
        };
        let search_output: Value = serde_json::from_str(search_output.as_str().unwrap()).unwrap();
        assert_eq!(search_output["tools"][0]["name"], "convert_currency");
        // The hidden tool is never advertised, but its call is answered.
        assert!(!tool_names(&seen[3]).contains(&"internal_audit".to_string()));
    }
}

#[tokio::test]
async fn a_model_that_searches_tools_gets_them_deferred() {
    let log: Log = Arc::default();
    let mut tools = weather_tools();
    let mut deferred = Sleepy::new("convert_currency", 1, &log);
    deferred.exposure = Exposure::Deferred;
    tools.register(deferred).unwrap();
    let (client, seen) = scripted(|index, _| {
        let output = match index {
            0 => vec![
                json!({"type": "tool_search_call", "id": "ts_1", "call_id": null, "execution": "server",
                       "arguments": {"query": "currency"}, "status": "completed"}),
                json!({"type": "tool_search_output", "id": "tso_1", "call_id": null, "execution": "server",
                       "tools": [{"type": "function", "name": "convert_currency", "description": "x",
                                  "parameters": {"type": "object", "properties": {}}, "strict": null,
                                  "defer_loading": true}],
                       "status": "completed"}),
                call("c", "convert_currency", &json!({})),
            ],
            _ => vec![message("done")],
        };
        Reply::sse(&events("resp", &output))
    });
    let model =
        OpenAiResponses::new(client, template()).with_info(ModelInfo::new().with_tool_search(true));
    let mut runner = Runner::new(model, tools);
    let outcome = runner
        .run(&mut Conversation::user("Convert"))
        .await
        .unwrap();
    assert!(outcome.is_completed());
    assert_eq!(
        runner.loaded_tools().cloned().collect::<Vec<_>>(),
        [ToolKey::new("convert_currency")]
    );
    let seen = seen.lock().unwrap();
    let tools = seen[0]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 3);
    let deferred = tools
        .iter()
        .find(|tool| tool["name"] == "convert_currency")
        .unwrap();
    assert_eq!(deferred["defer_loading"], true);
    assert!(tools.iter().any(|tool| tool["type"] == "tool_search"));
    // The search items are replayed (without their ids), and the deferred call was answered.
    let replayed = input_of(&seen[1], "tool_search_output");
    assert_eq!(replayed.len(), 1);
    assert!(replayed[0].get("id").is_none());
    assert_eq!(
        input_of(&seen[1], "function_call_output")[0]["output"],
        "\"convert_currency\""
    );
}

#[tokio::test]
async fn reasoning_is_replayed_encrypted_when_nothing_is_stored() {
    let (client, seen) = scripted(|index, _| {
        let output = match index {
            0 => vec![
                json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAAA"}),
                call("a", "get_weather", &json!({"city": "Ghent"})),
            ],
            _ => vec![message("done")],
        };
        Reply::json(&response("resp", &output))
    });
    let model =
        OpenAiResponses::new(client.clone(), template().reasoning_effort("low")).with_stream(false);
    let mut runner = Runner::new(model, weather_tools());
    let mut conversation = Conversation::user("hi");
    runner.run(&mut conversation).await.unwrap();
    assert!(matches!(conversation.items()[1], Item::Reasoning(_)));
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0]["include"], json!(["reasoning.encrypted_content"]));
        let reasoning = input_of(&seen[1], "reasoning");
        assert_eq!(reasoning[0]["encrypted_content"], "gAAAA");
        assert!(reasoning[0].get("id").is_none());
    }

    // Stored: no include (even for a reasoning model), the items keep their ids.
    let model = OpenAiResponses::new(client, template())
        .with_stream(false)
        .with_store(true)
        .with_info(ModelInfo::new().with_reasoning(true));
    let mut runner = Runner::new(model, weather_tools());
    let mut conversation = Conversation::user("hi");
    conversation.push(Message::text(Role::Assistant, "earlier").with_id("msg_1"));
    runner.run(&mut conversation).await.unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[2]["store"], true);
    assert!(seen[2].get("include").is_none());
    assert_eq!(seen[2]["input"][1]["id"], "msg_1");
}

#[tokio::test]
async fn an_incomplete_response_ends_the_run() {
    let incomplete = |id: &str, reason: &str, output: &[Value]| {
        let mut body = response(id, output);
        body["status"] = json!("incomplete");
        body["incomplete_details"] = json!({"reason": reason});
        Reply::sse(&[
            json!({"type": "response.incomplete", "sequence_number": 0, "response": body}),
        ])
    };
    let (client, seen) = scripted(move |index, _| match index {
        0 => Reply::sse(&events(
            "resp_0",
            &[call("a", "get_weather", &json!({"city": "Ghent"}))],
        )),
        _ => incomplete("resp_1", "max_output_tokens", &[message("Half an ans")]),
    });
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), weather_tools());
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
    assert_eq!(seen.lock().unwrap().len(), 2);

    // On Chat Completions, `length`.
    let (client, _) = scripted(|_, _| {
        let mut completion = chat_completion(Some("Half"), &[]);
        completion["choices"][0]["finish_reason"] = json!("length");
        Reply::json(&completion)
    });
    let model = OpenAiChat::new(client, ChatCompletionBuilder::new("gpt-5.4-mini"));
    let outcome = Runner::new(model, ToolRegistry::new())
        .run(&mut Conversation::user("Write"))
        .await
        .unwrap();
    assert_eq!(
        outcome.stop,
        RunStop::Incomplete {
            reason: Some("length".into())
        }
    );
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
    let (client, _) = scripted(|_, _| {
        Reply::json(&response(
            "resp",
            &[
                call("a", "get_weather", &json!({"city": "Ghent"})),
                call("b", "explode", &json!({})),
            ],
        ))
    });
    let model = OpenAiResponses::new(client, template()).with_stream(false);
    let mut runner =
        Runner::new(model, tools).with_config(RunConfig::new().with_parallel_tool_calls(false));
    let mut conversation = Conversation::user("hi");
    let error = runner.run(&mut conversation).await.unwrap_err();
    assert!(matches!(error, Error::Tool(_)), "{error:?}");
    assert_eq!(conversation.unanswered_calls().len(), 1);
    let report = conversation.normalize(&ModelInfo::default());
    assert_eq!(report.aborted, ["b"]);
}

#[tokio::test]
async fn the_history_is_repaired_before_the_request() {
    let (client, seen) = scripted(|_, _| Reply::sse(&events("resp", &[message("ok")])));
    let mut conversation = Conversation::user("hi");
    conversation.push_call(&ToolCall::function("lost", "get_weather", "{}"));
    conversation.push_output(&ToolOutput::new(
        &ToolCall::function("ghost", "get_weather", "{}"),
        "orphan",
        true,
    ));
    let mut runner = Runner::new(OpenAiResponses::new(client, template()), weather_tools());
    runner.run(&mut conversation).await.unwrap();
    let seen = seen.lock().unwrap();
    let outputs = input_of(&seen[0], "function_call_output");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0]["call_id"], "lost");
    assert_eq!(outputs[0]["output"], "aborted");
}

#[test]
fn the_run_future_is_send() {
    fn assert_send<T: Send>(_: &T) {}
    let (client, _) = {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let _guard = runtime.enter();
        scripted(|_, _| Reply::json(&json!({})))
    };
    let mut runner = Runner::new(
        OpenAiResponses::new(client.clone(), template()),
        ToolRegistry::new(),
    );
    let mut conversation = Conversation::new();
    let cancel = CancellationSignal::new();
    let future = runner.run_with(&mut conversation, &cancel, |_| {});
    assert_send(&future);
    drop(future);
    let mut runner = Runner::new(
        OpenAiChat::new(client, ChatCompletionBuilder::new("gpt-5.4-mini")),
        ToolRegistry::new(),
    );
    let future = runner.run_with(&mut conversation, &cancel, |_| {});
    assert_send(&future);
}

// ---------------------------------------------------------------------------------------------
// The client's interceptors, request settings, item origins, `Compat`, cut-off completions and
// details for the application.

/// Records the operations the interceptor hooks saw.
#[derive(Clone, Default)]
struct Hooks(Arc<Mutex<Vec<String>>>);

impl openai_ergonomic::Interceptor for Hooks {
    async fn before_request(
        &self,
        ctx: &mut openai_ergonomic::BeforeRequestContext<'_>,
    ) -> openai_ergonomic::Result<()> {
        self.0
            .lock()
            .unwrap()
            .push(format!("before {} {}", ctx.operation, ctx.model));
        Ok(())
    }

    async fn after_response(
        &self,
        ctx: &openai_ergonomic::AfterResponseContext<'_>,
    ) -> openai_ergonomic::Result<()> {
        self.0
            .lock()
            .unwrap()
            .push(format!("after {} {:?}", ctx.operation, ctx.output_tokens));
        Ok(())
    }

    async fn on_stream_terminal(
        &self,
        ctx: &openai_ergonomic::StreamTerminalContext<'_>,
    ) -> openai_ergonomic::Result<()> {
        self.0
            .lock()
            .unwrap()
            .push(format!("stream end {:?}", ctx.summary.output_tokens));
        Ok(())
    }
}

fn client_with(config: Config, hooks: &Hooks) -> Client {
    Client::builder(config)
        .unwrap()
        .with_interceptor(Box::new(hooks.clone()))
        .build()
}

#[tokio::test]
async fn requests_go_through_the_clients_interceptors_with_the_settings() {
    let (config, seen) = scripted_config(|index, _| {
        if index == 0 {
            Reply::sse(&events(
                "resp_1",
                &[call("c1", "get_weather", &json!({"city": "Ghent"}))],
            ))
        } else {
            Reply::sse(&events("resp_2", &[message("Rain.")]))
        }
    });
    let hooks = Hooks::default();
    let model = OpenAiResponses::new(
        client_with(config, &hooks),
        ResponsesBuilder::new("gpt-5.4-mini"),
    );
    assert_eq!(
        model.info().origin,
        Some(Origin::new(RESPONSES_API, PROVIDER, "gpt-5.4-mini"))
    );
    let mut runner = Runner::new(model, weather_tools()).with_config(
        RunConfig::new().with_model_settings(
            ModelSettings::new()
                .with_reasoning(ReasoningLevel::Off)
                .with_max_output_tokens(300),
        ),
    );
    let mut conversation = Conversation::user("Weather in Ghent?");
    runner.run(&mut conversation).await.unwrap();

    let bodies = seen.lock().unwrap().clone();
    assert_eq!(bodies[0]["reasoning"]["effort"], "none");
    assert_eq!(bodies[0]["max_output_tokens"], 300);
    // The interceptors saw both requests and the end of both streams.
    for _ in 0..50 {
        if hooks.0.lock().unwrap().len() >= 4 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // The stream hooks run on a task of their own: compare without the order.
    let mut seen_hooks = hooks.0.lock().unwrap().clone();
    seen_hooks.sort();
    assert_eq!(
        seen_hooks,
        [
            "before responses gpt-5.4-mini",
            "before responses gpt-5.4-mini",
            "stream end Some(5)",
            "stream end Some(5)"
        ]
    );
    // The model's items record it as their origin.
    let origin = conversation.calls()[0].meta.origin.clone().unwrap();
    assert_eq!(origin.model, "gpt-5.4-mini");
    assert_eq!(origin.api, RESPONSES_API);
}

#[tokio::test]
async fn reasoning_of_another_model_is_not_sent() {
    let (client, seen) = scripted(|_, _| Reply::sse(&events("resp_1", &[message("Ok.")])));
    let model = OpenAiResponses::new(client, ResponsesBuilder::new("gpt-5.4-mini"))
        .with_info(ModelInfo::new().with_reasoning(true));
    let mut runner = Runner::new(model, ToolRegistry::new());
    let mut conversation = Conversation::user("Think.");
    conversation.push(
        Reasoning::new(PROVIDER)
            .with_encrypted_content("theirs")
            .with_meta(ItemMeta::new().with_origin(Origin::new(
                RESPONSES_API,
                PROVIDER,
                "gpt-5.4",
            ))),
    );
    conversation.push(
        Reasoning::new(PROVIDER)
            .with_encrypted_content("ours")
            .with_meta(ItemMeta::new().with_origin(Origin::new(
                RESPONSES_API,
                PROVIDER,
                "gpt-5.4-mini",
            ))),
    );
    runner.run(&mut conversation).await.unwrap();
    let body = seen.lock().unwrap()[0].clone();
    let reasoning = input_of(&body, "reasoning");
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0]["encrypted_content"], "ours");
}

/// Counts its runs; reports the city in its details.
struct Counted(Arc<Mutex<usize>>);

impl FunctionTool for Counted {
    type Input = Value;
    type Output = Value;
    fn name(&self) -> &str {
        "get_weather"
    }
    fn description(&self) -> &str {
        "Get the current weather of a city."
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"city": {"type": "string"}}})
    }
    async fn execute(&self, input: Value) -> Result<Value> {
        *self.0.lock().unwrap() += 1;
        Ok(json!({"city": input["city"], "temperature": 12}))
    }
    fn details(&self, _output: &Value) -> Option<Value> {
        Some(json!({"secret": "for the application only"}))
    }
}

#[tokio::test]
async fn chat_requests_follow_the_servers_compat() {
    // Two runs: a call, then the answer, each.
    let (client, seen) = scripted(|index, _| {
        if index % 2 == 0 {
            Reply::json(&chat_completion(
                None,
                &[("c1", "get_weather", json!({"city": "Ghent"}))],
            ))
        } else {
            Reply::json(&chat_completion(Some("Rain."), &[]))
        }
    });
    let runs = Arc::new(Mutex::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(Counted(Arc::clone(&runs))).unwrap();
    let template = ChatCompletionBuilder::new("local-model");
    // The client's provider is OpenAI's (with another base URL): its preset.
    assert_eq!(
        *OpenAiChat::new(client.clone(), template.clone()).compat(),
        Compat::openai()
    );
    // A server that takes only what every compatible server takes.
    let model = OpenAiChat::new(client.clone(), template.clone()).with_compat(Compat::cautious());
    let settings = ModelSettings::new()
        .with_reasoning(ReasoningLevel::Low)
        .with_max_output_tokens(200);
    let mut runner =
        Runner::new(model, tools).with_config(RunConfig::new().with_model_settings(settings));
    let mut conversation = Conversation::new();
    conversation.push(Message::text(Role::Developer, "Use metric units."));
    conversation.push_user("Weather in Ghent?");
    runner.run(&mut conversation).await.unwrap();

    let bodies = seen.lock().unwrap().clone();
    assert_eq!(bodies[0]["messages"][0]["role"], "system");
    assert_eq!(bodies[0]["max_tokens"], 200);
    assert!(bodies[0].get("max_completion_tokens").is_none());
    assert!(bodies[0].get("reasoning_effort").is_none());
    // The tool's details stay with the application.
    let tool_message = &bodies[1]["messages"][3];
    assert_eq!(tool_message["role"], "tool");
    assert!(tool_message.get("name").is_none());
    assert!(!tool_message
        .to_string()
        .contains("for the application only"));
    let answer = conversation
        .items()
        .iter()
        .find_map(|item| match item {
            Item::ToolOutput(output) => Some(output.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        answer.details,
        Some(json!({"secret": "for the application only"}))
    );

    // OpenAI's own: developer messages, reasoning_effort, max_completion_tokens; and a server
    // that wants the tool's name on tool messages.
    let mut tools = ToolRegistry::new();
    tools.register(Counted(Arc::clone(&runs))).unwrap();
    let model =
        OpenAiChat::new(client, template).with_compat(Compat::openai().with_tool_result_name(true));
    let mut runner =
        Runner::new(model, tools).with_config(RunConfig::new().with_model_settings(settings));
    let mut conversation = Conversation::new();
    conversation.push(Message::text(Role::Developer, "Use metric units."));
    conversation.push_user("Weather in Ghent?");
    runner.run(&mut conversation).await.unwrap();
    let bodies = seen.lock().unwrap().clone();
    assert_eq!(bodies[2]["messages"][0]["role"], "developer");
    assert_eq!(bodies[2]["reasoning_effort"], "low");
    assert_eq!(bodies[2]["max_completion_tokens"], 200);
    assert_eq!(bodies[3]["messages"][3]["name"], "get_weather");
}

#[tokio::test]
async fn calls_of_a_completion_cut_off_at_the_output_limit_are_not_run() {
    let (client, _) = scripted(|_, _| {
        let mut completion = chat_completion(None, &[("c1", "get_weather", json!({"city": "Gh"}))]);
        completion["choices"][0]["finish_reason"] = json!("length");
        Reply::json(&completion)
    });
    let runs = Arc::new(Mutex::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(Counted(Arc::clone(&runs))).unwrap();
    let model = OpenAiChat::new(client, ChatCompletionBuilder::new("gpt-5.4-mini"));
    let mut runner = Runner::new(model, tools);
    let mut conversation = Conversation::user("Weather in Ghent?");
    let outcome = runner.run(&mut conversation).await.unwrap();
    assert!(matches!(outcome.stop, RunStop::Incomplete { .. }));
    assert_eq!(*runs.lock().unwrap(), 0);
    let answer = conversation
        .items()
        .iter()
        .find_map(|item| match item {
            Item::ToolOutput(output) => Some(output.clone()),
            _ => None,
        })
        .unwrap();
    assert!(!answer.success);
    assert_eq!(answer.body, ToolOutputBody::text(CUT_OFF_OUTPUT));
}

#[tokio::test]
async fn instructions_and_structured_output_reach_both_apis() {
    let schema = OutputSchema::new(
        "answer",
        json!({"type": "object", "properties": {"umbrella": {"type": "boolean"}}}),
    )
    .with_strict(true);
    let config = RunConfig::new()
        .with_instructions("You are a household assistant.")
        .with_output(schema.clone());

    // Responses: `instructions` replaces the template's, `text.format` carries the schema.
    let (client, seen) = scripted(|_, _| Reply::sse(&events("resp_1", &[message("{}")])));
    let mut runner = Runner::new(
        OpenAiResponses::new(client, template()),
        ToolRegistry::new(),
    )
    .with_config(config.clone());
    runner
        .run(&mut Conversation::user("Umbrella?"))
        .await
        .unwrap();
    let body = seen.lock().unwrap()[0].clone();
    assert_eq!(body["instructions"], "You are a household assistant.");
    assert_eq!(
        body["text"]["format"],
        json!({"type": "json_schema", "name": "answer", "schema": schema.schema, "strict": true})
    );

    // Chat Completions: a first developer (or system) message instead of the template's.
    let (client, seen) = scripted(|_, _| Reply::json(&chat_completion(Some("{}"), &[])));
    let template = ChatCompletionBuilder::new("gpt-5.4-mini").system("Be brief.");
    let mut runner = Runner::new(
        OpenAiChat::new(client.clone(), template.clone()),
        ToolRegistry::new(),
    )
    .with_config(config.clone());
    runner
        .run(&mut Conversation::user("Umbrella?"))
        .await
        .unwrap();
    let body = seen.lock().unwrap()[0].clone();
    assert_eq!(
        body["messages"][0],
        json!({"role": "developer", "content": "You are a household assistant."})
    );
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "answer");
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);

    // A server without the developer role gets a system message.
    let (client, seen) = scripted(|_, _| Reply::json(&chat_completion(Some("{}"), &[])));
    let model = OpenAiChat::new(client, template).with_compat(Compat::cautious());
    let mut runner = Runner::new(model, ToolRegistry::new()).with_config(config);
    runner
        .run(&mut Conversation::user("Umbrella?"))
        .await
        .unwrap();
    assert_eq!(seen.lock().unwrap()[0]["messages"][0]["role"], "system");
}
