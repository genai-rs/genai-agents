//! WebAssembly: the runner in a JavaScript host, with a model and tools that are not `Send`
//! (they hold `Rc`s): parallel calls answered in call order, a retried request (the host's timer),
//! and cancellation.
//!
//! `cargo test -p genai-agents --target wasm32-unknown-unknown --test wasm`, with
//! `wasm-bindgen-test-runner` as the target's runner (it runs the tests in Node.js).
#![cfg(target_family = "wasm")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(unknown_lints, clippy::unused_async_trait_impl)]
// Nothing here is `Send`: WebAssembly futures stay on the host's thread.
#![allow(clippy::future_not_send, clippy::unnecessary_literal_bound)]

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::rc::Rc;
use std::time::Duration;

use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{
    CancellationSignal, Conversation, Item, Model, ModelError, ModelEvent, ModelInfo, ModelRequest,
    ModelResponse, ModelStream, Result, RetryClass, RetryConfig, RunConfig, RunEvent, RunStop,
    Runner, ToolCall,
};
use serde_json::{json, Value};
use wasm_bindgen_test::wasm_bindgen_test;

/// A model whose replies come from a list, and that counts its requests in an `Rc`.
struct Scripted {
    replies: RefCell<Vec<std::result::Result<Vec<Item>, ModelError>>>,
    requests: Rc<Cell<usize>>,
}

impl Model for Scripted {
    fn info(&self) -> ModelInfo {
        ModelInfo::default()
    }

    fn stream<'a>(
        &'a self,
        _request: ModelRequest<'_>,
    ) -> impl Future<Output = std::result::Result<ModelStream<'a>, ModelError>> {
        self.requests.set(self.requests.get() + 1);
        let reply = self.replies.borrow_mut().remove(0);
        async move {
            let items = reply?;
            let mut events: Vec<ModelEvent> = items.into_iter().map(ModelEvent::ItemDone).collect();
            events.push(ModelEvent::Completed(ModelResponse::new()));
            Ok(ModelStream::from_events(events))
        }
    }
}

/// Counts its calls in an `Rc` and waits on the host's timer.
struct Counter {
    name: &'static str,
    calls: Rc<Cell<u32>>,
    millis: u32,
}

impl FunctionTool for Counter {
    type Input = Value;
    type Output = String;
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "Count, then say your name."
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _: Value) -> Result<String> {
        gloo_timers::future::TimeoutFuture::new(self.millis).await;
        self.calls.set(self.calls.get() + 1);
        Ok(self.name.to_string())
    }
    fn supports_parallel(&self) -> bool {
        true
    }
}

fn call(id: &str, name: &str) -> Item {
    Item::ToolCall(ToolCall::function(id, name, "{}"))
}

#[wasm_bindgen_test]
async fn the_runner_runs_tools_that_are_not_send() {
    let calls = Rc::new(Cell::new(0));
    let mut tools = ToolRegistry::new();
    for (name, millis) in [("slow", 30), ("fast", 1)] {
        tools
            .register(Counter {
                name,
                calls: calls.clone(),
                millis,
            })
            .unwrap();
    }
    let requests = Rc::new(Cell::new(0));
    let model = Scripted {
        replies: RefCell::new(vec![
            Ok(vec![call("a", "slow"), call("b", "fast")]),
            Err(ModelError::new("cut off").with_retry_class(RetryClass::Backoff)),
            Ok(vec![Item::assistant("done")]),
        ]),
        requests: requests.clone(),
    };
    let retry = RetryConfig::new(1).with_base_delay(Duration::from_millis(5));
    let mut runner = Runner::new(model, tools).with_config(RunConfig::new().with_retry(retry));
    let mut answered = Vec::new();
    let mut conversation = Conversation::user("go");
    let outcome = runner
        .run_with(&mut conversation, &CancellationSignal::new(), |event| {
            if let RunEvent::ToolAnswered(output) = event {
                answered.push(output.call_id.clone());
            }
        })
        .await
        .unwrap();
    assert_eq!(outcome.stop, RunStop::Completed);
    assert_eq!(outcome.output_text(), "done");
    assert_eq!(outcome.turns[1].retries, 1);
    assert_eq!(calls.get(), 2);
    assert_eq!(requests.get(), 3);
    // Answered as they finished, recorded in call order.
    assert_eq!(answered, ["b", "a"]);
    let outputs: Vec<&str> = conversation
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::ToolOutput(output) => Some(output.call_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(outputs, ["a", "b"]);
}

#[wasm_bindgen_test]
async fn a_cancelled_run_answers_every_call() {
    let calls = Rc::new(Cell::new(0));
    let mut tools = ToolRegistry::new();
    tools
        .register(Counter {
            name: "forever",
            calls: calls.clone(),
            millis: 60_000,
        })
        .unwrap();
    let model = Scripted {
        replies: RefCell::new(vec![Ok(vec![call("a", "forever")])]),
        requests: Rc::new(Cell::new(0)),
    };
    let mut runner = Runner::new(model, tools);
    let cancel = CancellationSignal::new();
    let trigger = cancel.clone();
    wasm_bindgen_futures::spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(20).await;
        trigger.cancel();
    });
    let mut conversation = Conversation::user("go");
    let outcome = runner
        .run_with(&mut conversation, &cancel, |_| {})
        .await
        .unwrap();
    assert_eq!(outcome.stop, RunStop::Cancelled);
    assert_eq!(calls.get(), 0);
    assert!(conversation.unanswered_calls().is_empty());
}
