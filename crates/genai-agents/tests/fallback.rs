//! Fallback models and credentials: which route takes a request after which failure, how long
//! a route rests, and which route stays in use.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(not(target_family = "wasm"))]

use std::time::Duration;

use genai_agents::fallback::{Fallback, FallbackConfig};
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::{
    Conversation, ErrorKind, Model, ModelError, ModelInfo, Origin, RetryClass, RetryConfig,
    RunConfig, Runner, ToolRegistry,
};

fn named(model: &str, replies: Vec<Reply>) -> ScriptedModel {
    ScriptedModel::replies(replies).with_info(ModelInfo::new().with_origin(Origin::new(
        "responses",
        "test",
        model,
    )))
}

fn fail(kind: ErrorKind) -> Reply {
    Reply::fail(ModelError::new(format!("{kind:?}")).with_kind(kind))
}

fn quick() -> RunConfig {
    RunConfig::new().with_retry(
        RetryConfig::default()
            .with_base_delay(Duration::from_millis(1))
            .with_max_delay(Duration::from_millis(1)),
    )
}

async fn ask(runner: &mut Runner<Fallback>) -> String {
    runner
        .run(&mut Conversation::user("Hi"))
        .await
        .unwrap()
        .output_text()
        .to_owned()
}

#[tokio::test]
async fn refused_credentials_move_to_the_next_and_it_stays() {
    let expired = named("a", vec![fail(ErrorKind::Authentication)]);
    let spare = named("a", vec![Reply::text("one"), Reply::text("two")]);
    let other = named("b", vec![]);
    let model = Fallback::new(expired.clone())
        .credential(spare.clone())
        .then(other.clone());
    let mut runner = Runner::new(model, ToolRegistry::new());
    assert_eq!(ask(&mut runner).await, "one");
    assert_eq!(ask(&mut runner).await, "two");
    // The expired key was tried once; the spare one kept the requests (and its cache).
    assert_eq!(expired.requests().len(), 1);
    assert_eq!(spare.requests().len(), 2);
    assert!(other.requests().is_empty());
    assert_eq!(runner.model().current(), 1);
}

#[tokio::test]
async fn an_unknown_model_leaves_its_group_for_good() {
    let gone = named("old", vec![fail(ErrorKind::ModelNotFound)]);
    let same_model_other_key = named("old", vec![]);
    let next = named("new", vec![Reply::text("from new"), Reply::text("again")]);
    let model = Fallback::new(gone.clone())
        .credential(same_model_other_key.clone())
        .then(next.clone());
    let mut runner = Runner::new(model, ToolRegistry::new());
    assert_eq!(ask(&mut runner).await, "from new");
    // Its other credential was not tried: the model is gone, whatever the key.
    assert!(same_model_other_key.requests().is_empty());
    // The model the runner sees is the one that answers now.
    assert_eq!(runner.model().info().origin.unwrap().model, "new");
    // Even preferred again, the gone group is skipped.
    runner.model().prefer(0);
    assert_eq!(ask(&mut runner).await, "again");
    assert_eq!(gone.requests().len(), 1);
}

#[tokio::test]
async fn a_rate_limit_rests_the_route_as_long_as_asked() {
    let limited = named(
        "a",
        vec![
            Reply::fail(
                ModelError::new("slow down")
                    .with_kind(ErrorKind::RateLimit)
                    .with_retry_after(Some(Duration::from_millis(80))),
            ),
            Reply::text("back on a"),
        ],
    );
    let spare = named("a", vec![Reply::text("spare")]);
    let model = Fallback::new(limited.clone()).credential(spare.clone());
    let mut runner = Runner::new(model, ToolRegistry::new());
    assert_eq!(ask(&mut runner).await, "spare");
    // Back to the first once its rest is over (and preferred, as a session's route).
    tokio::time::sleep(Duration::from_millis(100)).await;
    runner.model().prefer(0);
    assert_eq!(ask(&mut runner).await, "back on a");
}

#[tokio::test]
async fn server_failures_fall_back_after_some_in_a_row() {
    let flaky = named(
        "a",
        vec![
            fail(ErrorKind::Server).into_retryable(),
            fail(ErrorKind::Server).into_retryable(),
        ],
    );
    let other = named("b", vec![Reply::text("from b")]);
    let model = Fallback::new(flaky.clone())
        .then(other.clone())
        .with_config(FallbackConfig::new().with_max_failures(2));
    let mut runner = Runner::new(model, ToolRegistry::new()).with_config(quick());
    // The first failure is the runner's to retry; the second rests the route and falls back.
    let outcome = runner.run(&mut Conversation::user("Hi")).await.unwrap();
    assert_eq!(outcome.output_text(), "from b");
    assert_eq!(outcome.turns[0].retries, 1);
    assert_eq!(flaky.requests().len(), 2);
}

#[tokio::test]
async fn local_faults_and_rejected_requests_never_fall_back() {
    for kind in [
        ErrorKind::Local,
        ErrorKind::Other,
        ErrorKind::ContextOverflow,
    ] {
        let broken = named("a", vec![fail(kind)]);
        let other = named("b", vec![Reply::text("unused")]);
        let model = Fallback::new(broken).then(other.clone());
        let mut runner = Runner::new(model, ToolRegistry::new()).with_config(
            quick()
                .with_context(genai_agents::ContextConfig::new().with_compact_on_overflow(false)),
        );
        let error = runner.run(&mut Conversation::user("Hi")).await.unwrap_err();
        assert_eq!(error.as_model().unwrap().kind(), kind);
        assert!(other.requests().is_empty(), "{kind:?}");
    }
}

#[tokio::test]
async fn every_route_resting_is_a_retryable_error() {
    let a = named("a", vec![fail(ErrorKind::Authentication)]);
    let b = named("b", vec![fail(ErrorKind::Authentication)]);
    let model = Fallback::new(a.clone()).then(b.clone());
    let mut runner = Runner::new(model, ToolRegistry::new()).with_config(quick());
    // Both refused: the last refusal is the run's error (not retried: credentials).
    let error = runner.run(&mut Conversation::user("Hi")).await.unwrap_err();
    assert_eq!(error.as_model().unwrap().kind(), ErrorKind::Authentication);
    // While both rest, a request reaches no route: a retryable error, with retries that find
    // them resting still.
    let error = runner.run(&mut Conversation::user("Hi")).await.unwrap_err();
    let error = error.as_model().unwrap();
    assert!(error.message().contains("resting or gone"), "{error}");
    assert!(error.retry_class().is_retryable());
    assert_eq!(a.requests().len() + b.requests().len(), 2);
}

trait Retryable {
    fn into_retryable(self) -> Self;
}

impl Retryable for Reply {
    fn into_retryable(self) -> Self {
        match self {
            Reply::Fail(error) => Reply::Fail(error.with_retry_class(RetryClass::Backoff)),
            other => other,
        }
    }
}
