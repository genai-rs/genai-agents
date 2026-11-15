//! A Langfuse dataset run against a mock Langfuse: items run as cases, scored, linked.
#![cfg(feature = "langfuse")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::{Runner, ToolRegistry};
use genai_agents_eval::langfuse::run_dataset;
use genai_agents_eval::{target_fn, Case, ExactMatch, Run};
use langfuse_ergonomic::ClientBuilder;
use mockito::Matcher;
use serde_json::{json, Value};

const DATASET: &str = r#"{"id":"ds-1","name":"capitals","description":null,"metadata":null,"inputSchema":null,"expectedOutputSchema":null,"projectId":"proj-1","createdAt":"2026-11-01T00:00:00Z","updatedAt":"2026-11-01T00:00:00Z"}"#;

fn item(id: &str, input: &str, expected: &str) -> Value {
    json!({
        "id": id, "status": "ACTIVE", "input": input, "expectedOutput": expected,
        "datasetId": "ds-1", "datasetName": "capitals", "metadata": null, "sourceTraceId": null,
        "sourceObservationId": null, "createdAt": "2026-11-01T00:00:00Z",
        "updatedAt": "2026-11-01T00:00:00Z", "mediaReferences": []
    })
}

#[tokio::test]
async fn a_dataset_runs_as_a_dataset_run_with_scores() {
    let mut server = mockito::Server::new_async().await;
    let json_type = ("content-type", "application/json");
    let _dataset = server
        .mock("GET", "/api/public/v2/datasets/capitals")
        .with_header(json_type.0, json_type.1)
        .with_body(DATASET)
        .create_async()
        .await;
    let _items = server
        .mock("GET", "/api/public/dataset-items")
        .match_query(Matcher::Any)
        .with_header(json_type.0, json_type.1)
        .with_body(
            json!({
                "data": [item("i1", "France?", "Paris"), item("i2", "Belgium?", "Brussels")],
                "meta": {"page": 1, "limit": 100, "totalItems": 2, "totalPages": 1}
            })
            .to_string(),
        )
        .create_async()
        .await;
    // Whatever the client sends of traces and scores, kept to look at.
    let sent: Arc<Mutex<Vec<String>>> = Arc::default();
    let mut sinks = Vec::new();
    for (path, status, body) in [
        (
            "/api/public/ingestion",
            207,
            r#"{"successes":[],"errors":[]}"#,
        ),
        ("/api/public/otel/v1/traces", 200, "{}"),
        ("/api/public/scores", 200, r#"{"id":"score"}"#),
    ] {
        let sent = Arc::clone(&sent);
        sinks.push(
            server
                .mock("POST", path)
                .match_request(move |request| {
                    sent.lock()
                        .unwrap()
                        .push(request.utf8_lossy_body().unwrap().to_string());
                    true
                })
                .with_status(status)
                .with_header(json_type.0, json_type.1)
                .with_body(body)
                .expect_at_least(0)
                .create_async()
                .await,
        );
    }
    let run_items = server
        .mock("POST", "/api/public/dataset-run-items")
        .match_body(Matcher::PartialJson(json!({"runName": "baseline"})))
        .with_header(json_type.0, json_type.1)
        .with_body(r#"{"id":"dri","datasetRunId":"run-1","datasetRunName":"baseline","datasetItemId":"x","traceId":"t","observationId":null,"createdAt":"2026-11-01T00:00:00Z","updatedAt":"2026-11-01T00:00:00Z"}"#)
        .expect(2)
        .create_async()
        .await;

    let client = ClientBuilder::new()
        .public_key("pk-test")
        .secret_key("sk-test")
        .base_url(server.url())
        .build()
        .unwrap();
    // Knows France only.
    let target = target_fn(|case: Case| async move {
        let model = ScriptedModel::replies(vec![Reply::text("Paris")]);
        let mut runner = Runner::new(model, ToolRegistry::new());
        let mut conversation = case.input;
        let outcome = runner.run(&mut conversation).await?;
        Ok(Run::new(conversation, outcome))
    });
    let exact = ExactMatch;
    let run = run_dataset(&client, "capitals", "baseline", &target, &[&exact], 2)
        .await
        .unwrap();
    run_items.assert_async().await;

    assert_eq!(run.experiment.dataset_run_id.as_deref(), Some("run-1"));
    assert_eq!(run.experiment.items.len(), 2);
    assert_eq!(run.report.summary.cases, 2);
    assert_eq!(run.report.summary.pass_rate, 0.5);
    let france = run
        .report
        .cases
        .iter()
        .find(|case| case.id == "i1")
        .unwrap();
    assert!(france.pass_all_k);
    // The scores went to Langfuse.
    let sent = sent.lock().unwrap().join("\n");
    assert!(sent.contains("exact_match"), "{sent}");
}
