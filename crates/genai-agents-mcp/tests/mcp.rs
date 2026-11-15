//! The tools of a real MCP server (rmcp's server side, in-process over a pipe) in a registry:
//! listing and registering, namespaces, calls, `isError`, read-only tools, a changed tool list,
//! schemas, colliding names, images, the trace context, and a run with the runner.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::tools::{Advertise, Exposure, ToolKey, ToolRegistry};
use genai_agents::{
    Content, Conversation, Item, Message, Role, RunStop, Runner, ToolCall, ToolDefinition,
    ToolOutput, ToolOutputBody,
};
use genai_agents_mcp::{Grouping, McpOptions, McpServer, TraceContext};
use rmcp::ServiceExt;
use serde_json::{json, Value};
use support::{State, TestServer};

const NS: &str = "household";

/// Start the test server in this process and connect to it over a pipe.
async fn connect(options: McpOptions) -> (McpServer, Arc<State>) {
    let server = TestServer::default();
    let state = Arc::clone(&server.state);
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let running = server.serve(server_io).await.unwrap();
        let _ = running.waiting().await;
    });
    let client = McpServer::connect("household", client_io, options)
        .await
        .unwrap();
    (client, state)
}

async fn registered(options: McpOptions) -> (McpServer, Arc<State>, ToolRegistry) {
    let (server, state) = connect(options).await;
    let mut tools = ToolRegistry::new();
    server.register(&mut tools).await.unwrap();
    (server, state, tools)
}

fn call(id: &str, name: &str, arguments: &Value) -> ToolCall {
    ToolCall::function(id, name, arguments.to_string()).in_namespace(NS)
}

async fn answer(tools: &ToolRegistry, name: &str, arguments: &Value) -> ToolOutput {
    tools
        .answer_call(call("c1", name, arguments))
        .await
        .unwrap()
}

/// Every definition, as if every tool were direct.
fn all_definitions(tools: &ToolRegistry) -> Vec<ToolDefinition> {
    tools.definitions_where(|_, _| Some(Advertise::Plain))
}

fn namespaced_parameters(tools: &ToolRegistry, tool: &str) -> Value {
    all_definitions(tools)
        .into_iter()
        .find_map(|definition| match definition {
            ToolDefinition::Namespace { tools, .. } => {
                tools.into_iter().find_map(|member| match member {
                    ToolDefinition::Function {
                        name, parameters, ..
                    } if name == tool => Some(parameters),
                    _ => None,
                })
            }
            _ => None,
        })
        .unwrap()
}

#[tokio::test]
async fn the_tools_are_registered_deferred_in_a_namespace() {
    let (server, _, tools) = registered(McpOptions::new()).await;

    assert_eq!(server.name(), NS);
    assert_eq!(
        server.description(),
        "Weather, files and charts for the household."
    );
    assert_eq!(
        server.instructions(),
        Some("Ask for the weather before planning outside.")
    );
    let names: Vec<String> = server
        .tools()
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    assert_eq!(names.len(), 10);
    assert!(names.contains(&"get_weather".to_string()));
    assert_eq!(tools.len(), 10);
    assert!(tools.keys().all(|key| key.namespace.as_deref() == Some(NS)));
    assert!(tools
        .keys()
        .all(|key| tools.exposure(key) == Some(Exposure::Deferred)));

    // Deferred: not in the plain definitions, found by a tool search.
    assert!(tools.definitions().is_empty());
    let definitions = all_definitions(&tools);
    let [ToolDefinition::Namespace {
        name, description, ..
    }] = definitions.as_slice()
    else {
        panic!("one namespace expected: {definitions:?}");
    };
    assert_eq!(name, NS);
    assert_eq!(description, server.description());
    assert_eq!(
        tools.search_deferred("weather in a city", 1),
        [ToolKey::namespaced(NS, "get_weather")]
    );
    assert!(!server.tools_changed());
}

#[tokio::test]
async fn a_call_with_arguments_answers_with_untrusted_text() {
    let (_, _, tools) = registered(McpOptions::new()).await;
    let output = answer(&tools, "get_weather", &json!({"city": "Ghent"})).await;
    assert!(output.success);
    assert!(output.untrusted);
    assert_eq!(output.body.as_text(), Some("12 degrees in Ghent"));
}

#[tokio::test]
async fn an_error_result_is_a_failure_the_model_sees() {
    let (_, _, tools) = registered(McpOptions::new()).await;
    let output = answer(&tools, "find_city", &json!({"city": "Atlantis"})).await;
    assert!(!output.success);
    assert_eq!(output.body.as_text(), Some("no such city: Atlantis"));
}

#[tokio::test]
async fn read_only_tools_run_in_parallel_and_the_hints_are_kept() {
    let (server, _, tools) = registered(McpOptions::new()).await;
    assert!(tools.supports_parallel(&call("c", "get_weather", &json!({}))));
    assert!(!tools.supports_parallel(&call("c", "delete_file", &json!({}))));
    assert!(!tools.supports_parallel(&call("c", "find_city", &json!({}))));

    let hints = |name: &str| server.tool(&ToolKey::namespaced(NS, name)).unwrap();
    assert!(hints("get_weather").read_only());
    assert!(!hints("get_weather").destructive());
    assert!(hints("delete_file").destructive());
    assert_eq!(
        hints("delete_file").annotations.unwrap().destructive_hint,
        Some(true)
    );
    // Without hints, a tool may destroy something.
    assert!(hints("find_city").destructive());
    assert!(hints("find_city").open_world());
}

#[tokio::test]
async fn schemas_are_cleaned_up_for_the_models() {
    let (_, _, tools) = registered(McpOptions::new()).await;
    assert_eq!(
        namespaced_parameters(&tools, "search"),
        json!({
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["recipe"]},
                "query": {"type": "string"}
            }
        })
    );
    let output = answer(&tools, "search", &json!({"kind": "recipe"})).await;
    assert_eq!(output.body.as_text(), Some(r#"found "recipe""#));
}

#[tokio::test]
async fn colliding_names_get_a_hash() {
    let (server, _, tools) = registered(McpOptions::new()).await;
    let key_of = |raw: &str| {
        server
            .tools()
            .into_iter()
            .find(|tool| tool.name == raw)
            .unwrap()
            .key
    };
    let word = key_of("look.up");
    let place = key_of("look-up");
    assert_eq!(word, ToolKey::namespaced(NS, "look_up"));
    assert!(place.name.starts_with("look_up_"), "{place:?}");
    assert!(place.name.len() <= 64);

    let output = answer(&tools, &word.name, &json!({})).await;
    assert_eq!(output.body.as_text(), Some("a word"));
    let output = answer(&tools, &place.name, &json!({})).await;
    assert_eq!(output.body.as_text(), Some("a place"));
}

#[tokio::test]
async fn images_come_back_as_images() {
    let (_, _, tools) = registered(McpOptions::new()).await;
    let output = answer(&tools, "chart", &json!({})).await;
    assert_eq!(
        output.body,
        ToolOutputBody::Items(vec![
            Content::text("the temperature this week"),
            Content::image_url("data:image/png;base64,iVBORw0KGgo="),
        ])
    );
}

#[tokio::test]
async fn a_changed_tool_list_takes_effect_for_new_calls_only() {
    let (server, state, mut tools) = registered(McpOptions::new()).await;
    let slow = ToolKey::namespaced(NS, "slow");
    assert!(tools.exposure(&slow).is_some());

    // A call that started with the old list.
    let before = tools.clone();
    let running = tokio::spawn(async move {
        before
            .answer_call(call("c1", "slow", &json!({})))
            .await
            .unwrap()
    });
    state.started.notified().await;

    // The server installs a tool and drops `slow`; it says so.
    let output = answer(&tools, "install", &json!({})).await;
    assert_eq!(output.body.as_text(), Some("installed"));
    tokio::time::timeout(Duration::from_secs(5), server.changed())
        .await
        .unwrap();
    assert!(server.tools_changed());
    assert!(state.installed.load(Ordering::SeqCst));

    let registered = server.register(&mut tools).await.unwrap();
    assert!(!server.tools_changed());
    assert!(registered.iter().any(|tool| tool.name == "extra"));
    assert_eq!(tools.len(), 10);
    assert!(tools.exposure(&slow).is_none());
    assert_eq!(
        answer(&tools, "extra", &json!({})).await.body.as_text(),
        Some("extra")
    );
    // A new call of the old tool is unknown.
    let gone = answer(&tools, "slow", &json!({})).await;
    assert!(!gone.success);
    assert_eq!(
        gone.body.as_text(),
        Some("unsupported call: household.slow")
    );

    // The running call finishes with the tool it started with.
    state.release.notify_one();
    let output = running.await.unwrap();
    assert!(output.success);
    assert_eq!(output.body.as_text(), Some("done"));
}

#[tokio::test]
async fn the_trace_context_goes_to_the_server() {
    const TRACEPARENT: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    let options = McpOptions::new()
        .trace_context(|| Some(TraceContext::new(TRACEPARENT).with_tracestate("k=v")));
    let (_, _, tools) = registered(options).await;
    let output = answer(&tools, "trace", &json!({})).await;
    assert_eq!(output.body.as_text(), Some(TRACEPARENT));

    let (_, _, tools) = registered(McpOptions::new()).await;
    let output = answer(&tools, "trace", &json!({})).await;
    assert_eq!(output.body.as_text(), Some("none"));
}

#[tokio::test]
async fn prefixed_direct_trusted_tools() {
    let options = McpOptions::new()
        .grouping(Grouping::Prefix)
        .exposure(Exposure::Direct)
        .untrusted_output(false);
    let (server, _, mut tools) = registered(options).await;
    let definitions = tools.definitions();
    let names: Vec<&str> = definitions
        .iter()
        .filter_map(ToolDefinition::name)
        .collect();
    assert!(names.contains(&"household__get_weather"), "{names:?}");
    assert_eq!(names.len(), 10);

    let output = tools
        .answer_call(ToolCall::function(
            "c1",
            "household__get_weather",
            r#"{"city":"Ghent"}"#,
        ))
        .await
        .unwrap();
    assert!(!output.untrusted);
    assert_eq!(output.body.as_text(), Some("12 degrees in Ghent"));

    // Registering again replaces the server's tools.
    server.register(&mut tools).await.unwrap();
    assert_eq!(tools.len(), 10);
}

#[tokio::test]
async fn a_runner_calls_a_deferred_mcp_tool_after_a_tool_search() {
    let (server, _) = connect(McpOptions::new()).await;
    let model = ScriptedModel::new(|index, _| {
        let items = match index {
            0 => vec![Item::ToolCall(ToolCall::function(
                "s1",
                "tool_search",
                r#"{"query":"weather"}"#,
            ))],
            1 => vec![Item::ToolCall(call(
                "c1",
                "get_weather",
                &json!({"city": "Ghent"}),
            ))],
            _ => vec![Item::Message(Message::text(
                Role::Assistant,
                "12 degrees: no umbrella.",
            ))],
        };
        Reply::items(items)
    });
    let mut runner = Runner::new(model.clone(), ToolRegistry::new());
    server.register(runner.registry_mut()).await.unwrap();

    let mut conversation = Conversation::user("Umbrella in Ghent today?");
    let outcome = runner.run(&mut conversation).await.unwrap();
    assert_eq!(outcome.stop, RunStop::Completed);
    assert_eq!(outcome.output_text(), "12 degrees: no umbrella.");

    let requests = model.requests();
    assert_eq!(requests[0].tool_names(), ["tool_search"]);
    // Found by the search, the tool is advertised in its namespace.
    assert!(requests[1].tools.iter().any(|definition| matches!(
        definition,
        ToolDefinition::Namespace { name, .. } if name == NS
    )));
    let outputs = requests[2].outputs();
    let weather = outputs
        .iter()
        .find(|output| output.call_id == "c1")
        .unwrap();
    assert!(weather.success);
    assert!(weather.untrusted);
    assert_eq!(weather.body.as_text(), Some("12 degrees in Ghent"));
}
