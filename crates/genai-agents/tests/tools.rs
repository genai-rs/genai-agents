//! Tools: definitions, decoding and dispatch, answering every call (`answer_call`), rich and
//! custom tools, namespaces, dynamic tools and truncation.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(
    unknown_lints,
    clippy::unused_async_trait_impl,
    clippy::empty_enums,
    clippy::unnecessary_literal_bound,
    clippy::items_after_statements
)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use genai_agents::tools::{
    CustomFormat, CustomTool, DynamicTool, FunctionTool, GrammarSyntax, OutputBudget, RichTool,
    ToolError, ToolFailure, ToolKey, ToolRegistry, TruncationConfig,
};
use genai_agents::{Content, Error, Result, ToolCall, ToolDefinition, ToolKind, ToolOutput};
use genai_agents::{ToolOutputBody, ABORTED_OUTPUT};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------------------------
// Typed tools, decoding and dispatch.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    message: String,
    suffix: Option<String>,
}

struct Echo {
    name: String,
    calls: Arc<AtomicUsize>,
    schema: Value,
    description: String,
}

impl Echo {
    fn new(name: &str, calls: &Arc<AtomicUsize>) -> Self {
        Self {
            name: name.into(),
            calls: calls.clone(),
            description: "Echo a message with an optional suffix".into(),
            schema: json!({"type": "object", "properties": {
                "message": {"type": "string", "description": "Text to echo"},
                "suffix": {"type": ["string", "null"], "description": "Optional suffix"}
            }, "required": ["message"], "additionalProperties": false}),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("application rejected message")]
struct Rejected;

impl FunctionTool for Echo {
    type Input = Input;
    type Output = String;
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }
    async fn execute(&self, input: Input) -> Result<String> {
        tokio::task::yield_now().await;
        self.calls.fetch_add(1, Ordering::SeqCst);
        if input.message == "fail" {
            return Err(Error::application(Rejected));
        }
        Ok(input.message + &input.suffix.unwrap_or_default())
    }
}

fn names(definitions: &[ToolDefinition]) -> Vec<&str> {
    definitions
        .iter()
        .map(|definition| definition.name().unwrap_or("tool_search"))
        .collect()
}

#[tokio::test]
async fn definitions_are_sorted_and_calls_are_answered() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(Echo::new("z_echo", &calls)).unwrap();
    tools.register(Echo::new("a_echo", &calls)).unwrap();
    let definitions = tools.definitions();
    assert_eq!(names(&definitions), ["a_echo", "z_echo"]);
    let ToolDefinition::Function {
        description,
        parameters,
        defer_loading,
        ..
    } = &definitions[0]
    else {
        panic!("a function expected");
    };
    assert_eq!(description, "Echo a message with an optional suffix");
    assert_eq!(parameters, &Echo::new("unused", &calls).schema);
    assert!(!defer_loading);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let first = tools
        .answer_call(&ToolCall::function(
            "first",
            "z_echo",
            r#"{"message":"Hello"}"#,
        ))
        .await
        .unwrap();
    assert_eq!(first.call_id, "first");
    assert_eq!(first.body.as_text(), Some(r#""Hello""#));
    let second = tools
        .answer_call(&ToolCall::function(
            "second",
            "a_echo",
            r#"{"message":"Hi","suffix":"!"}"#,
        ))
        .await
        .unwrap();
    assert_eq!(second.body.as_text(), Some(r#""Hi!""#));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        tools
            .execute("z_echo", r#"{"message":"Hi","suffix":null}"#)
            .await
            .unwrap(),
        "Hi"
    );
}

#[tokio::test]
async fn rejects_bad_arguments_before_running_handler() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(Echo::new("echo", &calls)).unwrap();
    for args in [
        "{",
        "{}",
        r#"{"message":3}"#,
        r#"{"message":"a","extra":1}"#,
        r#"{"message":"a","message":"b"}"#,
        r#"{"message":"a"} trailing"#,
    ] {
        assert!(
            matches!(
                tools.execute("echo", args).await,
                Err(ToolError::Arguments { .. })
            ),
            "{args}"
        );
    }
    for args in ["[]", "null", "3", r#""hello""#] {
        assert!(matches!(
            tools.execute("echo", args).await,
            Err(ToolError::InvalidArguments { .. })
        ));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn failed_registration_preserves_existing_tools() {
    let calls = Arc::new(AtomicUsize::new(0));
    let replacement_calls = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(Echo::new("echo", &calls)).unwrap();
    assert!(
        matches!(tools.register(Echo::new("echo", &replacement_calls)), Err(ToolError::Duplicate { name }) if name == "echo")
    );
    for name in ["", "space name", "écho", &"a".repeat(65)] {
        assert!(matches!(
            tools.register(Echo::new(name, &calls)),
            Err(ToolError::InvalidDefinition { .. })
        ));
    }
    for schema in [Value::Null, json!([]), json!({}), json!({"type": "array"})] {
        let mut tool = Echo::new("invalid", &calls);
        tool.schema = schema;
        assert!(matches!(
            tools.register(tool),
            Err(ToolError::InvalidDefinition { .. })
        ));
    }
    let mut blank = Echo::new("blank", &calls);
    blank.description = "  ".into();
    assert!(matches!(
        tools.register(blank),
        Err(ToolError::InvalidDefinition { .. })
    ));
    assert_eq!(tools.definitions().len(), 1);
    tools.execute("echo", r#"{"message":"ok"}"#).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(replacement_calls.load(Ordering::SeqCst), 0);
}

struct Dynamic;
impl FunctionTool for Dynamic {
    type Input = Value;
    type Output = Value;
    fn name(&self) -> &'static str {
        "dynamic"
    }
    fn description(&self) -> &'static str {
        "Echo any object"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "additionalProperties": true})
    }
    async fn execute(&self, input: Value) -> Result<Value> {
        Ok(input)
    }
}

struct BadOutput;
impl Serialize for BadOutput {
    fn serialize<S: Serializer>(&self, _: S) -> std::result::Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom("cannot encode output"))
    }
}

struct BadEncoder;
impl FunctionTool for BadEncoder {
    type Input = Value;
    type Output = BadOutput;
    fn name(&self) -> &'static str {
        "bad_encoder"
    }
    fn description(&self) -> &'static str {
        "Demonstrate output failures"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _: Value) -> Result<BadOutput> {
        Ok(BadOutput)
    }
}

#[tokio::test]
async fn dynamic_json_and_output_encoding_failures() {
    let mut tools = ToolRegistry::default();
    assert_eq!(tools.definitions().len(), 0);
    tools.register(Dynamic).unwrap();
    tools.register(BadEncoder).unwrap();
    let input = json!({"nested": {"items": [true, null, 3]}});
    assert_eq!(
        tools.execute("dynamic", &input.to_string()).await.unwrap(),
        input
    );
    assert!(matches!(
        tools.execute("dynamic", "[]").await,
        Err(ToolError::InvalidArguments { .. })
    ));
    assert!(matches!(
        tools.execute("bad_encoder", "{}").await,
        Err(ToolError::Output { .. })
    ));
    // An output that cannot be encoded is fatal: it is not the model's to fix.
    assert!(matches!(
        tools
            .answer_call(ToolCall::function("c", "bad_encoder", "{}"))
            .await,
        Err(ToolError::Call { .. })
    ));
}

struct Stateful {
    revision: Arc<AtomicUsize>,
}
impl FunctionTool for Stateful {
    type Input = Value;
    type Output = usize;
    fn name(&self) -> &'static str {
        "stateful"
    }
    fn description(&self) -> &'static str {
        "Read the current application revision"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "description": format!("Revision {}", self.revision.load(Ordering::SeqCst))})
    }
    async fn execute(&self, _: Value) -> Result<usize> {
        tokio::task::yield_now().await;
        Ok(self.revision.load(Ordering::SeqCst))
    }
}

#[tokio::test]
async fn definitions_are_snapshots_while_handler_state_remains_live() {
    let revision = Arc::new(AtomicUsize::new(1));
    let mut tools = ToolRegistry::new();
    tools
        .register(Stateful {
            revision: revision.clone(),
        })
        .unwrap();
    let before = tools.definitions();
    revision.store(2, Ordering::SeqCst);
    assert_eq!(before, tools.definitions());
    assert_eq!(tools.execute("stateful", "{}").await.unwrap(), 2);
}

#[derive(Debug, thiserror::Error)]
#[error("record does not exist")]
struct MissingRecord;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

struct FailingTool;
impl FunctionTool for FailingTool {
    type Input = NoArgs;
    type Output = Value;
    fn name(&self) -> &'static str {
        "failure"
    }
    fn description(&self) -> &'static str {
        "Return an application error without arguments"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {}, "additionalProperties": false})
    }
    async fn execute(&self, _: NoArgs) -> Result<Value> {
        Err(Error::fatal(MissingRecord))
    }
}

#[tokio::test]
async fn crate_result_preserves_domain_error_and_call_context() {
    async fn application() -> Result<()> {
        let mut tools = ToolRegistry::new();
        tools.register(FailingTool)?;
        tools
            .answer_call(ToolCall::function("domain-call", "failure", "{}"))
            .await?;
        Ok(())
    }
    let error = application().await.unwrap_err();
    assert!(error.to_string().contains("domain-call"));
    assert!(error.to_string().contains("failure"));
    assert!(error.to_string().contains("record does not exist"));
    let mut source = std::error::Error::source(&error);
    let mut found_domain_error = false;
    while let Some(cause) = source {
        found_domain_error |= cause.downcast_ref::<MissingRecord>().is_some();
        source = cause.source();
    }
    assert!(found_domain_error);
    let mut tools = ToolRegistry::new();
    tools.register(FailingTool).unwrap();
    for args in ["", r#"{"unexpected":true}"#] {
        assert!(matches!(
            tools.execute("failure", args).await,
            Err(ToolError::Arguments { .. })
        ));
    }
}

#[tokio::test]
async fn registry_can_be_shared_for_concurrent_calls() {
    let count = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    let name = "a".repeat(64);
    tools.register(Echo::new(&name, &count)).unwrap();
    tools.register(Echo::new("A-0_b", &count)).unwrap();
    let tools = Arc::new(tools);
    let mut tasks = Vec::new();
    for index in 0..8 {
        let shared = tools.clone();
        let name = name.clone();
        tasks.push(tokio::spawn(async move {
            shared
                .answer_call(ToolCall::function(
                    index.to_string(),
                    name,
                    r#"{"message":"ok"}"#,
                ))
                .await
                .unwrap()
        }));
    }
    for (index, task) in tasks.into_iter().enumerate() {
        let output = task.await.unwrap();
        assert_eq!(output.call_id, index.to_string());
        assert_eq!(output.body.as_text(), Some(r#""ok""#));
    }
    assert_eq!(count.load(Ordering::SeqCst), 8);
}

// ---------------------------------------------------------------------------------------------
// Answering every call; rich, custom, namespaced and dynamic tools.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CityInput {
    city: String,
}

struct Weather;

impl FunctionTool for Weather {
    type Input = CityInput;
    type Output = Value;
    fn name(&self) -> &str {
        "get_weather"
    }
    fn description(&self) -> &str {
        "Current weather of a city"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"city": {"type": "string"}},
               "required": ["city"], "additionalProperties": false})
    }
    async fn execute(&self, input: CityInput) -> Result<Value> {
        match input.city.as_str() {
            "Atlantis" => Err(Error::application(std::io::Error::other("no such city"))),
            "Mordor" => Err(Error::fatal(std::io::Error::other(
                "weather service is down",
            ))),
            city => Ok(json!({"city": city, "sky": "rain"})),
        }
    }
    fn supports_parallel(&self) -> bool {
        true
    }
    fn output_budget(&self) -> Option<OutputBudget> {
        Some(OutputBudget::Bytes(100))
    }
}

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
        json!({"type": "object", "properties": {}})
    }
    async fn execute(&self, _input: Value) -> Result<ToolOutputBody> {
        Ok(vec![
            Content::text("Here is the chart"),
            Content::image_url("https://example.com/chart.png"),
        ]
        .into())
    }
}

struct Patch;

impl CustomTool for Patch {
    fn name(&self) -> &str {
        "apply_patch"
    }
    fn description(&self) -> &str {
        "Apply a patch to the workspace"
    }
    fn format(&self) -> CustomFormat {
        CustomFormat::lark("start: \"*** Begin Patch\" /.+/s")
    }
    async fn execute(&self, input: String) -> Result<ToolOutputBody> {
        Ok(ToolOutputBody::text(format!(
            "applied {} bytes",
            input.len()
        )))
    }
}

struct Search;

impl FunctionTool for Search {
    type Input = Value;
    type Output = String;
    fn name(&self) -> &str {
        "search"
    }
    fn description(&self) -> &str {
        "Search the documentation"
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"q": {"type": "string"}}})
    }
    async fn execute(&self, input: Value) -> Result<String> {
        Ok(format!("docs for {}", input["q"]))
    }
}

fn registry() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(Weather).unwrap();
    tools.register_rich(Chart).unwrap();
    tools.register_custom(Patch).unwrap();
    tools
        .namespace("docs", "Documentation tools")
        .unwrap()
        .register(Search)
        .unwrap();
    tools
}

#[tokio::test]
async fn failures_the_model_can_act_on_are_answered() {
    let tools = registry();
    let answer = |call: ToolCall| {
        let tools = &tools;
        async move { tools.answer_call(&call).await.unwrap() }
    };

    let good = answer(ToolCall::function(
        "c0",
        "get_weather",
        r#"{"city":"Ghent"}"#,
    ))
    .await;
    assert!(good.success);
    assert!(!good.untrusted);
    assert_eq!(good.kind, ToolKind::Function);
    assert_eq!(
        serde_json::from_str::<Value>(good.body.as_text().unwrap()).unwrap(),
        json!({"city": "Ghent", "sky": "rain"})
    );

    let bad = answer(ToolCall::function("c1", "get_weather", "{\"town\":1}")).await;
    assert!(!bad.success);
    assert!(bad
        .body
        .as_text()
        .unwrap()
        .starts_with("failed to parse function arguments:"));

    let not_object = answer(ToolCall::function("c2", "get_weather", "[]")).await;
    assert_eq!(
        not_object.body.as_text(),
        Some("function arguments must be a JSON object")
    );

    let unknown = answer(ToolCall::function("c3", "nope", "{}")).await;
    assert_eq!(unknown.body.as_text(), Some("unsupported call: nope"));

    let handler = answer(ToolCall::function(
        "c4",
        "get_weather",
        r#"{"city":"Atlantis"}"#,
    ))
    .await;
    assert_eq!(handler.body.as_text(), Some("no such city"));
    assert!(!handler.success);

    let wrong_kind = answer(ToolCall::custom("c5", "get_weather", "hello")).await;
    assert!(wrong_kind.body.as_text().unwrap().contains("function tool"));

    let other = answer(ToolCall::other("c6", "future_tool")).await;
    assert!(!other.success);
    assert_eq!(other.kind, ToolKind::Other("future_tool".into()));
}

#[tokio::test]
async fn fatal_errors_and_missing_ids_are_returned() {
    let tools = registry();
    let error = tools
        .answer_call(ToolCall::function(
            "call_9",
            "get_weather",
            r#"{"city":"Mordor"}"#,
        ))
        .await
        .unwrap_err();
    match &error {
        ToolError::Call { call_id, source } => {
            assert_eq!(call_id, "call_9");
            assert!(
                matches!(**source, ToolError::Execution { ref source, .. } if source.is_fatal())
            );
        }
        other => panic!("unexpected {other:?}"),
    }
    assert!(matches!(
        tools
            .answer_call(ToolCall::function("", "get_weather", "{}"))
            .await,
        Err(ToolError::Call { .. })
    ));

    // A classification of your own: hide handler errors from the model.
    let answer = tools
        .answer_call_with(
            ToolCall::function("c", "get_weather", r#"{"city":"Atlantis"}"#),
            |error| match error {
                ToolError::Execution { .. } => {
                    ToolFailure::RespondToModel("the tool failed".into())
                }
                other => other.classify(),
            },
        )
        .await
        .unwrap();
    assert_eq!(answer.body.as_text(), Some("the tool failed"));
}

#[tokio::test]
async fn custom_namespaced_and_rich_tools() {
    let tools = registry();

    let patch = tools
        .answer_call(ToolCall::custom("c1", "apply_patch", "*** Begin Patch"))
        .await
        .unwrap();
    assert_eq!(patch.body.as_text(), Some("applied 15 bytes"));
    assert_eq!(patch.kind, ToolKind::Custom);

    let search = tools
        .answer_call(ToolCall::function("c2", "search", r#"{"q":"tools"}"#).in_namespace("docs"))
        .await
        .unwrap();
    assert_eq!(search.body.as_text(), Some(r#""docs for \"tools\"""#));
    // Outside its namespace the tool is unknown.
    let outside = tools
        .answer_call(ToolCall::function("c3", "search", "{}"))
        .await
        .unwrap();
    assert_eq!(outside.body.as_text(), Some("unsupported call: search"));

    let chart = tools
        .answer_call(ToolCall::function("c4", "chart", "{}"))
        .await
        .unwrap();
    let ToolOutputBody::Items(items) = &chart.body else {
        panic!("content parts expected");
    };
    assert!(items[1].is_image());
    assert_eq!(chart.body.to_text(), None);

    assert!(tools.supports_parallel(&ToolCall::function("c", "get_weather", "{}")));
    assert!(!tools.supports_parallel(&ToolCall::function("c", "chart", "{}")));
    assert_eq!(
        tools.output_budget(&ToolCall::function("c", "get_weather", "{}")),
        Some(OutputBudget::Bytes(100))
    );
}

#[test]
fn definitions_of_every_kind() {
    let tools = registry();
    let definitions = tools.definitions();
    assert_eq!(definitions.len(), 4);
    assert_eq!(
        definitions[0],
        ToolDefinition::Custom {
            name: "apply_patch".into(),
            description: "Apply a patch to the workspace".into(),
            format: CustomFormat::lark("start: \"*** Begin Patch\" /.+/s"),
            defer_loading: false,
        }
    );
    assert_eq!(definitions[1].name(), Some("chart"));
    let ToolDefinition::Function { parameters, .. } = &definitions[2] else {
        panic!("a function expected");
    };
    assert_eq!(parameters["required"], json!(["city"]));
    let ToolDefinition::Namespace {
        name,
        description,
        tools: members,
    } = &definitions[3]
    else {
        panic!("a namespace expected");
    };
    assert_eq!(name, "docs");
    assert_eq!(description, "Documentation tools");
    assert_eq!(members[0].name(), Some("search"));

    let keys: Vec<&ToolKey> = tools.keys().collect();
    assert_eq!(keys.len(), 4);
    assert_eq!(keys[3], &ToolKey::namespaced("docs", "search"));
    assert_eq!(tools.kind(keys[0]), Some(&ToolKind::Custom));
    assert_eq!(
        CustomFormat::regex("a+"),
        CustomFormat::Grammar {
            syntax: GrammarSyntax::Regex,
            definition: "a+".into()
        }
    );
    assert_eq!(GrammarSyntax::Lark.as_str(), "lark");
}

#[test]
fn namespaces_and_tools_do_not_collide() {
    let mut tools = registry();
    assert!(matches!(
        tools.namespace("get_weather", "Weather"),
        Err(ToolError::Duplicate { .. })
    ));
    assert!(matches!(
        tools.namespace("docs", "Other description"),
        Err(ToolError::InvalidDefinition { .. })
    ));
    assert!(matches!(
        tools
            .namespace("docs", "Documentation tools")
            .unwrap()
            .register(Search),
        Err(ToolError::Duplicate { .. })
    ));
    struct Docs;
    impl FunctionTool for Docs {
        type Input = Value;
        type Output = Value;
        fn name(&self) -> &str {
            "docs"
        }
        fn description(&self) -> &str {
            "Clashes with the namespace"
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn execute(&self, input: Value) -> Result<Value> {
            Ok(input)
        }
    }
    assert!(matches!(
        tools.register(Docs),
        Err(ToolError::Duplicate { .. })
    ));
    assert!(matches!(
        tools.namespace("bad name", "x"),
        Err(ToolError::InvalidDefinition { .. })
    ));
}

#[tokio::test]
async fn dynamic_tools_are_sanitized() {
    let mut tools = ToolRegistry::new();
    let name = tools
        .register_dynamic(
            DynamicTool::new(
                "create-issue",
                "",
                json!({"properties": {"title": {"const": "bug"}, "labels": true}}),
                |arguments: Value| async move {
                    Ok(ToolOutputBody::text(format!(
                        "created {}",
                        arguments["title"]
                    )))
                },
            )
            .prefix("github")
            .supports_parallel(true)
            .untrusted_output(true),
        )
        .unwrap();
    assert_eq!(name, "github__create_issue");
    assert_eq!(tools.raw_name(&name), Some("create-issue"));
    let ToolDefinition::Function {
        description,
        parameters,
        ..
    } = &tools.definitions()[0]
    else {
        panic!("a function expected");
    };
    assert_eq!(description, "create-issue");
    assert_eq!(parameters["type"], "object");
    assert_eq!(
        parameters["properties"]["title"],
        json!({"type": "string", "enum": ["bug"]})
    );
    assert_eq!(
        parameters["properties"]["labels"],
        json!({"type": "string"})
    );

    let answer = tools
        .answer_call(ToolCall::function("c1", &name, r#"{"title":"x"}"#))
        .await
        .unwrap();
    assert_eq!(answer.body.as_text(), Some(r#"created "x""#));
    assert!(
        answer.untrusted,
        "the tool says its output comes from outside"
    );
    assert!(tools.supports_parallel(&ToolCall::function("c", &name, "{}")));
    assert_eq!(
        tools.execute(&name, r#"{"title":"y"}"#).await.unwrap(),
        json!("created \"y\"")
    );

    // The same raw name again gets another name.
    let again = tools
        .register_dynamic(
            DynamicTool::new(
                "create-issue",
                "Create",
                json!({"type": "object"}),
                |_| async { Ok(ToolOutputBody::text("again")) },
            )
            .prefix("github"),
        )
        .unwrap();
    assert_ne!(again, name);
    assert!(again.starts_with("github__create_issue_"));

    // An oversized schema falls back to any object.
    let properties: serde_json::Map<String, Value> = (0..400)
        .map(|i| {
            (
                format!("field_{i}"),
                json!({"anyOf": [{"type": "string"}, {"type": "integer"}]}),
            )
        })
        .collect();
    let big = tools
        .register_dynamic(DynamicTool::new(
            "big",
            "Big",
            json!({"type": "object", "properties": properties}),
            |_| async { Ok(ToolOutputBody::text("ok")) },
        ))
        .unwrap();
    let parameters = tools
        .definitions()
        .into_iter()
        .find_map(|definition| match definition {
            ToolDefinition::Function {
                name, parameters, ..
            } if name == big => Some(parameters),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        parameters,
        json!({"type": "object", "properties": {}, "additionalProperties": true})
    );

    assert!(matches!(
        tools.register_dynamic(DynamicTool::new(
            "list",
            "x",
            json!({"type": "array"}),
            |_| async { Ok(ToolOutputBody::text("")) }
        )),
        Err(ToolError::InvalidDefinition { .. })
    ));
}

#[test]
fn truncation_uses_the_tool_budget() {
    let call = ToolCall::function("c", "get_weather", "{}");
    let answer = ToolOutput::new(&call, "x".repeat(300), true);
    let config = TruncationConfig::default();
    let budget = config.budget_for(Some(OutputBudget::Bytes(100)));
    let cut = answer.truncated(budget, config.header);
    let text = cut.body.as_text().unwrap();
    assert!(text.starts_with("[truncated: original 300 bytes/~75 tokens, 1 lines]\n"));
    assert!(text.contains("…200 chars truncated…"));
    assert_eq!(
        ToolOutput::aborted(&call).body.as_text(),
        Some(ABORTED_OUTPUT)
    );
}

#[cfg(feature = "schemars")]
#[tokio::test]
async fn a_schema_from_the_input_type() {
    use genai_agents::tools::schema;

    /// Where to look.
    #[derive(Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct Lookup {
        /// The city.
        city: String,
        /// The unit: celsius or fahrenheit.
        unit: Option<Unit>,
    }

    #[derive(Deserialize, schemars::JsonSchema)]
    #[serde(rename_all = "lowercase")]
    enum Unit {
        Celsius,
        Fahrenheit,
    }

    struct Typed;
    impl FunctionTool for Typed {
        type Input = Lookup;
        type Output = String;
        fn name(&self) -> &str {
            "lookup"
        }
        fn description(&self) -> &str {
            "Look a city up"
        }
        fn parameters_schema(&self) -> Value {
            schema::for_type::<Lookup>()
        }
        async fn execute(&self, input: Lookup) -> Result<String> {
            let unit = match input.unit {
                Some(Unit::Fahrenheit) => "F",
                Some(Unit::Celsius) | None => "C",
            };
            Ok(format!("{} in {unit}", input.city))
        }
    }

    let schema = schema::for_type::<Lookup>();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["required"], json!(["city"]));
    assert_eq!(schema["additionalProperties"], false);
    assert!(schema.get("$schema").is_none());
    assert!(schema.get("$defs").is_none(), "definitions are inlined");
    assert_eq!(schema["properties"]["city"]["description"], "The city.");
    let mut tools = ToolRegistry::new();
    tools.register(Typed).unwrap();
    assert_eq!(
        tools
            .execute("lookup", r#"{"city":"Ghent","unit":"fahrenheit"}"#)
            .await
            .unwrap(),
        "Ghent in F"
    );
}
