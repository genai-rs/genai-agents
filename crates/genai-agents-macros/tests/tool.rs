//! `#[tool]` functions registered in a `ToolRegistry` and called through it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{tool, Error, ToolCall, ToolDefinition};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Units of temperature.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Units {
    /// Degrees Celsius.
    Celsius,
    /// Degrees Fahrenheit.
    Fahrenheit,
}

/// The weather in a city.
#[derive(Debug, Serialize)]
pub struct Weather {
    city: String,
    temperature: f64,
}

/// Current weather of a city.
///
/// Use it before suggesting what to wear.
#[tool]
pub async fn get_weather(
    /// The city, such as "Ghent".
    city: String,
    /// The units of the temperature, Celsius by default.
    units: Option<Units>,
) -> genai_agents::Result<Weather> {
    let temperature = match units.unwrap_or(Units::Celsius) {
        Units::Celsius => 12.0,
        Units::Fahrenheit => 53.6,
    };
    Ok(Weather { city, temperature })
}

/// Fails on purpose.
#[tool(name = "always_fails", parallel, untrusted)]
async fn failing(
    /// Ignored.
    #[serde(default)]
    reason: String,
) -> genai_agents::Result<()> {
    Err(Error::application(std::io::Error::other(format!(
        "no luck{reason}"
    ))))
}

use genai_agents::Result;

#[tool(description = "The answer to everything.")]
async fn r#answer() -> Result<u32> {
    Ok(42)
}

fn function(definition: &ToolDefinition) -> (&str, &str, &serde_json::Value) {
    match definition {
        ToolDefinition::Function {
            name,
            description,
            parameters,
            ..
        } => (name, description, parameters),
        other => panic!("not a function tool: {other:?}"),
    }
}

#[test]
fn definition_from_the_function() {
    let mut tools = ToolRegistry::new();
    tools.register(GetWeather).unwrap();
    let definitions = tools.definitions();
    let (name, description, parameters) = function(&definitions[0]);
    assert_eq!(name, "get_weather");
    assert_eq!(
        description,
        "Current weather of a city.\n\nUse it before suggesting what to wear."
    );
    assert_eq!(parameters["type"], "object");
    assert_eq!(parameters["required"], json!(["city"]));
    assert_eq!(parameters["additionalProperties"], json!(false));
    assert_eq!(
        parameters["properties"]["city"],
        json!({"type": "string", "description": "The city, such as \"Ghent\"."})
    );
    assert_eq!(
        parameters["properties"]["units"]["description"],
        "The units of the temperature, Celsius by default."
    );
    assert!(!GetWeather.supports_parallel());
    assert!(!GetWeather.untrusted_output());
}

#[test]
fn options() {
    assert_eq!(Failing.name(), "always_fails");
    assert!(Failing.supports_parallel());
    assert!(Failing.untrusted_output());
    // `#[serde(default)]` went to the field: not required.
    assert_eq!(Failing.parameters_schema().get("required"), None);

    assert_eq!(Answer.name(), "answer");
    assert_eq!(Answer.description(), "The answer to everything.");
    assert_eq!(
        Answer.parameters_schema(),
        json!({"type": "object", "properties": {}, "additionalProperties": false})
    );
}

#[tokio::test(flavor = "current_thread")]
async fn calls_through_the_registry() {
    let mut tools = ToolRegistry::new();
    tools.register(GetWeather).unwrap();
    tools.register(Failing).unwrap();
    tools.register(Answer).unwrap();

    let answer = tools
        .answer_call(ToolCall::function(
            "call_1",
            "get_weather",
            r#"{"city":"Ghent","units":"fahrenheit"}"#,
        ))
        .await
        .unwrap();
    assert!(answer.success);
    let output: serde_json::Value = serde_json::from_str(answer.body.as_text().unwrap()).unwrap();
    assert_eq!(output, json!({"city": "Ghent", "temperature": 53.6}));

    // An optional parameter left out.
    let answer = tools
        .answer_call(ToolCall::function(
            "call_2",
            "get_weather",
            r#"{"city":"Ghent"}"#,
        ))
        .await
        .unwrap();
    assert!(answer.success);

    // A missing and an unknown field: answered as failures, without running the tool.
    for arguments in [r#"{}"#, r#"{"city":"Ghent","country":"BE"}"#] {
        let answer = tools
            .answer_call(ToolCall::function("call_3", "get_weather", arguments))
            .await
            .unwrap();
        assert!(!answer.success, "{arguments}");
    }

    let answer = tools
        .answer_call(ToolCall::function("call_4", "always_fails", "{}"))
        .await
        .unwrap();
    assert!(!answer.success);
    assert_eq!(answer.body.as_text(), Some("no luck"));

    let answer = tools
        .answer_call(ToolCall::function("call_5", "answer", "{}"))
        .await
        .unwrap();
    assert_eq!(answer.body.as_text(), Some("42"));
}

#[tokio::test(flavor = "current_thread")]
async fn the_function_stays_callable() {
    let weather = get_weather("Ghent".to_owned(), None).await.unwrap();
    assert_eq!(weather.city, "Ghent");
}
