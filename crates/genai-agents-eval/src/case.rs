//! Cases and datasets.

use genai_agents::{Conversation, Item};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One thing to try: an input, optionally what the answer should be, and data of its own for
/// the scorers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Case {
    /// Its id, unique in its dataset.
    pub id: String,
    /// The conversation the target runs on.
    pub input: Conversation,
    /// What the answer should be, for the scorers that compare: a text, or JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<Value>,
    /// Anything else the scorers or the target need.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl Case {
    /// A case with this input.
    #[must_use]
    pub fn new(id: impl Into<String>, input: impl Into<Conversation>) -> Self {
        Self {
            id: id.into(),
            input: input.into(),
            expected: None,
            metadata: Value::Null,
        }
    }

    /// A case whose input is one user message.
    #[must_use]
    pub fn user(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::new(id, Conversation::user(text))
    }

    /// The same case, expecting this answer.
    #[must_use]
    pub fn with_expected(mut self, expected: impl Into<Value>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    /// The same case, with this metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// The text of the expected answer, when it is a string.
    #[must_use]
    pub fn expected_text(&self) -> Option<&str> {
        self.expected.as_ref().and_then(Value::as_str)
    }

    /// The input of a case from JSON: a string is a user message; an array is a conversation of
    /// items; an object with `messages` (`[{"role", "content"}]`) is a conversation of those
    /// messages.
    ///
    /// # Errors
    ///
    /// When the JSON is none of these.
    pub fn input_from_json(input: &Value) -> Result<Conversation, crate::EvalError> {
        match input {
            Value::String(text) => Ok(Conversation::user(text.clone())),
            Value::Array(_) => Ok(serde_json::from_value(input.clone())?),
            Value::Object(object) => {
                let Some(Value::Array(messages)) = object.get("messages") else {
                    return Err(crate::EvalError::Input(
                        "an object input needs `messages`".into(),
                    ));
                };
                let mut conversation = Conversation::new();
                for message in messages {
                    let role = message["role"].as_str().unwrap_or("user");
                    let content = message["content"].as_str().unwrap_or_default();
                    let role = match role {
                        "system" => genai_agents::Role::System,
                        "developer" => genai_agents::Role::Developer,
                        "assistant" => genai_agents::Role::Assistant,
                        _ => genai_agents::Role::User,
                    };
                    conversation.push(Item::Message(genai_agents::Message::text(role, content)));
                }
                Ok(conversation)
            }
            _ => Err(crate::EvalError::Input(format!(
                "an input is a string, an array of items or an object with messages, not {input}"
            ))),
        }
    }
}

/// A named list of cases.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Dataset {
    /// Its name, in reports.
    pub name: String,
    /// The cases.
    pub cases: Vec<Case>,
}

impl Dataset {
    /// A dataset of these cases.
    #[must_use]
    pub fn new(name: impl Into<String>, cases: Vec<Case>) -> Self {
        Self {
            name: name.into(),
            cases,
        }
    }

    /// A dataset from [JSON Lines](https://jsonlines.org), a case per line:
    /// `{"id": …, "input": …, "expected": …, "metadata": …}`, where the input is anything
    /// [`Case::input_from_json`] takes. A line without an id gets its line number.
    ///
    /// # Errors
    ///
    /// When a line is not such a case.
    pub fn from_jsonl(name: impl Into<String>, text: &str) -> Result<Self, crate::EvalError> {
        let mut cases = Vec::new();
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line)?;
            let id = match &value["id"] {
                Value::String(id) => id.clone(),
                Value::Number(id) => id.to_string(),
                _ => (number + 1).to_string(),
            };
            let input = Case::input_from_json(&value["input"])?;
            let mut case = Case::new(id, input);
            case.expected = value
                .get("expected")
                .filter(|value| !value.is_null())
                .cloned();
            case.metadata = value.get("metadata").cloned().unwrap_or(Value::Null);
            cases.push(case);
        }
        Ok(Self::new(name, cases))
    }

    /// Read a dataset from a JSON Lines file, named after the file.
    ///
    /// # Errors
    ///
    /// When it cannot be read or a line is not a case.
    #[cfg(not(target_family = "wasm"))]
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, crate::EvalError> {
        let path = path.as_ref();
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(path).map_err(crate::EvalError::io)?;
        Self::from_jsonl(name, &text)
    }
}
