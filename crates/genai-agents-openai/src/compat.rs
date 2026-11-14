//! [`Compat`]: what a server that speaks Chat Completions takes.

use openai_ergonomic::ProviderKind;

/// The field a Chat Completions request puts its output limit in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MaxTokensField {
    /// `max_tokens`, the older name, which most compatible servers take.
    MaxTokens,
    /// `max_completion_tokens`, OpenAI's name since its reasoning models.
    MaxCompletionTokens,
}

impl MaxTokensField {
    /// The field's name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

/// What a server that speaks Chat Completions takes. "OpenAI-compatible" differs per server:
/// [`OpenAiChat`](crate::OpenAiChat) leaves out or renames what its server does not take.
///
/// [`OpenAiChat::new`](crate::OpenAiChat::new) picks the preset of the client's provider
/// ([`Compat::for_provider`]); [`OpenAiChat::with_compat`](crate::OpenAiChat::with_compat) sets
/// another. The presets of the compatible providers are cautious: a server that takes more can
/// be given a `Compat` that says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(clippy::struct_excessive_bools)] // independent capabilities
pub struct Compat {
    /// Whether it takes `developer` messages; otherwise they are sent as `system` messages.
    pub supports_developer_role: bool,
    /// Whether it takes `reasoning_effort`; otherwise the request's reasoning level is left out.
    pub supports_reasoning_effort: bool,
    /// Whether it reports the usage at the end of a stream (`stream_options.include_usage`).
    /// `OpenAiChat` does not stream yet, so this is for a later version.
    pub supports_usage_in_streaming: bool,
    /// The field of the output limit.
    pub max_tokens_field: MaxTokensField,
    /// Whether a tool message must name its tool (`name`).
    pub requires_tool_result_name: bool,
    /// Whether it takes `store`; otherwise a template's `store` is left out.
    pub supports_store: bool,
}

impl Compat {
    /// OpenAI: everything.
    #[must_use]
    pub fn openai() -> Self {
        Self {
            supports_developer_role: true,
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            requires_tool_result_name: false,
            supports_store: true,
        }
    }

    /// Azure OpenAI: as OpenAI.
    #[must_use]
    pub fn azure() -> Self {
        Self::openai()
    }

    /// OpenRouter: `max_tokens`, `system` messages; reasoning goes through its own `reasoning`
    /// object (set it on the template), not `reasoning_effort`.
    #[must_use]
    pub fn openrouter() -> Self {
        Self {
            supports_usage_in_streaming: true,
            ..Self::cautious()
        }
    }

    /// Groq: `max_completion_tokens`, `system` messages.
    #[must_use]
    pub fn groq() -> Self {
        Self {
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            ..Self::cautious()
        }
    }

    /// Ollama: `max_tokens`, `system` messages, usage in streams.
    #[must_use]
    pub fn ollama() -> Self {
        Self {
            supports_usage_in_streaming: true,
            ..Self::cautious()
        }
    }

    /// Any other server: only what every compatible server takes (`system` messages,
    /// `max_tokens`, no `reasoning_effort`, no `store`).
    #[must_use]
    pub fn cautious() -> Self {
        Self {
            supports_developer_role: false,
            supports_reasoning_effort: false,
            supports_usage_in_streaming: false,
            max_tokens_field: MaxTokensField::MaxTokens,
            requires_tool_result_name: false,
            supports_store: false,
        }
    }

    /// The preset of a provider of openai-ergonomic.
    #[must_use]
    pub fn for_provider(kind: ProviderKind) -> Self {
        match kind {
            ProviderKind::OpenAi => Self::openai(),
            ProviderKind::Azure => Self::azure(),
            ProviderKind::OpenRouter => Self::openrouter(),
            ProviderKind::Groq => Self::groq(),
            ProviderKind::Ollama => Self::ollama(),
            _ => Self::cautious(),
        }
    }

    /// The same, with or without `developer` messages.
    #[must_use]
    pub fn with_developer_role(mut self, supported: bool) -> Self {
        self.supports_developer_role = supported;
        self
    }

    /// The same, with or without `reasoning_effort`.
    #[must_use]
    pub fn with_reasoning_effort(mut self, supported: bool) -> Self {
        self.supports_reasoning_effort = supported;
        self
    }

    /// The same, with or without the usage at the end of a stream.
    #[must_use]
    pub fn with_usage_in_streaming(mut self, supported: bool) -> Self {
        self.supports_usage_in_streaming = supported;
        self
    }

    /// The same, with this field for the output limit.
    #[must_use]
    pub fn with_max_tokens_field(mut self, field: MaxTokensField) -> Self {
        self.max_tokens_field = field;
        self
    }

    /// The same, with or without the tool's name on tool messages.
    #[must_use]
    pub fn with_tool_result_name(mut self, required: bool) -> Self {
        self.requires_tool_result_name = required;
        self
    }

    /// The same, with or without `store`.
    #[must_use]
    pub fn with_store(mut self, supported: bool) -> Self {
        self.supports_store = supported;
        self
    }
}

impl Default for Compat {
    /// OpenAI's.
    fn default() -> Self {
        Self::openai()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_provider_has_its_preset() {
        assert_eq!(Compat::for_provider(ProviderKind::OpenAi), Compat::openai());
        assert_eq!(Compat::for_provider(ProviderKind::Azure), Compat::openai());
        assert_eq!(
            Compat::for_provider(ProviderKind::Groq).max_tokens_field,
            MaxTokensField::MaxCompletionTokens
        );
        assert!(!Compat::for_provider(ProviderKind::Ollama).supports_developer_role);
        assert!(Compat::for_provider(ProviderKind::OpenRouter).supports_usage_in_streaming);
        assert_eq!(
            Compat::for_provider(ProviderKind::Custom),
            Compat::cautious()
        );
        assert_eq!(Compat::default(), Compat::openai());
        assert_eq!(MaxTokensField::MaxTokens.as_str(), "max_tokens");
    }
}
