//! The configuration of a [`Runner`](super::Runner).

use std::num::NonZeroU32;
use std::time::Duration;

use crate::model::ModelSettings;
use crate::tools::TruncationConfig;

/// The `prompt_cache_key` the runner sends with every request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum PromptCacheKey {
    /// A key made for each [`Runner`](super::Runner), the same for all its requests, so that
    /// they share the prompt cache.
    #[default]
    PerRunner,
    /// This key.
    Custom(String),
    /// No key.
    Omit,
}

/// How the runner retries a model request that failed, as a whole.
///
/// A connection that failed, a stream cut off or failed midway, a rate limit, a server error:
/// only errors whose [`retry_class`](crate::ModelError::retry_class) allows it are retried.
///
/// The delay is the one the server asked for (a `retry-after` header, or a "try again in"
/// message), else `base_delay · 2^(attempt − 1)` with ±10 % jitter, at most `max_delay`.
///
/// A failure before any of the body (a connection, a rate limit or server error status) is
/// retried here only when the provider does not retry it itself
/// ([`Model::retries_before_body`](crate::Model::retries_before_body)), so a request is not
/// retried by both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct RetryConfig {
    /// Retries of one request after its first attempt. Default 5; 0 disables them.
    pub max_retries: u32,
    /// The first backoff delay. Default 200 ms.
    pub base_delay: Duration,
    /// The longest delay, server-directed ones included. Default 60 s.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(60),
        }
    }
}

impl RetryConfig {
    /// At most `max_retries` retries, with the default delays.
    #[must_use]
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Self::default()
        }
    }

    /// No retries.
    #[must_use]
    pub fn none() -> Self {
        Self::new(0)
    }

    /// The first backoff delay.
    #[must_use]
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// The longest delay.
    #[must_use]
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// The backoff delay before retry `attempt` (1 for the first retry), with `jitter` in
    /// `[-1, 1]` scaling ±10 %.
    pub(crate) fn backoff(&self, attempt: u32, jitter: f64) -> Duration {
        let exponent = attempt.saturating_sub(1).min(30);
        let delay = self.base_delay.saturating_mul(1_u32 << exponent);
        let factor = jitter.clamp(-1.0, 1.0).mul_add(0.1, 1.0);
        delay.mul_f64(factor).min(self.max_delay)
    }
}

/// The runner's own tool search over deferred tools.
///
/// A `tool_search` function the model calls to find
/// [deferred](crate::tools::Exposure::Deferred) tools, answered with
/// [`ToolRegistry::search_deferred`](crate::tools::ToolRegistry::search_deferred). The tools
/// found are advertised from the next request on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ToolSearchConfig {
    /// The most tools one search loads. Default 8.
    pub limit: usize,
    /// The description of the `tool_search` tool the model sees.
    pub description: String,
}

impl Default for ToolSearchConfig {
    fn default() -> Self {
        Self {
            limit: 8,
            description: "Search for more tools by what they do. The tools found can be called \
                          from the next step on."
                .to_string(),
        }
    }
}

impl ToolSearchConfig {
    /// The default search, with at most `limit` tools per search.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            ..Self::default()
        }
    }

    /// The description of the `tool_search` tool.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

/// The configuration of a [`Runner`](super::Runner).
///
/// The defaults: one prompt cache key per runner, parallel tool calls, at most 16 requests per
/// run, 5 retries, the runner's own tool search.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RunConfig {
    /// The most model requests of one run (retries not counted); `None` for no limit. When the
    /// model still calls tools after the last one, the run ends with
    /// [`RunStop::MaxIterations`](super::RunStop::MaxIterations). Default 16.
    pub max_iterations: Option<NonZeroU32>,
    /// The `prompt_cache_key` of the requests. Default [`PromptCacheKey::PerRunner`].
    pub prompt_cache_key: PromptCacheKey,
    /// Whether the model may call several tools at once. The runner runs the calls of tools that
    /// allow it ([`FunctionTool::supports_parallel`](crate::tools::FunctionTool::supports_parallel))
    /// at the same time and the others alone, and answers in call order. Default `true`.
    pub parallel_tool_calls: bool,
    /// How much of a tool's output the model sees. Default 10 000 bytes, with a header.
    pub truncation: TruncationConfig,
    /// The retries of a failed request. Default 5 retries from 200 ms.
    pub retry: RetryConfig,
    /// The runner's own tool search over deferred tools; `None` to offer none (deferred tools are
    /// then only advertised once [loaded](super::Runner::load_tool)). Default on.
    pub tool_search: Option<ToolSearchConfig>,
    /// The settings every request carries: the reasoning level, the output limit. Default: the
    /// provider's.
    pub model_settings: ModelSettings,
    /// The agent's name in traces (`invoke_agent {name}`, `gen_ai.agent.name`). Default none.
    pub name: Option<String>,
    /// Whether the spans record the content: the messages of each request and response, and the
    /// arguments and results of tool calls (`gen_ai.input.messages`, `gen_ai.output.messages`,
    /// `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`). They can hold personal or
    /// confidential data, so OpenTelemetry's GenAI conventions advise leaving them out unless
    /// asked for. Default `false`.
    pub trace_content: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_iterations: NonZeroU32::new(16),
            prompt_cache_key: PromptCacheKey::PerRunner,
            parallel_tool_calls: true,
            truncation: TruncationConfig::default(),
            retry: RetryConfig::default(),
            tool_search: Some(ToolSearchConfig::default()),
            model_settings: ModelSettings::default(),
            name: None,
            trace_content: false,
        }
    }
}

impl RunConfig {
    /// The defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// At most `max` model requests per run; 0 for no limit.
    #[must_use]
    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = NonZeroU32::new(max);
        self
    }

    /// The prompt cache key.
    #[must_use]
    pub fn with_prompt_cache_key(mut self, key: PromptCacheKey) -> Self {
        self.prompt_cache_key = key;
        self
    }

    /// Whether the model may call several tools at once.
    #[must_use]
    pub fn with_parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = parallel;
        self
    }

    /// The truncation of tool outputs.
    #[must_use]
    pub fn with_truncation(mut self, truncation: TruncationConfig) -> Self {
        self.truncation = truncation;
        self
    }

    /// The retries of failed requests.
    #[must_use]
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// The runner's own tool search; `None` to offer none.
    #[must_use]
    pub fn with_tool_search(mut self, search: Option<ToolSearchConfig>) -> Self {
        self.tool_search = search;
        self
    }

    /// The settings every request carries.
    #[must_use]
    pub fn with_model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = settings;
        self
    }

    /// The agent's name in traces.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Whether the spans record the content of messages and tool calls.
    #[must_use]
    pub fn with_trace_content(mut self, trace_content: bool) -> Self {
        self.trace_content = trace_content;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_and_is_capped() {
        let retry = RetryConfig::default().with_max_delay(Duration::from_secs(1));
        assert_eq!(retry.backoff(1, 0.0), Duration::from_millis(200));
        assert_eq!(retry.backoff(2, 0.0), Duration::from_millis(400));
        assert_eq!(retry.backoff(3, 1.0), Duration::from_millis(880));
        assert_eq!(retry.backoff(4, 0.0), Duration::from_secs(1));
        assert_eq!(retry.backoff(100, -1.0), Duration::from_secs(1));
    }
}
