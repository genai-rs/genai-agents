//! Errors: of a run ([`Error`]) and of a model request ([`ModelError`]).

use std::time::Duration;

use crate::tools::ToolError;

/// A boxed error that can cross threads.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// The result of a run, and of a tool handler.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Why a run failed, or why a tool handler did.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A model request failed and was not retried, or still failed after its retries.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A tool failed in a way the model cannot act on
    /// ([`ToolFailure::Fatal`](crate::tools::ToolFailure::Fatal)), or a tool could not be
    /// registered.
    #[error(transparent)]
    Tool(Box<ToolError>),
    /// An error of the application, with its source kept: a tool handler returns it with
    /// [`Error::application`], and the model sees its message.
    #[error("{0}")]
    Application(#[source] BoxError),
    /// A failure that must stop the run instead of being reported to the model: a tool handler
    /// returns it with [`Error::fatal`].
    #[error("fatal error: {0}")]
    Fatal(#[source] BoxError),
    /// JSON that could not be encoded or decoded.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<ToolError> for Error {
    fn from(error: ToolError) -> Self {
        Self::Tool(Box::new(error))
    }
}

impl Error {
    /// An application error, with its source kept. Returned by a tool handler, its message is
    /// what the model sees.
    pub fn application(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Application(Box::new(error))
    }

    /// A failure that must stop the run: returned by a tool handler, it ends
    /// [`ToolRegistry::answer_call`](crate::tools::ToolRegistry::answer_call) with an error
    /// instead of an answer for the model.
    pub fn fatal(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Fatal(Box::new(error))
    }

    /// Whether this is an [`Error::Fatal`].
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::Fatal(_))
    }

    /// The model error, if this is one.
    #[must_use]
    pub fn as_model(&self) -> Option<&ModelError> {
        match self {
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

/// How a failed model request may be retried ([`ModelError::retry_class`]).
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RetryClass {
    /// Retrying will not help: a client error, an exhausted quota, a context that is too long.
    Never,
    /// Retry after the delay the server asked for.
    ServerDirected {
        /// The delay before the next attempt.
        after: Duration,
    },
    /// Retry with exponential backoff: a transient failure (a rate limit without a known delay,
    /// a server error, a cut-off stream, a connection failure).
    Backoff,
}

impl RetryClass {
    /// Whether a retry may help.
    #[must_use]
    pub fn is_retryable(self) -> bool {
        !matches!(self, Self::Never)
    }
}

/// What kind of failure a [`ModelError`] is, for what to do about it beyond a retry.
///
/// New variants may be added in minor releases: match with a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The request is too long for the model's context window.
    ContextOverflow,
    /// The model does not exist, or this account cannot use it.
    ModelNotFound,
    /// The credentials were refused (status 401 or 403).
    Authentication,
    /// A rate limit or an exhausted quota (status 429).
    RateLimit,
    /// The provider failed (status 5xx, or a failed response).
    Server,
    /// The response stopped, or the stream ended, before it was complete.
    Interrupted,
    /// Nothing came from the provider: the request could not be built or sent, or its answer
    /// could not be read.
    Local,
    /// Anything else, such as a request the provider rejected.
    Other,
}

/// A model request that failed, with what a retry needs to know.
///
/// A provider builds it from its own error, which stays available as the
/// [source](std::error::Error::source) ([`provider_error`](Self::provider_error)). The runner
/// retries from the rest:
///
/// - the [`retry_class`](Self::retry_class): whether a retry may help, and after which delay
///   when the error itself says so;
/// - the [`retry_after`](Self::retry_after) delay the response's headers asked for;
/// - whether it failed [before the body](Self::is_before_body) (a refused connection, an error
///   status): the provider's HTTP layer may have retried those already
///   ([`Model::retries_before_body`](crate::Model::retries_before_body)).
#[derive(Debug)]
pub struct ModelError(Box<Inner>);

#[derive(Debug)]
struct Inner {
    message: String,
    kind: ErrorKind,
    retry: RetryClass,
    retry_after: Option<Duration>,
    before_body: bool,
    status: Option<u16>,
    code: Option<String>,
    source: Option<BoxError>,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.message)
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0
            .source
            .as_deref()
            .map(|source| source as &(dyn std::error::Error + 'static))
    }
}

impl ModelError {
    /// An error with this message that is not retried.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self(Box::new(Inner {
            message: message.into(),
            kind: ErrorKind::Other,
            retry: RetryClass::Never,
            retry_after: None,
            before_body: false,
            status: None,
            code: None,
            source: None,
        }))
    }

    /// An error of the provider, with its message; not retried until
    /// [`with_retry_class`](Self::with_retry_class) says otherwise.
    pub fn from_source(source: impl std::error::Error + Send + Sync + 'static) -> Self {
        let mut error = Self::new(source.to_string());
        error.0.source = Some(Box::new(source));
        error
    }

    /// A stream that ended before its response did: retried with backoff.
    #[must_use]
    pub fn stream_ended() -> Self {
        Self::new("the stream ended before its response did")
            .with_retry_class(RetryClass::Backoff)
            .with_kind(ErrorKind::Interrupted)
    }

    /// The same error, of this kind.
    #[must_use]
    pub fn with_kind(mut self, kind: ErrorKind) -> Self {
        self.0.kind = kind;
        self
    }

    /// What kind of failure it is.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }

    /// The same error, retried as `retry` says.
    #[must_use]
    pub fn with_retry_class(mut self, retry: RetryClass) -> Self {
        self.0.retry = retry;
        self
    }

    /// The same error, with the delay the response's headers asked for before a retry
    /// (`retry-after`).
    #[must_use]
    pub fn with_retry_after(mut self, after: Option<Duration>) -> Self {
        self.0.retry_after = after;
        self
    }

    /// The same error, failed before any of the body arrived (or not).
    #[must_use]
    pub fn with_before_body(mut self, before_body: bool) -> Self {
        self.0.before_body = before_body;
        self
    }

    /// The same error, with the HTTP status of the response.
    #[must_use]
    pub fn with_status(mut self, status: u16) -> Self {
        self.0.status = Some(status);
        self
    }

    /// The same error, with the provider's error code (such as `rate_limit_exceeded`).
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.0.code = Some(code.into());
        self
    }

    /// The message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.0.message
    }

    /// How the request may be retried.
    #[must_use]
    pub fn retry_class(&self) -> RetryClass {
        self.0.retry
    }

    /// The delay the response's headers asked for before a retry.
    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        self.0.retry_after
    }

    /// Whether the request failed before any of the body arrived: the connection, or an error
    /// status.
    #[must_use]
    pub fn is_before_body(&self) -> bool {
        self.0.before_body
    }

    /// The HTTP status of the response, if there was one.
    #[must_use]
    pub fn status(&self) -> Option<u16> {
        self.0.status
    }

    /// The provider's error code, if it sent one.
    #[must_use]
    pub fn code(&self) -> Option<&str> {
        self.0.code.as_deref()
    }

    /// Whether this is a rate limit: of [`ErrorKind::RateLimit`], status 429, or the code
    /// `rate_limit_exceeded`.
    #[must_use]
    pub fn is_rate_limit(&self) -> bool {
        self.0.kind == ErrorKind::RateLimit
            || self.0.status == Some(429)
            || self.code() == Some("rate_limit_exceeded")
    }

    /// Whether this is a server error (status 5xx).
    #[must_use]
    pub fn is_server_error(&self) -> bool {
        self.0
            .status
            .is_some_and(|status| (500..600).contains(&status))
    }

    /// The provider's own error, if it is an `E`.
    #[must_use]
    pub fn provider_error<E: std::error::Error + 'static>(&self) -> Option<&E> {
        self.0.source.as_deref()?.downcast_ref::<E>()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn model_errors_keep_their_source() {
        let error = ModelError::from_source(std::io::Error::other("refused"))
            .with_retry_class(RetryClass::Backoff)
            .with_before_body(true)
            .with_status(503);
        assert_eq!(error.to_string(), "refused");
        assert!(error.retry_class().is_retryable());
        assert!(error.is_before_body());
        assert!(error.is_server_error());
        assert!(!error.is_rate_limit());
        assert_eq!(
            error.provider_error::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::Other
        );
        let error = Error::from(error);
        assert!(error.as_model().is_some());
        assert!(!error.is_fatal());
    }
}
