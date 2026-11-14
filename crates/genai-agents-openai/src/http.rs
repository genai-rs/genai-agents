//! Sending a request through the client, the way its own calls go (its interceptors, its HTTP
//! stack with the retries and the credentials), and keeping what the response said: the rate
//! limits, and for a failure the delay the server asked for.

use std::collections::{BTreeSet, VecDeque};
use std::time::Duration;

use futures::StreamExt;
use genai_agents::{
    ErrorKind, ModelError, ModelEvent, ModelResponse, RateLimitWindow, RateLimits, RetryClass,
    StopReason, Usage,
};
use openai_ergonomic::{CallFailure, Client, Error, JsonEventStream, RateLimitSnapshot};
use serde_json::Value;

use crate::convert;

/// A [`ModelError`] from openai-ergonomic's error: its retry class, its status and code,
/// whether it came before the body, and the delay the failed response asked for.
pub(crate) fn model_error(
    error: Error,
    before_body: bool,
    retry_after: Option<Duration>,
) -> ModelError {
    let retry = match error.retry_class() {
        openai_ergonomic::RetryClass::ServerDirected { after } => {
            RetryClass::ServerDirected { after }
        }
        openai_ergonomic::RetryClass::Backoff => RetryClass::Backoff,
        _ => RetryClass::Never,
    };
    let (status, code) = match &error {
        Error::Api {
            status, error_code, ..
        } => (Some(*status), error_code.clone()),
        Error::ResponseFailed { code, .. } => (None, code.clone()),
        _ => (None, None),
    };
    let kind = error_kind(&error, status, code.as_deref());
    let mut model = ModelError::from_source(error)
        .with_kind(kind)
        .with_retry_class(retry)
        .with_retry_after(retry_after)
        .with_before_body(before_body);
    if let Some(status) = status {
        model = model.with_status(status);
    }
    if let Some(code) = code {
        model = model.with_code(code);
    }
    model
}

/// A [`ModelError`] from a failed call: its error, whether it came before the body, and the
/// delay its response asked for.
pub(crate) fn call_failure(failure: CallFailure) -> ModelError {
    let before_body = failure.is_before_body();
    let retry_after = failure.meta().retry_after();
    model_error(failure.into_error(), before_body, retry_after)
}

/// The kind of an openai-ergonomic error.
fn error_kind(error: &Error, status: Option<u16>, code: Option<&str>) -> ErrorKind {
    if error.is_context_length_exceeded() {
        return ErrorKind::ContextOverflow;
    }
    match (status, code) {
        (_, Some("model_not_found")) | (Some(404), _) => ErrorKind::ModelNotFound,
        (Some(401 | 403), _) | (_, Some("invalid_api_key")) => ErrorKind::Authentication,
        (Some(429), _) | (_, Some("rate_limit_exceeded" | "insufficient_quota")) => {
            ErrorKind::RateLimit
        }
        (Some(500..=599), _) => ErrorKind::Server,
        _ => match error {
            Error::ResponseFailed { .. } => ErrorKind::Server,
            Error::StreamConnection { .. } => ErrorKind::Interrupted,
            Error::Http(error) if error.is_body() => ErrorKind::Interrupted,
            Error::Http(_)
            | Error::Json(_)
            | Error::StreamParsing { .. }
            | Error::InvalidRequest(_)
            | Error::Builder(_)
            | Error::Config(_) => ErrorKind::Local,
            _ => ErrorKind::Other,
        },
    }
}

/// The rate limits a response reported (`x-ratelimit-*`).
pub(crate) fn rate_limits(snapshot: Option<RateLimitSnapshot>) -> Option<RateLimits> {
    let snapshot = snapshot?;
    let window = |window: Option<openai_ergonomic::RateLimitWindow>| {
        window.map(|window| RateLimitWindow::new(window.limit, window.remaining, window.reset))
    };
    Some(RateLimits::new(
        window(snapshot.requests),
        window(snapshot.tokens),
    ))
}

/// Whether the client retries a request that fails before its body itself: its default HTTP
/// client does, with `max_retries` retries; a client given with `ConfigBuilder::http_client`
/// brings its own layers.
pub(crate) fn retries_before_body<T>(client: &Client<T>) -> bool {
    client.config().http_client().is_none() && client.config().max_retries() > 0
}

/// `POST {api_base}{path}` with a JSON body, as the client's own calls go: through its
/// interceptors (named `operation`) and its HTTP stack. The JSON reply and its rate limits.
pub(crate) async fn call<T>(
    client: &Client<T>,
    operation: &str,
    path: &str,
    body: &Value,
) -> Result<(Value, Option<RateLimits>), ModelError>
where
    T: Default + openai_ergonomic::MaybeSend + openai_ergonomic::MaybeSync + 'static,
{
    let response = client
        .json_call(operation, path, body)
        .await
        .map_err(call_failure)?;
    Ok((response.body, rate_limits(response.meta.rate_limits())))
}

/// `POST {api_base}{path}` that answers with server-sent events, as for [`call`]. The events
/// and the rate limits.
pub(crate) async fn event_call<T>(
    client: &Client<T>,
    operation: &str,
    path: &str,
    body: &Value,
) -> Result<(JsonEventStream, Option<RateLimits>), ModelError>
where
    T: Default + openai_ergonomic::MaybeSend + openai_ergonomic::MaybeSync + 'static,
{
    let (events, meta) = client
        .json_event_call(operation, path, body)
        .await
        .map_err(call_failure)?;
    Ok((events, rate_limits(meta.rate_limits())))
}

/// The failure of a failed Responses API response (`status: failed`, or an `error`).
fn response_failure(response: &Value) -> Option<ModelError> {
    let failed = response.get("status").and_then(Value::as_str) == Some("failed");
    let error = response.get("error").filter(|error| !error.is_null());
    if !failed && error.is_none() {
        return None;
    }
    let code = error
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("the response failed");
    Some(model_error(
        Error::response_failed(code, message),
        false,
        None,
    ))
}

/// The output items of a Responses API response that were not seen yet, then its end; or its
/// failure.
pub(crate) fn response_events(
    response: &Value,
    seen: &mut BTreeSet<u64>,
) -> Vec<Result<ModelEvent, ModelError>> {
    if let Some(failure) = response_failure(response) {
        return vec![Err(failure)];
    }
    let mut events = Vec::new();
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for (index, item) in (0_u64..).zip(output) {
        if seen.insert(index) {
            events.extend(item_events(item.clone()));
        }
    }
    let calls = output.iter().any(|item| {
        matches!(
            item.get("type").and_then(Value::as_str),
            Some("function_call" | "custom_tool_call")
        )
    });
    let mut end = ModelResponse::new();
    if let Some(id) = response.get("id").and_then(Value::as_str) {
        end = end.with_id(id);
    }
    if let Some(usage) = response.get("usage").filter(|usage| usage.is_object()) {
        end = end.with_usage(usage_of(
            usage,
            ["input_tokens", "output_tokens"],
            ["input_tokens_details", "output_tokens_details"],
        ));
    }
    if response.get("status").and_then(Value::as_str) == Some("incomplete") {
        let reason = response
            .get("incomplete_details")
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        end = end.with_stop(stop_reason(reason)).with_incomplete(reason);
    } else {
        end = end.with_stop(if calls {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        });
    }
    events.push(Ok(ModelEvent::Completed(end)));
    events
}

/// Why the model stopped, from OpenAI's names (an incomplete reason or a `finish_reason`).
pub(crate) fn stop_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "max_output_tokens" | "length" => StopReason::Length,
        "content_filter" => StopReason::ContentFilter,
        other => StopReason::Other(other.to_string()),
    }
}

/// The usage of a response or completion: the input and output tokens under `totals`, the cached
/// and reasoning tokens in the `details` objects.
pub(crate) fn usage_of(usage: &Value, totals: [&str; 2], details: [&str; 2]) -> Usage {
    let tokens = |key: &str| usage.get(key).and_then(Value::as_i64).unwrap_or_default();
    let detail = |object: &str, key: &str| {
        usage
            .get(object)
            .and_then(|details| details.get(key))
            .and_then(Value::as_i64)
    };
    let mut result = Usage::new(tokens(totals[0]), tokens(totals[1]));
    if let Some(cached) = detail(details[0], "cached_tokens") {
        result = result.with_cache_read_tokens(cached);
    }
    if let Some(reasoning) = detail(details[1], "reasoning_tokens") {
        result = result.with_reasoning_tokens(reasoning);
    }
    result
}

/// An output item as events: the item, and the tools it loaded when it is a tool search's
/// output.
fn item_events(item: Value) -> Vec<Result<ModelEvent, ModelError>> {
    let loaded = convert::loaded_tools(&item);
    let mut events = vec![Ok(ModelEvent::ItemDone(convert::output_item(item)))];
    if !loaded.is_empty() {
        events.push(Ok(ModelEvent::ToolsLoaded(loaded)));
    }
    events
}

/// The events of a streamed Responses API response.
pub(crate) fn response_stream(
    events: JsonEventStream,
) -> genai_agents::BoxStream<'static, Result<ModelEvent, ModelError>> {
    let state = StreamState {
        events,
        seen: BTreeSet::new(),
        pending: VecDeque::new(),
        done: false,
    };
    Box::pin(futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.pending.pop_front() {
                return Some((event, state));
            }
            if state.done {
                return None;
            }
            match state.events.next().await {
                None => return None,
                Some(Err(error)) => {
                    state.done = true;
                    state
                        .pending
                        .push_back(Err(model_error(error, false, None)));
                }
                Some(Ok(event)) => state.read(event.0),
            }
        }
    }))
}

struct StreamState {
    events: JsonEventStream,
    seen: BTreeSet<u64>,
    pending: VecDeque<Result<ModelEvent, ModelError>>,
    done: bool,
}

impl StreamState {
    /// Read one event.
    fn read(&mut self, event: Value) {
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        match kind.as_str() {
            "response.output_text.delta" => {
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.pending
                    .push_back(Ok(ModelEvent::TextDelta(delta.to_string())));
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.pending
                    .push_back(Ok(ModelEvent::ReasoningDelta(delta.to_string())));
            }
            "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let id = event
                    .get("item_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.pending.push_back(Ok(ModelEvent::ToolCallArgsDelta {
                    id: id.to_string(),
                    delta: delta.to_string(),
                }));
            }
            "response.output_item.done" => {
                let index = event.get("output_index").and_then(Value::as_u64);
                if let (Some(index), Some(item)) = (index, event.get("item")) {
                    if self.seen.insert(index) {
                        self.pending.extend(item_events(item.clone()));
                    }
                }
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                self.done = true;
                let response = event.get("response").cloned().unwrap_or(Value::Null);
                let mut response = response;
                if kind == "response.failed" && response.get("status").is_none() {
                    response["status"] = Value::String("failed".into());
                }
                self.pending
                    .extend(response_events(&response, &mut self.seen));
            }
            "error" => {
                self.done = true;
                let code = event
                    .get("code")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let message = event
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("the response failed");
                self.pending.push_back(Err(model_error(
                    Error::response_failed(code, message),
                    false,
                    None,
                )));
            }
            _ => self
                .pending
                .push_back(Ok(ModelEvent::Other { kind, data: event })),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]
    use super::*;
    use serde_json::json;

    #[test]
    fn api_errors_keep_their_class_status_and_code() {
        let error = Error::api_detailed(
            429,
            "Rate limit reached. Please try again in 1.5s.",
            Some("requests".into()),
            Some("rate_limit_exceeded".into()),
        );
        let error = model_error(error, true, None);
        assert!(error.is_rate_limit());
        assert_eq!(error.kind(), ErrorKind::RateLimit);
        assert!(error.is_before_body());
        assert_eq!(
            error.retry_class(),
            RetryClass::ServerDirected {
                after: Duration::from_millis(1500)
            }
        );
        assert!(error.provider_error::<Error>().is_some());
        let quota = model_error(
            Error::api_detailed(429, "quota", None, Some("insufficient_quota".into())),
            true,
            None,
        );
        assert_eq!(quota.retry_class(), RetryClass::Never);
        let overflow = model_error(
            Error::response_failed(Some("context_length_exceeded".into()), "too long"),
            false,
            None,
        );
        assert_eq!(overflow.kind(), ErrorKind::ContextOverflow);
        assert_eq!(overflow.retry_class(), RetryClass::Never);
        let missing = model_error(
            Error::api_detailed(404, "no", None, Some("model_not_found".into())),
            true,
            None,
        );
        assert_eq!(missing.kind(), ErrorKind::ModelNotFound);
        let auth = model_error(Error::api(401, "who are you"), true, None);
        assert_eq!(auth.kind(), ErrorKind::Authentication);
    }

    #[test]
    fn failed_and_incomplete_responses() {
        let mut seen = BTreeSet::new();
        let events = response_events(
            &json!({"status": "failed", "error": {"code": "server_error", "message": "boom"}}),
            &mut seen,
        );
        let Err(error) = &events[0] else {
            panic!("a failure expected");
        };
        assert_eq!(error.retry_class(), RetryClass::Backoff);
        assert_eq!(error.code(), Some("server_error"));
        let events = response_events(
            &json!({"id": "resp_1", "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [{"type": "message", "role": "assistant",
                                "content": [{"type": "output_text", "text": "Half"}]}],
                    "usage": {"input_tokens": 3, "output_tokens": 4}}),
            &mut seen,
        );
        assert_eq!(events.len(), 2);
        let Ok(ModelEvent::Completed(end)) = &events[1] else {
            panic!("the end expected");
        };
        assert_eq!(end.incomplete.as_deref(), Some("max_output_tokens"));
        assert_eq!(end.stop, Some(StopReason::Length));
        assert_eq!(end.usage, Some(Usage::new(3, 4)));
        assert_eq!(end.id.as_deref(), Some("resp_1"));
    }
}
