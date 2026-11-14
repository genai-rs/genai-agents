//! A scripted [`Model`] for tests: it answers each request as a script says, and records the
//! requests. Feature `testing`.
//!
//! ```
//! use genai_agents::testing::{Reply, ScriptedModel};
//! use genai_agents::{Conversation, Item, Runner, ToolCall, ToolRegistry};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> genai_agents::Result<()> {
//! let model = ScriptedModel::new(|index, _request| match index {
//!     0 => Reply::items(vec![Item::ToolCall(ToolCall::function("call_1", "missing", "{}"))]),
//!     _ => Reply::items(vec![Item::assistant("Done.")]),
//! });
//! let mut runner = Runner::new(model.clone(), ToolRegistry::new());
//! let outcome = runner.run(&mut Conversation::user("Go")).await?;
//! assert_eq!(outcome.output_text(), "Done.");
//! // The second request carried the answer to the call.
//! let requests = model.requests();
//! assert_eq!(requests.len(), 2);
//! assert_eq!(
//!     requests[1].outputs()[0].body.as_text(),
//!     Some("unsupported call: missing")
//! );
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use futures::StreamExt;

use crate::error::ModelError;
use crate::item::{Item, ToolOutput};
use crate::model::{
    Model, ModelEvent, ModelInfo, ModelRequest, ModelResponse, ModelSettings, ModelStream,
    RateLimitWindow, RateLimits, ToolDefinition, Usage,
};
use crate::platform::{sleep, MaybeSend};

/// The answer to one request of a [`ScriptedModel`].
#[derive(Debug)]
#[non_exhaustive]
pub enum Reply {
    /// The request fails before its body (the [`ModelError`] is marked so).
    Fail(ModelError),
    /// The response: these events, each after its delay.
    Events {
        /// The rate limits its headers report.
        rate_limits: Option<RateLimits>,
        /// The events, each with the delay before it.
        events: Vec<(Duration, Result<ModelEvent, ModelError>)>,
    },
}

impl Reply {
    /// A response of these items (a text delta before each message), then its end with
    /// 10 input and 5 output tokens.
    #[must_use]
    pub fn items(items: Vec<Item>) -> Self {
        let mut events = Vec::new();
        for item in items {
            if let Item::Message(message) = &item {
                events.push((
                    Duration::ZERO,
                    Ok(ModelEvent::TextDelta(message.text_content())),
                ));
            }
            events.push((Duration::ZERO, Ok(ModelEvent::ItemDone(item))));
        }
        events.push((
            Duration::ZERO,
            Ok(ModelEvent::Completed(
                ModelResponse::new().with_usage(Usage::new(10, 5)),
            )),
        ));
        Self::Events {
            rate_limits: None,
            events,
        }
    }

    /// A response with one assistant message.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::items(vec![Item::assistant(text)])
    }

    /// A failure before the body.
    #[must_use]
    pub fn fail(error: ModelError) -> Self {
        Self::Fail(error)
    }

    /// The same response, with this response id on its end.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        let id = id.into();
        if let Self::Events { events, .. } = &mut self {
            for (_, event) in events {
                if let Ok(ModelEvent::Completed(end)) = event {
                    end.id = Some(id.clone());
                }
            }
        }
        self
    }

    /// The same response, reporting this many requests remaining of 100.
    #[must_use]
    pub fn with_remaining_requests(mut self, remaining: u64) -> Self {
        if let Self::Events { rate_limits, .. } = &mut self {
            *rate_limits = Some(RateLimits::new(
                Some(RateLimitWindow::new(Some(100), Some(remaining), None)),
                None,
            ));
        }
        self
    }

    /// The events, to change them (delay one, drop the end to cut the stream off, …); `None`
    /// for a [`Fail`](Self::Fail).
    pub fn events_mut(&mut self) -> Option<&mut Vec<(Duration, Result<ModelEvent, ModelError>)>> {
        match self {
            Self::Events { events, .. } => Some(events),
            Self::Fail(_) => None,
        }
    }
}

/// What one request to a [`ScriptedModel`] carried.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SeenRequest {
    /// The conversation.
    pub items: Vec<Item>,
    /// The tools.
    pub tools: Vec<ToolDefinition>,
    /// Whether parallel tool calls were allowed.
    pub parallel_tool_calls: bool,
    /// The prompt cache key.
    pub prompt_cache_key: Option<String>,
    /// The settings.
    pub settings: ModelSettings,
}

impl SeenRequest {
    /// The names of the tools (`<tool search>` for the provider's own tool search).
    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        self.tools
            .iter()
            .map(|tool| tool.name().unwrap_or("<tool search>").to_string())
            .collect()
    }

    /// The tool outputs of the conversation, in order.
    #[must_use]
    pub fn outputs(&self) -> Vec<ToolOutput> {
        self.items
            .iter()
            .filter_map(|item| match item {
                Item::ToolOutput(output) => Some(output.clone()),
                _ => None,
            })
            .collect()
    }
}

type Script = Arc<dyn Fn(usize, &SeenRequest) -> Reply + Send + Sync>;

/// A [`Model`] that answers request `n` (from 0) with `script(n, request)` and records every
/// request ([`requests`](Self::requests)). Clones share the script and the record.
#[derive(Clone)]
pub struct ScriptedModel {
    script: Script,
    seen: Arc<Mutex<Vec<SeenRequest>>>,
    info: ModelInfo,
    retries_before_body: bool,
}

impl std::fmt::Debug for ScriptedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedModel")
            .field("requests", &self.requests().len())
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl ScriptedModel {
    /// A model that answers request `n` with `script(n, request)`.
    pub fn new(script: impl Fn(usize, &SeenRequest) -> Reply + Send + Sync + 'static) -> Self {
        Self {
            script: Arc::new(script),
            seen: Arc::default(),
            info: ModelInfo::default(),
            retries_before_body: false,
        }
    }

    /// A model that answers with these replies in order, then fails every request.
    #[must_use]
    pub fn replies(replies: Vec<Reply>) -> Self {
        let replies: Mutex<Vec<Option<Reply>>> =
            Mutex::new(replies.into_iter().map(Some).collect());
        Self::new(move |index, _| {
            replies
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_mut(index)
                .and_then(Option::take)
                .unwrap_or_else(|| {
                    Reply::Fail(ModelError::new(format!(
                        "no reply scripted for request {index}"
                    )))
                })
        })
    }

    /// The same model, with this information.
    #[must_use]
    pub fn with_info(mut self, info: ModelInfo) -> Self {
        self.info = info;
        self
    }

    /// The same model, saying that its HTTP layer retries failures before the body
    /// ([`Model::retries_before_body`]).
    #[must_use]
    pub fn with_retries_before_body(mut self, retries: bool) -> Self {
        self.retries_before_body = retries;
        self
    }

    /// The requests so far, in order.
    #[must_use]
    pub fn requests(&self) -> Vec<SeenRequest> {
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Model for ScriptedModel {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn stream<'a>(
        &'a self,
        request: ModelRequest<'_>,
    ) -> impl Future<Output = Result<ModelStream<'a>, ModelError>> + MaybeSend {
        let seen = SeenRequest {
            items: request.items.to_vec(),
            tools: request.tools.to_vec(),
            parallel_tool_calls: request.parallel_tool_calls,
            prompt_cache_key: request.prompt_cache_key.map(str::to_string),
            settings: request.settings,
        };
        let index = {
            let mut log = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
            log.push(seen.clone());
            log.len() - 1
        };
        let reply = (self.script)(index, &seen);
        async move {
            match reply {
                Reply::Fail(error) => Err(error.with_before_body(true)),
                Reply::Events {
                    rate_limits,
                    events,
                } => {
                    let events = futures::stream::iter(events).then(|(delay, event)| async move {
                        if !delay.is_zero() {
                            sleep(delay).await;
                        }
                        event
                    });
                    Ok(ModelStream::new(Box::pin(events)).with_rate_limits(rate_limits))
                }
            }
        }
    }

    fn retries_before_body(&self) -> bool {
        self.retries_before_body
    }
}
