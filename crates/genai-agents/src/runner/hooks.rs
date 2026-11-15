//! Hooks: the application's code around a request, a call, a compaction and the end of a run.

use std::sync::Arc;

use crate::conversation::Conversation;
use crate::error::Result;
use crate::item::{Item, ToolCall, ToolOutput};
use crate::model::{ModelSettings, OutputSchema, ToolDefinition};
use crate::platform::{BoxFuture, MaybeSend, MaybeSync};

/// What a request is about to carry, for [`Hooks::before_request`] to change.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RequestPlan {
    /// The settings.
    pub settings: ModelSettings,
    /// The tools advertised.
    pub tools: Vec<ToolDefinition>,
    /// The instructions.
    pub instructions: Option<String>,
    /// The schema of the answer, if one is asked for.
    pub output: Option<OutputSchema>,
}

/// What [`Hooks::before_call`] decides about a call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CallDecision {
    /// Run it (with the arguments as the hook left them).
    Run,
    /// Do not run it: this text is its answer, a failure. Later hooks are not asked.
    Block(String),
}

/// Code of the application's around the steps of a run, with its context `C` (see
/// [`Agent`](crate::agent::Agent)). Every method has a default that does nothing; implement
/// the ones you need. A hook that fails ends the run with its error.
///
/// Several hooks run in the order they were added: each sees what the earlier ones changed.
pub trait Hooks<C>: MaybeSend + MaybeSync {
    /// Before each request: change its settings, its tools or its instructions.
    fn before_request<'a>(
        &'a self,
        context: &'a C,
        request: &'a mut RequestPlan,
    ) -> BoxFuture<'a, Result<()>> {
        let _ = (context, request);
        Box::pin(async { Ok(()) })
    }

    /// Before a call runs: change its arguments, or block it.
    fn before_call<'a>(
        &'a self,
        context: &'a C,
        call: &'a mut ToolCall,
    ) -> BoxFuture<'a, Result<CallDecision>> {
        let _ = (context, call);
        Box::pin(async { Ok(CallDecision::Run) })
    }

    /// After a call ran (not when it was blocked): change its answer before the model sees it
    /// and the conversation keeps it (redact it, add [`details`](ToolOutput::details)).
    fn after_call<'a>(
        &'a self,
        context: &'a C,
        call: &'a ToolCall,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        let _ = (context, call, output);
        Box::pin(async { Ok(()) })
    }

    /// When the model answered and the run would stop: items to go on with (a check that
    /// failed, say), or none to stop.
    fn on_stop<'a>(
        &'a self,
        context: &'a C,
        conversation: &'a Conversation,
    ) -> BoxFuture<'a, Result<Vec<Item>>> {
        let _ = (context, conversation);
        Box::pin(async { Ok(Vec::new()) })
    }

    /// Before a compaction, with the conversation it is about to summarise: to save what should
    /// outlive it. It cannot stop the compaction.
    fn before_compaction<'a>(
        &'a self,
        context: &'a C,
        conversation: &'a Conversation,
    ) -> BoxFuture<'a, ()> {
        let _ = (context, conversation);
        Box::pin(async {})
    }
}

/// The hooks of a run, bound to its context: what the runner calls.
pub(crate) trait RunHooks: MaybeSend + MaybeSync {
    fn before_request<'a>(&'a self, request: &'a mut RequestPlan) -> BoxFuture<'a, Result<()>>;
    fn before_call<'a>(&'a self, call: &'a mut ToolCall) -> BoxFuture<'a, Result<CallDecision>>;
    fn after_call<'a>(
        &'a self,
        call: &'a ToolCall,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>>;
    fn on_stop<'a>(&'a self, conversation: &'a Conversation) -> BoxFuture<'a, Result<Vec<Item>>>;
    fn before_compaction<'a>(&'a self, conversation: &'a Conversation) -> BoxFuture<'a, ()>;
    /// Whether there is any hook: the runner skips the work when there is none.
    fn is_empty(&self) -> bool;
}

/// No hooks: a [`Runner`](super::Runner) on its own.
pub(crate) struct NoHooks;

impl RunHooks for NoHooks {
    fn before_request<'a>(&'a self, _: &'a mut RequestPlan) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
    fn before_call<'a>(&'a self, _: &'a mut ToolCall) -> BoxFuture<'a, Result<CallDecision>> {
        Box::pin(async { Ok(CallDecision::Run) })
    }
    fn after_call<'a>(
        &'a self,
        _: &'a ToolCall,
        _: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
    fn on_stop<'a>(&'a self, _: &'a Conversation) -> BoxFuture<'a, Result<Vec<Item>>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn before_compaction<'a>(&'a self, _: &'a Conversation) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }
    fn is_empty(&self) -> bool {
        true
    }
}

/// Hooks with their context.
pub(crate) struct Bound<'c, C> {
    pub(crate) context: &'c C,
    pub(crate) hooks: &'c [Arc<dyn Hooks<C>>],
}

impl<C: MaybeSync> RunHooks for Bound<'_, C> {
    fn before_request<'a>(&'a self, request: &'a mut RequestPlan) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            for hook in self.hooks {
                hook.before_request(self.context, request).await?;
            }
            Ok(())
        })
    }

    fn before_call<'a>(&'a self, call: &'a mut ToolCall) -> BoxFuture<'a, Result<CallDecision>> {
        Box::pin(async move {
            for hook in self.hooks {
                if let CallDecision::Block(answer) = hook.before_call(self.context, call).await? {
                    return Ok(CallDecision::Block(answer));
                }
            }
            Ok(CallDecision::Run)
        })
    }

    fn after_call<'a>(
        &'a self,
        call: &'a ToolCall,
        output: &'a mut ToolOutput,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            for hook in self.hooks {
                hook.after_call(self.context, call, output).await?;
            }
            Ok(())
        })
    }

    fn on_stop<'a>(&'a self, conversation: &'a Conversation) -> BoxFuture<'a, Result<Vec<Item>>> {
        Box::pin(async move {
            let mut more = Vec::new();
            for hook in self.hooks {
                more.extend(hook.on_stop(self.context, conversation).await?);
            }
            Ok(more)
        })
    }

    fn before_compaction<'a>(&'a self, conversation: &'a Conversation) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            for hook in self.hooks {
                hook.before_compaction(self.context, conversation).await;
            }
        })
    }

    fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }
}
