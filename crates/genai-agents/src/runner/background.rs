//! Work a run carries along without waiting for it: sub-agents. The runner polls it with the
//! response and the calls, on its own task; what finishes comes back as input.

use std::task::{Context, Poll};

use crate::item::{Item, ToolCall, ToolOutput};
use crate::model::ToolDefinition;
use crate::platform::{MaybeSend, MaybeSync};

/// A source of background work for a run.
pub(crate) trait Background: MaybeSend + MaybeSync {
    /// The tools that start work.
    fn definitions(&self) -> Vec<ToolDefinition>;
    /// Start the work a call asks for, and answer it at once; `None` when the call is not one of
    /// these tools.
    fn start(&self, call: &ToolCall) -> Option<ToolOutput>;
    /// The next piece of work that finished, as an input item; pending while none did.
    fn poll_finished(&self, cx: &mut Context<'_>) -> Poll<Item>;
    /// How many pieces of work are running.
    fn running(&self) -> usize;
}

/// No background work: a run of a plain [`Runner`](super::Runner).
pub(crate) struct NoBackground;

impl Background for NoBackground {
    fn definitions(&self) -> Vec<ToolDefinition> {
        Vec::new()
    }
    fn start(&self, _: &ToolCall) -> Option<ToolOutput> {
        None
    }
    fn poll_finished(&self, _: &mut Context<'_>) -> Poll<Item> {
        Poll::Pending
    }
    fn running(&self) -> usize {
        0
    }
}
