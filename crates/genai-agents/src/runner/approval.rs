//! Approvals: which calls need a person's yes before they run.

use serde::{Deserialize, Serialize};

use super::policy::ToolPolicy;
use crate::item::ToolCall;
use crate::model::fnv1a;
use crate::platform::{BoxFuture, MaybeSend, MaybeSync};
use crate::tools::ToolKey;

/// What an [`ApprovalPolicy`] decides about a call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ApprovalDecision {
    /// Run it.
    Approve,
    /// Do not run it: this is why (the call's answer).
    Deny(String),
    /// Ask: the run stops with the call pending ([`RunStop::Approval`](super::RunStop::Approval)).
    Ask,
}

/// Decides, before a call runs, whether it may: approve, deny, or ask a person. A policy can
/// look at the tool and the arguments, keep its own record, or ask another model (an LLM
/// reviewer); it is asked only for calls not approved already (by key for the session, or
/// one by one on resume).
pub trait ApprovalPolicy: MaybeSend + MaybeSync {
    /// Decide about a call.
    fn decide<'a>(&'a self, call: &'a ToolCall) -> BoxFuture<'a, ApprovalDecision>;

    /// What an "approve for the session" covers of a call: by default its tool
    /// (`namespace.name`); a policy can narrow it to a file or a command found in the
    /// arguments, so that the next call with the same key is not asked again.
    fn key(&self, call: &ToolCall) -> String {
        match &call.namespace {
            Some(namespace) => format!("{namespace}.{}", call.name),
            None => call.name.clone(),
        }
    }
}

/// Ask for the calls of the tools that match these patterns (as [`ToolPolicy`]'s: `*` for
/// any run of characters, `namespace.name`), approve the others.
#[derive(Debug, Clone, Default)]
pub struct AskFor {
    patterns: ToolPolicy,
}

impl AskFor {
    /// Ask for the tools that match `patterns`.
    #[must_use]
    pub fn tools(patterns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut policy = ToolPolicy::new();
        for pattern in patterns {
            policy = policy.allow(pattern);
        }
        Self { patterns: policy }
    }
}

impl ApprovalPolicy for AskFor {
    fn decide<'a>(&'a self, call: &'a ToolCall) -> BoxFuture<'a, ApprovalDecision> {
        let ask = !self.patterns.allow.is_empty() && self.patterns.permits(&ToolKey::of(call));
        Box::pin(async move {
            if ask {
                ApprovalDecision::Ask
            } else {
                ApprovalDecision::Approve
            }
        })
    }
}

/// A call waiting for a person's decision: what to show them, and what binds the decision to
/// this call with these arguments. `Serialize` and `Deserialize`: the decision can come much
/// later, from another process.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PendingCall {
    /// The call's id.
    pub call_id: String,
    /// The tool.
    pub name: String,
    /// The tool's namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// The arguments, as the model wrote them.
    pub arguments: String,
    /// What an approval for the session covers ([`ApprovalPolicy::key`]).
    pub key: String,
    /// A hash of the arguments: a decision applies only to a call that still has them.
    pub hash: String,
}

impl PendingCall {
    pub(crate) fn new(call: &ToolCall, key: String) -> Self {
        Self {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            namespace: call.namespace.clone(),
            arguments: call.input.clone(),
            key,
            hash: arguments_hash(&call.input),
        }
    }

    /// Run this call.
    #[must_use]
    pub fn approve(&self) -> Approval {
        self.decide(Decision::Approve)
    }

    /// Run this call, and every later call with the same [key](Self::key) without asking.
    #[must_use]
    pub fn approve_for_session(&self) -> Approval {
        self.decide(Decision::ApproveForSession)
    }

    /// Do not run this call; the model gets the reason as its answer, and the run goes on.
    #[must_use]
    pub fn deny(&self, reason: impl Into<String>) -> Approval {
        self.decide(Decision::Deny(reason.into()))
    }

    /// Stop the run: no pending call runs.
    #[must_use]
    pub fn abort(&self) -> Approval {
        self.decide(Decision::Abort)
    }

    fn decide(&self, decision: Decision) -> Approval {
        Approval {
            call_id: self.call_id.clone(),
            hash: self.hash.clone(),
            decision,
        }
    }
}

/// A person's decision about one call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Decision {
    /// Run it.
    Approve,
    /// Run it, and the later calls with the same key.
    ApproveForSession,
    /// Do not run it, for this reason.
    Deny(String),
    /// Stop the run.
    Abort,
}

/// A decision bound to a pending call ([`PendingCall::approve`] and the like), for
/// [`Runner::resume`](super::Runner::resume).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Approval {
    /// The call.
    pub call_id: String,
    /// The hash of its arguments when it was pending.
    pub hash: String,
    /// The decision.
    pub decision: Decision,
}

/// The hash a pending call is bound to.
pub(crate) fn arguments_hash(arguments: &str) -> String {
    format!("{:016x}", fnv1a(arguments.as_bytes()))
}
