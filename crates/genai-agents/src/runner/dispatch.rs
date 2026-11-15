//! Running the tool calls of a turn: concurrently where the tools allow it, answered in call
//! order.
//!
//! The gate is a FIFO read/write lock without a lock: a call of a tool that
//! [`supports_parallel`](crate::tools::FunctionTool::supports_parallel) starts while only such calls
//! run; any other call starts once everything before it finished, and everything after it waits
//! for it. The futures run on the loop's own task (no spawning), so nothing needs to be `Send` on
//! WebAssembly.

use std::collections::VecDeque;
use std::task::{Context, Poll};

use futures::stream::{FuturesUnordered, Stream};
use tracing::field::Empty;
use tracing::Instrument;

use super::approval::{ApprovalDecision, ApprovalPolicy};
use super::hooks::{CallDecision, RunHooks};
use crate::item::{ToolCall, ToolKind, ToolOutput};
use crate::platform::{BoxFuture, Instant};
use crate::tools::{ToolError, ToolRegistry};

/// A running call: its slot, and its answer (`None`: waiting for a person's approval).
type Running<'r> = BoxFuture<'r, (usize, Result<Option<ToolOutput>, ToolError>)>;

struct Slot {
    call: ToolCall,
    parallel: bool,
    /// Whether the approval policy decides before it runs.
    check: bool,
    started: Option<Instant>,
    answer: Option<ToolOutput>,
    /// The policy asked a person: no answer until the run resumes.
    pending: bool,
}

pub(super) struct Dispatcher<'r> {
    registry: &'r ToolRegistry,
    slots: Vec<Slot>,
    queue: VecDeque<usize>,
    running: FuturesUnordered<Running<'r>>,
    parallel_running: usize,
    exclusive_running: bool,
    /// The run's span: the parent of the calls' spans.
    span: tracing::Span,
    trace_content: bool,
    hooks: &'r dyn RunHooks,
    approvals: Option<&'r dyn ApprovalPolicy>,
}

impl<'r> Dispatcher<'r> {
    pub(super) fn new(
        registry: &'r ToolRegistry,
        span: tracing::Span,
        trace_content: bool,
        hooks: &'r dyn RunHooks,
        approvals: Option<&'r dyn ApprovalPolicy>,
    ) -> Self {
        Self {
            span,
            trace_content,
            hooks,
            approvals,
            registry,
            slots: Vec::new(),
            queue: VecDeque::new(),
            running: FuturesUnordered::new(),
            parallel_running: 0,
            exclusive_running: false,
        }
    }

    /// Queue a call; it starts as soon as the gate lets it. With `check`, the approval policy
    /// decides first.
    pub(super) fn push(&mut self, call: ToolCall, parallel: bool, check: bool) {
        self.slots.push(Slot {
            call,
            parallel,
            check,
            started: None,
            answer: None,
            pending: false,
        });
        self.queue.push_back(self.slots.len() - 1);
        self.start_ready();
    }

    /// Add a call answered already (such as the loop's own `tool_search`).
    pub(super) fn push_answered(&mut self, call: ToolCall, answer: ToolOutput) {
        self.slots.push(Slot {
            call,
            parallel: true,
            check: false,
            pending: false,
            started: Some(Instant::now()),
            answer: Some(answer),
        });
    }

    fn start_ready(&mut self) {
        while let Some(&index) = self.queue.front() {
            let parallel = self.slots[index].parallel;
            let may_start = if parallel {
                !self.exclusive_running
            } else {
                !self.exclusive_running && self.parallel_running == 0
            };
            if !may_start {
                return;
            }
            self.queue.pop_front();
            if parallel {
                self.parallel_running += 1;
            } else {
                self.exclusive_running = true;
            }
            let slot = &mut self.slots[index];
            slot.started = Some(Instant::now());
            let call = slot.call.clone();
            let check = slot.check;
            let registry = self.registry;
            let hooks = self.hooks;
            let approvals = self.approvals;
            let span = tool_span(&self.span, &call, self.trace_content);
            let trace_content = self.trace_content;
            self.running.push(Box::pin(
                async move {
                    let decision = match (check, approvals) {
                        (true, Some(policy)) => policy.decide(&call).await,
                        _ => ApprovalDecision::Approve,
                    };
                    let answer = match decision {
                        ApprovalDecision::Ask => Ok(None),
                        ApprovalDecision::Deny(reason) => Ok(Some(ToolOutput::failure(
                            &call,
                            format!("denied: {reason}"),
                        ))),
                        _ => answer_hooked(registry, hooks, call).await.map(Some),
                    };
                    let current = tracing::Span::current();
                    match &answer {
                        Ok(None) => {
                            current.record("genai_agents.approval", "pending");
                        }
                        Ok(Some(output)) => {
                            if !output.success {
                                current.record("error.type", "tool_error");
                            }
                            if trace_content {
                                if let Some(text) = output.body.to_text() {
                                    current.record("gen_ai.tool.call.result", text.as_str());
                                }
                            }
                        }
                        Err(_) => {
                            current.record("otel.status_code", "ERROR");
                            current.record("error.type", "fatal");
                        }
                    }
                    (index, answer)
                }
                .instrument(span),
            ));
        }
    }

    /// Whether every call was answered.
    pub(super) fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.running.is_empty()
    }

    /// Poll for the next call to finish and return its slot (a fatal failure is the error):
    /// pending while no call runs.
    pub(super) fn poll_finished(&mut self, cx: &mut Context<'_>) -> Poll<Result<usize, ToolError>> {
        let (index, result) = match std::pin::Pin::new(&mut self.running).poll_next(cx) {
            Poll::Ready(Some(finished)) => finished,
            Poll::Ready(None) | Poll::Pending => return Poll::Pending,
        };
        Poll::Ready(self.record(index, result))
    }

    fn record(
        &mut self,
        index: usize,
        result: Result<Option<ToolOutput>, ToolError>,
    ) -> Result<usize, ToolError> {
        if self.slots[index].parallel {
            self.parallel_running -= 1;
        } else {
            self.exclusive_running = false;
        }
        match result? {
            Some(answer) => self.slots[index].answer = Some(answer),
            None => self.slots[index].pending = true,
        }
        self.start_ready();
        Ok(index)
    }

    /// The calls waiting for a person's approval, in call order.
    pub(super) fn pending(&self) -> Vec<ToolCall> {
        self.slots
            .iter()
            .filter(|slot| slot.pending)
            .map(|slot| slot.call.clone())
            .collect()
    }

    /// The answer of a finished slot.
    pub(super) fn answer(&self, index: usize) -> Option<&ToolOutput> {
        self.slots.get(index)?.answer.as_ref()
    }

    /// The calls, in order, with their answers so far; the dispatcher is emptied.
    pub(super) fn take(&mut self) -> Vec<(ToolCall, Option<ToolOutput>)> {
        self.queue.clear();
        self.running = FuturesUnordered::new();
        self.parallel_running = 0;
        self.exclusive_running = false;
        std::mem::take(&mut self.slots)
            .into_iter()
            .map(|slot| (slot.call, slot.answer))
            .collect()
    }

    /// Stop every call: the answers of the finished calls, `"aborted by user after …s"` for the
    /// running ones and `"aborted"` for the others, in call order. The dispatcher is emptied.
    pub(super) fn abort(&mut self) -> Vec<ToolOutput> {
        // A call waiting for approval never ran: "aborted", as one that never started.
        let started: Vec<Option<Instant>> = self
            .slots
            .iter()
            .map(|slot| slot.started.filter(|_| !slot.pending))
            .collect();
        self.take()
            .into_iter()
            .zip(started)
            .map(|((call, answer), started)| match (answer, started) {
                (Some(answer), _) => answer,
                (None, Some(started)) => ToolOutput::failure(
                    &call,
                    format!(
                        "aborted by user after {:.1}s",
                        started.elapsed().as_secs_f64()
                    ),
                ),
                (None, None) => ToolOutput::aborted(&call),
            })
            .collect()
    }

    /// Answer the calls that have not started yet with `answer(call)` instead of running them;
    /// the running ones go on. Returns how many were answered so.
    pub(super) fn answer_unstarted(&mut self, answer: impl Fn(&ToolCall) -> ToolOutput) -> usize {
        let unstarted: Vec<usize> = self.queue.drain(..).collect();
        for &index in &unstarted {
            let slot = &mut self.slots[index];
            slot.started = Some(Instant::now());
            slot.answer = Some(answer(&slot.call));
        }
        unstarted.len()
    }

    /// The finished answers, in call order, of a dispatcher stopped by a fatal failure. The
    /// dispatcher is emptied.
    pub(super) fn finished_answers(&mut self) -> Vec<ToolOutput> {
        self.take()
            .into_iter()
            .filter_map(|(_, answer)| answer)
            .collect()
    }
}

/// The `execute_tool` span of a call, a child of the run's span.
fn tool_span(parent: &tracing::Span, call: &ToolCall, trace_content: bool) -> tracing::Span {
    let kind = match call.kind {
        ToolKind::Custom => "custom",
        _ => "function",
    };
    tracing::info_span!(
        parent: parent,
        "execute_tool",
        otel.name = super::trace::span_name("execute_tool", &call.name),
        otel.kind = "internal",
        otel.status_code = Empty,
        gen_ai.operation.name = "execute_tool",
        gen_ai.tool.name = call.name.as_str(),
        gen_ai.tool.namespace = call.namespace.as_deref(),
        gen_ai.tool.call.id = call.call_id.as_str(),
        gen_ai.tool.type = kind,
        gen_ai.tool.call.arguments = trace_content.then_some(call.input.as_str()),
        gen_ai.tool.call.result = Empty,
        genai_agents.approval = Empty,
        error.type = Empty,
    )
}

/// Answer a call through the hooks: they may change its arguments or block it before it runs,
/// and change its answer after. A hook that fails stops the run, as a fatal tool failure does.
async fn answer_hooked(
    registry: &ToolRegistry,
    hooks: &dyn RunHooks,
    mut call: ToolCall,
) -> Result<ToolOutput, ToolError> {
    if hooks.is_empty() {
        return registry.answer_call(call).await;
    }
    let hook_failed = |call: &ToolCall, error: crate::Error| ToolError::Call {
        call_id: call.call_id.clone(),
        source: Box::new(ToolError::Execution {
            name: call.name.clone(),
            source: error,
        }),
    };
    match hooks.before_call(&mut call).await {
        Err(error) => return Err(hook_failed(&call, error)),
        Ok(CallDecision::Block(answer)) => return Ok(ToolOutput::failure(&call, answer)),
        Ok(CallDecision::Run) => {}
    }
    let mut output = registry.answer_call(call.clone()).await?;
    hooks
        .after_call(&call, &mut output)
        .await
        .map_err(|error| hook_failed(&call, error))?;
    Ok(output)
}
