//! Agents: a [`Runner`] with a name, instructions, a context, hooks, guardrails and an output
//! type.
//!
//! ```
//! use genai_agents::agent::{Agent, Verdict};
//! use genai_agents::testing::{Reply, ScriptedModel};
//! use genai_agents::{Conversation, RunStop};
//!
//! /// What the application passes to every run: the hooks, guardrails and instructions see it.
//! struct User {
//!     name: String,
//! }
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> genai_agents::Result<()> {
//! let model = ScriptedModel::replies(vec![Reply::text("Take an umbrella, Ann.")]);
//! let mut agent = Agent::new("household", model)
//!     .dynamic_instructions(|user: &User| format!("You help {} around the house.", user.name))
//!     .input_guardrail_fn("no passwords", |_: &User, conversation: &Conversation| {
//!         let asks = conversation.items().iter().any(|item| match item {
//!             genai_agents::Item::Message(message) => message.text_content().contains("password"),
//!             _ => false,
//!         });
//!         if asks { Verdict::trip("asks for a password") } else { Verdict::Pass }
//!     });
//!
//! let user = User { name: "Ann".into() };
//! let outcome = agent.run(&user, &mut Conversation::user("Umbrella today?")).await?;
//! assert_eq!(outcome.output().map(String::as_str), Some("Take an umbrella, Ann."));
//! assert_eq!(
//!     agent.runner().model().requests()[0].instructions.as_deref(),
//!     Some("You help Ann around the house.")
//! );
//!
//! let tripped = agent.run(&user, &mut Conversation::user("What is my password?")).await?;
//! assert!(matches!(tripped.run.stop, RunStop::Guardrail { ref name, .. } if name == "no passwords"));
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use tracing::Instrument;

use crate::conversation::Conversation;
use crate::error::{Error, Result};
use crate::item::Item;
use crate::model::{Model, OutputSchema};
use crate::platform::{BoxFuture, MaybeSend, MaybeSync};
use crate::runner::{
    Background, Bound, CancellationSignal, Extras, Hooks, NoSink, RunConfig, RunEvent, RunOutcome,
    RunStop, Runner, SessionSink, Sink, ToolPolicy,
};
use crate::session::{Session, SessionStore};
use crate::tools::ToolRegistry;

/// What a guardrail decides.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Verdict {
    /// Nothing wrong.
    Pass,
    /// Stop the run, for this reason.
    Trip(String),
}

impl Verdict {
    /// A verdict that stops the run for `reason`.
    #[must_use]
    pub fn trip(reason: impl Into<String>) -> Self {
        Self::Trip(reason.into())
    }
}

/// A check of a run's input, which runs at the same time as the first request: when it trips,
/// the run is cancelled and ends with [`RunStop::Guardrail`].
pub trait InputGuardrail<C>: MaybeSend + MaybeSync {
    /// Its name, in [`RunStop::Guardrail`].
    fn name(&self) -> &str;

    /// Check the conversation as the run starts.
    fn check<'a>(
        &'a self,
        context: &'a C,
        input: &'a Conversation,
    ) -> BoxFuture<'a, Result<Verdict>>;
}

/// A check of the agent's answer, before the run returns it: when it trips, the run ends with
/// [`RunStop::Guardrail`] and no output.
pub trait OutputGuardrail<C>: MaybeSend + MaybeSync {
    /// Its name, in [`RunStop::Guardrail`].
    fn name(&self) -> &str;

    /// Check the answer's text (the JSON of a structured answer).
    fn check<'a>(&'a self, context: &'a C, output: &'a str) -> BoxFuture<'a, Result<Verdict>>;
}

#[cfg(not(target_family = "wasm"))]
type CheckFn<C, T> = dyn Fn(&C, &T) -> Verdict + Send + Sync;
#[cfg(target_family = "wasm")]
type CheckFn<C, T> = dyn Fn(&C, &T) -> Verdict;

/// A guardrail from a name and a function.
struct FnGuardrail<C, T: ?Sized> {
    name: String,
    check: Box<CheckFn<C, T>>,
}

impl<C: MaybeSync> InputGuardrail<C> for FnGuardrail<C, Conversation> {
    fn name(&self) -> &str {
        &self.name
    }

    fn check<'a>(
        &'a self,
        context: &'a C,
        input: &'a Conversation,
    ) -> BoxFuture<'a, Result<Verdict>> {
        let verdict = (self.check)(context, input);
        Box::pin(async move { Ok(verdict) })
    }
}

impl<C: MaybeSync> OutputGuardrail<C> for FnGuardrail<C, str> {
    fn name(&self) -> &str {
        &self.name
    }

    fn check<'a>(&'a self, context: &'a C, output: &'a str) -> BoxFuture<'a, Result<Verdict>> {
        let verdict = (self.check)(context, output);
        Box::pin(async move { Ok(verdict) })
    }
}

#[cfg(not(target_family = "wasm"))]
type InstructionsFn<C> = dyn Fn(&C) -> String + Send + Sync;
#[cfg(target_family = "wasm")]
type InstructionsFn<C> = dyn Fn(&C) -> String;

/// An agent's instructions: fixed, or made from the run's context.
enum Instructions<C> {
    Text(String),
    Dynamic(Arc<InstructionsFn<C>>),
}

/// The result of an agent's run: the run's [outcome](RunOutcome), and the answer as the agent's
/// output type when the run completed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AgentOutcome<O> {
    /// The run: why it stopped, its requests.
    pub run: RunOutcome,
    /// The answer, when the model answered ([`RunStop::Completed`]) and the answer passed the
    /// output guardrails.
    pub output: Option<O>,
}

impl<O> AgentOutcome<O> {
    /// The answer, if there is one.
    #[must_use]
    pub fn output(&self) -> Option<&O> {
        self.output.as_ref()
    }

    /// The answer.
    ///
    /// # Errors
    ///
    /// [`Error::Output`] with why the run stopped, when it gave no answer.
    pub fn into_output(self) -> Result<O> {
        self.output
            .ok_or_else(|| Error::Output(format!("the run gave no answer: {:?}", self.run.stop)))
    }
}

/// How an answer's text becomes the output type.
type Parse<O> = fn(&str) -> std::result::Result<O, String>;

/// An agent: a name, instructions, a model with tools (a [`Runner`]), and what the application
/// adds around its runs:
///
/// - a context `C` the application passes to every run, which the instructions, the
///   [hooks](Hooks) and the guardrails see;
/// - [hooks](Hooks) around requests, calls, compactions and the end of a run;
/// - input guardrails, which run at the same time as the first request, and output guardrails,
///   which check the answer;
/// - an output type `O` (default `String`): with [`output`](Self::output), the model is asked
///   for JSON of `O`'s schema, and the answer is parsed into it, with a retry when it does not
///   parse.
///
/// See the [module documentation](self) for an example.
pub struct Agent<M, C = (), O = String> {
    name: String,
    description: String,
    handoffs: Vec<Handoff<C>>,
    sub_agents: Vec<SubAgent<C>>,
    sub_agent_limits: SubAgentLimits,
    /// How deep a sub-agent this agent is: 0 for one the application runs.
    depth: u32,
    runner: Runner<M>,
    instructions: Option<Instructions<C>>,
    hooks: Vec<Arc<dyn Hooks<C>>>,
    input_guardrails: Vec<Arc<dyn InputGuardrail<C>>>,
    output_guardrails: Vec<Arc<dyn OutputGuardrail<C>>>,
    output_schema: Option<OutputSchema>,
    parse: Parse<O>,
    output_retries: u32,
}

impl<M, C, O> std::fmt::Debug for Agent<M, C, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("name", &self.name)
            .field("hooks", &self.hooks.len())
            .field("input_guardrails", &self.input_guardrails.len())
            .field("output_guardrails", &self.output_guardrails.len())
            .field("output_schema", &self.output_schema)
            .finish_non_exhaustive()
    }
}

impl<M: Model, C> Agent<M, C, String> {
    /// An agent named `name` on `model`, without tools, answering in text. The context type `C`
    /// is what its runs get (`()` when there is none).
    pub fn new(name: impl Into<String>, model: M) -> Self {
        let name = name.into();
        let runner = Runner::new(model, ToolRegistry::new())
            .with_config(RunConfig::new().with_name(name.clone()));
        Self {
            name,
            description: String::new(),
            handoffs: Vec::new(),
            sub_agents: Vec::new(),
            sub_agent_limits: SubAgentLimits::default(),
            depth: 0,
            runner,
            instructions: None,
            hooks: Vec::new(),
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            output_schema: None,
            parse: |text| Ok(text.to_owned()),
            output_retries: 1,
        }
    }
}

impl<M: Model, C: MaybeSync, O> Agent<M, C, O> {
    /// The agent's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The runner: the model, the tools and the configuration.
    #[must_use]
    pub fn runner(&self) -> &Runner<M> {
        &self.runner
    }

    /// The runner, to change its configuration or model between runs.
    pub fn runner_mut(&mut self) -> &mut Runner<M> {
        &mut self.runner
    }

    /// The same agent, with what it does: what another agent sees of it in a handoff.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// The same agent, able to hand the conversation over to `agent`: the model gets a
    /// `transfer_to_<name>` tool, and when it calls it, `agent` runs on the conversation and its
    /// answer is this run's (parsed as this agent's output type, and checked by this agent's
    /// output guardrails).
    #[must_use]
    pub fn handoff(self, agent: impl DynAgent<C> + 'static) -> Self {
        self.handoff_filtered(agent, None)
    }

    /// [`handoff`](Self::handoff), giving the other agent the conversation as `filter` makes it
    /// (such as [`without_tool_calls`]); what it adds goes into the whole conversation.
    #[must_use]
    pub fn handoff_filtered(
        mut self,
        agent: impl DynAgent<C> + 'static,
        filter: Option<fn(&Conversation) -> Conversation>,
    ) -> Self {
        self.runner
            .add_handoff(agent.name().to_owned(), agent.description().to_owned());
        self.handoffs.push(Handoff {
            agent: Box::new(agent),
            filter,
        });
        self
    }

    /// The same agent, able to start the agent `name` as a sub-agent: the model gets a
    /// `start_<name>` tool that starts it on a task and answers at once; the run goes on, and
    /// the sub-agent's answer comes back as a user message when it finishes. A run that would
    /// stop waits for its sub-agents first; a run that ends (cancelled, failed) stops them.
    ///
    /// `make` makes a fresh agent for each task. A sub-agent may use only the tools this agent's
    /// [policy](Self::tool_policy) allows, and its own sub-agents count towards
    /// [the depth limit](Self::sub_agent_limits).
    #[cfg(not(target_family = "wasm"))]
    #[must_use]
    pub fn sub_agent<A>(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        make: impl Fn() -> A + Send + Sync + 'static,
    ) -> Self
    where
        A: DynAgent<C> + 'static,
    {
        self.sub_agents.push(SubAgent::new(
            name.into(),
            description.into(),
            Box::new(move || Box::new(make()) as Box<dyn DynAgent<C>>),
        ));
        self
    }

    /// The same agent, able to start the agent `name` as a sub-agent (see the native version).
    #[cfg(target_family = "wasm")]
    #[must_use]
    pub fn sub_agent<A>(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        make: impl Fn() -> A + 'static,
    ) -> Self
    where
        A: DynAgent<C> + 'static,
    {
        self.sub_agents.push(SubAgent::new(
            name.into(),
            description.into(),
            Box::new(move || Box::new(make()) as Box<dyn DynAgent<C>>),
        ));
        self
    }

    /// The same agent, with these limits on its sub-agents.
    #[must_use]
    pub fn sub_agent_limits(mut self, limits: SubAgentLimits) -> Self {
        self.sub_agent_limits = limits;
        self
    }

    /// The same agent, asking `policy` before a call runs ([`Runner::with_approvals`]).
    #[must_use]
    pub fn approvals(mut self, policy: impl crate::runner::ApprovalPolicy + 'static) -> Self {
        self.runner.set_approvals(policy);
        self
    }

    /// The same agent with these tools.
    #[must_use]
    pub fn with_tools(mut self, tools: impl Into<Arc<ToolRegistry>>) -> Self {
        let config = self.runner.config().clone();
        self.runner = Runner::new(self.runner.into_model(), tools).with_config(config);
        self
    }

    /// The same agent with this configuration (its name stays the agent's, unless the
    /// configuration names one).
    #[must_use]
    pub fn with_config(mut self, mut config: RunConfig) -> Self {
        if config.name.is_none() {
            config.name = Some(self.name.clone());
        }
        *self.runner.config_mut() = config;
        self
    }

    /// The same agent with these instructions (the system prompt).
    #[must_use]
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(Instructions::Text(instructions.into()));
        self
    }

    /// The same agent with instructions made from each run's context.
    #[cfg(not(target_family = "wasm"))]
    #[must_use]
    pub fn dynamic_instructions(
        mut self,
        instructions: impl Fn(&C) -> String + Send + Sync + 'static,
    ) -> Self {
        self.instructions = Some(Instructions::Dynamic(Arc::new(instructions)));
        self
    }

    /// The same agent with instructions made from each run's context.
    #[cfg(target_family = "wasm")]
    #[must_use]
    pub fn dynamic_instructions(mut self, instructions: impl Fn(&C) -> String + 'static) -> Self {
        self.instructions = Some(Instructions::Dynamic(Arc::new(instructions)));
        self
    }

    /// The same agent with this hook, after the ones added before.
    #[must_use]
    pub fn hook(mut self, hook: impl Hooks<C> + 'static) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// The same agent with this input guardrail.
    #[must_use]
    pub fn input_guardrail(mut self, guardrail: impl InputGuardrail<C> + 'static) -> Self {
        self.input_guardrails.push(Arc::new(guardrail));
        self
    }

    /// The same agent with this output guardrail.
    #[must_use]
    pub fn output_guardrail(mut self, guardrail: impl OutputGuardrail<C> + 'static) -> Self {
        self.output_guardrails.push(Arc::new(guardrail));
        self
    }

    /// The same agent with an input guardrail from a function.
    #[cfg(not(target_family = "wasm"))]
    #[must_use]
    pub fn input_guardrail_fn(
        self,
        name: impl Into<String>,
        check: impl Fn(&C, &Conversation) -> Verdict + Send + Sync + 'static,
    ) -> Self
    where
        C: 'static,
    {
        self.input_guardrail(FnGuardrail {
            name: name.into(),
            check: Box::new(check),
        })
    }

    /// The same agent with an input guardrail from a function.
    #[cfg(target_family = "wasm")]
    #[must_use]
    pub fn input_guardrail_fn(
        self,
        name: impl Into<String>,
        check: impl Fn(&C, &Conversation) -> Verdict + 'static,
    ) -> Self
    where
        C: 'static,
    {
        self.input_guardrail(FnGuardrail {
            name: name.into(),
            check: Box::new(check),
        })
    }

    /// The same agent with an output guardrail from a function.
    #[cfg(not(target_family = "wasm"))]
    #[must_use]
    pub fn output_guardrail_fn(
        self,
        name: impl Into<String>,
        check: impl Fn(&C, &str) -> Verdict + Send + Sync + 'static,
    ) -> Self
    where
        C: 'static,
    {
        self.output_guardrail(FnGuardrail {
            name: name.into(),
            check: Box::new(check),
        })
    }

    /// The same agent with an output guardrail from a function.
    #[cfg(target_family = "wasm")]
    #[must_use]
    pub fn output_guardrail_fn(
        self,
        name: impl Into<String>,
        check: impl Fn(&C, &str) -> Verdict + 'static,
    ) -> Self
    where
        C: 'static,
    {
        self.output_guardrail(FnGuardrail {
            name: name.into(),
            check: Box::new(check),
        })
    }

    /// The same agent with this tool policy.
    #[must_use]
    pub fn tool_policy(mut self, policy: ToolPolicy) -> Self {
        self.runner.config_mut().tool_policy = policy;
        self
    }

    /// How many times an answer that does not parse as the output type is asked for again.
    /// Default 1.
    #[must_use]
    pub fn with_output_retries(mut self, retries: u32) -> Self {
        self.output_retries = retries;
        self
    }

    /// The same agent, answering with `T`: the model is asked for JSON of `T`'s schema
    /// (feature `schemars`), and the answer is parsed into `T`.
    #[cfg(feature = "schemars")]
    #[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
    #[must_use]
    pub fn output<T>(self) -> Agent<M, C, T>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        self.output_with_schema(OutputSchema::for_type::<T>())
    }

    /// The same agent, answering with `T`, asking the model for JSON of this schema.
    #[must_use]
    pub fn output_with_schema<T>(self, schema: OutputSchema) -> Agent<M, C, T>
    where
        T: serde::de::DeserializeOwned,
    {
        Agent {
            name: self.name,
            description: self.description,
            handoffs: self.handoffs,
            sub_agents: self.sub_agents,
            sub_agent_limits: self.sub_agent_limits,
            depth: self.depth,
            runner: self.runner,
            instructions: self.instructions,
            hooks: self.hooks,
            input_guardrails: self.input_guardrails,
            output_guardrails: self.output_guardrails,
            output_schema: Some(schema),
            parse: |text| serde_json::from_str(text).map_err(|error| error.to_string()),
            output_retries: self.output_retries,
        }
    }

    /// Run on `conversation` with `context`: until the model answers (and the answer parses
    /// as the output type), a guardrail trips, or the run stops as a [`Runner`]'s does.
    ///
    /// # Errors
    ///
    /// As [`Runner::run_with`]; a hook's or a guardrail's error; [`Error::Output`] when the
    /// answer still does not parse after the retries.
    pub async fn run(
        &mut self,
        context: &C,
        conversation: &mut Conversation,
    ) -> Result<AgentOutcome<O>> {
        self.run_with(context, conversation, &CancellationSignal::new(), |_| {})
            .await
    }

    /// [`run`](Self::run), with a cancellation signal and the run's events.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run).
    pub async fn run_with<F>(
        &mut self,
        context: &C,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
        on_event: F,
    ) -> Result<AgentOutcome<O>>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        self.run_any(
            context,
            conversation,
            cancel,
            on_event,
            &mut NoSink,
            None,
            true,
        )
        .await
    }

    /// [`run`](Self::run) on a [`Session`], as [`Runner::run_session`] does.
    ///
    /// # Errors
    ///
    /// As [`run`](Self::run), and [`Error::Session`].
    pub async fn run_session<S, F>(
        &mut self,
        context: &C,
        session: &mut Session<S>,
        cancel: &CancellationSignal,
        on_event: F,
    ) -> Result<AgentOutcome<O>>
    where
        S: SessionStore,
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let mut conversation = session.conversation();
        let id = session.id().as_str().to_owned();
        let mut sink = SessionSink {
            synced: conversation.len(),
            session,
        };
        self.run_any(
            context,
            &mut conversation,
            cancel,
            on_event,
            &mut sink,
            Some(&id),
            true,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_any<F, K>(
        &mut self,
        context: &C,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
        mut on_event: F,
        sink: &mut K,
        conversation_id: Option<&str>,
        guard: bool,
    ) -> Result<AgentOutcome<O>>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
        K: Sink,
    {
        let instructions = match &self.instructions {
            Some(Instructions::Text(text)) => Some(text.clone()),
            Some(Instructions::Dynamic(make)) => Some(make(context)),
            None => self.runner.config().instructions.clone(),
        };
        let config = self.runner.config_mut();
        config.instructions = instructions;
        config.output.clone_from(&self.output_schema);

        let hooks = Bound {
            context,
            hooks: &self.hooks,
        };
        let policies = self.runner.policies();
        let pool = SubAgentPool::new(
            context,
            &self.sub_agents,
            self.depth,
            self.sub_agent_limits,
            &policies,
        );
        let extras = Extras {
            hooks: &hooks,
            background: &pool,
        };
        let mut retries = 0;
        let mut first = guard;
        loop {
            let outcome = if first && !self.input_guardrails.is_empty() {
                Self::run_guarded(
                    &mut self.runner,
                    &self.input_guardrails,
                    &extras,
                    context,
                    conversation,
                    cancel,
                    &mut on_event,
                    sink,
                    conversation_id,
                )
                .await?
            } else {
                let span = self.runner.run_span(conversation_id);
                self.runner
                    .run_inner(conversation, cancel, &mut on_event, sink, &span, &extras)
                    .instrument(span.clone())
                    .await?
            };
            first = false;
            let outcome = match &outcome.stop {
                RunStop::Handoff { to } => {
                    let to = to.clone();
                    let outcome = Self::hand_over(
                        &mut self.handoffs,
                        &to,
                        outcome,
                        context,
                        conversation,
                        cancel,
                    )
                    .await?;
                    // What the other agent added goes into the session too.
                    let settings = self.runner.turn_settings();
                    sink.commit(conversation, None, &settings).await?;
                    outcome
                }
                _ => outcome,
            };
            if outcome.stop != RunStop::Completed {
                return Ok(AgentOutcome {
                    run: outcome,
                    output: None,
                });
            }
            let text = outcome.output_text().to_owned();
            let output = match (self.parse)(&text) {
                Ok(output) => output,
                Err(error) if retries < self.output_retries => {
                    retries += 1;
                    conversation.push(crate::runner::note_item(&format!(
                        "Your answer does not match the required format ({error}). Answer again, \
                         with only JSON that matches the schema."
                    )));
                    continue;
                }
                Err(error) => return Err(Error::Output(error)),
            };
            for guardrail in &self.output_guardrails {
                if let Verdict::Trip(reason) = guardrail.check(context, &text).await? {
                    let mut run = outcome;
                    run.stop = RunStop::Guardrail {
                        name: guardrail.name().to_owned(),
                        reason,
                    };
                    return Ok(AgentOutcome { run, output: None });
                }
            }
            return Ok(AgentOutcome {
                run: outcome,
                output: Some(output),
            });
        }
    }

    /// Resume a run that stopped for approvals ([`RunStop::Approval`]), as
    /// [`Runner::resume`] does, with this agent's hooks (the input guardrails do not run
    /// again).
    ///
    /// # Errors
    ///
    /// As [`Runner::resume`] and [`run`](Self::run).
    pub async fn resume<F>(
        &mut self,
        context: &C,
        conversation: &mut Conversation,
        approvals: &[crate::runner::Approval],
        cancel: &CancellationSignal,
        mut on_event: F,
    ) -> Result<AgentOutcome<O>>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let hooks = Bound {
            context,
            hooks: &self.hooks,
        };
        if self
            .runner
            .apply(conversation, approvals, cancel, &hooks, &mut on_event)
            .await?
        {
            return Ok(AgentOutcome {
                run: RunOutcome::new(RunStop::Cancelled, Vec::new()),
                output: None,
            });
        }
        self.run_any(
            context,
            conversation,
            cancel,
            on_event,
            &mut NoSink,
            None,
            false,
        )
        .await
    }

    /// [`resume`](Self::resume) on a [`Session`]: the decisions' answers and what the run adds
    /// go into the session. The session holds the pending calls: a process that opens it later
    /// can resume.
    ///
    /// # Errors
    ///
    /// As [`resume`](Self::resume), and [`Error::Session`].
    pub async fn resume_session<S, F>(
        &mut self,
        context: &C,
        session: &mut Session<S>,
        approvals: &[crate::runner::Approval],
        cancel: &CancellationSignal,
        mut on_event: F,
    ) -> Result<AgentOutcome<O>>
    where
        S: SessionStore,
        F: FnMut(RunEvent<'_>) + MaybeSend,
    {
        let mut conversation = session.conversation();
        let id = session.id().as_str().to_owned();
        let mut sink = SessionSink {
            synced: conversation.len(),
            session,
        };
        let hooks = Bound {
            context,
            hooks: &self.hooks,
        };
        let aborted = self
            .runner
            .apply(&mut conversation, approvals, cancel, &hooks, &mut on_event)
            .await?;
        if aborted {
            let settings = self.runner.turn_settings();
            sink.commit(&conversation, None, &settings).await?;
            return Ok(AgentOutcome {
                run: RunOutcome::new(RunStop::Cancelled, Vec::new()),
                output: None,
            });
        }
        self.run_any(
            context,
            &mut conversation,
            cancel,
            on_event,
            &mut sink,
            Some(&id),
            false,
        )
        .await
    }

    /// Run the agent a handoff named, on the conversation (filtered, if the handoff says so),
    /// and give its outcome, with this run's requests first.
    async fn hand_over(
        handoffs: &mut [Handoff<C>],
        to: &str,
        before: RunOutcome,
        context: &C,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
    ) -> Result<RunOutcome> {
        let Some(handoff) = handoffs
            .iter_mut()
            .find(|handoff| handoff.agent.name() == to)
        else {
            return Ok(before);
        };
        let after = match handoff.filter {
            None => handoff.agent.run_dyn(context, conversation, cancel).await?,
            Some(filter) => {
                let mut filtered = filter(conversation);
                let start = filtered.len();
                let after = handoff
                    .agent
                    .run_dyn(context, &mut filtered, cancel)
                    .await?;
                conversation.extend(filtered.items()[start..].iter().cloned());
                after
            }
        };
        let mut turns = before.turns;
        turns.extend(after.turns);
        Ok(RunOutcome::new(after.stop, turns))
    }

    /// A run with the input guardrails checking at the same time: the first that trips cancels
    /// it.
    #[allow(clippy::too_many_arguments)]
    async fn run_guarded<F, K>(
        runner: &mut Runner<M>,
        guardrails: &[Arc<dyn InputGuardrail<C>>],
        extras: &Extras<'_>,
        context: &C,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
        on_event: &mut F,
        sink: &mut K,
        conversation_id: Option<&str>,
    ) -> Result<RunOutcome>
    where
        F: FnMut(RunEvent<'_>) + MaybeSend,
        K: Sink,
    {
        let input = conversation.clone();
        let inner = CancellationSignal::new();
        let span = runner.run_span(conversation_id);
        let check = async {
            for guardrail in guardrails {
                if let Verdict::Trip(reason) = guardrail.check(context, &input).await? {
                    inner.cancel();
                    return Ok(Some((guardrail.name().to_owned(), reason)));
                }
            }
            Ok::<_, Error>(None)
        };
        let run = runner
            .run_inner(conversation, &inner, on_event, sink, &span, extras)
            .instrument(span.clone());
        // The checks first: one that decides at once stops the run before its request.
        let both = futures::future::join(check, run);
        let forward = async {
            cancel.cancelled().await;
            inner.cancel();
        };
        futures::pin_mut!(both, forward);
        let (check, run) = match futures::future::select(both, forward).await {
            futures::future::Either::Left((done, _)) => done,
            // Cancelled from outside: the run stops on its own signal.
            futures::future::Either::Right(((), both)) => both.await,
        };
        // A tripped guardrail decides the outcome, whatever the run it cancelled came to.
        match (check?, run) {
            (Some((name, reason)), run) => {
                let turns = run.map(|outcome| outcome.turns).unwrap_or_default();
                Ok(RunOutcome::new(RunStop::Guardrail { name, reason }, turns))
            }
            (None, run) => run,
        }
    }
}

/// An agent another can hand over to.
struct Handoff<C> {
    agent: Box<dyn DynAgent<C>>,
    filter: Option<fn(&Conversation) -> Conversation>,
}

/// An agent of any model and output type, as another agent uses it: the other end of a
/// [handoff](Agent::handoff), or behind a [tool](AgentTool). [`Agent`] implements it.
pub trait DynAgent<C>: MaybeSend + MaybeSync {
    /// Its name.
    fn name(&self) -> &str;

    /// What it does.
    fn description(&self) -> &str;

    /// Become a sub-agent at `depth`: starting sub-agents of its own only above `max_depth`
    /// (the stricter of its own limit and this), and using only the tools `parent` allows as
    /// well as its own policy (`parent`: its parent's policies, those it inherited included).
    /// Default: nothing changes.
    fn enter_sub_agent(&mut self, depth: u32, max_depth: u32, parent: &[ToolPolicy]) {
        let _ = (depth, max_depth, parent);
    }

    /// Run on `conversation`, as [`Agent::run_with`] does; the answer is the outcome's text.
    fn run_dyn<'a>(
        &'a mut self,
        context: &'a C,
        conversation: &'a mut Conversation,
        cancel: &'a CancellationSignal,
    ) -> BoxFuture<'a, Result<RunOutcome>>;
}

impl<M, C, O> DynAgent<C> for Agent<M, C, O>
where
    M: Model,
    C: MaybeSync,
    O: MaybeSend + MaybeSync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn enter_sub_agent(&mut self, depth: u32, max_depth: u32, parent: &[ToolPolicy]) {
        self.depth = depth;
        self.sub_agent_limits.max_depth = self.sub_agent_limits.max_depth.min(max_depth);
        for policy in parent {
            self.runner.restrict(policy.clone());
        }
    }

    fn run_dyn<'a>(
        &'a mut self,
        context: &'a C,
        conversation: &'a mut Conversation,
        cancel: &'a CancellationSignal,
    ) -> BoxFuture<'a, Result<RunOutcome>> {
        Box::pin(async move {
            self.run_with(context, conversation, cancel, |_| {})
                .await
                .map(|outcome| outcome.run)
        })
    }
}

/// A handoff filter: only the messages of the conversation (no tool calls, outputs or
/// reasoning), for an agent that needs only what was said.
#[must_use]
pub fn without_tool_calls(conversation: &Conversation) -> Conversation {
    Conversation::from_items(
        conversation
            .items()
            .iter()
            .filter(|item| matches!(item, crate::item::Item::Message(_)))
            .cloned(),
    )
}

/// An agent as a tool: a call runs the agent on a conversation of its own (the call's `input`
/// as the user's message) and answers with its answer. Calls run one at a time.
pub struct AgentTool<C> {
    name: String,
    description: String,
    agent: tokio::sync::Mutex<Box<dyn DynAgent<C>>>,
    context: Arc<C>,
}

impl<C> std::fmt::Debug for AgentTool<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTool")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<C> AgentTool<C> {
    /// `agent` as the tool `name`, its runs getting `context`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: impl DynAgent<C> + 'static,
        context: Arc<C>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            agent: tokio::sync::Mutex::new(Box::new(agent)),
            context,
        }
    }
}

/// The input of an [`AgentTool`].
#[derive(Debug, serde::Deserialize)]
pub struct AgentToolInput {
    /// What to ask the agent.
    pub input: String,
}

impl<C: MaybeSend + MaybeSync + 'static> crate::tools::FunctionTool for AgentTool<C> {
    type Input = AgentToolInput;
    type Output = String;

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "What to ask, with everything the agent needs to know"
                }
            },
            "required": ["input"],
            "additionalProperties": false,
        })
    }

    async fn execute(&self, input: AgentToolInput) -> Result<String> {
        let mut agent = self.agent.lock().await;
        let mut conversation = Conversation::user(input.input);
        let outcome = agent
            .run_dyn(&self.context, &mut conversation, &CancellationSignal::new())
            .await?;
        if outcome.stop == RunStop::Completed {
            Ok(outcome.output_text().to_owned())
        } else {
            Err(Error::application(std::io::Error::other(format!(
                "the agent gave no answer: {:?}",
                outcome.stop
            ))))
        }
    }
}

#[cfg(not(target_family = "wasm"))]
type MakeAgent<C> = Box<dyn Fn() -> Box<dyn DynAgent<C>> + Send + Sync>;
#[cfg(target_family = "wasm")]
type MakeAgent<C> = Box<dyn Fn() -> Box<dyn DynAgent<C>>>;

/// An agent another can start in the background ([`Agent::sub_agent`]).
struct SubAgent<C> {
    name: String,
    tool: String,
    description: String,
    make: MakeAgent<C>,
}

impl<C> SubAgent<C> {
    fn new(name: String, description: String, make: MakeAgent<C>) -> Self {
        let tool = format!(
            "start_{}",
            crate::tools::sanitize_tool_name(&name).to_lowercase()
        );
        Self {
            name,
            tool,
            description,
            make,
        }
    }
}

/// The limits on an agent's sub-agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SubAgentLimits {
    /// The sub-agents of one run that may run at once. Default 4.
    pub max_running: usize,
    /// How deep sub-agents may start sub-agents: 1 lets the agent start sub-agents that start
    /// none. Default 2.
    pub max_depth: u32,
}

impl Default for SubAgentLimits {
    fn default() -> Self {
        Self {
            max_running: 4,
            max_depth: 2,
        }
    }
}

impl SubAgentLimits {
    /// These limits.
    #[must_use]
    pub fn new(max_running: usize, max_depth: u32) -> Self {
        Self {
            max_running,
            max_depth,
        }
    }
}

/// The sub-agents of one run: started by calls, polled by the runner, dropped (stopped) with the
/// run.
struct SubAgentPool<'a, C> {
    context: &'a C,
    specs: &'a [SubAgent<C>],
    depth: u32,
    limits: SubAgentLimits,
    policies: &'a [ToolPolicy],
    running: std::sync::Mutex<futures::stream::FuturesUnordered<BoxFuture<'a, Item>>>,
    started: std::sync::atomic::AtomicUsize,
}

impl<'a, C> SubAgentPool<'a, C> {
    fn new(
        context: &'a C,
        specs: &'a [SubAgent<C>],
        depth: u32,
        limits: SubAgentLimits,
        policies: &'a [ToolPolicy],
    ) -> Self {
        Self {
            context,
            specs,
            depth,
            limits,
            policies,
            running: std::sync::Mutex::new(futures::stream::FuturesUnordered::new()),
            started: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn lock(
        &self,
    ) -> std::sync::MutexGuard<'_, futures::stream::FuturesUnordered<BoxFuture<'a, Item>>> {
        self.running
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl<C: MaybeSync> Background for SubAgentPool<'_, C> {
    fn definitions(&self) -> Vec<crate::model::ToolDefinition> {
        self.specs
            .iter()
            .map(|spec| crate::model::ToolDefinition::Function {
                name: spec.tool.clone(),
                description: format!(
                    "Start the agent {} on a task, in the background: {} You get its answer as a \
                     message when it finishes; go on with other work meanwhile.",
                    spec.name, spec.description
                ),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task, with everything the agent needs to know"
                        }
                    },
                    "required": ["task"],
                    "additionalProperties": false,
                }),
                defer_loading: false,
            })
            .collect()
    }

    fn start(&self, call: &crate::item::ToolCall) -> Option<crate::item::ToolOutput> {
        use crate::item::ToolOutput;
        let spec = self.specs.iter().find(|spec| spec.tool == call.name)?;
        if self.depth >= self.limits.max_depth {
            return Some(ToolOutput::failure(
                call,
                "sub-agents cannot start sub-agents this deep; do the task yourself",
            ));
        }
        if self.running() >= self.limits.max_running {
            return Some(ToolOutput::failure(
                call,
                format!(
                    "{} sub-agents are running already, the most at once; wait for one to finish",
                    self.limits.max_running
                ),
            ));
        }
        let task = serde_json::from_str::<serde_json::Value>(&call.input)
            .ok()
            .and_then(|input| {
                input
                    .get("task")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            });
        let Some(task) = task else {
            return Some(ToolOutput::failure(
                call,
                "failed to parse function arguments: a `task` string is required",
            ));
        };
        let number = self
            .started
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        let mut agent = (spec.make)();
        agent.enter_sub_agent(self.depth + 1, self.limits.max_depth, self.policies);
        let context = self.context;
        let name = spec.name.clone();
        self.lock().push(Box::pin(async move {
            let mut conversation = Conversation::user(task);
            let cancel = CancellationSignal::new();
            let result = agent.run_dyn(context, &mut conversation, &cancel).await;
            let text = match result {
                Ok(outcome) if outcome.stop == RunStop::Completed => format!(
                    "The sub-agent {name} finished task {number}:\n{}",
                    outcome.output_text()
                ),
                Ok(outcome) => format!(
                    "The sub-agent {name} stopped task {number} without an answer: {:?}",
                    outcome.stop
                ),
                Err(error) => format!("The sub-agent {name} failed task {number}: {error}"),
            };
            Item::user(text)
        }));
        Some(ToolOutput::new(
            call,
            format!(
                "Started {} on task {number}; its answer comes as a message.",
                spec.name
            ),
            true,
        ))
    }

    fn poll_finished(&self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Item> {
        use futures::StreamExt;
        match self.lock().poll_next_unpin(cx) {
            std::task::Poll::Ready(Some(item)) => std::task::Poll::Ready(item),
            _ => std::task::Poll::Pending,
        }
    }

    fn running(&self) -> usize {
        self.lock().len()
    }
}
