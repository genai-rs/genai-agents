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
use crate::model::{Model, OutputSchema};
use crate::platform::{BoxFuture, MaybeSend, MaybeSync};
use crate::runner::{
    Bound, CancellationSignal, Hooks, NoSink, RunConfig, RunEvent, RunOutcome, RunStop, Runner,
    SessionSink, Sink, ToolPolicy,
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
        self.run_any(context, conversation, cancel, on_event, &mut NoSink, None)
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
        )
        .await
    }

    async fn run_any<F, K>(
        &mut self,
        context: &C,
        conversation: &mut Conversation,
        cancel: &CancellationSignal,
        mut on_event: F,
        sink: &mut K,
        conversation_id: Option<&str>,
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

        let mut retries = 0;
        let mut first = true;
        loop {
            let outcome = if first && !self.input_guardrails.is_empty() {
                self.run_guarded(
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
                let hooks = Bound {
                    context,
                    hooks: &self.hooks,
                };
                self.runner
                    .run_inner(conversation, cancel, &mut on_event, sink, &span, &hooks)
                    .instrument(span.clone())
                    .await?
            };
            first = false;
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

    /// A run with the input guardrails checking at the same time: the first that trips cancels
    /// it.
    async fn run_guarded<F, K>(
        &mut self,
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
        let span = self.runner.run_span(conversation_id);
        let hooks = Bound {
            context,
            hooks: &self.hooks,
        };
        let guardrails = &self.input_guardrails;
        let check = async {
            for guardrail in guardrails {
                if let Verdict::Trip(reason) = guardrail.check(context, &input).await? {
                    inner.cancel();
                    return Ok(Some((guardrail.name().to_owned(), reason)));
                }
            }
            Ok::<_, Error>(None)
        };
        let run = self
            .runner
            .run_inner(conversation, &inner, on_event, sink, &span, &hooks)
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
