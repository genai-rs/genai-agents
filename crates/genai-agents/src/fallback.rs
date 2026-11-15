//! Fallback models and credentials: a [`Model`] that sends each request to the first of its
//! routes that can take it.
//!
//! Routes come in groups: the routes of a group are the same model with other credentials (other
//! API keys or accounts of the same provider); the next group is another model, or another
//! provider.
//!
//! - **Refused or expired credentials** ([`ErrorKind::Authentication`]): the next credential of
//!   the same group; the refused one rests (for as long as the provider asked, or
//!   [`FallbackConfig::auth_rest`]).
//! - **A rate limit or a server failure** ([`ErrorKind::RateLimit`], [`ErrorKind::Server`]):
//!   the route rests for as long as the provider asked, or [`FallbackConfig::failure_rest`]
//!   once it failed [`FallbackConfig::max_failures`] times in a row, and the request goes to the
//!   next route that is not resting.
//! - **An unknown model** ([`ErrorKind::ModelNotFound`]): its whole group is left for good.
//! - **Local faults** ([`ErrorKind::Local`]: the request could not be built or sent) and
//!   everything else (a request the provider rejected, a context too long) never fall back: the
//!   next route would fail the same way.
//!
//! The route that answered last stays in use ([`Fallback::current`]): the provider keeps a
//! prompt cache per account, which another credential would not share. Only failures that come
//! before the response's body fall back; a stream that fails half-way is the runner's to retry.
//!
//! ```
//! use genai_agents::fallback::Fallback;
//! use genai_agents::testing::{Reply, ScriptedModel};
//! use genai_agents::{Conversation, ErrorKind, ModelError, Runner, ToolRegistry};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> genai_agents::Result<()> {
//! let expired = ScriptedModel::replies(vec![Reply::fail(
//!     ModelError::new("invalid api key").with_kind(ErrorKind::Authentication).with_status(401),
//! )]);
//! let spare_key = ScriptedModel::replies(vec![Reply::text("Hello.")]);
//! let other_provider = ScriptedModel::replies(vec![]);
//! let model = Fallback::new(expired).credential(spare_key).then(other_provider);
//! let mut runner = Runner::new(model, ToolRegistry::new());
//! let outcome = runner.run(&mut Conversation::user("Hi")).await?;
//! assert_eq!(outcome.output_text(), "Hello.");
//! assert_eq!(runner.model().current(), 1); // the spare key, from now on
//! # Ok(())
//! # }
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use crate::error::{ErrorKind, ModelError};
use crate::model::{Model, ModelInfo, ModelRequest, ModelStream};
use crate::platform::{BoxFuture, Instant, MaybeSend, MaybeSync};

/// A [`Model`] behind a box, for routes of different types.
trait DynModel: MaybeSend + MaybeSync {
    fn info(&self) -> ModelInfo;
    fn stream_dyn<'a, 'f>(
        &'a self,
        request: ModelRequest<'f>,
    ) -> BoxFuture<'f, Result<ModelStream<'a>, ModelError>>
    where
        'a: 'f;
}

impl<M: Model> DynModel for M {
    fn info(&self) -> ModelInfo {
        Model::info(self)
    }

    fn stream_dyn<'a, 'f>(
        &'a self,
        request: ModelRequest<'f>,
    ) -> BoxFuture<'f, Result<ModelStream<'a>, ModelError>>
    where
        'a: 'f,
    {
        Box::pin(self.stream(request))
    }
}

/// How long routes rest, and when a failing one gives way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct FallbackConfig {
    /// How long refused credentials rest when the provider does not say. Default 10 minutes.
    pub auth_rest: Duration,
    /// How long a failing route rests when the provider does not say. Default 30 seconds.
    pub failure_rest: Duration,
    /// The rate limits and server failures in a row after which a route rests, when the
    /// provider does not say how long to wait (a wait it asks for rests the route at once).
    /// Default 2.
    pub max_failures: u32,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            auth_rest: Duration::from_secs(600),
            failure_rest: Duration::from_secs(30),
            max_failures: 2,
        }
    }
}

impl FallbackConfig {
    /// The defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How long refused credentials rest.
    #[must_use]
    pub fn with_auth_rest(mut self, rest: Duration) -> Self {
        self.auth_rest = rest;
        self
    }

    /// How long a failing route rests.
    #[must_use]
    pub fn with_failure_rest(mut self, rest: Duration) -> Self {
        self.failure_rest = rest;
        self
    }

    /// The failures in a row after which a route rests.
    #[must_use]
    pub fn with_max_failures(mut self, failures: u32) -> Self {
        self.max_failures = failures.max(1);
        self
    }
}

/// What is known about a route.
#[derive(Debug, Default)]
struct Health {
    /// Until when it rests.
    resting_until: Option<Instant>,
    /// Left for good: its model does not exist.
    gone: bool,
    /// Rate limits and server failures in a row.
    failures: u32,
}

struct Route {
    model: Box<dyn DynModel>,
    group: usize,
}

/// A [`Model`] of routes, each request sent to the first that can take it. See the
/// [module documentation](self).
pub struct Fallback {
    routes: Vec<Route>,
    health: Mutex<Vec<Health>>,
    current: AtomicUsize,
    config: FallbackConfig,
}

impl std::fmt::Debug for Fallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fallback")
            .field("routes", &self.routes.len())
            .field("current", &self.current())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Fallback {
    /// Routes that start with `model`.
    pub fn new(model: impl Model + 'static) -> Self {
        Self {
            routes: vec![Route {
                model: Box::new(model),
                group: 0,
            }],
            health: Mutex::new(vec![Health::default()]),
            current: AtomicUsize::new(0),
            config: FallbackConfig::default(),
        }
    }

    /// The same routes, with `model` as another credential of the last group: the same model,
    /// another API key or account.
    #[must_use]
    pub fn credential(self, model: impl Model + 'static) -> Self {
        let group = self.routes.last().map_or(0, |route| route.group);
        self.push(model, group)
    }

    /// The same routes, with `model` as the start of a new group: another model or provider,
    /// tried when the groups before cannot answer.
    #[must_use]
    pub fn then(self, model: impl Model + 'static) -> Self {
        let group = self.routes.last().map_or(0, |route| route.group + 1);
        self.push(model, group)
    }

    fn push(mut self, model: impl Model + 'static, group: usize) -> Self {
        self.routes.push(Route {
            model: Box::new(model),
            group,
        });
        self.lock().push(Health::default());
        self
    }

    /// The same routes, with this configuration.
    #[must_use]
    pub fn with_config(mut self, config: FallbackConfig) -> Self {
        self.config = config;
        self
    }

    /// The route in use, from 0 in the order the routes were added.
    #[must_use]
    pub fn current(&self) -> usize {
        self.current.load(Ordering::SeqCst)
    }

    /// Use this route from now on (one a session started with, say), when there is one at that
    /// index.
    pub fn prefer(&self, route: usize) {
        if route < self.routes.len() {
            self.current.store(route, Ordering::SeqCst);
        }
    }

    /// The number of routes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Whether there is no route (never: [`new`](Self::new) takes the first).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Health>> {
        self.health.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The routes to try, in order: the current one first, then the others of its group, then
    /// the next groups and the earlier ones. Resting and gone routes are left out.
    fn order(&self) -> Vec<usize> {
        let now = Instant::now();
        let health = self.lock();
        let current = self.current().min(self.routes.len() - 1);
        let group = self.routes[current].group;
        let mut order: Vec<usize> = (0..self.routes.len()).collect();
        // Within the current group from the current route on, then the following groups, then
        // wrapping around.
        order.sort_by_key(|&index| {
            let route = &self.routes[index];
            let groups = self.routes.last().map_or(1, |route| route.group + 1);
            let group_distance = (route.group + groups - group) % groups;
            let index_distance = (index + self.routes.len() - current) % self.routes.len();
            (group_distance, index_distance)
        });
        order.retain(|&index| available(&health[index], now));
        order
    }

    /// Whether a route can take a request now.
    fn is_available(&self, route: usize) -> bool {
        available(&self.lock()[route], Instant::now())
    }

    /// Record a failure of `route`; returns whether the next route may take the request.
    fn failed(&self, route: usize, error: &ModelError) -> bool {
        let now = Instant::now();
        let mut health = self.lock();
        match error.kind() {
            ErrorKind::Authentication => {
                let rest = error.retry_after().unwrap_or(self.config.auth_rest);
                health[route].resting_until = Some(now + rest);
                true
            }
            ErrorKind::ModelNotFound => {
                let group = self.routes[route].group;
                for (index, other) in self.routes.iter().enumerate() {
                    if other.group == group {
                        health[index].gone = true;
                    }
                }
                true
            }
            ErrorKind::RateLimit | ErrorKind::Server => {
                let health = &mut health[route];
                health.failures += 1;
                if let Some(after) = error.retry_after() {
                    health.resting_until = Some(now + after);
                    true
                } else if health.failures >= self.config.max_failures {
                    health.resting_until = Some(now + self.config.failure_rest);
                    health.failures = 0;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn succeeded(&self, route: usize) {
        let mut health = self.lock();
        health[route].failures = 0;
        health[route].resting_until = None;
        drop(health);
        self.current.store(route, Ordering::SeqCst);
    }
}

impl Model for Fallback {
    fn info(&self) -> ModelInfo {
        let current = self.current().min(self.routes.len() - 1);
        self.routes[current].model.info()
    }

    async fn stream<'a>(
        &'a self,
        request: ModelRequest<'_>,
    ) -> Result<ModelStream<'a>, ModelError> {
        let order = self.order();
        let Some(&first) = order.first() else {
            return Err(
                ModelError::new("every route of the fallback is resting or gone")
                    .with_kind(ErrorKind::Server)
                    .with_retry_class(crate::error::RetryClass::Backoff),
            );
        };
        let mut last = None;
        for &route in &order {
            // A failure of an earlier route may have left this one (its group) for good.
            if !self.is_available(route) {
                continue;
            }
            if route != first {
                tracing::info!(
                    from = first,
                    to = route,
                    error = %last.as_ref().map_or(String::new(), ModelError::to_string),
                    "falling back to another route"
                );
            }
            match self.routes[route].model.stream_dyn(request).await {
                Ok(stream) => {
                    self.succeeded(route);
                    return Ok(stream);
                }
                Err(error) => {
                    let next = self.failed(route, &error);
                    if !next {
                        return Err(error);
                    }
                    last = Some(error);
                }
            }
        }
        Err(last.unwrap_or_else(|| ModelError::new("no route answered")))
    }
}

fn available(health: &Health, now: Instant) -> bool {
    !health.gone && health.resting_until.is_none_or(|until| until <= now)
}
