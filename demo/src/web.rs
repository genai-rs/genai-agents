//! `serve`: the web pages, on the server generated from `demo/api/openapi.yaml`.
//!
//! `/` is one conversation with the assistant, typed or spoken, with the tool calls it made:
//!
//! - Typed: a chat in a session, with buttons for the calls that need approval.
//! - Spoken: a call with GPT-Live in the browser (WebRTC). GPT-Live hands the questions it
//!   should not answer itself to the application (client delegation); the page sends them to
//!   `/api/live/delegations`, where the assistant answers them, and gives GPT-Live the answer to
//!   say (`session.commentary.append`).
//!
//! The OpenAI API key stays on the server: it creates the Live session for the browser's offer.

use std::collections::HashMap;
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

use genai_agents::agent::Agent;
use genai_agents::fallback::Fallback;
use genai_agents::runner::{Approval, PendingCall};
use genai_agents::session::{Session, SessionId};
use genai_agents::{CancellationSignal, Conversation, Role, RunStop};
use genai_agents_demo_api::models;
use genai_agents_demo_api::server::assistant_api::{
    AssistantApi, DecideApprovalsRequest, DecideApprovalsResponse, GetInfoRequest, GetInfoResponse,
    GetSessionRequest, GetSessionResponse, SendMessageRequest, SendMessageResponse,
};
use genai_agents_demo_api::server::live_api::{
    CreateLiveSessionRequest, CreateLiveSessionResponse, LiveApi, RunDelegationRequest,
    RunDelegationResponse,
};
use genai_agents_demo_api::server::{router, ErrorHandler, ServerError};
use genai_agents_sqlite::SqliteStore;
use openai_client_base::apis::configuration::{Configuration, SecretString};
use openai_client_base::apis::live_api;
use openai_client_base::models as openai;
use tokio::sync::Mutex;
use tracing::Instrument as _;

use crate::assistant::{
    self, assistant, store, transcript, BoxError, CallLog, CallReport, Channel, Input, TurnReport,
};
use crate::model::Tracing;
use crate::tools::Workspace;

/// GPT-Live's instructions: how it talks, and what it hands to the assistant. Following OpenAI's
/// Live prompting guide, the household's facts are the backend's; the voice only talks.
const LIVE_INSTRUCTIONS: &str = "You are the friendly voice of a household's assistant. Speak \
briefly and naturally, one or two sentences at a time, in the caller's language.

# Delegation policy
A backend knows the household's notes: its plans, dates, amounts, addresses and who to call. \
Delegate every question about the household to the backend, and every request to write a \
note. Never guess an answer about the household; wait for the backend's result before you say \
a date, an amount or a name. Small talk and questions about how you work need no backend.";

/// The page, built into the binary. `/voice` is the same page: the voice and the chat are one
/// conversation now, and old links keep working.
const PAGES: [(&str, &str, &str); 4] = [
    (
        "/",
        "text/html; charset=utf-8",
        include_str!("../web/index.html"),
    ),
    (
        "/voice",
        "text/html; charset=utf-8",
        include_str!("../web/index.html"),
    ),
    (
        "/app.css",
        "text/css; charset=utf-8",
        include_str!("../web/app.css"),
    ),
    (
        "/app.js",
        "text/javascript; charset=utf-8",
        include_str!("../web/app.js"),
    ),
];

/// Serve the pages on `DEMO_ADDR` (default `127.0.0.1:8080`) until Ctrl-C.
pub async fn serve(workspace: Workspace, tracing: Arc<Tracing>) -> Result<(), BoxError> {
    let addr: SocketAddr = env::var("DEMO_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8080".into())
        .parse()?;
    let store = store(&workspace).await?;
    let (_, label) = assistant(&workspace, Channel::Text)?;
    let server = Server {
        workspace,
        store,
        tracing,
        label,
        live: Live::from_env(),
        agents: Mutex::new(HashMap::new()),
    };
    let mut app = router(server);
    for (path, content_type, body) in PAGES {
        app = app.route(
            path,
            axum::routing::get(
                move || async move { ([(http_content_type(), content_type)], body) },
            ),
        );
    }
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("[serving http://{addr}: the page at /; Ctrl-C stops]");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn http_content_type() -> axum::http::HeaderName {
    axum::http::header::CONTENT_TYPE
}

/// The Live model, when there is an OpenAI API key to create its sessions with.
struct Live {
    configuration: Configuration,
    model: String,
}

impl Live {
    /// `OPENAI_API_KEY`, and `DEMO_LIVE_MODEL` (default `gpt-live-1`).
    fn from_env() -> Option<Self> {
        let key = env::var("OPENAI_API_KEY")
            .ok()
            .filter(|key| !key.is_empty())?;
        let mut configuration = Configuration::new();
        configuration.bearer_access_token = Some(SecretString::from(key));
        if let Ok(base) = env::var("OPENAI_BASE_URL") {
            configuration.base_path = base.trim_end_matches('/').to_owned();
        }
        let model = env::var("DEMO_LIVE_MODEL").unwrap_or_else(|_| "gpt-live-1".into());
        Some(Self {
            configuration,
            model,
        })
    }
}

struct Server {
    workspace: Workspace,
    store: SqliteStore,
    tracing: Arc<Tracing>,
    label: String,
    live: Option<Live>,
    /// An assistant per chat session: its turns go one at a time, and its fallback remembers
    /// which route answered last.
    agents: Mutex<HashMap<String, Arc<Mutex<Agent<Fallback>>>>>,
}

impl Server {
    async fn agent(&self, session: &str) -> Result<Arc<Mutex<Agent<Fallback>>>, BoxError> {
        let mut agents = self.agents.lock().await;
        if let Some(agent) = agents.get(session) {
            return Ok(Arc::clone(agent));
        }
        let (agent, _) = assistant(&self.workspace, Channel::Text)?;
        let agent = Arc::new(Mutex::new(agent));
        agents.insert(session.to_owned(), Arc::clone(&agent));
        Ok(agent)
    }

    /// Run a turn of a chat session.
    async fn chat_turn(&self, session: &str, input: Input<'_>) -> Result<models::Turn, BoxError> {
        let agent = self.agent(session).await?;
        let mut agent = agent.lock().await;
        let mut session = Session::open(&self.store, SessionId::new(session)).await?;
        let report =
            assistant::turn(&mut agent, &mut session, &self.tracing, input, |_| {}).await?;
        Ok(self.turn_model(report))
    }

    fn turn_model(&self, report: TurnReport) -> models::Turn {
        let (status, detail) = match &report.stop {
            RunStop::Completed => (models::turn::Status::Answered, None),
            RunStop::Approval { .. } => (models::turn::Status::Approval, None),
            other => (models::turn::Status::Stopped, Some(format!("{other:?}"))),
        };
        let mut turn = models::Turn::new(
            status,
            calls(&report),
            i32::try_from(report.requests).unwrap_or(i32::MAX),
            report.tokens,
        );
        turn.answer = report.answer.clone();
        turn.detail = detail;
        turn.pending = Some(report.pending().iter().map(pending_model).collect());
        turn.trace_url = report.trace_id.as_deref().and_then(trace_url);
        turn.trace_id = report.trace_id;
        turn
    }

    /// Answer what GPT-Live delegated: a run of the assistant on the transcript, without a
    /// session. Writes are not approved by voice: the assistant hears that the chat page does
    /// them.
    async fn delegation(
        &self,
        delegation_id: &str,
        messages: &[models::Message],
    ) -> Result<models::DelegationResult, BoxError> {
        let (mut agent, _) = assistant(&self.workspace, Channel::Voice)?;
        let mut conversation = Conversation::new();
        for message in messages {
            match message.role {
                models::Role::User => conversation.push_user(message.text.clone()),
                models::Role::Assistant => conversation.push_assistant(message.text.clone()),
                // A role this version of the spec does not know: not part of what was said.
                models::Role::Unknown => {}
            }
        }
        let span = tracing::info_span!("delegation", delegation_id);
        let cancel = CancellationSignal::new();
        let mut log = CallLog::default();
        let mut outcome = agent
            .run_with(&(), &mut conversation, &cancel, |event| log.observe(&event))
            .instrument(span.clone())
            .await?;
        if let RunStop::Approval { pending } = &outcome.run.stop {
            let denied: Vec<Approval> = pending
                .iter()
                .map(|call| call.deny("notes are written on the chat page, not by voice"))
                .collect();
            outcome = agent
                .resume(&(), &mut conversation, &denied, &cancel, |event| {
                    log.observe(&event)
                })
                .instrument(span.clone())
                .await?;
        }
        let answer = match (&outcome.run.stop, outcome.output()) {
            (RunStop::Completed, Some(answer)) => answer.clone(),
            (stop, _) => format!("The assistant could not answer ({stop:?})."),
        };
        let calls = log
            .finish(conversation.items())
            .iter()
            .map(call_model)
            .collect();
        let mut result = models::DelegationResult::new(answer, calls);
        if self.tracing.enabled() {
            let trace = Tracing::trace_id(&span);
            result.trace_url = trace_url(&trace);
            result.trace_id = Some(trace);
        }
        Ok(result)
    }
}

fn calls(report: &TurnReport) -> Vec<models::Call> {
    report.calls.iter().map(call_model).collect()
}

fn call_model(call: &CallReport) -> models::Call {
    let mut model = models::Call::new(call.name.clone(), call.arguments.clone());
    model.output = call.output.clone();
    model.ok = call.ok;
    model.duration_ms = call
        .duration_ms
        .map(|ms| i64::try_from(ms).unwrap_or(i64::MAX));
    model
}

/// A pending call for the page. Both sides serialize the same fields; the page sends it back
/// unchanged with its decision.
fn pending_model(call: &PendingCall) -> models::PendingCall {
    let mut model = models::PendingCall::new(
        call.call_id.clone(),
        call.name.clone(),
        call.arguments.clone(),
        call.key.clone(),
        call.hash.clone(),
    );
    model.namespace = call.namespace.clone();
    model
}

/// The page's pending call, back as genai-agents' own: the decision is bound to its hash, so a
/// call whose arguments the page changed is not approved.
fn pending_call(model: &models::PendingCall) -> Result<PendingCall, BoxError> {
    Ok(serde_json::from_value(serde_json::to_value(model)?)?)
}

/// The trace in the Langfuse UI, when `LANGFUSE_PROJECT_ID` says which project it is in.
fn trace_url(trace_id: &str) -> Option<String> {
    let project = env::var("LANGFUSE_PROJECT_ID").ok()?;
    let host = env::var("LANGFUSE_HOST").unwrap_or_else(|_| "https://cloud.langfuse.com".into());
    Some(format!(
        "{}/project/{project}/traces/{trace_id}",
        host.trim_end_matches('/')
    ))
}

/// A session id the store and the URLs take: letters, digits, `-` and `_`, as the spec says.
/// (The generated server does not validate parameters yet.)
fn valid_session(session: &str) -> bool {
    (1..=64).contains(&session.len())
        && session
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn problem(status: u16, message: impl Into<String>) -> (axum::http::StatusCode, models::Problem) {
    (
        axum::http::StatusCode::from_u16(status).unwrap_or(axum::http::StatusCode::BAD_REQUEST),
        models::Problem::new(message.into()),
    )
}

impl ErrorHandler for Server {
    /// Log the error, and answer with its message: the demo runs for its developer.
    fn server_error(&self, error: ServerError) -> axum::response::Response {
        eprintln!("error: {error}");
        let (status, body) = problem(500, error.to_string());
        axum::response::IntoResponse::into_response((status, axum::Json(body)))
    }
}

impl AssistantApi for Server {
    async fn get_info(&self, _: GetInfoRequest) -> Result<GetInfoResponse, ServerError> {
        let mut info = models::Info::new(
            self.label.clone(),
            self.tracing.enabled(),
            self.live.is_some(),
        );
        info.live_model = self.live.as_ref().map(|live| live.model.clone());
        Ok(GetInfoResponse::Status200(info))
    }

    async fn get_session(
        &self,
        request: GetSessionRequest,
    ) -> Result<GetSessionResponse, ServerError> {
        if !valid_session(&request.session) {
            let (status, body) = problem(400, "a session id is 1 to 64 letters, digits, - or _");
            return Ok(GetSessionResponse::Default { status, body });
        }
        let session = Session::open(&self.store, SessionId::new(request.session))
            .await
            .map_err(fail)?;
        let messages = transcript(session.conversation().items())
            .into_iter()
            .map(|(role, text)| {
                let role = if role == Role::User {
                    models::Role::User
                } else {
                    models::Role::Assistant
                };
                models::Message::new(role, text)
            })
            .collect();
        Ok(GetSessionResponse::Status200(models::History::new(
            messages,
        )))
    }

    async fn send_message(
        &self,
        request: SendMessageRequest,
    ) -> Result<SendMessageResponse, ServerError> {
        if !valid_session(&request.session) {
            let (status, body) = problem(400, "a session id is 1 to 64 letters, digits, - or _");
            return Ok(SendMessageResponse::Default { status, body });
        }
        let key = format!(
            "web-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(fail)?
                .as_nanos()
        );
        let input = Input::Question {
            key,
            text: &request.body.text,
        };
        let turn = self
            .chat_turn(&request.session, input)
            .await
            .map_err(fail)?;
        Ok(SendMessageResponse::Status200(turn))
    }

    async fn decide_approvals(
        &self,
        request: DecideApprovalsRequest,
    ) -> Result<DecideApprovalsResponse, ServerError> {
        if !valid_session(&request.session) {
            let (status, body) = problem(400, "a session id is 1 to 64 letters, digits, - or _");
            return Ok(DecideApprovalsResponse::Default { status, body });
        }
        let mut decisions = Vec::new();
        for decision in &request.body.decisions {
            let Ok(call) = pending_call(&decision.call) else {
                let (status, body) = problem(400, "a pending call does not decode");
                return Ok(DecideApprovalsResponse::Default { status, body });
            };
            decisions.push(if decision.approve {
                call.approve()
            } else {
                call.deny("the user said no")
            });
        }
        match self
            .chat_turn(&request.session, Input::Decisions(&decisions))
            .await
        {
            Ok(turn) => Ok(DecideApprovalsResponse::Status200(turn)),
            // A decision for a call that is not pending (any more), or whose arguments changed.
            Err(error) => match error.downcast_ref::<genai_agents::Error>() {
                Some(genai_agents::Error::Approval(message)) => {
                    let (status, body) = problem(409, message.clone());
                    Ok(DecideApprovalsResponse::Default { status, body })
                }
                _ => Err(fail(error)),
            },
        }
    }
}

impl LiveApi for Server {
    async fn create_live_session(
        &self,
        request: CreateLiveSessionRequest,
    ) -> Result<CreateLiveSessionResponse, ServerError> {
        let Some(live) = &self.live else {
            let (status, body) = problem(503, "set OPENAI_API_KEY to use GPT-Live");
            return Ok(CreateLiveSessionResponse::Default { status, body });
        };
        let mut session = openai::LiveMediaSessionCreateParams::new(live.model.clone().into());
        session.instructions = Some(Some(LIVE_INSTRUCTIONS.to_owned()));
        session.delegation = Some(Some(openai::Delegation::LiveClientDelegationParam(
            openai::LiveClientDelegationParam::new(
                openai::live_client_delegation_param::Type::Client,
            ),
        )));
        if let Some(voice) = request.body.voice {
            let mut output = openai::LiveInitialSessionAudioOutputParam::new();
            output.voice = Some(voice);
            let mut audio = openai::LiveMediaSessionAudioParam::new();
            audio.output = Some(output);
            session.audio = Some(audio);
        }
        let transport = openai::LiveSessionCreateRequestTransport::LiveWebRtcTransport(
            openai::LiveWebRtcTransport::new(
                openai::live_web_rtc_transport::Type::Webrtc,
                request.body.sdp,
            ),
        );
        let create = openai::LiveSessionCreateRequest::new(session, transport);
        let created = live_api::create_live()
            .configuration(&live.configuration)
            .live_session_create_request(&create)
            .call()
            .await;
        let created = match created {
            Ok(created) => created,
            Err(error) => {
                // OpenAI's own message is in the body of its error response.
                let message = match &error {
                    openai_client_base::apis::Error::ResponseError(response) => {
                        format!("GPT-Live: {} {}", response.status, response.content)
                    }
                    other => format!("GPT-Live: {other}"),
                };
                eprintln!("{message}");
                let (status, body) = problem(502, message);
                return Ok(CreateLiveSessionResponse::Default { status, body });
            }
        };
        let openai::LiveSessionCreateResponseTransport::LiveWebRtcTransport(answer) =
            created.transport
        else {
            let (status, body) = problem(502, "GPT-Live answered without a WebRTC answer");
            return Ok(CreateLiveSessionResponse::Default { status, body });
        };
        Ok(CreateLiveSessionResponse::Status201(
            models::LiveSession::new(created.session.id, answer.sdp),
        ))
    }

    async fn run_delegation(
        &self,
        request: RunDelegationRequest,
    ) -> Result<RunDelegationResponse, ServerError> {
        let result = self
            .delegation(&request.body.delegation_id, &request.body.transcript)
            .await
            .map_err(fail)?;
        Ok(RunDelegationResponse::Status200(result))
    }
}

/// A `ServerError` from any error, the demo's boxed ones too.
fn fail(error: impl std::fmt::Display) -> ServerError {
    ServerError::new(error.to_string())
}
