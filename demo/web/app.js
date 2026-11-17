// The demo page: one conversation with the household assistant, typed or spoken.
//
// - Typed: a chat session through the demo's API (demo/api/openapi.yaml), with buttons for the calls
//   that need approval.
// - Spoken: a GPT-Live call over WebRTC. The server creates the Live session for this page's offer
//   (its API key never comes here); GPT-Live delegates the household's questions to this application
//   (client delegation), which asks the assistant (`/api/live/delegations`) and gives GPT-Live the
//   answer to say (`session.commentary.append`).
"use strict";

const $ = (id) => document.getElementById(id);
const log = $("log");
const orb = $("orb");
const status = $("status");
const form = $("ask");
const question = $("question");
const send = $("send");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

// ---- Small helpers -----------------------------------------------------------------------------

function stored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Kept for as long as the page then.
  }
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const SVG = "http://www.w3.org/2000/svg";

// An icon from path data: `[["path", {d: ...}], ...]`.
function icon(className, shapes, viewBox = "0 0 24 24") {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", viewBox);
  svg.setAttribute("aria-hidden", "true");
  if (className) svg.setAttribute("class", className);
  for (const [tag, attributes] of shapes) {
    const shape = document.createElementNS(SVG, tag);
    for (const [name, value] of Object.entries(attributes)) shape.setAttribute(name, value);
    svg.append(shape);
  }
  return svg;
}

const icons = {
  avatar: () =>
    icon(
      "avatar",
      [
        ["path", { d: "M16 2c1.2 6.6 3.4 8.8 10 10-6.6 1.2-8.8 3.4-10 10-1.2-6.6-3.4-8.8-10-10 6.6-1.2 8.8-3.4 10-10Z" }],
        ["path", { d: "M25 19c.6 3.2 1.8 4.4 5 5-3.2.6-4.4 1.8-5 5-.6-3.2-1.8-4.4-5-5 3.2-.6 4.4-1.8 5-5Z", opacity: ".7" }],
      ],
      "0 0 32 32"
    ),
  wrench: () =>
    icon("", [["path", { d: "M14.7 6.3a4 4 0 0 0 5 5L21 13a6 6 0 0 1-8.2 1.6L6 21.4a2 2 0 0 1-2.8-2.8l6.8-6.8A6 6 0 0 1 11.6 3.6Z" }]]),
  check: () => icon("", [["path", { d: "M5 12.5 10 17 19 7" }]]),
  cross: () => icon("", [["path", { d: "M6 6l12 12M18 6 6 18" }]]),
  chevron: () => icon("chevron", [["path", { d: "M6 9l6 6 6-6" }]]),
};

function scrollDown() {
  log.scrollTop = log.scrollHeight;
}

// A message of the conversation: the person's on the right, the assistant's on the left.
function addMessage(role, text, className) {
  const row = element("div", "row " + role);
  if (role === "assistant") row.append(icons.avatar());
  const bubble = element("div", "bubble" + (className ? " " + className : ""), text);
  row.append(bubble);
  log.append(row);
  scrollDown();
  return bubble;
}

// JSON indented; a JSON string (a function tool's text output) as its text.
function pretty(text) {
  try {
    const value = JSON.parse(text);
    return typeof value === "string" ? value : JSON.stringify(value, null, 2);
  } catch {
    return text;
  }
}

function duration(ms) {
  if (ms < 1) return "<1 ms";
  return ms < 1000 ? ms + " ms" : (ms / 1000).toFixed(1) + " s";
}

// A tool call: a pill with its name and how it went, that opens into its input and output.
function toolCard(call) {
  const card = element("details", "tool");
  const summary = element("summary");
  summary.append(icons.wrench(), element("span", "", call.name));
  const state = element("span", "state");
  if (call.ok === true) {
    state.classList.add("ok");
    state.append(icons.check());
  } else if (call.ok === false) {
    state.classList.add("failed");
    state.append(icons.cross(), "failed");
  } else {
    state.append(call.pending ? call.pending : "waits for approval");
  }
  if (call.duration_ms != null) state.append(duration(call.duration_ms));
  summary.append(state, icons.chevron());
  card.append(summary);
  const section = (label, text) => {
    const part = element("div", "section");
    part.append(element("p", "label", label), element("pre", "", pretty(text)));
    card.append(part);
  };
  section("Input", call.arguments);
  if (call.output != null) section("Output", call.output);
  log.append(card);
  scrollDown();
  return card;
}

// A line under a message or a card: what it cost, and its trace.
function addMeta(after, result, text) {
  const meta = element("div", "meta indent", text);
  if (result.trace_url) {
    if (text) meta.append(" · ");
    const link = element("a", "", "trace");
    link.href = result.trace_url;
    link.target = "_blank";
    link.rel = "noopener";
    meta.append(link);
  } else if (result.trace_id) {
    meta.append((text ? " · " : "") + "trace " + result.trace_id);
  }
  after.after(meta);
}

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body ? { "content-type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.message || (data.error && data.error.message) || response.statusText);
  }
  return data;
}

function showNotice(text) {
  const notice = $("notice");
  notice.textContent = text;
  notice.hidden = false;
}

// ---- The theme, the settings and the text input ------------------------------------------------

const themeSwitch = $("theme");

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeSwitch.setAttribute("aria-checked", String(theme === "dark"));
}

applyTheme(document.documentElement.dataset.theme || "dark");
themeSwitch.addEventListener("click", () => {
  const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(theme);
  store("demo-theme", theme);
});

function openSettings(open) {
  $("drawer").hidden = !open;
  $("scrim").hidden = !open;
  if (open) $("close-settings").focus();
}

$("open-settings").addEventListener("click", () => openSettings(true));
$("close-settings").addEventListener("click", () => openSettings(false));
$("scrim").addEventListener("click", () => openSettings(false));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("drawer").hidden) openSettings(false);
});

const voiceSelect = $("voice");
voiceSelect.value = stored("demo-voice") || "";
voiceSelect.addEventListener("change", () => store("demo-voice", voiceSelect.value));

const keyboard = $("keyboard");

function showInput(show) {
  form.hidden = !show;
  keyboard.setAttribute("aria-pressed", String(show));
  store("demo-input", show ? "shown" : "hidden");
  if (show) question.focus();
}

showInput(stored("demo-input") !== "hidden");
keyboard.addEventListener("click", () => showInput(form.hidden));

// ---- The chat ----------------------------------------------------------------------------------

function newSessionId() {
  const bytes = new Uint8Array(6);
  crypto.getRandomValues(bytes);
  return "web-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

let session = stored("demo-session") || newSessionId();
store("demo-session", session);

function greet() {
  addMessage(
    "assistant",
    "Hello! Ask me about the household's notes: tap the microphone to talk, or type below. For example: " +
      "“When are we painting the kitchen?”"
  );
}

function showTurn(turn) {
  for (const call of turn.calls || []) {
    // A call that waits for approval shows in the approval card.
    if (call.ok == null && turn.status === "approval") continue;
    toolCard(call);
  }
  const cost = `${turn.requests} requests, ${turn.tokens} tokens`;
  if (turn.status === "answered") {
    addMeta(addMessage("assistant", turn.answer || "").parentElement, turn, cost);
  } else if (turn.status === "approval") {
    const card = element("div", "approval");
    card.append(element("p", "", "The assistant wants to run:"));
    for (const call of turn.pending) {
      card.append(element("pre", "", call.name + " " + pretty(call.arguments)));
    }
    const buttons = element("div", "buttons");
    const approve = element("button", "primary", "Approve");
    const deny = element("button", "secondary", "Deny");
    approve.type = deny.type = "button";
    const decide = async (ok) => {
      approve.disabled = deny.disabled = true;
      buttons.replaceWith(element("div", "meta", ok ? "Approved." : "Denied."));
      const decisions = turn.pending.map((call) => ({ call, approve: ok }));
      await run(() => api("POST", `/api/sessions/${session}/approvals`, { decisions }));
    };
    approve.addEventListener("click", () => decide(true));
    deny.addEventListener("click", () => decide(false));
    buttons.append(approve, deny);
    card.append(buttons);
    log.append(card);
    scrollDown();
  } else {
    addMessage("assistant", "The run stopped: " + (turn.detail || "unknown"), "error");
  }
}

async function run(request) {
  send.disabled = true;
  const thinking = addMessage("assistant", "Thinking…", "pending");
  try {
    const turn = await request();
    thinking.parentElement.remove();
    showTurn(turn);
  } catch (error) {
    thinking.parentElement.remove();
    addMessage("assistant", error.message, "error");
  } finally {
    send.disabled = false;
    if (!form.hidden) question.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) return;
  question.value = "";
  addMessage("user", text);
  run(() => api("POST", `/api/sessions/${session}/messages`, { text }));
});

$("new-session").addEventListener("click", () => {
  session = newSessionId();
  store("demo-session", session);
  $("session").textContent = session;
  log.replaceChildren();
  greet();
  openSettings(false);
});

// ---- The stage: a waveform through the orb, and a ring around it while someone speaks ----------

class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.mic = null; // analysers, while there is a call
    this.speaker = null;
    this.levels = { mic: 0, speaker: 0 };
    this.live = false;
    this.frame = null;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    const { width, height } = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = width;
    this.height = height;
    this.draw(performance.now());
  }

  // The loudness of an analyser's signal, from 0 to about 1.
  static level(analyser) {
    if (!analyser) return 0;
    const data = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (const sample of data) sum += sample * sample;
    return Math.min(1, Math.sqrt(sum / data.length) * 6);
  }

  start() {
    this.live = true;
    if (!this.frame && !reducedMotion.matches) this.frame = requestAnimationFrame((t) => this.tick(t));
  }

  stop() {
    this.live = false;
    this.mic = this.speaker = null;
    this.levels = { mic: 0, speaker: 0 };
  }

  tick(time) {
    const ease = (from, to) => from + (to - from) * (to > from ? 0.35 : 0.08);
    this.levels.mic = ease(this.levels.mic, Stage.level(this.mic));
    this.levels.speaker = ease(this.levels.speaker, Stage.level(this.speaker));
    this.draw(time);
    const quiet = this.levels.mic < 0.01 && this.levels.speaker < 0.01;
    this.frame = this.live || !quiet ? requestAnimationFrame((t) => this.tick(t)) : null;
  }

  draw(time) {
    const { context: c, width, height } = this;
    if (!width) return;
    c.clearRect(0, 0, width, height);
    const cx = width / 2;
    const cy = height / 2 - 8;
    const orbRadius = 48;
    const level = Math.max(this.levels.mic, this.levels.speaker);
    const t = reducedMotion.matches ? 0 : time / 1000;

    // The waveform: calm when idle, larger the louder the call.
    const gradient = c.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, "rgba(244, 114, 182, 0.15)");
    gradient.addColorStop(0.25, "#f472b6");
    gradient.addColorStop(0.45, "#fbbf24");
    gradient.addColorStop(0.55, "#4ade80");
    gradient.addColorStop(0.75, "#a78bfa");
    gradient.addColorStop(1, "rgba(167, 139, 250, 0.15)");
    c.strokeStyle = gradient;
    c.lineWidth = 2;
    c.shadowColor = "rgba(168, 85, 247, 0.6)";
    c.shadowBlur = 8;
    const amplitude = (this.live ? 3 : 1.5) + level * 34;
    const wave = (x) => {
      // Fades out towards the edges and into the orb.
      const edge = Math.min(1, Math.min(x, width - x) / (width * 0.25));
      const y =
        Math.sin(x * 0.018 + t * 2.1) * 0.6 + Math.sin(x * 0.041 - t * 3.3) * 0.3 + Math.sin(x * 0.09 + t * 5) * 0.1;
      return cy + y * amplitude * edge;
    };
    for (const [from, to] of [
      [0, cx - orbRadius - 6],
      [cx + orbRadius + 6, width],
    ]) {
      c.beginPath();
      for (let x = from; x <= to; x += 3) {
        if (x === from) c.moveTo(x, wave(x));
        else c.lineTo(x, wave(x));
      }
      c.stroke();
    }

    // The ring: amber while the assistant speaks, violet while the caller does.
    if (this.live && level > 0.02) {
      const speaking = this.levels.speaker >= this.levels.mic;
      c.strokeStyle = speaking ? "#facc15" : "#c084fc";
      c.shadowColor = speaking ? "rgba(250, 204, 21, 0.8)" : "rgba(192, 132, 252, 0.8)";
      c.shadowBlur = 16;
      c.lineWidth = 3;
      c.beginPath();
      for (let i = 0; i <= 96; i++) {
        const angle = (i / 96) * Math.PI * 2;
        const wobble = Math.sin(angle * 9 + t * 4) * 0.6 + Math.sin(angle * 5 - t * 3) * 0.4;
        const r = orbRadius + 8 + level * 10 + wobble * level * 7;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) c.moveTo(x, y);
        else c.lineTo(x, y);
      }
      c.stroke();
    }
    c.shadowBlur = 0;
  }
}

const stage = new Stage($("wave"));

// ---- The voice call ----------------------------------------------------------------------------

// What was said, for the assistant: GPT-Live's delegation event carries no task text.
class Transcript {
  constructor() {
    this.messages = [];
    this.bubbles = [];
  }

  // A transcript delta of `role`: it continues that role's last message, or starts a new one when
  // the other side spoke in between.
  add(role, delta) {
    const last = this.messages[this.messages.length - 1];
    if (last && last.role === role) {
      last.text += delta;
      this.bubbles[this.bubbles.length - 1].textContent = last.text;
      scrollDown();
    } else {
      this.messages.push({ role, text: delta });
      this.bubbles.push(addMessage(role, delta));
    }
  }

  recent() {
    return this.messages
      .filter((message) => message.text.trim())
      .slice(-20)
      .map((message) => ({ role: message.role, text: message.text.trim() }));
  }

  lastQuestion() {
    const asked = this.recent().filter((message) => message.role === "user");
    return asked.length ? asked[asked.length - 1].text : "";
  }
}

// Resolves once ICE gathering is complete (or after two seconds), so the offer holds the
// candidates: the server posts it once and gets one answer.
function gathered(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", done);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", done);
    setTimeout(resolve, 2000);
  });
}

function setOrb(state, label, text) {
  orb.dataset.state = state;
  orb.setAttribute("aria-label", label);
  status.textContent = text;
}

let call = null;

class Call {
  constructor() {
    this.transcript = new Transcript();
    this.started = false;
    this.closed = null;
  }

  async start(voice) {
    setOrb("connecting", "Connecting", "Asking for the microphone…");
    this.mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audio = new AudioContext();
    stage.mic = this.analyser(this.mic);
    this.pc = new RTCPeerConnection();
    this.pc.addEventListener("track", (event) => {
      $("speaker").srcObject = event.streams[0];
      stage.speaker = this.analyser(event.streams[0]);
    });
    for (const track of this.mic.getAudioTracks()) this.pc.addTrack(track, this.mic);
    this.channel = this.pc.createDataChannel("oai-events");
    this.channel.addEventListener("message", (event) => this.receive(JSON.parse(event.data)));
    this.channel.addEventListener("close", () => this.ended("The connection closed."));

    status.textContent = "Connecting…";
    await this.pc.setLocalDescription(await this.pc.createOffer());
    await gathered(this.pc);
    const session = await api("POST", "/api/live/sessions", {
      sdp: this.pc.localDescription.sdp,
      voice: voice || undefined,
    });
    this.id = session.id;
    await this.pc.setRemoteDescription({ type: "answer", sdp: session.sdp });
  }

  analyser(stream) {
    const analyser = this.audio.createAnalyser();
    analyser.fftSize = 1024;
    this.audio.createMediaStreamSource(stream).connect(analyser);
    return analyser;
  }

  send(event) {
    if (this.channel && this.channel.readyState === "open") this.channel.send(JSON.stringify(event));
  }

  receive(event) {
    switch (event.type) {
      case "session.started":
        this.started = true;
        setOrb("live", "Hang up", "Listening: speak");
        stage.start();
        break;
      case "session.input_transcript.delta":
        this.transcript.add("user", event.delta);
        break;
      case "session.output_transcript.delta":
        this.transcript.add("assistant", event.delta);
        break;
      case "session.delegation.created":
        this.delegate(event.delegation.id);
        break;
      case "session.closed":
        if (this.closed) this.closed();
        this.ended("The call ended.");
        break;
      case "error":
        addMessage("assistant", "GPT-Live: " + ((event.error && event.error.message) || JSON.stringify(event)), "error");
        break;
      default:
        break;
    }
  }

  // GPT-Live handed a task to the application: the assistant answers it from the transcript.
  async delegate(id) {
    const asked = this.transcript.lastQuestion();
    const card = toolCard({ name: "household assistant", arguments: JSON.stringify({ question: asked }), pending: "working…" });
    const started = performance.now();
    status.textContent = "Looking it up…";
    try {
      const result = await api("POST", "/api/live/delegations", {
        delegation_id: id,
        transcript: this.transcript.recent(),
      });
      card.remove();
      for (const c of result.calls) toolCard(c);
      const done = toolCard({
        name: "household assistant",
        arguments: JSON.stringify({ question: asked }),
        output: result.answer,
        ok: true,
        duration_ms: Math.round(performance.now() - started),
      });
      if (result.trace_url || result.trace_id) addMeta(done, result, "");
      this.send({ type: "session.commentary.append", delegation_id: id, content: result.answer });
    } catch (error) {
      card.remove();
      toolCard({
        name: "household assistant",
        arguments: JSON.stringify({ question: asked }),
        output: error.message,
        ok: false,
        duration_ms: Math.round(performance.now() - started),
      });
      this.send({
        type: "session.commentary.append",
        delegation_id: id,
        content: "The household's notes cannot be reached right now; say so and offer to try again.",
      });
    } finally {
      if (this.started && !this.done) status.textContent = "Listening: speak";
    }
  }

  // Close gracefully: `session.close`, then wait for `session.closed` (or five seconds).
  async hangUp() {
    if (this.started) {
      const closed = new Promise((resolve) => {
        this.closed = resolve;
        setTimeout(resolve, 5000);
      });
      this.send({ type: "session.close" });
      setOrb("connecting", "Hanging up", "Hanging up…");
      await closed;
    }
    this.ended("The call ended.");
  }

  ended(message) {
    if (this.done) return;
    this.done = true;
    if (this.channel) this.channel.close();
    if (this.pc) this.pc.close();
    if (this.mic) for (const track of this.mic.getTracks()) track.stop();
    if (this.audio) this.audio.close();
    stage.stop();
    setOrb("idle", "Start a voice call", message + " Tap to talk again.");
    call = null;
  }
}

orb.addEventListener("click", async () => {
  if (call) {
    if (call.started) call.hangUp();
    return;
  }
  call = new Call();
  try {
    await call.start(voiceSelect.value);
  } catch (error) {
    addMessage("assistant", error.message, "error");
    if (call) call.ended("Not connected.");
  }
});

// ---- Start ------------------------------------------------------------------------------------

async function start() {
  $("session").textContent = session;
  if (!window.isSecureContext || !navigator.mediaDevices) {
    showNotice(
      "The browser only gives a page the microphone on https or on localhost: open this page as " +
        "http://localhost:<port>/ (through an SSH tunnel, for instance) to talk."
    );
    orb.disabled = true;
    status.textContent = "Voice needs https or localhost";
  }
  try {
    const info = await api("GET", "/api/info");
    $("model").textContent = info.model;
    $("live-model").textContent = info.live_model || "off: set OPENAI_API_KEY";
    $("tracing").textContent = info.tracing ? "to Langfuse" : "off";
    if (!info.live) {
      orb.disabled = true;
      status.textContent = "Voice is off: start the demo with OPENAI_API_KEY";
    }
  } catch {
    // The page works without it.
  }
  greet();
  try {
    const history = await api("GET", `/api/sessions/${session}`);
    for (const message of history.messages) addMessage(message.role, message.text);
  } catch (error) {
    addMessage("assistant", error.message, "error");
  }
  if (!form.hidden) question.focus();
}

start();
