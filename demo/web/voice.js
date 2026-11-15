// The voice page: a GPT-Live call over WebRTC. The server creates the session for this page's
// offer (its API key never comes here); GPT-Live delegates the household's questions to this
// application (client delegation), which asks the assistant (`/api/live/delegations`) and gives
// GPT-Live the answer to say (`session.commentary.append`).
"use strict";

const log = document.getElementById("log");
const status = document.getElementById("status");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const notice = document.getElementById("notice");

let call = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStatus(text) {
  status.textContent = text;
}

function showNotice(text) {
  notice.textContent = text;
  notice.hidden = false;
}

function addEntry(className, text) {
  const node = element("div", "msg " + className, text);
  log.append(node);
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  return node;
}

// What was said, for the assistant: GPT-Live's delegation event carries no task text.
class Transcript {
  constructor() {
    this.messages = [];
    this.nodes = [];
  }

  // A transcript delta of `role`: it continues that role's last message, or starts a new one
  // when the other side spoke in between.
  add(role, delta) {
    const last = this.messages[this.messages.length - 1];
    if (last && last.role === role) {
      last.text += delta;
      this.nodes[this.nodes.length - 1].textContent = last.text;
    } else {
      this.messages.push({ role, text: delta });
      this.nodes.push(addEntry(role, delta));
    }
  }

  recent() {
    return this.messages
      .filter((message) => message.text.trim())
      .slice(-20)
      .map((message) => ({ role: message.role, text: message.text.trim() }));
  }
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.message || (data.error && data.error.message) || response.statusText);
  }
  return data;
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

class Call {
  constructor() {
    this.transcript = new Transcript();
    this.started = false;
    this.closed = null;
  }

  async start(voice) {
    setStatus("Asking for the microphone…");
    this.mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.pc = new RTCPeerConnection();
    this.pc.addEventListener("track", (event) => {
      document.getElementById("speaker").srcObject = event.streams[0];
    });
    for (const track of this.mic.getAudioTracks()) this.pc.addTrack(track, this.mic);
    this.channel = this.pc.createDataChannel("oai-events");
    this.channel.addEventListener("message", (event) => this.receive(JSON.parse(event.data)));
    this.channel.addEventListener("close", () => this.ended("The connection closed."));

    setStatus("Connecting…");
    await this.pc.setLocalDescription(await this.pc.createOffer());
    await gathered(this.pc);
    const session = await post("/api/live/sessions", { sdp: this.pc.localDescription.sdp, voice: voice || undefined });
    this.id = session.id;
    await this.pc.setRemoteDescription({ type: "answer", sdp: session.sdp });
  }

  send(event) {
    if (this.channel && this.channel.readyState === "open") {
      this.channel.send(JSON.stringify(event));
    }
  }

  receive(event) {
    switch (event.type) {
      case "session.started":
        this.started = true;
        setStatus("Connected: speak");
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
        addEntry("assistant error", "GPT-Live: " + ((event.error && event.error.message) || JSON.stringify(event)));
        break;
      default:
        break;
    }
  }

  // GPT-Live handed a task to the application: the assistant answers it from the transcript.
  async delegate(id) {
    const entry = addEntry("assistant", "Asking the assistant…");
    const meta = element("div", "meta");
    entry.append(meta);
    try {
      const result = await post("/api/live/delegations", {
        delegation_id: id,
        transcript: this.transcript.recent(),
      });
      entry.firstChild.textContent = result.answer;
      if (result.calls.length) {
        const calls = element("div", "calls");
        for (const c of result.calls) calls.append(element("span", "call", c.name + "(" + c.arguments + ")"));
        entry.insertBefore(calls, meta);
      }
      meta.textContent = "The assistant's answer, for GPT-Live to say";
      if (result.trace_url) {
        meta.append(" · ");
        const link = element("a", "", "trace");
        link.href = result.trace_url;
        link.target = "_blank";
        link.rel = "noopener";
        meta.append(link);
      } else if (result.trace_id) {
        meta.append(" · trace " + result.trace_id);
      }
      this.send({ type: "session.commentary.append", delegation_id: id, content: result.answer });
    } catch (error) {
      entry.classList.add("error");
      entry.firstChild.textContent = "The assistant failed: " + error.message;
      this.send({
        type: "session.commentary.append",
        delegation_id: id,
        content: "The household's notes cannot be reached right now; say so and offer to try again.",
      });
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
      setStatus("Hanging up…");
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
    setStatus(message);
    startButton.disabled = false;
    stopButton.disabled = true;
    call = null;
  }
}

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  stopButton.disabled = false;
  call = new Call();
  try {
    await call.start(document.getElementById("voice").value);
  } catch (error) {
    addEntry("assistant error", error.message);
    if (call) call.ended("Not connected");
  }
});

stopButton.addEventListener("click", () => {
  if (call) call.hangUp();
});

async function start() {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    showNotice(
      "The browser only gives a page the microphone on https or on localhost. Open this page as " +
        "http://localhost:<port>/voice on the machine that runs the demo, or serve it over https."
    );
    startButton.disabled = true;
  }
  try {
    const response = await fetch("/api/info");
    const info = await response.json();
    document.getElementById("info").textContent =
      (info.live_model ? info.live_model + " with " : "") + info.model + (info.tracing ? " · traced to Langfuse" : "");
    if (!info.live) {
      showNotice("GPT-Live needs an OpenAI API key: start the demo with OPENAI_API_KEY set.");
      startButton.disabled = true;
    }
  } catch {
    // The page works without it.
  }
}

start();
