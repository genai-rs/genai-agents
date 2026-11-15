// The chat page: a session with the assistant, through the demo's API (demo/api/openapi.yaml).
"use strict";

const log = document.getElementById("log");
const form = document.getElementById("ask");
const question = document.getElementById("question");
const send = document.getElementById("send");

function storedSession() {
  try {
    return localStorage.getItem("demo-session");
  } catch {
    return null;
  }
}

function storeSession(id) {
  try {
    localStorage.setItem("demo-session", id);
  } catch {
    // The session lasts as long as the page then.
  }
}

function newSessionId() {
  const bytes = new Uint8Array(6);
  crypto.getRandomValues(bytes);
  return "web-" + Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

let session = storedSession() || newSessionId();
storeSession(session);

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body ? { "content-type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = data.message || (data.error && data.error.message) || response.statusText;
    throw new Error(message);
  }
  return data;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollDown() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function addMessage(role, text) {
  document.getElementById("empty").hidden = true;
  const node = element("div", "msg " + role, text);
  log.append(node);
  scrollDown();
  return node;
}

function addMeta(node, turn) {
  if (turn.calls && turn.calls.length) {
    const calls = element("div", "calls");
    for (const call of turn.calls) {
      calls.append(element("span", "call", call.name + "(" + call.arguments + ")"));
    }
    node.append(calls);
  }
  const meta = element("div", "meta", `${turn.requests} requests, ${turn.tokens} tokens`);
  if (turn.trace_url) {
    meta.append(" · ");
    const link = element("a", "", "trace");
    link.href = turn.trace_url;
    link.target = "_blank";
    link.rel = "noopener";
    meta.append(link);
  } else if (turn.trace_id) {
    meta.append(" · trace " + turn.trace_id);
  }
  node.append(meta);
}

function showTurn(turn) {
  if (turn.status === "answered") {
    addMeta(addMessage("assistant", turn.answer || ""), turn);
  } else if (turn.status === "approval") {
    const card = element("div", "approval");
    card.append(element("p", "", "The assistant wants to run:"));
    for (const call of turn.pending) {
      card.append(element("pre", "", call.name + " " + prettyArguments(call.arguments)));
    }
    const buttons = element("div", "buttons");
    const approve = element("button", "", "Approve");
    const deny = element("button", "secondary", "Deny");
    approve.type = deny.type = "button";
    const decide = async (ok) => {
      approve.disabled = deny.disabled = true;
      card.append(element("div", "meta", ok ? "Approved." : "Denied."));
      const decisions = turn.pending.map((call) => ({ call, approve: ok }));
      await run(() => api("POST", `/api/sessions/${session}/approvals`, { decisions }));
    };
    approve.addEventListener("click", () => decide(true));
    deny.addEventListener("click", () => decide(false));
    buttons.append(approve, deny);
    card.append(buttons);
    addMeta(card, turn);
    log.append(card);
    scrollDown();
  } else {
    addMessage("assistant error", "The run stopped: " + (turn.detail || "unknown"));
  }
}

function prettyArguments(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

async function run(request) {
  send.disabled = true;
  const thinking = addMessage("assistant", "…");
  try {
    const turn = await request();
    thinking.remove();
    showTurn(turn);
  } catch (error) {
    thinking.remove();
    addMessage("assistant error", error.message);
  } finally {
    send.disabled = false;
    question.focus();
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

document.getElementById("new-session").addEventListener("click", () => {
  session = newSessionId();
  storeSession(session);
  log.replaceChildren();
  document.getElementById("empty").hidden = false;
  document.getElementById("session").textContent = session;
});

async function start() {
  document.getElementById("session").textContent = session;
  try {
    const info = await api("GET", "/api/info");
    document.getElementById("info").textContent = info.model + (info.tracing ? " · traced to Langfuse" : "");
  } catch {
    // The page works without it.
  }
  try {
    const history = await api("GET", `/api/sessions/${session}`);
    for (const message of history.messages) addMessage(message.role, message.text);
  } catch (error) {
    addMessage("assistant error", error.message);
  }
  question.focus();
}

start();
