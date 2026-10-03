// Minimal fake ACP server standing in for `eve acp` in harness tests.
// Scenarios are selected with the FAKE_SCENARIO env var; see test/eve-harness.test.ts.
import readline from "node:readline";

const scenario = process.env.FAKE_SCENARIO ?? "reply";
const rl = readline.createInterface({ input: process.stdin });
let nextId = 1;
let sessionCount = 0;
const sessions = new Set();
const closed = new Set();
let cancelRequested = false;
const permissionOutcomes = [];

const send = (message) => process.stdout.write(`${JSON.stringify(message)}\n`);
const respond = (id, result) => send({ id, result });
const notify = (method, params) => send({ method, params });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function streamUpdate(sessionId, update) {
  notify("session/update", { sessionId, update });
}

async function runPrompt(sessionId, id) {
  const chunk = (text, messageId) =>
    streamUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text },
      messageId,
    });
  if (scenario === "reply") {
    chunk("hello from eve", `${sessionId}:message:0`);
    return respond(id, { stopReason: "end_turn" });
  }
  if (scenario === "steps") {
    chunk("intermediate note ", `${sessionId}:message:0`);
    await sleep(10);
    chunk("final answer", `${sessionId}:message:1`);
    return respond(id, { stopReason: "end_turn" });
  }
  if (scenario === "tools") {
    streamUpdate(sessionId, {
      sessionUpdate: "tool_call",
      toolCallId: "c1",
      title: "run_query",
      kind: "other",
      status: "pending",
    });
    await sleep(10);
    streamUpdate(sessionId, {
      sessionUpdate: "tool_call_update",
      toolCallId: "c1",
      status: "completed",
      content: [{ type: "content", content: { type: "text", text: "42 rows" } }],
    });
    chunk("query done", `${sessionId}:message:0`);
    return respond(id, { stopReason: "end_turn" });
  }
  if (scenario === "tools-failed") {
    streamUpdate(sessionId, { sessionUpdate: "tool_call", toolCallId: "c1", title: "run_query", status: "pending" });
    await sleep(10);
    streamUpdate(sessionId, {
      sessionUpdate: "tool_call_update",
      toolCallId: "c1",
      status: "failed",
      rawOutput: "boom",
    });
    chunk("failed", `${sessionId}:message:0`);
    return respond(id, { stopReason: "end_turn" });
  }
  if (scenario === "thought") {
    streamUpdate(sessionId, {
      sessionUpdate: "agent_thought_chunk",
      content: { type: "text", text: "pondering" },
      messageId: `${sessionId}:thought:0`,
    });
    chunk("the answer", `${sessionId}:message:0`);
    return respond(id, { stopReason: "end_turn" });
  }
  if (scenario === "permission") {
    const requestId = nextId++;
    send({
      id: requestId,
      method: "session/request_permission",
      params: {
        sessionId,
        toolCall: { toolCallId: "c1", title: "eve_tool", kind: "other", status: "pending" },
        options: [
          { kind: "allow_once", name: "Approve", optionId: "opt-approve" },
          { kind: "reject_once", name: "Deny", optionId: "opt-deny" },
        ],
      },
    });
    const outcome = await new Promise((resolve) => {
      permissionOutcomes.push(resolve);
    });
    chunk(`decision:${outcome}`, `${sessionId}:message:0`);
    return respond(id, { stopReason: "end_turn" });
  }
  if (scenario === "cancel") {
    chunk("working", `${sessionId}:message:0`);
    for (let waited = 0; waited < 5000 && !cancelRequested; waited += 20) await sleep(20);
    return respond(id, { stopReason: cancelRequested ? "cancelled" : "end_turn" });
  }
  if (scenario === "hang") {
    for (let waited = 0; waited < 5000 && !cancelRequested; waited += 20) await sleep(20);
    return respond(id, { stopReason: cancelRequested ? "cancelled" : "end_turn" });
  }
  if (scenario === "echo-session") {
    chunk(`sid:${sessionId}`, `${sessionId}:message:0`);
    return respond(id, { stopReason: "end_turn" });
  }
  if (scenario === "turn-error") {
    return send({ id, error: { code: -32002, message: "eve app exploded" } });
  }
  chunk(`unknown scenario ${scenario}`, `${sessionId}:message:0`);
  return respond(id, { stopReason: "end_turn" });
}

rl.on("line", (line) => {
  if (!line.trim()) return;
  let message;
  try {
    message = JSON.parse(line);
  } catch {
    return;
  }
  if (message.id !== undefined && message.method === undefined) {
    // Response to a server-initiated request (session/request_permission).
    const resolve = permissionOutcomes.shift();
    resolve?.(message.result?.outcome?.optionId ?? "cancelled");
    return;
  }
  if (message.method === "initialize") {
    return respond(message.id, {
      protocolVersion: 1,
      agentCapabilities: { loadSession: false, promptCapabilities: {}, sessionCapabilities: { close: {} } },
      agentInfo: { name: "eve-fake", title: "eve-fake", version: "0.0.0" },
      authMethods: [],
    });
  }
  if (message.method === "session/new") {
    sessionCount += 1;
    const sessionId = `acp-${sessionCount}`;
    sessions.add(sessionId);
    return respond(message.id, { sessionId });
  }
  if (message.method === "session/prompt") return void runPrompt(message.params.sessionId, message.id);
  if (message.method === "session/cancel") {
    cancelRequested = true;
    return;
  }
  if (message.method === "session/close") {
    closed.add(message.params.sessionId);
    return respond(message.id, {});
  }
  if (message.method === "elicitation/create") return respond(message.id, { action: "cancel" });
  if (message.id !== undefined) return respond(message.id, {});
});

// Keep the process alive; stdin close ends it.
rl.on("close", () => process.exit(0));
