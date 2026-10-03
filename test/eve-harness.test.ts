import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createEveHarness } from "../src/harness/eve-harness.ts";
import { NonRetryableTurnError } from "../src/core/turn-error.ts";
import { modelSupportedByHarness, HARNESS_IDS, thinkingLevelsForHarness } from "../src/model/pi-models.ts";
import type { Harness, HarnessTurnInput, HarnessTurnResult } from "../src/harness/harness.ts";
import type { ScopeId, SessionEntry } from "../src/types.ts";

const FAKE_SERVER = new URL("./fixtures/eve-acp-fake-server.mjs", import.meta.url).pathname;

function eveHarness(scenario: string): Harness {
  return createEveHarness({
    binaryPath: process.execPath,
    args: [FAKE_SERVER],
    // eve's ACP server enforces cwd === the eve application root, and spawn
    // fails with ENOENT when the cwd does not exist.
    appRoot: mkdtempSync(join(tmpdir(), "eve-fake-app-")),
    env: { ...process.env, FAKE_SCENARIO: scenario },
    turnWallClockMs: 5_000,
  });
}

function makeTurn(overrides: Partial<HarnessTurnInput> = {}): {
  turn: HarnessTurnInput;
  entries: SessionEntry[];
  deltas: string[];
  toolStarts: string[];
} {
  const scope = { kind: "org", id: "test" } as unknown as ScopeId;
  const entries: SessionEntry[] = [];
  const deltas: string[] = [];
  const toolStarts: string[] = [];
  const turn = {
    session: { id: "s1" } as HarnessTurnInput["session"],
    input: "hi",
    systemPrompt: "sys",
    history: [],
    tools: {} as HarnessTurnInput["tools"],
    scopeLabel: scope,
    orgScopeId: scope,
    emit: async (entry: { type: string; payload: unknown; scopeLabel: ScopeId }) => {
      const saved = { ...entry, sessionId: "s1", seq: entries.length + 1, createdAt: Date.now() } as SessionEntry;
      entries.push(saved);
      return saved;
    },
    recordModelCall: () => {},
    onDelta: (chunk: string) => deltas.push(chunk),
    onToolCallStart: (name: string) => toolStarts.push(name),
    ...overrides,
  } as HarnessTurnInput;
  return { turn, entries, deltas, toolStarts };
}

function payloads(entries: SessionEntry[], type: string): unknown[] {
  return entries.filter((entry) => entry.type === type).map((entry) => entry.payload);
}

test("eve harness registers in the harness catalog with pi-like model support", () => {
  assert.ok(HARNESS_IDS.includes("eve"));
  assert.ok(modelSupportedByHarness("claude-opus-5", "eve"));
  assert.ok(!modelSupportedByHarness("claude-opus-5", "nonexistent"));
  assert.deepEqual(thinkingLevelsForHarness("eve"), ["auto"]);
});

test("reply scenario: streams deltas and emits the final assistant entry", async () => {
  const harness = eveHarness("reply");
  try {
    const { turn, entries, deltas } = makeTurn();
    const result = await harness.turns.runTurn(turn);
    assert.equal(result.reply, "hello from eve");
    assert.ok(deltas.join("").includes("hello from eve"));
    assert.deepEqual(payloads(entries, "user").at(0), { text: "hi" });
    assert.deepEqual(payloads(entries, "assistant").at(-1), { text: "hello from eve" });
  } finally {
    await harness.turns.close?.();
  }
});

test("steps scenario: intermediate messages surface as entries, the last becomes the reply", async () => {
  const harness = eveHarness("steps");
  try {
    const { turn, entries } = makeTurn();
    const result = await harness.turns.runTurn(turn);
    assert.equal(result.reply, "final answer");
    assert.deepEqual(payloads(entries, "assistant"), [{ text: "intermediate note" }, { text: "final answer" }]);
  } finally {
    await harness.turns.close?.();
  }
});

test("tools scenario: tool_call and tool_result entries with titles and results", async () => {
  const harness = eveHarness("tools");
  try {
    const { turn, entries, toolStarts } = makeTurn();
    const result = await harness.turns.runTurn(turn);
    assert.equal(result.reply, "query done");
    assert.deepEqual(toolStarts, ["run_query"]);
    assert.deepEqual(payloads(entries, "tool_call"), [{ tool: "run_query", callId: "c1" }]);
    assert.deepEqual(payloads(entries, "tool_result"), [
      { tool: "run_query", callId: "c1", result: "42 rows", isError: false },
    ]);
  } finally {
    await harness.turns.close?.();
  }
});

test("tools-failed scenario: failed tool results carry isError", async () => {
  const harness = eveHarness("tools-failed");
  try {
    const { turn, entries } = makeTurn();
    await harness.turns.runTurn(turn);
    assert.deepEqual(payloads(entries, "tool_result"), [
      { tool: "run_query", callId: "c1", result: "boom", isError: true },
    ]);
  } finally {
    await harness.turns.close?.();
  }
});

test("thought scenario: thinking entries precede the final message", async () => {
  const harness = eveHarness("thought");
  try {
    const { turn, entries } = makeTurn();
    const result = await harness.turns.runTurn(turn);
    assert.equal(result.reply, "the answer");
    assert.deepEqual(payloads(entries, "thinking"), [{ thinking: "pondering" }]);
  } finally {
    await harness.turns.close?.();
  }
});

test("permission scenario: the approval gate selects allow or reject", async () => {
  const harness = eveHarness("permission");
  try {
    const allowed = makeTurn({ toolApprovalGate: () => true });
    const allowedResult = await harness.turns.runTurn(allowed.turn);
    assert.equal(allowedResult.reply, "decision:opt-approve");

    const denied = makeTurn({ session: { id: "s2" } as HarnessTurnInput["session"], toolApprovalGate: () => false });
    const deniedResult = await harness.turns.runTurn(denied.turn);
    assert.equal(deniedResult.reply, "decision:opt-deny");

    const ungated = makeTurn({ session: { id: "s3" } as HarnessTurnInput["session"] });
    const ungatedResult = await harness.turns.runTurn(ungated.turn);
    assert.equal(ungatedResult.reply, "decision:opt-approve");
  } finally {
    await harness.turns.close?.();
  }
});

test("cancel scenario: user abort stops the turn and reports stoppedByUser", async () => {
  const harness = eveHarness("cancel");
  try {
    const controller = new AbortController();
    const { turn } = makeTurn({ cancel: controller.signal });
    const settled = harness.turns.runTurn(turn);
    setTimeout(() => controller.abort(), 150);
    const result: HarnessTurnResult = await settled;
    assert.equal(result.stopped, true);
    assert.equal(result.stoppedByUser, true);
  } finally {
    await harness.turns.close?.();
  }
});

test("hang scenario: wall clock cancel resolves the turn as stopped", async () => {
  const harness = eveHarness("hang");
  try {
    const { turn } = makeTurn({ turnWallClockMs: 200 });
    const startedAt = Date.now();
    const result = await harness.turns.runTurn(turn);
    assert.equal(result.stopped, true);
    assert.ok(Date.now() - startedAt < 4_000, "cancel should resolve well before the 5s server cap");
  } finally {
    await harness.turns.close?.();
  }
});

test("echo-session scenario: one acp session per qm session, reset creates a new one", async () => {
  const harness = eveHarness("echo-session");
  try {
    const first = makeTurn();
    assert.equal((await harness.turns.runTurn(first.turn)).reply, "sid:acp-1");
    const second = makeTurn({ session: { id: "s1" } as HarnessTurnInput["session"] });
    assert.equal((await harness.turns.runTurn(second.turn)).reply, "sid:acp-1");
    await harness.turns.resetSession?.("s1");
    const third = makeTurn({ session: { id: "s1" } as HarnessTurnInput["session"] });
    assert.equal((await harness.turns.runTurn(third.turn)).reply, "sid:acp-2");
  } finally {
    await harness.turns.close?.();
  }
});

test("turn-error scenario: server errors surface as turn failures", async () => {
  const harness = eveHarness("turn-error");
  try {
    const { turn } = makeTurn();
    await assert.rejects(harness.turns.runTurn(turn), /eve app exploded/);
  } finally {
    await harness.turns.close?.();
  }
});

test("missing configuration: turns refuse with a NonRetryableTurnError", async () => {
  const harness = createEveHarness({
    binaryPath: process.execPath,
    args: [FAKE_SERVER],
    turnWallClockMs: 5_000,
  });
  try {
    const { turn } = makeTurn();
    await assert.rejects(
      harness.turns.runTurn(turn),
      (error: unknown) => error instanceof NonRetryableTurnError && /EVE_APP_ROOT/.test(String(error.message)),
    );
  } finally {
    await harness.turns.close?.();
  }
});
