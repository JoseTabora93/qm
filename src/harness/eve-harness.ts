import { CONFIG_DEFAULTS, type Config } from "../config.ts";
import { documentFallbackText } from "../core/document-inputs.ts";
import { NonRetryableTurnError } from "../core/turn-error.ts";
import { startSignalPoll, type RunSignalStore } from "../runs/run-signal-store.ts";
import { swallow } from "../util/errors.ts";
import {
  defineHarness,
  type Harness,
  type HarnessTurnInput,
  type HarnessTurnResult,
} from "./harness.ts";
import { EveAcpProcess, type AcpContentBlock, type AcpPermissionOption, type AcpPermissionRequest, type AcpSessionUpdate } from "./eve-acp.ts";
import { recordSteerIntake, tapeReplyCheckpoint, type SteerIntake } from "./harness-shared.ts";
import { recordedMessageTimestamps } from "./replay.ts";

const EVE_CANCEL_GRACE_MS = 10_000;

export interface EveHarnessOptions {
  binaryPath?: string;
  /** Root of the local eve application; `session/new.cwd` must point here. */
  appRoot?: string;
  /** Bridge an already-running eve deployment via `eve acp --url` instead of a local app root. */
  url?: string;
  /** Overrides the default `["acp"]` argv; test seam for fake ACP servers. */
  args?: string[];
  env?: NodeJS.ProcessEnv;
  turnWallClockMs?: number;
  signals?: RunSignalStore;
}

export function eveHarnessConfigOptions(config: Config): EveHarnessOptions {
  return {
    ...(config.eveBinPath ? { binaryPath: config.eveBinPath } : {}),
    ...(config.eveAppRoot ? { appRoot: config.eveAppRoot } : {}),
    ...(config.eveUrl ? { url: config.eveUrl } : {}),
    env: config.eveProcessEnv,
    turnWallClockMs: config.turnWallClockMs,
  };
}

function blockText(block: AcpContentBlock): string {
  if (block.type === "text" && typeof block.text === "string") return block.text;
  if (block.content) return blockText(block.content);
  return "";
}

function contentText(content: AcpSessionUpdate["content"]): string {
  const blocks = Array.isArray(content) ? content : content ? [content] : [];
  return blocks.map(blockText).join("");
}

function permissionOptionFor(request: AcpPermissionRequest, allow: boolean): AcpPermissionOption | undefined {
  const kinds = allow ? ["allow_once", "allow_always"] : ["reject_once", "reject_always"];
  return (request.options ?? []).find((option) => option.kind !== undefined && kinds.includes(option.kind));
}

/**
 * Per-turn transcript assembly: buffers the current assistant message and
 * pending thinking so message boundaries surface earlier steps as entries and
 * the final message becomes the turn reply.
 */
class TurnAssembly {
  private messageId: string | undefined;
  private messageText = "";
  private readonly thinking: string[] = [];
  private readonly toolTitles = new Map<string, string>();
  private readonly turn: HarnessTurnInput;

  constructor(turn: HarnessTurnInput) {
    this.turn = turn;
  }

  async appendMessage(text: string, messageId?: string): Promise<void> {
    if (messageId && messageId !== this.messageId) await this.flushMessage(false);
    this.messageId = messageId ?? this.messageId;
    this.messageText += text;
  }

  appendThinking(text: string): void {
    this.thinking.push(text);
  }

  noteTool(callId: string, title: string): void {
    this.toolTitles.set(callId, title);
  }

  toolTitle(callId: string): string | undefined {
    return this.toolTitles.get(callId);
  }

  private async flushThinking(): Promise<void> {
    for (const value of this.thinking.splice(0))
      await this.turn.emit({ type: "thinking", payload: { thinking: value }, scopeLabel: this.turn.scopeLabel });
  }

  /** Emits the buffered message; `final` returns it as the reply instead of emitting. */
  async flushMessage(final: boolean): Promise<string> {
    await this.flushThinking();
    const text = this.messageText.trim();
    this.messageText = "";
    if (!text || final) return text;
    await this.turn.emit({ type: "assistant", payload: { text }, scopeLabel: this.turn.scopeLabel });
    return text;
  }
}

async function handleUpdate(turn: HarnessTurnInput, assembly: TurnAssembly, update: AcpSessionUpdate): Promise<void> {
  switch (update.sessionUpdate) {
    case "agent_message_chunk": {
      const text = contentText(update.content);
      if (!text) return;
      turn.onDelta?.(text);
      await assembly.appendMessage(text, update.messageId);
      return;
    }
    case "agent_thought_chunk": {
      const text = contentText(update.content);
      if (text) assembly.appendThinking(text);
      return;
    }
    case "tool_call": {
      const callId = update.toolCallId ?? "";
      const title = update.title ?? "tool";
      assembly.noteTool(callId, title);
      turn.onToolCallStart?.(title);
      await turn.emit({ type: "tool_call", payload: { tool: title, callId }, scopeLabel: turn.scopeLabel });
      return;
    }
    case "tool_call_update": {
      const callId = update.toolCallId ?? "";
      await turn.emit({
        type: "tool_result",
        payload: {
          tool: assembly.toolTitle(callId) ?? update.title ?? "tool",
          callId,
          result: contentText(update.content) || (typeof update.rawOutput === "string" ? update.rawOutput : ""),
          isError: update.status === "failed",
        },
        scopeLabel: turn.scopeLabel,
      });
      return;
    }
    default:
      return;
  }
}

export function createEveHarness(opts: EveHarnessOptions = {}): Harness {
  const defaultTurnWallClockMs = opts.turnWallClockMs ?? CONFIG_DEFAULTS.turnWallClockSec * 1000;
  let acp: EveAcpProcess | null = null;
  let starting: Promise<EveAcpProcess> | null = null;
  const sessions = new Map<string, string>();
  const active = new Map<string, HarnessTurnInput>();
  const assemblyByTurn = new WeakMap<HarnessTurnInput, TurnAssembly>();

  const ensureProcess = (): Promise<EveAcpProcess> => {
    if (acp) return Promise.resolve(acp);
    starting ??= (async () => {
      const args = opts.args ?? ["acp", ...(opts.url ? ["--url", opts.url] : [])];
      const spawned = new EveAcpProcess({
        binaryPath: opts.binaryPath ?? "eve",
        args,
        cwd: opts.appRoot ?? process.cwd(),
        ...(opts.env ? { env: opts.env } : {}),
        handlers: {
          onPermissionRequest: async (request) => {
            const turn = active.get(request.sessionId);
            const allow = turn ? turn.toolApprovalGate?.(request.toolCall?.title ?? "tool") !== false : true;
            const option = permissionOptionFor(request, allow);
            if (!option) return { outcome: { outcome: "cancelled" } };
            return { outcome: { outcome: "selected", optionId: option.optionId } };
          },
          onElicitationRequest: async () =>
            // qm has no mid-turn form channel; declining makes eve cancel the
            // pending question and end the turn instead of hanging.
            ({ action: "cancel" }),
        },
      });
      spawned.process.once("close", () => {
        if (acp === spawned) {
          acp = null;
          starting = null;
          sessions.clear();
        }
      });
      spawned.onUpdate = (sessionId, update) => {
        const turn = active.get(String(sessionId ?? ""));
        if (!turn) return;
        const assembly = assemblyByTurn.get(turn);
        if (!assembly) return;
        void handleUpdate(turn, assembly, update as AcpSessionUpdate).catch((error) =>
          swallow("eve: session update", error),
        );
      };
      await spawned.initialize();
      acp = spawned;
      return spawned;
    })();
    return starting.catch((error) => {
      starting = null;
      throw error;
    });
  };

  const ensureSession = async (turn: HarnessTurnInput): Promise<string> => {
    const existing = sessions.get(turn.session.id);
    if (existing) return existing;
    if (!opts.appRoot && !opts.url)
      throw new NonRetryableTurnError(
        "eve is not configured — set EVE_APP_ROOT to a local eve application root, or EVE_URL to a deployed agent",
      );
    const spawned = await ensureProcess();
    const sessionId = await spawned.newSession(opts.appRoot ?? process.cwd());
    sessions.set(turn.session.id, sessionId);
    return sessionId;
  };

  const runPromptOnce = async (
    spawned: EveAcpProcess,
    acpSessionId: string,
    text: string,
    turn: HarnessTurnInput,
    assembly: TurnAssembly,
    cancelState: { cancelled: boolean; stoppedByUser: boolean },
  ): Promise<string> => {
    const wallMs = turn.turnWallClockMs ?? defaultTurnWallClockMs;
    let timedOut = false;
    const timer =
      wallMs > 0
        ? setTimeout(() => {
            timedOut = true;
            cancelState.cancelled = true;
            void spawned.cancel(acpSessionId).catch(() => undefined);
          }, wallMs)
        : null;
    try {
      return await Promise.race([
        spawned.prompt(acpSessionId, text),
        ...(timer
          ? [
              new Promise<never>((_, reject) =>
                setTimeout(
                  () => reject(new NonRetryableTurnError(`eve turn exceeded ${Math.round(wallMs / 1000)}s wall clock`)),
                  wallMs + EVE_CANCEL_GRACE_MS,
                ),
              ),
            ]
          : []),
      ]);
    } finally {
      if (timer) clearTimeout(timer);
      if (timedOut) {
        const partial = await assembly.flushMessage(true);
        if (partial)
          await turn.emit({
            type: "assistant",
            payload: { text: partial, stopped: true },
            scopeLabel: turn.scopeLabel,
          });
      }
    }
  };

  return defineHarness(
    {
      id: "eve",
      controlTransport: "json-rpc",
      toolTransport: "plugin",
      transcriptFormat: "acp-ndjson",
      capabilities: new Set(["abort"]),
    },
    {
      async runTurn(turn: HarnessTurnInput): Promise<HarnessTurnResult> {
        if (turn.cancel?.aborted) return { reply: "", stopped: true };
        const acpSessionId = await ensureSession(turn);
        if (active.has(acpSessionId)) throw new NonRetryableTurnError("eve session already has an active turn");
        const spawned = await ensureProcess();

        await turn.emit({
          type: "user",
          payload: {
            text: turn.input,
            ...((turn.triggerTs ?? turn.entryTs) ? { ts: turn.triggerTs ?? turn.entryTs } : {}),
            ...(turn.attachments?.length ? { attachments: turn.attachments } : {}),
          },
          scopeLabel: turn.scopeLabel,
        });

        let promptText = turn.input;
        for (const document of turn.documents ?? []) {
          const text = await documentFallbackText(document, turn.cancel);
          promptText += `\n\n${text}`;
        }

        const assembly = new TurnAssembly(turn);
        assemblyByTurn.set(turn, assembly);
        const cancelState = { cancelled: false, stoppedByUser: false };
        active.set(acpSessionId, turn);
        const queued: Array<{ text: string; intake?: SteerIntake }> = [{ text: promptText }];

        const abort = (stoppedByUser: boolean) => {
          cancelState.cancelled = true;
          if (stoppedByUser) cancelState.stoppedByUser = true;
          void spawned.cancel(acpSessionId).catch(() => undefined);
        };
        const onUserCancel = () => abort(true);
        turn.cancel?.addEventListener("abort", onUserCancel, { once: true });
        const stopSignals =
          opts.signals && turn.runId
            ? startSignalPoll(
                opts.signals,
                turn.runId,
                {
                  onAbort: async () => abort(false),
                  onSteer: async (text, ts, request, acknowledge) => {
                    if (ts && recordedMessageTimestamps(turn.history).has(ts)) return false;
                    const prepared = await turn.prepareSteer?.(text, request);
                    queued.push({
                      text: prepared?.text ?? text,
                      intake: {
                        text,
                        ...(ts ? { ts } : {}),
                        ...(prepared?.attachments?.length ? { attachments: prepared.attachments } : {}),
                        acknowledge,
                      },
                    });
                    return true;
                  },
                },
                { onError: (error) => swallow("eve signal poll", error), drainOnStop: true },
              )
            : null;

        let stopReason = "end_turn";
        let failure: unknown = null;
        try {
          while (queued.length && !cancelState.cancelled) {
            const next = queued.shift()!;
            if (next.intake) await recordSteerIntake(turn, next.intake);
            try {
              stopReason = await runPromptOnce(spawned, acpSessionId, next.text, turn, assembly, cancelState);
            } catch (error) {
              failure = error;
              break;
            }
          }
        } finally {
          turn.cancel?.removeEventListener("abort", onUserCancel);
          if (stopSignals) await stopSignals();
          active.delete(acpSessionId);
        }

        if (failure) throw failure;
        const reply = await assembly.flushMessage(true);
        if (stopReason === "cancelled" || cancelState.cancelled) {
          if (reply)
            await turn.emit({
              type: "assistant",
              payload: { text: reply, stopped: true },
              scopeLabel: turn.scopeLabel,
            });
          return {
            reply,
            stopped: true,
            ...(cancelState.stoppedByUser ? { stoppedByUser: true as const } : {}),
          };
        }
        if (reply) {
          const finalEntry = await turn.emit({
            type: "assistant",
            payload: { text: reply },
            scopeLabel: turn.scopeLabel,
          });
          await tapeReplyCheckpoint(turn, finalEntry);
        }
        return { reply };
      },
      async resetSession(sessionId: string): Promise<void> {
        const acpSessionId = sessions.get(sessionId);
        if (!acpSessionId) return;
        sessions.delete(sessionId);
        if (!acp) return;
        await acp.closeSession(acpSessionId).catch(() => undefined);
      },
      async close(): Promise<void> {
        sessions.clear();
        await acp?.close();
        acp = null;
        starting = null;
      },
    },
  );
}
