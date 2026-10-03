import { spawn, type ChildProcess } from "node:child_process";
import { errMessage } from "../util/errors.ts";
import { redactSecrets } from "./redact-secrets.ts";

/** ACP v1 wire protocol version (see @agentclientprotocol/sdk PROTOCOL_VERSION). */
export const ACP_PROTOCOL_VERSION = 1;

export interface AcpContentBlock {
  type: string;
  text?: string;
  /** tool_call_update content wraps the payload block in `{ type: "content", content }`. */
  content?: AcpContentBlock;
}

/**
 * The `update` member of a `session/update` notification. eve only emits
 * agent_message_chunk, agent_thought_chunk, tool_call, and tool_call_update,
 * so the shape stays loose beyond those.
 */
export interface AcpSessionUpdate {
  sessionUpdate: string;
  content?: AcpContentBlock | AcpContentBlock[];
  messageId?: string;
  toolCallId?: string;
  title?: string;
  kind?: string;
  status?: string;
  rawInput?: unknown;
  rawOutput?: unknown;
}

export interface AcpPermissionOption {
  kind?: string;
  name?: string;
  optionId: string;
}

export interface AcpPermissionRequest {
  sessionId: string;
  toolCall?: { toolCallId?: string; title?: string };
  options?: AcpPermissionOption[];
}

export interface AcpElicitationRequest {
  sessionId: string;
  mode?: string;
  message?: string;
}

export type AcpStopReason = "end_turn" | "cancelled" | string;

export interface EveAcpHandlers {
  /** Server-initiated `session/request_permission` (tool approval). */
  onPermissionRequest(request: AcpPermissionRequest): Promise<{ outcome: Record<string, unknown> }>;
  /** Server-initiated `elicitation/create` (eve questions). */
  onElicitationRequest(request: AcpElicitationRequest): Promise<{ action: string }>;
}

export interface EveAcpProcessOptions {
  binaryPath: string;
  args: string[];
  cwd: string;
  env?: NodeJS.ProcessEnv;
  handlers: EveAcpHandlers;
}

interface AcpMessage {
  id?: number | string;
  method?: string;
  params?: Record<string, unknown>;
  result?: unknown;
  error?: { code?: number; message?: string; data?: unknown };
}

/** Looser shape for messages we write; inbound messages are validated as AcpMessage. */
interface AcpOutgoingMessage {
  id?: number | string;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: { code?: number; message?: string; data?: unknown };
}

function isAcpMessage(value: unknown): value is AcpMessage {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const message = value as Record<string, unknown>;
  if (typeof message.method !== "string" && message.id === undefined) return false;
  if (message.params !== undefined && (typeof message.params !== "object" || message.params === null)) return false;
  return true;
}

/**
 * JSON-RPC over stdio transport for one `eve acp` process (ACP v1), following
 * the CodexAppServer pattern: a lazily-seeded id counter, serialized stdin
 * writes, notification/request tails, and fail-all on process exit.
 */
export class EveAcpProcess {
  readonly process: ChildProcess;
  private readonly options: EveAcpProcessOptions;
  /** Routed `session/update` notifications; the harness assigns the active-turn sink. */
  onUpdate: (sessionId: unknown, update: unknown) => void = () => {};
  private nextId = 1;
  private readonly pending = new Map<number | string, { resolve(value: unknown): void; reject(error: Error): void }>();
  private writeTail = Promise.resolve();
  private receiveTail = Promise.resolve();
  private stderr = "";
  private closed = false;
  private closeError: Error | null = null;
  private readonly processClosed: Promise<void>;

  constructor(options: EveAcpProcessOptions) {
    this.options = options;
    this.process = spawn(options.binaryPath, options.args, {
      cwd: options.cwd,
      ...(options.env ? { env: options.env } : {}),
      stdio: ["pipe", "pipe", "pipe"],
    });
    let resolveProcessClosed!: () => void;
    this.processClosed = new Promise<void>((resolve) => {
      resolveProcessClosed = resolve;
    });
    let pendingOutput = "";
    const receiveLine = (line: string) => {
      this.receiveTail = this.receiveTail.then(() => this.receive(line)).catch((error) => this.failTransport(error));
    };
    this.process.stdout!.setEncoding("utf8");
    this.process.stdout!.on("data", (chunk: string) => {
      const lines = (pendingOutput + chunk).split("\n");
      pendingOutput = lines.pop()!;
      for (const line of lines) receiveLine(line);
    });
    this.process.stdout!.on("end", () => {
      if (pendingOutput) receiveLine(pendingOutput);
      pendingOutput = "";
    });
    this.process.stderr?.on("data", (chunk: Buffer) => {
      this.stderr = `${this.stderr}${chunk.toString()}`.slice(-16_384);
    });
    this.process.once("error", (error) => {
      this.closed = true;
      this.closeError = error;
      this.failAll(error);
      resolveProcessClosed();
    });
    this.process.once("close", (code, signal) => {
      this.closed = true;
      const stderr = redactSecrets(this.stderr.trim());
      this.closeError = new Error(`eve acp exited (${code ?? signal ?? "unknown"})${stderr ? `: ${stderr}` : ""}`);
      this.failAll(this.closeError);
      resolveProcessClosed();
    });
  }

  error(): Error | null {
    return this.closeError;
  }

  async initialize(): Promise<void> {
    await this.request("initialize", {
      protocolVersion: ACP_PROTOCOL_VERSION,
      clientInfo: { name: "qm", title: "QM", version: "1" },
      // Form elicitation is deliberately not advertised: qm has no mid-turn
      // form channel, so eve questions surface as a cancelled turn instead of
      // a hang.
      clientCapabilities: {},
    });
  }

  async newSession(cwd: string): Promise<string> {
    const result = (await this.request("session/new", { cwd, mcpServers: [] })) as { sessionId?: unknown };
    if (typeof result?.sessionId !== "string") throw new Error("eve acp session/new returned no sessionId");
    return result.sessionId;
  }

  async prompt(sessionId: string, text: string, signal?: AbortSignal): Promise<AcpStopReason> {
    const result = (await this.request(
      "session/prompt",
      { sessionId, prompt: [{ type: "text", text }] },
      signal,
    )) as { stopReason?: unknown };
    return typeof result?.stopReason === "string" ? result.stopReason : "end_turn";
  }

  async cancel(sessionId: string): Promise<void> {
    await this.notify("session/cancel", { sessionId });
  }

  async closeSession(sessionId: string): Promise<void> {
    await this.request("session/close", { sessionId });
  }

  async close(): Promise<void> {
    if (this.closed) return await this.processClosed;
    this.closed = true;
    this.process.kill("SIGTERM");
    const timer = setTimeout(() => this.process.kill("SIGKILL"), 2_000);
    await this.processClosed;
    clearTimeout(timer);
  }

  private async receive(line: string): Promise<void> {
    if (!line.trim()) return;
    let message: AcpMessage;
    try {
      const parsed: unknown = JSON.parse(line);
      if (!isAcpMessage(parsed)) throw new Error("eve acp emitted an invalid JSON-RPC message");
      message = parsed;
    } catch (error) {
      if (error instanceof SyntaxError)
        throw new Error(redactSecrets(`eve acp emitted invalid JSON: ${line.slice(0, 500)}`));
      throw error;
    }
    if (message.id !== undefined && !message.method) {
      const waiter = this.pending.get(message.id);
      if (!waiter) return;
      this.pending.delete(message.id);
      if ("error" in message) {
        const detail = message.error?.message ?? JSON.stringify(message.error?.data) ?? "error";
        waiter.reject(new Error(redactSecrets(`eve acp ${message.error?.code ?? "error"}: ${detail}`)));
      } else waiter.resolve(message.result);
      return;
    }
    if (!message.method) return;
    if (message.method === "session/update") {
      const update = message.params?.update;
      if (update && typeof update === "object") this.onUpdate(message.params!.sessionId, update);
      return;
    }
    if (message.id === undefined) return;
    void this.respond(message.id, message.method, message.params ?? {}).catch((error) =>
      this.failTransport(error),
    );
  }

  private async respond(id: number | string, method: string, params: Record<string, unknown>): Promise<void> {
    let response: AcpOutgoingMessage;
    try {
      if (method === "session/request_permission") {
        const outcome = await this.options.handlers.onPermissionRequest(params as unknown as AcpPermissionRequest);
        response = { id, result: { outcome: outcome.outcome } };
      } else if (method === "elicitation/create") {
        const outcome = await this.options.handlers.onElicitationRequest(params as unknown as AcpElicitationRequest);
        response = { id, result: { action: outcome.action } };
      } else {
        response = { id, error: { code: -32601, message: `eve acp client method ${method} is not handled` } };
      }
    } catch (error) {
      response = { id, error: { code: -32000, message: errMessage(error) } };
    }
    if (!this.closed) await this.send(response);
  }

  private request(method: string, params?: unknown, signal?: AbortSignal): Promise<unknown> {
    if (this.closed) return Promise.reject(new Error("eve acp process is closed"));
    if (signal?.aborted) return Promise.reject(new Error("eve acp request cancelled"));
    const id = this.nextId++;
    let rejectResult!: (error: Error) => void;
    const result = new Promise<unknown>((resolve, reject) => {
      rejectResult = reject;
      this.pending.set(id, { resolve, reject });
    });
    if (signal) {
      const onAbort = () => {
        if (this.pending.delete(id)) rejectResult(new Error("eve acp request cancelled"));
      };
      signal.addEventListener("abort", onAbort, { once: true });
      void result.then(
        () => signal.removeEventListener("abort", onAbort),
        () => signal.removeEventListener("abort", onAbort),
      );
    }
    void this.send({ id, method, ...(params === undefined ? {} : { params }) }).catch((error) => {
      const waiter = this.pending.get(id);
      this.pending.delete(id);
      waiter?.reject(error instanceof Error ? error : new Error(String(error)));
    });
    return result;
  }

  private async notify(method: string, params?: unknown): Promise<void> {
    await this.send({ method, ...(params === undefined ? {} : { params }) });
  }

  private send(message: AcpOutgoingMessage): Promise<void> {
    const line = `${JSON.stringify(message)}\n`;
    const operation = this.writeTail.then(async () => {
      if (this.closed || !this.process.stdin?.writable) throw new Error("eve acp stdin is closed");
      await new Promise<void>((resolve, reject) => {
        this.process.stdin!.write(line, (error) => (error ? reject(error) : resolve()));
      });
    });
    this.writeTail = operation.catch(() => undefined);
    return operation;
  }

  private failTransport(error: unknown): void {
    if (this.closed) return;
    this.failAll(error instanceof Error ? error : new Error(String(error)));
    this.process.kill("SIGTERM");
  }

  private failAll(error: Error): void {
    for (const waiter of this.pending.values()) waiter.reject(error);
    this.pending.clear();
  }
}
