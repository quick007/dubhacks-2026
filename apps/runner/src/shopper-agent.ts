/** Bounded JSON-lines adapter for an independently launched shopper process.
 * No evaluator catalog, IDs, gold labels or prior shopper decisions enter this wire.
 */
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import type { ShopperAssignment } from "./shopper-contract.js";

export type AgentAction = { type: "click"; ref: string } | { type: "search"; query: string }
  | { type: "scroll"; direction: "up" | "down" } | { type: "abandon"; reason: string };
export type ShopperObservation = { text: string; screenshotBase64: string;
  controls: { ref: string; role: string; name: string }[]; url: string; cartItemCount?: number;
  viewport?: { scrollY: number; height: number; documentHeight: number } };
export interface ShopperAgent {
  decide(observation: ShopperObservation, remaining: { actions: number; wallMs: number }): Promise<AgentAction>;
  close(): Promise<void>;
}

export function parseAgentAction(value: unknown, assignedQuery: string): AgentAction {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected one shopper action object");
  const a = value as Record<string, unknown>;
  const keys = (...expected: string[]) => Object.keys(a).length === expected.length && expected.every((k) => k in a);
  if (a.type === "click" && keys("type", "ref") && typeof a.ref === "string" && /^control-\d+$/.test(a.ref)) return a as AgentAction;
  if (a.type === "search" && keys("type", "query") && a.query === assignedQuery) return a as AgentAction;
  if (a.type === "scroll" && keys("type", "direction") && ["up", "down"].includes(String(a.direction))) return a as AgentAction;
  if (a.type === "abandon" && keys("type", "reason") && typeof a.reason === "string" && a.reason.trim() && a.reason.length <= 4000) return a as AgentAction;
  throw new Error("Invalid shopper action or private/unassigned search query");
}

/** One fresh child per session. The caller supplies an installed authorized adapter.
 * There is deliberately no shell, implicit API credential, model download or fake fallback.
 */
export class StdioShopperAgent implements ShopperAgent {
  private child: ChildProcessWithoutNullStreams;
  private pending: { resolve: (a: AgentAction) => void; reject: (e: Error) => void } | null = null;
  private failed: Error | null = null;
  constructor(command: string, args: string[], private assignment: ShopperAssignment,
    private log: (event: unknown) => void) {
    this.child = spawn(command, args, { shell: false, stdio: "pipe" });
    const fail = (error: Error) => { this.failed = error; this.pending?.reject(error); this.pending = null; };
    this.child.on("error", fail);
    this.child.stdin.on("error", fail);
    this.child.stdout.on("error", fail);
    this.child.stderr.on("error", fail);
    this.child.on("exit", (code, signal) => fail(new Error(`Shopper adapter exited (${code ?? signal})`)));
    let stderrBytes = 0;
    this.child.stderr.on("data", (chunk: Buffer) => {
      stderrBytes += chunk.length;
      if (stderrBytes <= 64 * 1024) this.log({ type: "agent-stderr", text: chunk.toString("utf8") });
    });
    let lineBytes = 0;
    this.child.stdout.on("data", (chunk: Buffer) => {
      lineBytes += chunk.length;
      if (lineBytes > 64 * 1024) { fail(new Error("Shopper adapter response exceeded 64 KiB")); this.child.kill(); }
    });
    createInterface({ input: this.child.stdout, crlfDelay: Infinity }).on("line", (line) => {
      lineBytes = 0;
      if (!this.pending) { fail(new Error("Unsolicited shopper adapter output")); return; }
      const pending = this.pending; this.pending = null;
      try { pending.resolve(parseAgentAction(JSON.parse(line), assignment.recommenderContext.query)); }
      catch (error) { pending.reject(error instanceof Error ? error : new Error(String(error))); }
    });
  }
  async decide(observation: ShopperObservation, remaining: { actions: number; wallMs: number }) {
    if (this.failed) throw this.failed;
    if (this.pending) throw new Error("Only one shopper decision may be outstanding");
    return new Promise<AgentAction>((resolve, reject) => {
      this.pending = { resolve, reject };
      this.child.stdin.write(JSON.stringify({ version: "visible-shopper-agent.v1", synthetic: true,
        assignment: this.assignment, observation, remaining,
        instructions: "Return exactly one JSON action line: click(ref), search(query), scroll(direction), or abandon(reason). Use only current observed controls. No real purchases. Never read repository/evaluator files. Only the assigned public query may enter search." }) + "\n", (error) => {
          if (error) { this.pending?.reject(error); this.pending = null; }
        });
    });
  }
  async close() {
    this.pending?.reject(new Error("Shopper adapter stopped")); this.pending = null;
    this.child.stdin.end();
    if (this.child.exitCode !== null || this.child.signalCode !== null) return;
    this.child.kill("SIGTERM");
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => { this.child.kill("SIGKILL"); resolve(); }, 1000);
      this.child.once("exit", () => { clearTimeout(timer); resolve(); });
    });
  }
}
