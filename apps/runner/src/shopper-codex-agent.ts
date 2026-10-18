/** Official Codex CLI adapter. Reuses CLI-managed login; never opens or copies credentials. */
import { spawn } from "node:child_process";
import { access, constants, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import os from "node:os";
import path from "node:path";
import { parseAgentAction, type AgentAction, type ShopperAgent, type ShopperObservation } from "./shopper-agent.js";
import type { ShopperAssignment } from "./shopper-contract.js";

export async function findCodexBinary(explicit?: string) {
  const candidates = explicit ? [explicit] : [
    "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex",
    ...(process.env.PATH ?? "").split(path.delimiter).filter(Boolean).map((p) => path.join(p, "codex")),
  ];
  for (const candidate of candidates) {
    if (!path.isAbsolute(candidate)) continue;
    try { await access(candidate, constants.X_OK); return candidate; } catch { /* next installed official CLI candidate */ }
  }
  throw new Error("No installed Codex CLI found. Install the official @openai/codex package and use codex login, or provide --codex-binary. No browser dispatched.");
}

export async function codexModelAvailability(binary: string, model: string): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, ["app-server", "--stdio"], { stdio: "pipe" });
    const send = (value: unknown) => child.stdin.write(JSON.stringify(value) + "\n");
    let done = false;
    const finish = (error?: Error, value?: Record<string, unknown>) => {
      if (done) return; done = true; clearTimeout(timer); child.kill("SIGTERM");
      if (error) reject(error); else resolve(value!);
    };
    const timer = setTimeout(() => finish(new Error("Codex model availability request timed out; check codex login status")), 20000);
    child.on("error", (error) => finish(error));
    child.stdin.on("error", (error) => finish(error));
    child.stdout.on("error", (error) => finish(error));
    child.stderr.on("data", () => {});
    child.on("exit", (code) => { if (!done) finish(new Error(`Codex metadata process exited (${code})`)); });
    createInterface({ input: child.stdout }).on("line", (line) => {
      let response: { id?: number; error?: unknown; result?: { data?: Record<string, unknown>[] } };
      try { response = JSON.parse(line); } catch { return; }
      if (response.error) { finish(new Error(`Codex metadata request failed: ${JSON.stringify(response.error)}`)); return; }
      if (response.id === 1) { send({ method: "initialized", params: {} }); send({ id: 2, method: "model/list", params: { limit: 100, includeHidden: false } }); }
      if (response.id === 2) {
        const available = response.result?.data?.find((entry) => entry.model === model);
        if (!available || !Array.isArray(available.inputModalities) || !available.inputModalities.includes("image")) {
          finish(new Error(`Requested ${model} with image input is not available for the signed-in Codex account. No model substituted.`));
        } else finish(undefined, available);
      }
    });
    send({ id: 1, method: "initialize", params: { clientInfo: { name: "storefront_shopper_evaluation", title: "Synthetic shopper evaluation", version: "0.1.0" } } });
  });
}

const actionSchema = { type: "object", properties: { action: { anyOf: [
  { type: "object", properties: { type: { const: "click" }, ref: { type: "string" } }, required: ["type", "ref"], additionalProperties: false },
  { type: "object", properties: { type: { const: "search" }, query: { type: "string" } }, required: ["type", "query"], additionalProperties: false },
  { type: "object", properties: { type: { const: "scroll" }, direction: { enum: ["up", "down"] } }, required: ["type", "direction"], additionalProperties: false },
  { type: "object", properties: { type: { const: "abandon" }, reason: { type: "string" } }, required: ["type", "reason"], additionalProperties: false },
] } }, required: ["action"], additionalProperties: false };

/** A shopper owns only its own action history. Every decision gets actual current pixels and visible controls.
 * The CLI runs from an empty temporary directory with user config disabled and a read-only sandbox.
 */
export class CodexShopperAgent implements ShopperAgent {
  private directory: Promise<string>;
  private child: ReturnType<typeof spawn> | undefined;
  private stopped = false;
  private turn = 0;
  private history: { action: AgentAction; viewport: ShopperObservation["viewport"]; cartItemCount: number | null }[] = [];
  constructor(private binary: string, private model: string, private assignment: ShopperAssignment,
    private log: (event: unknown) => void) {
    this.directory = mkdtemp(path.join(os.tmpdir(), "codex-shopper-"));
  }
  async decide(observation: ShopperObservation, remaining: { actions: number; wallMs: number }) {
    if (this.stopped) throw new Error("Codex shopper stopped");
    const directory = await this.directory;
    await mkdir(directory, { recursive: true });
    const sequence = ++this.turn, image = path.join(directory, `screen-${sequence}.png`), schema = path.join(directory, "action-schema.json");
    await writeFile(image, Buffer.from(observation.screenshotBase64, "base64"));
    await writeFile(schema, JSON.stringify(actionSchema));
    const prompt = JSON.stringify({ task: "Act as the assigned synthetic shopper using the attached current Chrome screenshot and visible controls. Return one next action. All executedActions below already succeeded in the browser. An Add to cart action adds another unit; it does not open checkout. Quantity one is required: after adding one qualifying item, find the cart control (scroll if needed), inspect the cart, then confirm only the fake demo order. Do not add the same item repeatedly. viewport.scrollY is the actual distance from the page top, not a product hint; one upward scroll may leave you below the header. Use the observed scroll positions to avoid reversing direction repeatedly without reaching the intended area. Do not call tools, access files, use shell or search outside the storefront. No repository or evaluator access. No real payment/purchase. Only public query may enter store search. Inspect actual visible product/color/price and stop if hard facts cannot be verified.",
      assignment: this.assignment, executedActions: this.history.map((item) => item.action),
      previousDecisionViews: this.history.slice(-4).map(({ viewport, cartItemCount, action }) => ({ viewport: viewport ?? null, cartItemCount, action })),
      observation: { text: observation.text, controls: observation.controls, url: observation.url, viewport: observation.viewport ?? null,
        cartItemCount: observation.cartItemCount ?? null }, remaining });
    const args = ["--no-daemon", "exec", "--ignore-user-config", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
      "--model", this.model, "--json", "--color", "never", "--output-schema", schema, "--image", image,
      "--config", "features.shell_tool=false", "--config", "features.apps=false", "--config", "features.multi_agent=false",
      "--config", "features.skill_search=false", "--config", 'web_search="disabled"',
      "--config", 'model_reasoning_effort="low"', "-"];
    this.log({ type: "codex-decision-start", model: this.model, sequence, auth: "CLI-managed existing login", toolsRequested: false });
    const child = spawn(this.binary, args, { cwd: directory, stdio: "pipe" });
    this.child = child;
    const result = await new Promise<AgentAction>((resolve, reject) => {
      let output: string | undefined, stderr = "", bytes = 0, failure: Error | undefined;
      const fail = (error: Error) => { failure ??= error; child.kill("SIGTERM"); };
      child.on("error", reject);
      child.stdin!.on("error", (error) => fail(error)); child.stdout!.on("error", fail); child.stderr!.on("error", fail);
      child.stderr!.on("data", (chunk: Buffer) => { if (stderr.length < 8000) stderr += chunk.toString("utf8"); });
      child.stdout!.on("data", (chunk: Buffer) => { bytes += chunk.length; if (bytes > 1024 * 1024) fail(new Error("Codex decision exceeded bounded event output")); });
      createInterface({ input: child.stdout! }).on("line", (line) => {
        try {
          const event = JSON.parse(line);
          if (event.type === "item.completed" && event.item?.type === "agent_message") output = event.item.text;
          if (["item.started", "item.completed"].includes(event.type) && ["command_execution", "file_change", "mcp_tool_call", "web_search"].includes(event.item?.type)) fail(new Error("Shopper attempted an unavailable external tool instead of a visible-store action"));
          if (event.type === "turn.failed" || event.type === "error") fail(new Error(`Codex decision failed: ${JSON.stringify(event.error ?? event.message)}`));
        } catch { /* retain bounded non-JSON diagnostic output through the exit error */ }
      });
      child.on("exit", (code, signal) => {
        if (failure) { reject(failure); return; }
        if (code !== 0 || !output) { reject(new Error(`Codex decision exited ${code ?? signal}: ${stderr.trim().slice(-4000)}`)); return; }
        try { resolve(parseAgentAction(JSON.parse(output).action, this.assignment.recommenderContext.query)); }
        catch (error) { reject(error); }
      });
      child.stdin!.end(prompt);
    });
    this.history.push({ action: result, viewport: observation.viewport, cartItemCount: observation.cartItemCount ?? null });
    this.log({ type: "codex-decision-finish", model: this.model, sequence, action: result });
    return result;
  }
  async close() {
    this.stopped = true;
    const child = this.child;
    if (child && child.exitCode === null && child.signalCode === null) {
      child.kill("SIGTERM");
      await new Promise<void>((resolve) => { const timer = setTimeout(() => { child.kill("SIGKILL"); resolve(); }, 1000);
        child.once("exit", () => { clearTimeout(timer); resolve(); }); });
    }
    await rm(await this.directory, { recursive: true, force: true });
  }
}
