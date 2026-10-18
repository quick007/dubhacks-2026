/** Explicit post-run export only. No browser, model, credentials, or environment-file access. */
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const SHOPPER_RESULTS_ENDPOINT = "http://127.0.0.1:4311/api/experiment-results";
export const SHOPPER_EXPORT_LIMIT = 8 * 1024 * 1024;
const ARMS = ["generic", "local-model"] as const;
type JsonObject = Record<string, unknown>;
export type ShopperExportEnvelope = { version: "experiment-results.v1"; reportId: string; experimentId: string;
  synthetic: true; humanConversionClaim: false; report: JsonObject };
export type ShopperExportReceipt = { reportId: string; sha256: string; durableLocal: true;
  duplicate: boolean; cloud: "asynchronous" };
export type ExportFetch = (url: string, init: RequestInit) => Promise<Response>;

function requireValue(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
function object(value: unknown): JsonObject {
  requireValue(value && typeof value === "object" && !Array.isArray(value), "Expected export JSON object");
  return value as JsonObject;
}
function safeId(value: unknown): asserts value is string {
  requireValue(typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$/.test(value), "Safe report/experiment/case/run ID required");
}
function duration(value: unknown) {
  requireValue(typeof value === "number" && Number.isFinite(value) && value >= 0, "Finite nonnegative observed duration required");
}

/** Retains the entire JSON report, including missing/invalid rows and all evidence caveats. */
export function buildEnvelope(value: unknown, options: { reportId: string; experimentId?: string }): ShopperExportEnvelope {
  safeId(options.reportId);
  const report = object(value);
  requireValue(report.version === "shopper-evidence-aggregate.v1" && report.fixedCases === 10 && report.fixedSessions === 20,
    "Fixed ten-pair shopper-evidence-aggregate.v1 report required");
  requireValue(report.humanConversionClaim === false && (report.synthetic === undefined || report.synthetic === true)
    && (report.realPurchase === undefined || report.realPurchase === false)
    && (report.realPurchases === undefined || report.realPurchases === false), "Synthetic evidence without real purchases or human conversion claims required");
  requireValue(Array.isArray(report.rows) && report.rows.length === 20, "All twenty result rows must be retained");
  const runs = new Set<string>(), slots = new Set<string>(), cases = new Map<string, Set<string>>(), experiments = new Set<string>();
  for (const value of report.rows) {
    const row = object(value); safeId(row.caseId); safeId(row.expectedRunId);
    requireValue(ARMS.some((arm) => arm === row.arm), "Known shopper arm required");
    const arm = row.arm as string, slot = `${row.caseId}/${arm}`;
    requireValue(!runs.has(row.expectedRunId) && !slots.has(slot), "Unique paired case/arm and run IDs required");
    runs.add(row.expectedRunId); slots.add(slot);
    const paired = cases.get(row.caseId) ?? new Set<string>(); paired.add(arm); cases.set(row.caseId, paired);
    requireValue(["recorded", "unattempted", "invalid-sidecar"].includes(String(row.status))
      && ["recorded", "unattempted"].includes(String(row.formalStatus)), "Known result and formal states required");
    requireValue(row.formalStatus !== "recorded" || row.status === "recorded", "Formal state requires recorded evidence");
    if (row.session !== undefined) {
      const session = object(row.session);
      requireValue(session.realPurchase === false, "Every retained session must explicitly exclude real purchases");
      const run = object(session.run); safeId(run.experimentId); experiments.add(run.experimentId);
      if (row.status === "recorded") {
        requireValue(session.contractVersion === "shopper-session.v1" && session.caseId === row.caseId && session.arm === row.arm
          && run.id === row.expectedRunId, "Recorded session identity mismatch");
        duration(session.elapsedMs);
        const model = object(session.model);
        requireValue(Array.isArray(model.readiness), "Observed model readiness array required");
        for (const value of model.readiness) {
          const measure = object(value);
          requireValue(["rank", "firstThree", "complete"].includes(String(measure.stage)), "Known readiness stage required");
          duration(measure.durationMs);
        }
      }
    } else requireValue(row.status !== "recorded", "Recorded row requires its full session");
  }
  requireValue(cases.size === 10 && [...cases.values()].every((arms) => arms.size === 2), "Ten cases with both arms required");
  const arms = object(report.arms);
  for (const arm of ARMS) requireValue(object(arms[arm]).fixedDenominator === 10, "Both arm summaries must retain fixed denominator ten");
  requireValue(experiments.size <= 1, "Retained sessions have different experiment IDs");
  const experimentId = options.experimentId ?? experiments.values().next().value;
  safeId(experimentId);
  requireValue(experiments.size === 0 || experiments.has(experimentId), "Envelope experiment ID differs from recorded session");
  const envelope: ShopperExportEnvelope = { version: "experiment-results.v1", reportId: options.reportId,
    experimentId, synthetic: true, humanConversionClaim: false, report };
  const bytes = JSON.stringify(envelope);
  requireValue(Buffer.byteLength(bytes) <= SHOPPER_EXPORT_LIMIT, "Export envelope exceeds 8 MiB");
  // A separate snapshot ensures caller mutation cannot change the validated evidence before posting.
  return JSON.parse(bytes) as ShopperExportEnvelope;
}

/** Only the fixed unauthenticated loopback intake is allowed; redirects cannot forward report data. */
export async function postShopperResults(envelope: ShopperExportEnvelope, options: {
  endpoint?: string; timeoutMs?: number; fetchImpl?: ExportFetch;
} = {}): Promise<ShopperExportReceipt> {
  const endpoint = new URL(options.endpoint ?? SHOPPER_RESULTS_ENDPOINT);
  requireValue(endpoint.origin === "http://127.0.0.1:4311" && endpoint.pathname === "/api/experiment-results"
    && !endpoint.search && !endpoint.hash && !endpoint.username && !endpoint.password, "Only the fixed local experiment-results endpoint is allowed");
  const checked = buildEnvelope(envelope.report, { reportId: envelope.reportId, experimentId: envelope.experimentId });
  requireValue(envelope.version === checked.version && envelope.synthetic === true && envelope.humanConversionClaim === false,
    "Versioned synthetic envelope required");
  const timeoutMs = options.timeoutMs ?? 10_000;
  requireValue(Number.isInteger(timeoutMs) && timeoutMs > 0 && timeoutMs <= 60_000, "Export timeout must be 1..60000 milliseconds");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error("Local results export timed out")), timeoutMs);
  try {
    const response = await (options.fetchImpl ?? fetch)(endpoint.href, { method: "POST", redirect: "error",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(checked), signal: controller.signal });
    requireValue(response.status === 202, response.status === 409
      ? "Report ID already exists with different evidence; use a new snapshot report ID"
      : `Local results export failed with HTTP ${response.status}`);
    const receipt = object(await response.json());
    requireValue(receipt.reportId === checked.reportId && typeof receipt.sha256 === "string" && /^[a-f0-9]{64}$/.test(receipt.sha256)
      && receipt.durableLocal === true && typeof receipt.duplicate === "boolean" && receipt.cloud === "asynchronous",
      "Local results endpoint did not confirm a durable asynchronous receipt");
    return receipt as ShopperExportReceipt;
  } finally { clearTimeout(timer); }
}

export async function shopperExportMain(args: string[], fetchImpl?: ExportFetch) {
  const options: Record<string, string> = {};
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index], value = args[index + 1], key = flag.replace(/^--/, "");
    requireValue(flag.startsWith("--") && ["report", "report-id", "experiment-id"].includes(key)
      && value && !value.startsWith("--") && !options[key], "Unknown, repeated or missing export CLI option");
    options[key] = value;
  }
  requireValue(options.report && options["report-id"], "Use --report COMPLETED_AGGREGATE_JSON --report-id UNIQUE_SNAPSHOT_ID [--experiment-id ID for empty evidence]");
  requireValue((await stat(options.report)).size <= SHOPPER_EXPORT_LIMIT, "Aggregate input exceeds 8 MiB");
  const report: unknown = JSON.parse(await readFile(options.report, "utf8"));
  const envelope = buildEnvelope(report, { reportId: options["report-id"], experimentId: options["experiment-id"] });
  return postShopperResults(envelope, { fetchImpl });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  shopperExportMain(process.argv.slice(2)).then((receipt) => process.stdout.write(`${JSON.stringify(receipt)}\n`))
    .catch((error: unknown) => { process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
}
