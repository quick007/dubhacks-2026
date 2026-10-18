/** Development-only real shopper smoke; excluded from the held-out cohort. */
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { loadShopperPacket } from "../../../../apps/runner/src/shopper-eval-cli.ts";
import { executeShopperSession } from "../../../../apps/runner/src/shopper-runner.ts";
import { ChromeShopperBrowser } from "../../../../apps/runner/src/shopper-chrome.ts";
import { CodexShopperAgent, findCodexBinary, codexModelAvailability } from "../../../../apps/runner/src/shopper-codex-agent.ts";
const root = path.dirname(fileURLToPath(import.meta.url));
const packetPath = path.resolve(root, "../store_shopper_eval_v1");
const packet = await loadShopperPacket(packetPath);
const scenario = packet.cases.find((item) => item.caseId === "shopper-pair-02")!;
const binary = await findCodexBinary();
const availability = await codexModelAvailability(binary, "gpt-6.1-sol");
await writeFile(path.join(root, "model-availability.json"), JSON.stringify(availability, null, 2) + "\n", { flag: "wx" });
const session = await executeShopperSession(packetPath, packet, scenario, "generic", path.join(root, "generic"), {
  openBrowser: (url, directory) => ChromeShopperBrowser.open(url, directory),
  openAgent: (assignment, log) => new CodexShopperAgent(binary, "gpt-6.1-sol", assignment, log),
});
const report = { schema: "shopper-server-checkout-development-smoke.v1", synthetic: true, heldoutCohortMember: false,
  purpose: "Actual Sol visible-store navigation and local server receipt plumbing on a prior development case",
  status: session.status, protocolPassed: session.protocol.passed, elapsedMs: session.elapsedMs, steps: session.steps.length,
  fakeOrderConfirmed: session.fakeOrderConfirmed, choice: session.choice, artifacts: session.artifacts, error: session.error };
await writeFile(path.join(root, "smoke-result.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
process.stdout.write(JSON.stringify(report) + "\n");
