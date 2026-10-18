import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const sources = {};
async function json(path) {
  const bytes = await readFile(resolve(root, path));
  sources[path] = createHash("sha256").update(bytes).digest("hex");
  return JSON.parse(bytes);
}
const rank = await json("experiments/ranker/pretrained/function_v5_audit_v1/INDEPENDENT_AUDIT.json");
const browser = await json("experiments/ranker/pretrained/STORE_PAIRED_INCREMENTAL_PROBE_V1_INDEPENDENT_AUDIT.json");
const profile = await json("experiments/ranker/pretrained/STORE_FAST_PATH_PROFILE_V1_INDEPENDENT_REVIEW.json");
if (rank.version !== "function-v5-native-independent-audit.v1" || rank.totalChecks !== 1317
  || browser.version !== "paired-incremental-independent-audit.v1" || browser.checksPassed !== 3291
  || browser.denominators.fixedPairs !== 20 || browser.denominators.failed !== 0 || browser.denominators.unattempted !== 0
  || profile.status !== "PASS" || profile.exactLogicalParityPairs !== 20 || profile.copyFieldsPerArm !== 20) {
  throw new Error("Completed measured source scope required; refusing an incomplete presentation projection.");
}
const value = {
  version: "current-model-presentation.v1", synthetic: true, humanConversionClaim: false,
  rank: { cases: rank.results.learned.cases, catalogProducts: 16,
    before: rank.results["frozen-v3"], after: rank.results.learned,
    scope: "Independent known-catalog wordings held out of the fixed v5 fit; prior trained ranker versus balanced trained ranker." },
  loadedBrowser: { pairs: 20, generic: browser.statistics.genericFirstUseful,
    local: browser.statistics.localFirstUseful, incremental: browser.statistics.pairedIncrementalLocalMinusGeneric,
    scope: "Paired loaded/pre-scrolled collection requests; no cold navigation or physical paint claim.", imageCache: "Eligible from each isolated context's unscored bootstrap.",
    physicalPaintMeasured: false },
  server: { fullSix: profile.timingMs["A-original"].fullSixWireReadyMs,
    cards: profile.cardsPerArm, freshFields: profile.copyFieldsPerArm,
    scope: "Warm local model; fresh query, ranking, six-card generation, guards and serialization. No browser/images/startup." },
  sources,
};
const output = resolve(root, "apps/web/src/fieldwork/current-model.json");
const serialized = JSON.stringify(value, null, 2) + "\n";
if (process.argv.includes("--check")) {
  if (await readFile(output, "utf8") !== serialized) throw new Error("Presentation projection differs from measured source files.");
} else await writeFile(output, serialized);
console.log(JSON.stringify({ output, sources: Object.keys(sources).length, checked: process.argv.includes("--check") }));
