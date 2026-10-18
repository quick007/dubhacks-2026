/** Deterministic positive-choice training inputs; never treats nonclicks as dislikes. */
import { createHash } from "node:crypto";
import { readFile, writeFile, unlink } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const key = (session, view) => `${session}/${view}`;
export function choiceExamples(rows, { allowSynthetic = false } = {}) {
  const sessions = new Map(), views = new Map(), invalidViews = new Set(), events = new Set();
  const eligible = [];
  for (const row of rows) {
    if (row.schema !== "fieldwork.training-observation.v1" || row.provenance?.splitVersion !== "fieldwork.session-split.v1") throw new Error("Versioned provenance required");
    const p = row.provenance, e = row.observation;
    if (!allowSynthetic && (p.classification?.environment !== "production" || p.classification?.traffic !== "real")) continue;
    const bucket = createHash("sha256").update(`fieldwork.session-split.v1/${p.sessionId}`).digest()[0] % 10;
    const split = bucket === 8 ? "validation" : bucket === 9 ? "test" : "train";
    if (row.split !== split || sessions.has(p.sessionId) && sessions.get(p.sessionId) !== split) throw new Error("Session split mismatch");
    if (!e?.eventId || p.eventId !== e.eventId || events.has(e.eventId)) throw new Error("Duplicate or mismatched observation ID");
    events.add(e.eventId); sessions.set(p.sessionId, split); eligible.push(row);
    if (e.action === "catalog-view") {
      const id = key(p.sessionId, e.data.viewId), previous = views.get(id);
      if (previous && JSON.stringify(previous.observation.data.products) !== JSON.stringify(e.data.products)) invalidViews.add(id);
      else views.set(id, row);
    }
  }
  const examples = [];
  for (const row of eligible) {
    const p = row.provenance, e = row.observation;
    if (!["selection", "cart-add"].includes(e.action)) continue;
    const id = key(p.sessionId, e.data.viewId), view = views.get(id);
    if (!view || invalidViews.has(id) || view.observation.sequence >= e.sequence) continue;
    const candidates = view.observation.data.products;
    const rendered = candidates.find((c) => c.productId === e.data.productId);
    if (!rendered || e.action === "cart-add" && rendered.variantId !== e.data.variantId) continue;
    examples.push({ schema: "fieldwork.shopping-choice.v1", split: row.split,
      provenance: { ...p, catalogEventId: view.observation.eventId, catalogBatchId: view.provenance.batchId,
        catalogSourceSha256: view.provenance.sourceSha256, derivation: "explicit-positive-choice.v1" },
      context: view.observation.context, candidates,
      chosen: { productId: e.data.productId, variantId: e.data.variantId },
      choiceScope: rendered.variantId === e.data.variantId ? "rendered-variant" : "product-with-observed-variant-change",
      signal: e.action, negativeLabels: [], realPurchaseObserved: false,
      selectionMechanism: "observational-not-randomized" });
  }
  return examples.sort((a, b) => a.provenance.sessionId.localeCompare(b.provenance.sessionId) || a.provenance.eventId.localeCompare(b.provenance.eventId));
}
export async function main(args) {
  const options = {}; let allowSynthetic = false;
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    if (flag === "--allow-synthetic") { allowSynthetic = true; continue; }
    if (!["--input", "--out"].includes(flag) || !args[i + 1] || options[flag]) throw new Error("Use --input OBSERVATIONS_JSONL --out NEW_CHOICES_JSONL [--allow-synthetic]");
    options[flag] = args[++i];
  }
  if (!options["--input"] || !options["--out"]) throw new Error("Input and new output required");
  const bytes = await readFile(options["--input"]);
  if (bytes.length > 64 * 1024 * 1024) throw new Error("Input exceeds 64 MiB; narrow export range");
  const rows = bytes.toString("utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
  const examples = choiceExamples(rows, { allowSynthetic });
  await writeFile(options["--out"], examples.map((e) => JSON.stringify(e) + "\n").join(""), { flag: "wx", mode: 0o600 });
  const manifest = { schema: "fieldwork.shopping-choice-export.v1", inputSha256: createHash("sha256").update(bytes).digest("hex"),
    inputObservations: rows.length, examples: examples.length, allowSynthetic, derivation: "explicit-positive-choice.v1",
    splitVersion: "fieldwork.session-split.v1", negativeLabelsInferred: false, replayIncluded: false };
  try { await writeFile(options["--out"] + ".manifest.json", JSON.stringify(manifest, null, 2) + "\n", { flag: "wx", mode: 0o600 }); }
  catch (error) { await unlink(options["--out"]); throw error; }
  return manifest;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main(process.argv.slice(2))
  .then((manifest) => process.stdout.write(JSON.stringify(manifest) + "\n"))
  .catch((error) => { process.stderr.write(error.message + "\n"); process.exitCode = 1; });
