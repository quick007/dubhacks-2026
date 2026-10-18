// Protocol fixtures only: no weights, native execution, HTTP or trained quality evidence.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import path from "node:path";
import { ADAPTER_VERSION, ADAPTER_MODULE, FIXED, SOURCES, EVIDENCE, CATALOG_RUN,
  validateAdapterProfile, verifyAdapterArtifacts, adapterArguments, assertAdapterHealth } from "./personalized-adapter-profile.mjs";
import { acceptedProfile, main, nativeArguments } from "./dev-personalized.mjs";
const digest = bytes => createHash("sha256").update(bytes).digest("hex");

function fixture() {
  const root = "/cpu-fixture/code", assets = "/cpu-fixture/assets", files = new Map(), overrides = new Map();
  function put(filename, value, pinned) {
    const bytes = Buffer.from(typeof value === "string" ? value : JSON.stringify(value)); files.set(filename, bytes);
    if (pinned) overrides.set(digest(bytes), pinned); return pinned || digest(bytes);
  }
  const ref = (name, value, pinned) => ({ root: "assets", file: `data/ranker/pretrained/accepted/${name}`, sha256: put(`${assets}/data/ranker/pretrained/accepted/${name}`, value, pinned) });
  const profile = { version: ADAPTER_VERSION, status: "accepted", kind: "lexical-choice-lora-query-gate.v2", queryIntentSourceManifestSha256: FIXED.queryTrain,
    catalogFeatureRoot: { root: "assets", directory: CATALOG_RUN }, sourceSha256: {} };
  for (const name of SOURCES) profile.sourceSha256[name] = put(`${root}/${name}`, `CPU-only source fixture ${name}`);
  put(`${root}/experiments/ranker/pretrained/PERSONA_SPECIFIC_QUERY_REGRESSION_POLICY_V1.json`, {
    version: "persona-specific-query-regression-policy.v1", adapterPublic320PlanSha256: FIXED.httpPlan, originalPromotionGateSha256: FIXED.qualityGate,
    fixedCases: 320, requirements: { sameGuardV5Top1Agreements: 320, sourceOrHardConstraintViolations: 0 },
  }, FIXED.queryRegressionPolicy);
  profile.queryIntentHead = ref("query-head.pt", "not a real checkpoint");
  profile.colorHead = ref("color-head.pt", "not real color weights", FIXED.colorHead);
  profile.colorCache = ref("color-cache.pt", "not a real color cache", FIXED.colorCache);
  const directory = `${assets}/data/ranker/pretrained/accepted`;
  const adapterSha = put(`${directory}/fit/adapter.safetensors`, "not real adapter weights");
  const parity = { version: "persona-choice-inference-parity.v1", adapterSha256: adapterSha, cachedScoringMode: "plain", dynamicMode: "gradient-public-XHY-last16",
    cachedScoreMatchesTrainingFeaturesExact: true, dynamicMatchesCachedFeatureScoresExact: true, zeroAndNonzeroControlPassed: true, learnedPostfitPassed: true,
    helperSourceSha256: { score: profile.sourceSha256["experiments/ranker/pretrained/store_persona_choice_scoring_v1.py"], capture: profile.sourceSha256["experiments/ranker/pretrained/store_persona_choice_capture_v1.py"] },
    zeroNonzeroProofSha256: digest("proof"), postfitProofSha256: [0, 1, 2, 3].map(i => digest(`postfit-${i}`)) };
  const paritySha = put(`${directory}/inference-parity.json`, parity);
  const routes = Array.from({ length: 16 }, (_, i) => i + 100);
  const index = { version: "persona-choice-public-feature-cache.v1", teacherOrTargetsIncluded: false, modelRevision: FIXED.revision, entries: {},
    key: "exact public prompt SHA", baseModelFiles: { fixture: "not weights" }, featureSourceSha256: profile.sourceSha256["experiments/ranker/pretrained/store_persona_choice_features_v2.py"],
    formatterSha256: profile.sourceSha256["experiments/ranker/pretrained/store_qwen_lexical_choice_inputs_v2.py"], scoreSourceSha256: parity.helperSourceSha256.score };
  for (let i = 0; i < 20; i++) {
    const key = digest(`public-prompt-${i}`), archive = `${key}.npz`;
    index.entries[key] = { archive, archiveSha256: put(`${directory}/public-features/${archive}`, `fixture pure archive ${i}`), dtypes: { x: "float16", h: "float16", y: "float16" },
      routeTokenIds: routes, promptTokenIdsSha256: key, promptTokens: 900, prefixPath: "fixture prefix", nativeInputSuffixTokens: 16 };
  }
  const indexSha = put(`${directory}/public-features/index.json`, index);
  const serving = { version: "persona-choice-adapter-serving.v1", actualSuffixTokens: 16, trainedUpdates: 3584, base927Unchanged: true, postfitNativeControlsExact: true,
    inferenceTargetFieldsRequired: false, cachedScoringMode: "plain", dynamicMode: "gradient-public-XHY-last16", modelRevision: index.modelRevision, catalogSha256: FIXED.catalog,
    adapter: { file: "fit/adapter.safetensors", sha256: adapterSha, rank: 4, scale: 20, target: "language_model.model.layers.31.mlp.down_proj" },
    inferenceParity: { file: "inference-parity.json", sha256: paritySha }, publicCache: { file: "public-features/index.json", sha256: indexSha },
    routeTokenIds: routes, productIds: routes.map(x => `fixture-product-${x}`), scoreSourceSha256: parity.helperSourceSha256.score, formatterSha256: index.formatterSha256 };
  profile.servingManifest = ref("serving.json", serving);
  const rankHeadSha = put(`${directory}/v5-head.pt`, "not real rank weights");
  profile.rankExport = ref("v5-export.json", { head: { file: "v5-head.pt", sha256: rankHeadSha } }, FIXED.rankExport);
  profile.copyManifest = ref("copy-serving.json", { all48BindingsSupervised: true, runtimeTeacherLookup: false, headSha256: digest("copy head fixture") }, FIXED.copyManifest);
  profile.integrationManifest = ref("integration.json", { fixtureOnly: true });
  const proofs = {
    offlineAudit: { version: "persona-choice-combined-independent-audit.v1", status: "completed-offline-adjudication", offlineCoreAndReplacementPass: true, overallPromotionPass: false,
      privateQualityJSONParsed: true, gateSha256: FIXED.qualityGate, sourceSha256: profile.sourceSha256["experiments/ranker/pretrained/audit_persona_choice_combined_v1.py"] },
    admissionBeforeGold: { privateQualityJSONParsed: false, adapterCheckpointSha256: adapterSha, packetManifestSha256: FIXED.publicPacket,
      adapterFreezeSha256: digest("adapter freeze fixture"), combinedFreezeSha256: digest("combined freeze fixture") },
    queryMechanicalAudit: { status: "passed-independent-TRAIN-replay", headSha256: profile.queryIntentHead.sha256, sourceManifestSha256: FIXED.queryTrain,
      auditSourceSha256: profile.sourceSha256["experiments/ranker/pretrained/store_query_intent_audit_v3.py"], privateQualityLabelsRead: false,
      replay: { observedReplayOptimizerSteps: 250, everyFinalParameterBitwiseMatches: true }, twoReloadsExactLogitParity: true, reloadSavedParity: { rows: 1272, exactSavedProbabilityParity: true } },
    queryCombinationReplay: { status: "passed", adapterCheckpointSha256: adapterSha, gateCheckpointSha256: profile.queryIntentHead.sha256,
      version: "persona-combined-query-replay-audit.v1", v5HeadSha256: rankHeadSha, rows: 344, uniqueQueryVectorsReplayed: 292, nativeCalls: 0, optimizerUpdates: 0,
      combinedFreezeSha256: digest("combined freeze fixture"), overallPromotionPass: false, servingPending: true, privateQualityLabelsRead: false,
      sourceSha256: { "/historical/provenance/only/audit_persona_combined_query_replay_v1.py": profile.sourceSha256["experiments/ranker/pretrained/audit_persona_combined_query_replay_v1.py"] },
      exactQueryProbabilityAndV5Parity: true, actualFloat32BlendAndGuardsExact: true, modelRowsChecked: 1376 },
    httpValidation: { version: "persona-adapter-http-semantic-probe.v2", status: "completed-fixed320-validation", fixedDenominator: 320,
      summary: { completed: 320, failedOrTimeout: 0, unattempted: 0, protocolPassed: 320, uniqueActualRequestIds: 320, sameGuardV5Comparisons: 320, sameGuardV5Top1Agreements: 320 },
      cases: Array.from({ length: 320 }, (_, i) => ({ probeCaseId: `fixture-case-${i}`, status: "completed", analysis: {
        protocolPassed: true, qualityPassed: true, sameV5Top1UnderSameGuards: true, learnedGateCheckpointSha256: profile.queryIntentHead.sha256,
        errors: [], qualityFailures: [], requestId: `fixture-request-${i}` } })),
    },
    browserValidation: { version: "store-persona-browser-probe.v2", sourcesUnchanged: true, startupUnchanged: true,
      summary: { protocolPassed: 13, loadedSwitchFirst3: { valid: 10 }, loadedSwitchFull6: { valid: 10 } }, functionalArms: [{ passed: true }, { passed: true }] },
    shutdown: Object.fromEntries(["frozen927HashesUnchanged", "copyCheckpointUnchanged", "v5ServingExportUnchanged", "v5HeadCheckpointUnchanged", "rankAdapterRestoredToCopy",
      "rankAdapterAndPublicCacheUnchanged", "learnedVariantAndQueryIntentAssetsUnchanged", "adapterEngineSourcesUnchanged"].map(key => [key, true])),
  };
  profile.evidence = {};
  put(`${directory}/plan.json`, { fixedDenominator: 320, cases: proofs.httpValidation.cases.map(row => ({ probeCaseId: row.probeCaseId })) }, FIXED.httpPlan);
  for (const key of EVIDENCE) {
    if (key === "queryCombinationReplay") proofs[key].mechanicalQueryAuditSha256 = profile.evidence.queryMechanicalAudit.sha256;
    if (key === "httpValidation") proofs[key].servingSelection = Object.fromEntries(["offlineAudit", "queryMechanicalAudit", "queryCombinationReplay"].map(name => [name + "Sha256", profile.evidence[name].sha256]));
    profile.evidence[key] = ref(`${key}.json`, proofs[key]);
  }
  const acceptance = { version: "persona-adapter-live-acceptance.v1", status: "accepted", overallPromotionPass: true, browserPassed: true, freshCopyFactsPassed: true, shutdownIntegrityPassed: true,
    specificQueryRegressionPassed: true, specificQueryRegressionPolicySha256: FIXED.queryRegressionPolicy,
    specificQueryRegressionPlanSha256: FIXED.httpPlan, v5CheckpointSha256: rankHeadSha,
    adapterSha256: adapterSha, servingManifestSha256: profile.servingManifest.sha256, gateCheckpointSha256: profile.queryIntentHead.sha256, integrationManifestSha256: profile.integrationManifest.sha256,
    evidenceSha256: Object.fromEntries(EVIDENCE.map(key => [key, profile.evidence[key].sha256])), sourceByteEquivalenceVerified: true, sourceSha256: { ...profile.sourceSha256 },
    artifactSha256: Object.fromEntries(["servingManifest", "queryIntentHead", "colorHead", "colorCache", "rankExport", "copyManifest", "integrationManifest"].map(key => [key, profile[key].sha256])) };
  profile.evidence.acceptance = ref("acceptance.json", acceptance);
  const deps = { read: async file => { if (!files.has(file)) throw Object.assign(new Error("fixture missing"), { code: "ENOENT" }); return files.get(file); },
    inspect: async file => { if (!files.has(file)) throw new Error("fixture missing"); return { isFile: () => true, isSymbolicLink: () => false, size: files.get(file).length }; },
    resolve: async file => file, digestBytes: bytes => overrides.get(digest(bytes)) || digest(bytes) };
  return { root, assets, files, profile, serving, parity, index, proofs, acceptance, deps, put,
    normalized: () => validateAdapterProfile(profile, root, assets),
    repinEvidence(key, value) { profile.evidence[key] = ref(`${key}.json`, value); acceptance.evidenceSha256[key] = profile.evidence[key].sha256; profile.evidence.acceptance = ref("acceptance.json", acceptance); },
  };
}

test("accepted adapter kind produces exact new engine flags without legacy head/port flags", async () => {
  const f = fixture(), profile = await verifyAdapterArtifacts(f.normalized(), f.deps);
  const args = nativeArguments({ assetRoot: f.assets, profile, candidate: "/assets/copy-run" });
  assert.equal(args[1], ADAPTER_MODULE);
  for (const flag of ["--serving-manifest", "--serving-manifest-sha256", "--catalog-feature-root", "--color-head", "--color-cache", "--query-intent-head", "--query-intent-head-sha256", "--query-intent-source-manifest-sha256", "--integration-manifest", "--rank-head-export", "--rank-head-export-sha256", "--asset-root", "--candidate-run"]) assert.ok(args.includes(flag));
  for (const flag of ["--persona-head", "--persona-feature-root", "--port"]) assert.equal(args.includes(flag), false);
  assert.equal(args[args.indexOf("--catalog-feature-root") + 1], `${f.assets}/${CATALOG_RUN}`);
  assert.equal(profile.assets.filter(a => a.file.endsWith(".npz")).length, 20);
  assert.equal(profile.personaHead.sha256, f.serving.adapter.sha256);
});

test("missing/failed/partial acceptance causes zero runner calls even with --run", async () => {
  const f = fixture(); let started = 0;
  for (const bad of [undefined, { ...f.profile, status: "pending" }, { ...f.profile, evidence: { ...f.profile.evidence, acceptance: undefined } }]) {
    await assert.rejects(() => main(["--run"], { free: async () => true,
      buildPlan: async () => acceptedProfile(f.root, f.assets, async () => {
        if (!bad) throw Object.assign(new Error("no accepted profile"), { code: "ENOENT" });
        return Buffer.from(JSON.stringify(bad));
      }), run: async () => { started++; } }));
  }
  for (const [key, bad] of [["httpValidation", { ...f.proofs.httpValidation, summary: { ...f.proofs.httpValidation.summary, completed: 319, unattempted: 1 } }],
    ["offlineAudit", { ...f.proofs.offlineAudit, offlineCoreAndReplacementPass: false }],
    ["admissionBeforeGold", { ...f.proofs.admissionBeforeGold, adapterCheckpointSha256: "1".repeat(64) }],
    ["queryCombinationReplay", { ...f.proofs.queryCombinationReplay, combinedFreezeSha256: "2".repeat(64) }],
    ["shutdown", { ...f.proofs.shutdown, copyCheckpointUnchanged: false }],
    ["browserValidation", { ...f.proofs.browserValidation, functionalArms: [{ passed: true }, { passed: false }] }]]) {
    const x = fixture(); x.repinEvidence(key, bad);
    await assert.rejects(() => main(["--run"], { free: async () => true, buildPlan: async () => verifyAdapterArtifacts(x.normalized(), x.deps), run: async () => { started++; } }));
  }
  assert.equal(started, 0);
});

test("co-located20 NPZ/cache/certificate/adapter checks reject missing or altered bytes without original-path fallback", async () => {
  for (const select of [f => `${f.assets}/data/ranker/pretrained/accepted/fit/adapter.safetensors`,
    f => `${f.assets}/data/ranker/pretrained/accepted/public-features/index.json`,
    f => `${f.assets}/data/ranker/pretrained/accepted/inference-parity.json`,
    f => `${f.assets}/data/ranker/pretrained/accepted/public-features/${Object.keys(f.index.entries)[0]}.npz`]) {
    const f = fixture(); f.files.set(select(f), Buffer.from("altered bytes"));
    await assert.rejects(() => verifyAdapterArtifacts(f.normalized(), f.deps), /checksum differs/);
    f.files.delete(select(f)); await assert.rejects(() => verifyAdapterArtifacts(f.normalized(), f.deps), /missing/);
  }
  const f = fixture(); f.serving.publicCache.file = "../original-worktree/index.json";
  f.profile.servingManifest.sha256 = f.put(f.normalized().servingManifest.file, f.serving);
  await assert.rejects(() => verifyAdapterArtifacts(f.normalized(), f.deps), /co-located relative artifact/);
});

test("pure public index rejects teacher metadata, count loss and route mismatch even when rehashed", async () => {
  for (const mutate of [i => { i.entries[Object.keys(i.entries)[0]].targetWeights = [1]; }, i => { delete i.entries[Object.keys(i.entries)[0]]; },
    i => { i.entries[Object.keys(i.entries)[0]].routeTokenIds = [...i.entries[Object.keys(i.entries)[0]].routeTokenIds].reverse(); }]) {
    const f = fixture(); mutate(f.index);
    f.serving.publicCache.sha256 = f.put(`${f.assets}/data/ranker/pretrained/accepted/public-features/index.json`, f.index);
    f.profile.servingManifest.sha256 = f.put(`${f.assets}/data/ranker/pretrained/accepted/serving.json`, f.serving);
    await assert.rejects(() => verifyAdapterArtifacts(f.normalized(), f.deps), /public cache|target-free20/);
  }
});

test("source drift, unknown roots, extraction-root confusion and inference-certificate mismatch fail closed", async () => {
  const f = fixture(); f.files.set(`${f.root}/${SOURCES[0]}`, Buffer.from("source drift"));
  await assert.rejects(() => verifyAdapterArtifacts(f.normalized(), f.deps), /checksum differs/);
  for (const mutate of [p => { p.catalogFeatureRoot.directory = "data/ranker/pretrained/new-extraction"; }, p => { p.colorHead.root = "remote"; },
    p => { p.servingManifest.file = "data/ranker/pretrained/../../.env"; }, p => { p.sourceSha256[".env"] = "a".repeat(64); }]) {
    const x = fixture(); mutate(x.profile); assert.throws(x.normalized);
  }
  const x = fixture(); x.parity.learnedPostfitPassed = false;
  x.serving.inferenceParity.sha256 = x.put(`${x.assets}/data/ranker/pretrained/accepted/inference-parity.json`, x.parity);
  x.profile.servingManifest.sha256 = x.put(`${x.assets}/data/ranker/pretrained/accepted/serving.json`, x.serving);
  await assert.rejects(() => verifyAdapterArtifacts(x.normalized(), x.deps), /inference certificate/);
});

test("ready health must bind accepted adapter, learned query gate, color and20 public entries", async () => {
  const f = fixture(), profile = await verifyAdapterArtifacts(f.normalized(), f.deps);
  const health = { ready: true, status: "ready", startup: { rankCheckpointSha256: profile.healthExpected.v5HeadSha256, copyCheckpointSha256: FIXED.nativeCopy,
    copyCandidate: { checkpointSha256: profile.healthExpected.copyHeadSha256 }, rankSessionProvenance: { rankHead: { servingExportSha256: profile.rankExport.sha256 } },
    sourceTensorsHashed: 927, memory: { processBudgetBytes: 7 * 2 ** 30 }, imageDeliveryManifestSha256: profile.integrationManifest.sha256,
    persona: { kind: "persona-adapter-engine.v2", checkpointSha256: profile.personaHead.sha256,
    servingManifestSha256: profile.servingManifest.sha256, queryIntent: { checkpointSha256: profile.queryIntentHead.sha256, sourceManifestSha256: FIXED.queryTrain,
      sourceSha256: profile.sourceSha256["experiments/ranker/pretrained/store_query_intent_gate_v3.py"], phraseOrPersonRules: false },
    color: { checkpointSha256: FIXED.colorHead, failedOldProductOrderingUsed: false }, publicDecisionFeatureCacheEntries: 20, privateTargetOrTeacherInputs: false } } };
  assert.doesNotThrow(() => assertAdapterHealth(health, profile));
  for (const mutate of [p => { p.queryIntent.checkpointSha256 = "0".repeat(64); }, p => { p.kind = "legacy-context"; }, p => { p.servingManifestSha256 = "1".repeat(64); }]) {
    const bad = structuredClone(health); mutate(bad.startup.persona); assert.throws(() => assertAdapterHealth(bad, profile), /identity differs/);
  }
});

test("query regression rejects summary-only claims, one changed choice/guard failure, wrong identity and duplicate request", async () => {
  const mutations = [
    h => { h.summary.sameGuardV5Comparisons = 319; }, h => { h.summary.sameGuardV5Top1Agreements = 319; },
    h => { h.cases[17].analysis.sameV5Top1UnderSameGuards = false; },
    h => { h.cases[17].analysis.qualityFailures = ["hard color violated"]; },
    h => { h.cases[17].analysis.errors = ["guard mismatch"]; },
    h => { h.cases[17].analysis.learnedGateCheckpointSha256 = "0".repeat(64); },
    h => { h.cases[17].analysis.requestId = h.cases[16].analysis.requestId; },
    h => { h.cases[17].probeCaseId = "replacement-outcome-dependent-case"; },
    h => { h.servingSelection.queryCombinationReplaySha256 = "0".repeat(64); },
  ];
  let spawned = 0;
  for (const mutate of mutations) {
    const f = fixture(); mutate(f.proofs.httpValidation); f.repinEvidence("httpValidation", f.proofs.httpValidation);
    await assert.rejects(() => main(["--run"], { free: async () => true,
      buildPlan: async () => verifyAdapterArtifacts(f.normalized(), f.deps), run: async () => { spawned++; } }), /specific-query regression policy/);
  }
  assert.equal(spawned, 0);
  const f = fixture(); f.acceptance.specificQueryRegressionPassed = false;
  f.profile.evidence.acceptance.sha256 = f.put(f.normalized().evidence.acceptance.file, f.acceptance);
  await assert.rejects(() => verifyAdapterArtifacts(f.normalized(), f.deps), /final root-owned acceptance/);
});

test("policy and actual co-located fixed320 plan bytes cannot drift behind passing booleans", async () => {
  const f = fixture();
  f.files.set(`${f.root}/experiments/ranker/pretrained/PERSONA_SPECIFIC_QUERY_REGRESSION_POLICY_V1.json`, Buffer.from("altered policy"));
  await assert.rejects(() => verifyAdapterArtifacts(f.normalized(), f.deps), /checksum differs/);
  const x = fixture(); x.files.set(`${x.assets}/data/ranker/pretrained/accepted/plan.json`, Buffer.from("altered fixed roster"));
  await assert.rejects(() => verifyAdapterArtifacts(x.normalized(), x.deps), /checksum differs/);
});
