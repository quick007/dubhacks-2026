// CPU-only admission for the explicitly accepted adapter launcher kind.
// This module never creates acceptance evidence or imports a native runtime.
import { createHash } from "node:crypto";
import { readFile, lstat, realpath } from "node:fs/promises";
import path from "node:path";

export const ADAPTER_VERSION = "personalized-local-adapter-profile.v2";
export const ADAPTER_MODULE = "experiments.ranker.pretrained.store_persona_adapter_engine_v2";
export const CATALOG_RUN = "data/ranker/pretrained/actual-store-function-v3-attempt-1";
export const FIXED = Object.freeze({
  colorHead: "2ec600745b19086b4bedcc4f66c283b5870b65eebce43f8715f0e8cbb154ea56",
  colorCache: "dbf8c335a533120f5220a19f88ad163e4984631e7d9a7e80e5894080d02af28e",
  queryTrain: "25fdb0c976dc3b6e01e826e7f30ea63fc65e3fbb85f56f24f4c83e8538703232",
  rankExport: "17a0a15a2da6e2fc90e651e1f5bb2c237d61241795ec0cca548a4f3ce319d703",
  copyManifest: "4bf86af880a59ffb420646bbaf24cd200b743dccb5d0cf9274190d3d0a4acfae",
  revision: "8b2b98c00a6b4d291155e4890773ca8f769aee53",
  catalog: "bdd7e2f5f5bc41c147830b5ecd03b550794a237ad633d4e4233688ec223b8450",
  nativeCopy: "56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8",
  qualityGate: "73fa0d52c5ef4fe983834f610ef00cf89a899cf420cdb81008dd5a7909298c6e",
  publicPacket: "2a0801c73440dfbcea50f73c0c1b2de3a96abc8c975caf8d54e92010a84fe312",
  queryRegressionPolicy: "b1436edfd3f9641c9431cb41aab9fb04b99c3a5f7606f3d456d2c6b87bfe4734",
  httpPlan: "a06cfbdb157d343cf0d435dcfdab6e936d4182656d0ef04801d8a57fe759c8e1",
});
export const EVIDENCE = Object.freeze(["offlineAudit", "admissionBeforeGold", "queryMechanicalAudit", "queryCombinationReplay", "httpValidation", "browserValidation", "shutdown"]);
export const SOURCES = Object.freeze([
  "scripts/dev-personalized.mjs", "scripts/personalized-adapter-profile.mjs",
  ...["store_persona_adapter_engine_v2.py", "store_persona_adapter_runtime_v2.py", "store_persona_adapter_contract_v2.py",
    "store_persona_choice_scoring_v1.py", "store_persona_choice_capture_v1.py", "store_persona_choice_catalog_cache_v1.py", "store_persona_choice_features_v2.py",
    "store_qwen_lexical_choice_inputs_v2.py", "store_query_intent_gate_v3.py", "store_query_intent_audit_v3.py",
    "audit_persona_choice_combined_v1.py", "audit_persona_combined_query_replay_v1.py"].map(x => `experiments/ranker/pretrained/${x}`),
]);
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const need = (ok, message) => { if (!ok) throw new Error(`Accepted adapter profile: ${message}`); };
const exact = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
function relative(name) {
  return typeof name === "string" && name.length > 0 && !path.isAbsolute(name) && !name.includes("\\") && !name.split("/").some(x => ["..", ".", ""].includes(x));
}
function location(record, root, assets, directory = false) {
  const base = record?.root === "worktree" ? root : record?.root === "assets" ? assets : null;
  const name = record?.[directory ? "directory" : "file"];
  need(base && relative(name) && /^(data\/ranker\/pretrained\/|experiments\/ranker\/pretrained\/)/.test(name), "explicit local research artifact path required");
  if (!directory) need(exact(record.sha256), "exact artifact SHA256 required");
  return { ...record, file: path.resolve(base, name), base: path.resolve(base), ...(directory ? {} : { sha256: record.sha256 }) };
}

export function validateAdapterProfile(value, root, assetRoot) {
  need(value?.version === ADAPTER_VERSION && value.status === "accepted" && value.kind === "lexical-choice-lora-query-gate.v2", "quality and live acceptance are pending; no model was launched");
  need(value.queryIntentSourceManifestSha256 === FIXED.queryTrain, "reviewed query TRAIN manifest required");
  const result = { kind: value.kind, version: value.version, queryIntentSourceManifestSha256: FIXED.queryTrain };
  for (const key of ["servingManifest", "queryIntentHead", "colorHead", "colorCache", "rankExport", "copyManifest", "integrationManifest"])
    result[key] = location(value[key], root, assetRoot);
  for (const [key, expected] of Object.entries({ colorHead: FIXED.colorHead, colorCache: FIXED.colorCache, rankExport: FIXED.rankExport, copyManifest: FIXED.copyManifest }))
    need(result[key].sha256 === expected, `reviewed ${key} checksum required`);
  result.catalogFeatureRoot = location(value.catalogFeatureRoot, root, assetRoot, true).file;
  need(result.catalogFeatureRoot === path.resolve(assetRoot, CATALOG_RUN), "catalogFeatureRoot must be the original16 catalog asset run, not context extraction");
  need(value.evidence && typeof value.evidence === "object", "completed acceptance evidence required");
  result.evidence = Object.fromEntries([...EVIDENCE, "acceptance"].map(key => [key, location(value.evidence[key], root, assetRoot)]));
  need(value.sourceSha256 && typeof value.sourceSha256 === "object" && SOURCES.every(name => exact(value.sourceSha256[name])), "current launcher/runtime source inventory required");
  need(Object.keys(value.sourceSha256).every(name => relative(name) && /^(scripts\/.*\.mjs|experiments\/ranker\/pretrained\/.*\.py)$/.test(name)), "source inventory uses explicit code-root relative module paths");
  result.sourceSha256 = { ...value.sourceSha256 }; result.root = root;
  return result;
}

export async function verifyAdapterArtifacts(profile, { read = readFile, inspect = lstat, resolve = realpath, digestBytes = sha } = {}) {
  const assets = [];
  async function checked(file, expected, base) {
    const info = await inspect(file);
    need(info.isFile() && !info.isSymbolicLink() && info.size > 0, `regular nonempty artifact required: ${file}`);
    const actual = await resolve(file); const resolvedBase = await resolve(base);
    need(actual.startsWith(resolvedBase + path.sep), `artifact escaped selected root: ${file}`);
    const bytes = await read(file); need(exact(expected) && digestBytes(bytes) === expected, `checksum differs: ${file}`);
    assets.push({ file, bytes: info.size, sha256: expected }); return bytes;
  }
  async function artifact(record) { return checked(record.file, record.sha256, record.base); }
  async function jsonArtifact(record) { return JSON.parse(await artifact(record)); }
  async function colocated(root, name, checksum) {
    need(relative(name), "co-located relative artifact required; no original-path fallback");
    return checked(path.join(root, name), checksum, root);
  }
  for (const [name, checksum] of Object.entries(profile.sourceSha256))
    await checked(path.join(profile.root, name), checksum, profile.root);
  const regressionPolicy = JSON.parse(await checked(path.join(profile.root, "experiments/ranker/pretrained/PERSONA_SPECIFIC_QUERY_REGRESSION_POLICY_V1.json"), FIXED.queryRegressionPolicy, profile.root));
  need(regressionPolicy.version === "persona-specific-query-regression-policy.v1" && regressionPolicy.adapterPublic320PlanSha256 === FIXED.httpPlan &&
    regressionPolicy.originalPromotionGateSha256 === FIXED.qualityGate && regressionPolicy.fixedCases === 320 &&
    regressionPolicy.requirements?.sameGuardV5Top1Agreements === 320 && regressionPolicy.requirements.sourceOrHardConstraintViolations === 0,
    "exact preregistered conservative query regression policy required");
  const serving = await jsonArtifact(profile.servingManifest);
  need(serving.version === "persona-choice-adapter-serving.v1" && serving.actualSuffixTokens === 16 && serving.trainedUpdates === 3584 &&
    serving.base927Unchanged === true && serving.postfitNativeControlsExact === true && serving.inferenceTargetFieldsRequired === false &&
    serving.cachedScoringMode === "plain" && serving.dynamicMode === "gradient-public-XHY-last16" && serving.modelRevision === FIXED.revision &&
    serving.catalogSha256 === FIXED.catalog, "completed native adapter and fixed inference proofs required");
  const directory = path.dirname(profile.servingManifest.file); const adapter = serving.adapter;
  need(adapter?.rank === 4 && adapter.scale === 20 && adapter.target === "language_model.model.layers.31.mlp.down_proj", "rank4/scale20 final31 adapter required");
  await colocated(directory, adapter.file, adapter.sha256);
  const parity = JSON.parse(await colocated(directory, serving.inferenceParity?.file, serving.inferenceParity?.sha256));
  need(parity.version === "persona-choice-inference-parity.v1" && parity.adapterSha256 === adapter.sha256 &&
    parity.cachedScoringMode === "plain" && parity.dynamicMode === "gradient-public-XHY-last16" &&
    ["cachedScoreMatchesTrainingFeaturesExact", "dynamicMatchesCachedFeatureScoresExact", "zeroAndNonzeroControlPassed", "learnedPostfitPassed"].every(k => parity[k] === true) &&
    exact(parity.zeroNonzeroProofSha256) && Array.isArray(parity.postfitProofSha256) && parity.postfitProofSha256.length === 4 &&
    new Set(parity.postfitProofSha256).size === 4 && parity.postfitProofSha256.every(exact), "complete adapter-bound learned inference certificate required");
  need(parity.helperSourceSha256?.score === profile.sourceSha256["experiments/ranker/pretrained/store_persona_choice_scoring_v1.py"] &&
    parity.helperSourceSha256?.capture === profile.sourceSha256["experiments/ranker/pretrained/store_persona_choice_capture_v1.py"] &&
    serving.scoreSourceSha256 === parity.helperSourceSha256.score && serving.formatterSha256 === profile.sourceSha256["experiments/ranker/pretrained/store_qwen_lexical_choice_inputs_v2.py"], "certificate/helper/formatter source pins differ");
  const index = JSON.parse(await colocated(directory, serving.publicCache?.file, serving.publicCache?.sha256));
  const indexDirectory = path.dirname(path.join(directory, serving.publicCache.file));
  const indexFields = ["version", "entries", "teacherOrTargetsIncluded", "key", "modelRevision", "formatterSha256", "baseModelFiles", "scoreSourceSha256", "featureSourceSha256"].sort();
  const entryFields = ["archive", "archiveSha256", "dtypes", "routeTokenIds", "promptTokenIdsSha256", "promptTokens", "prefixPath", "nativeInputSuffixTokens"].sort();
  need(JSON.stringify(Object.keys(index).sort()) === JSON.stringify(indexFields) && index.version === "persona-choice-public-feature-cache.v1" && index.teacherOrTargetsIncluded === false && Object.keys(index.entries || {}).length === 20 &&
    index.formatterSha256 === serving.formatterSha256 && index.scoreSourceSha256 === serving.scoreSourceSha256 && index.modelRevision === serving.modelRevision &&
    index.featureSourceSha256 === profile.sourceSha256["experiments/ranker/pretrained/store_persona_choice_features_v2.py"], "exact target-free20 public feature index required");
  need(Array.isArray(serving.routeTokenIds) && serving.routeTokenIds.length === 16 && new Set(serving.routeTokenIds).size === 16 &&
    Array.isArray(serving.productIds) && serving.productIds.length === 16 && new Set(serving.productIds).size === 16, "canonical16 route/product identities required");
  for (const [key, entry] of Object.entries(index.entries)) {
    need(exact(key) && JSON.stringify(Object.keys(entry).sort()) === JSON.stringify(entryFields) && entry.promptTokenIdsSha256 === key &&
      entry.archive === key + ".npz" && entry.nativeInputSuffixTokens === 16 && Number.isInteger(entry.promptTokens) && entry.promptTokens >= 738 && entry.promptTokens <= 2048 &&
      JSON.stringify(entry.routeTokenIds) === JSON.stringify(serving.routeTokenIds), "public cache must contain pure co-located X/H/Y metadata keyed by prompt SHA");
    await colocated(indexDirectory, entry.archive, entry.archiveSha256);
  }
  for (const key of ["queryIntentHead", "colorHead", "colorCache"]) await artifact(profile[key]);
  const rank = await jsonArtifact(profile.rankExport);
  await colocated(path.dirname(profile.rankExport.file), rank.head?.file, rank.head?.sha256);
  const copy = await jsonArtifact(profile.copyManifest);
  need(copy.all48BindingsSupervised === true && copy.runtimeTeacherLookup === false && exact(copy.headSha256), "reviewed source-bound public48 copy manifest required");
  await jsonArtifact(profile.integrationManifest);
  const evidence = {};
  for (const key of [...EVIDENCE, "acceptance"]) evidence[key] = await jsonArtifact(profile.evidence[key]);
  const offline = evidence.offlineAudit, admission = evidence.admissionBeforeGold, query = evidence.queryMechanicalAudit, replay = evidence.queryCombinationReplay;
  need(offline.version === "persona-choice-combined-independent-audit.v1" && offline.status === "completed-offline-adjudication" &&
    offline.offlineCoreAndReplacementPass === true && offline.overallPromotionPass === false && offline.privateQualityJSONParsed === true &&
    offline.sourceSha256 === profile.sourceSha256["experiments/ranker/pretrained/audit_persona_choice_combined_v1.py"] && offline.gateSha256 === FIXED.qualityGate,
    "actual independent offline quality pass required");
  need(admission.privateQualityJSONParsed === false && admission.adapterCheckpointSha256 === adapter.sha256 &&
    admission.packetManifestSha256 === FIXED.publicPacket && exact(admission.adapterFreezeSha256) && exact(admission.combinedFreezeSha256),
    "before-gold admission must bind selected adapter, frozen predictions and exact independent public packet");
  need(query.status === "passed-independent-TRAIN-replay" && query.headSha256 === profile.queryIntentHead.sha256 && query.sourceManifestSha256 === FIXED.queryTrain &&
    query.auditSourceSha256 === profile.sourceSha256["experiments/ranker/pretrained/store_query_intent_audit_v3.py"] && query.privateQualityLabelsRead === false &&
    query.replay?.observedReplayOptimizerSteps === 250 && query.replay.everyFinalParameterBitwiseMatches === true && query.twoReloadsExactLogitParity === true &&
    query.reloadSavedParity?.rows === 1272 && query.reloadSavedParity.exactSavedProbabilityParity === true, "actual query TRAIN replay required");
  need(replay.version === "persona-combined-query-replay-audit.v1" && replay.status === "passed" && replay.adapterCheckpointSha256 === adapter.sha256 && replay.gateCheckpointSha256 === profile.queryIntentHead.sha256 &&
    replay.v5HeadSha256 === rank.head.sha256 && replay.rows === 344 && replay.uniqueQueryVectorsReplayed === 292 && replay.nativeCalls === 0 && replay.optimizerUpdates === 0 &&
    replay.combinedFreezeSha256 === admission.combinedFreezeSha256 && replay.overallPromotionPass === false && replay.servingPending === true &&
    replay.privateQualityLabelsRead === false && Object.values(replay.sourceSha256 || {}).includes(profile.sourceSha256["experiments/ranker/pretrained/audit_persona_combined_query_replay_v1.py"]) &&
    replay.exactQueryProbabilityAndV5Parity === true && replay.actualFloat32BlendAndGuardsExact === true && replay.modelRowsChecked === 1376 &&
    replay.mechanicalQueryAuditSha256 === profile.evidence.queryMechanicalAudit.sha256, "actual same-adapter/query combined replay required");
  const http = evidence.httpValidation;
  need(http.version === "persona-adapter-http-semantic-probe.v2" && http.status === "completed-fixed320-validation" && http.fixedDenominator === 320 && http.summary?.completed === 320 &&
    http.summary.failedOrTimeout === 0 && http.summary.unattempted === 0 && http.summary.protocolPassed === 320 && http.summary.uniqueActualRequestIds === 320,
    "actual complete320 HTTP validation required; failures and unattempted requests cannot be accepted");
  const httpPlan = JSON.parse(await colocated(path.dirname(profile.evidence.httpValidation.file), "plan.json", FIXED.httpPlan));
  need(httpPlan.fixedDenominator === 320 && Array.isArray(httpPlan.cases) && httpPlan.cases.length === 320 &&
    http.summary.sameGuardV5Comparisons === 320 && http.summary.sameGuardV5Top1Agreements === 320 &&
    http.servingSelection?.offlineAuditSha256 === profile.evidence.offlineAudit.sha256 &&
    http.servingSelection.queryMechanicalAuditSha256 === profile.evidence.queryMechanicalAudit.sha256 &&
    http.servingSelection.queryCombinationReplaySha256 === profile.evidence.queryCombinationReplay.sha256 &&
    Array.isArray(http.cases) && http.cases.length === 320 && http.cases.every((row, i) => row.probeCaseId === httpPlan.cases[i].probeCaseId &&
      row.status === "completed" && row.analysis?.protocolPassed === true && row.analysis.qualityPassed === true && row.analysis.sameV5Top1UnderSameGuards === true &&
      row.analysis.learnedGateCheckpointSha256 === profile.queryIntentHead.sha256 && Array.isArray(row.analysis.errors) && row.analysis.errors.length === 0 &&
      Array.isArray(row.analysis.qualityFailures) && row.analysis.qualityFailures.length === 0 && typeof row.analysis.requestId === "string" && row.analysis.requestId.length > 0) &&
    new Set(http.cases.map(row => row.analysis.requestId)).size === 320,
    "specific-query regression policy requires all320 same-guard v5 top choices/abstentions, exact roster/identities and zero factual/guard failures; no exclusions");
  const browser = evidence.browserValidation;
  need(browser.version === "store-persona-browser-probe.v2" && !browser.error && browser.sourcesUnchanged === true && browser.startupUnchanged === true &&
    browser.summary?.protocolPassed === 13 && browser.summary.loadedSwitchFirst3?.valid === 10 && browser.summary.loadedSwitchFull6?.valid === 10 &&
    Array.isArray(browser.functionalArms) && browser.functionalArms.length === 2 && browser.functionalArms.every(arm => arm.passed === true),
    "actual13 browser requests, ten loaded switches and both functional arms must pass");
  const acceptance = evidence.acceptance;
  need(acceptance.version === "persona-adapter-live-acceptance.v1" && acceptance.status === "accepted" && acceptance.overallPromotionPass === true &&
    acceptance.browserPassed === true && acceptance.freshCopyFactsPassed === true && acceptance.shutdownIntegrityPassed === true &&
    acceptance.adapterSha256 === adapter.sha256 && acceptance.servingManifestSha256 === profile.servingManifest.sha256 &&
    acceptance.gateCheckpointSha256 === profile.queryIntentHead.sha256 && acceptance.integrationManifestSha256 === profile.integrationManifest.sha256 &&
    acceptance.sourceByteEquivalenceVerified === true && Object.keys(acceptance.sourceSha256 || {}).length === Object.keys(profile.sourceSha256).length &&
    acceptance.specificQueryRegressionPassed === true && acceptance.specificQueryRegressionPolicySha256 === FIXED.queryRegressionPolicy &&
    acceptance.specificQueryRegressionPlanSha256 === FIXED.httpPlan && acceptance.v5CheckpointSha256 === rank.head.sha256 &&
    Object.entries(profile.sourceSha256).every(([key, value]) => acceptance.sourceSha256[key] === value) &&
    ["servingManifest", "queryIntentHead", "colorHead", "colorCache", "rankExport", "copyManifest", "integrationManifest"].every(key => acceptance.artifactSha256?.[key] === profile[key].sha256) &&
    EVIDENCE.every(key => acceptance.evidenceSha256?.[key] === profile.evidence[key].sha256), "final root-owned acceptance must bind all actual offline/live/browser/shutdown evidence");
  need(["frozen927HashesUnchanged", "copyCheckpointUnchanged", "v5ServingExportUnchanged", "v5HeadCheckpointUnchanged",
    "rankAdapterRestoredToCopy", "rankAdapterAndPublicCacheUnchanged", "learnedVariantAndQueryIntentAssetsUnchanged", "adapterEngineSourcesUnchanged"]
    .every(key => evidence.shutdown[key] === true), "actual native927/copy/v5/rank/cache/source shutdown integrity proof required");
  return { ...profile, personaHead: { file: path.join(directory, adapter.file), sha256: adapter.sha256 }, assets,
    acceptanceSha256: profile.evidence.acceptance.sha256, serving, parity,
    healthExpected: { v5HeadSha256: rank.head.sha256, copyHeadSha256: copy.headSha256 } };
}

export function assertAdapterHealth(value, profile) {
  const startup = value?.startup, persona = startup?.persona;
  need(value?.ready === true && value.status === "ready" && persona?.kind === "persona-adapter-engine.v2" && persona.checkpointSha256 === profile.personaHead.sha256 &&
    persona.servingManifestSha256 === profile.servingManifest.sha256 && persona.queryIntent?.checkpointSha256 === profile.queryIntentHead.sha256 &&
    persona.queryIntent.sourceManifestSha256 === FIXED.queryTrain && persona.queryIntent.phraseOrPersonRules === false &&
    persona.queryIntent.sourceSha256 === profile.sourceSha256["experiments/ranker/pretrained/store_query_intent_gate_v3.py"] &&
    persona.color?.checkpointSha256 === FIXED.colorHead && persona.color.failedOldProductOrderingUsed === false &&
    persona.publicDecisionFeatureCacheEntries === 20 && persona.privateTargetOrTeacherInputs === false &&
    startup.rankCheckpointSha256 === profile.healthExpected.v5HeadSha256 && startup.copyCheckpointSha256 === FIXED.nativeCopy &&
    startup.copyCandidate?.checkpointSha256 === profile.healthExpected.copyHeadSha256 &&
    startup.rankSessionProvenance?.rankHead?.servingExportSha256 === profile.rankExport.sha256 &&
    startup.sourceTensorsHashed === 927 && startup.memory?.processBudgetBytes === 7 * 2 ** 30 &&
    startup.imageDeliveryManifestSha256 === profile.integrationManifest.sha256,
    "live ready engine adapter/query/color identity differs from accepted profile");
}

export function adapterArguments({ assetRoot, profile, candidate }) {
  return ["-m", ADAPTER_MODULE, "--asset-root", assetRoot,
    "--serving-manifest", profile.servingManifest.file, "--serving-manifest-sha256", profile.servingManifest.sha256,
    "--catalog-feature-root", profile.catalogFeatureRoot, "--color-head", profile.colorHead.file, "--color-cache", profile.colorCache.file,
    "--query-intent-head", profile.queryIntentHead.file, "--query-intent-head-sha256", profile.queryIntentHead.sha256,
    "--query-intent-source-manifest-sha256", profile.queryIntentSourceManifestSha256,
    "--candidate-run", candidate, "--integration-manifest", profile.integrationManifest.file,
    "--rank-head-export", profile.rankExport.file, "--rank-head-export-sha256", profile.rankExport.sha256];
}
