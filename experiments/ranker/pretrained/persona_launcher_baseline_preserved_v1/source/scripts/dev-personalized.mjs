import { spawn, spawnSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createWriteStream } from "node:fs";
import { access, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const ROOT = fileURLToPath(new URL("../", import.meta.url));
export const PORTS = Object.freeze({ storefront: 4313, native: 8767, checkout: 4331 });
export const NATIVE_MODULE = "experiments.ranker.pretrained.store_persona_engine_v1";
export const MANIFEST = "experiments/ranker/pretrained/store_demo_integration_personalized_v1/manifest.json";
export const PROFILE = "experiments/ranker/pretrained/store_persona_serving_profile_v1.json";
export const ORIGINAL_CATALOG_RUN = "data/ranker/pretrained/actual-store-function-v3-attempt-1";
const UNPROMOTED_WEAK_HEAD = "12629e50562305ffe1f5e7497f02e649f10e0d14d149e5f46115bd6b5561310e";
const RANK_EXPORT_SHA = "17a0a15a2da6e2fc90e651e1f5bb2c237d61241795ec0cca548a4f3ce319d703";
const allowedEnvironment = ["PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "SHELL", "TERM"];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const shutdownDelay = ms => new Promise(resolve => setTimeout(resolve, ms).unref());
const digest = value => createHash("sha256").update(value).digest("hex");

export async function assertNoFrontendDotenv(root, inspect = stat) {
  for (const name of [".env", ".env.local", ".env.development", ".env.development.local"]) {
    const filename = path.join(root, "apps/storefront", name);
    const exists = await inspect(filename).then(() => true, error => {
      if (error.code === "ENOENT") return false;
      throw error;
    });
    if (exists) throw new Error(`Personalized launcher does not load secret environment files; review the existing storefront ${name} before using this isolated entry. Its contents were not read.`);
  }
}

export function validateReviewedProfile(value, root, assetRoot) {
  if (value?.version !== "personalized-local-profile.v1" || value.status !== "accepted") {
    throw new Error("Personalized store is pending its reviewed model quality gate; no accepted serving profile is available and no model was launched.");
  }
  function location(record, key) {
    const base = record?.root === "worktree" ? root : record?.root === "assets" ? assetRoot : null;
    const filename = record?.[key];
    if (!base || typeof filename !== "string" || !filename.startsWith("data/ranker/pretrained/") ||
      path.isAbsolute(filename) || filename.split(/[\\/]/).includes("..")) throw new Error("Reviewed profile requires exact local model artifact paths.");
    return path.resolve(base, filename);
  }
  const head = value.personaHead;
  if (!/^[a-f0-9]{64}$/.test(head?.sha256 || "") || head.sha256 === UNPROMOTED_WEAK_HEAD ||
      typeof value.contextKind !== "string" || !value.contextKind.trim()) {
    throw new Error("Reviewed profile does not select a new accepted persona head; the earlier weak experimental head cannot start this product entry.");
  }
  const headFile = location(head, "file");
  if (!headFile.endsWith(".pt")) throw new Error("Reviewed local persona PyTorch checkpoint required.");
  const personaFeatureRoot = location(value.personaFeatureRoot, "directory");
  if (personaFeatureRoot !== path.resolve(assetRoot, ORIGINAL_CATALOG_RUN)) {
    throw new Error(`personaFeatureRoot must select the original16-product catalog feature run at ${path.resolve(assetRoot, ORIGINAL_CATALOG_RUN)}. It is not the new context-extraction directory or the head's adjacent public-demo-features.pt.`);
  }
  return { contextKind: value.contextKind, personaHead: { file: headFile, sha256: head.sha256 },
    personaFeatureRoot };
}

export function nativeArguments({ assetRoot, profile, manifestFile, rankExport, candidate }) {
  return ["-m", NATIVE_MODULE, "--asset-root", assetRoot, "--persona-head", profile.personaHead.file,
    "--persona-head-sha256", profile.personaHead.sha256, "--persona-feature-root", profile.personaFeatureRoot,
    "--integration-manifest", manifestFile, "--rank-head-export", rankExport,
    "--rank-head-export-sha256", RANK_EXPORT_SHA, "--candidate-run", candidate, "--port", String(PORTS.native)];
}

export async function acceptedProfile(root, assetRoot, read = readFile) {
  const filename = path.join(root, PROFILE);
  const bytes = await read(filename).catch(error => {
    if (error.code === "ENOENT") throw new Error(`Personalized store is pending its reviewed model quality gate: ${filename} is not available. No model was launched.`);
    throw error;
  });
  return { ...validateReviewedProfile(JSON.parse(bytes), root, assetRoot), file: filename, sha256: digest(bytes) };
}

export function safeEnvironment(input = process.env, extra = {}) {
  return { ...Object.fromEntries(allowedEnvironment.filter(key => input[key] !== undefined).map(key => [key, input[key]])),
    HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", HF_DATASETS_OFFLINE: "1",
    PYTHONDONTWRITEBYTECODE: "1", PYTHONUNBUFFERED: "1", OMP_NUM_THREADS: "1", ...extra };
}

export async function portIsFree(port) {
  const server = net.createServer();
  return new Promise((resolve, reject) => {
    server.once("error", error => error.code === "EADDRINUSE" ? resolve(false) : reject(error));
    server.listen({ host: "127.0.0.1", port, exclusive: true }, () => server.close(() => resolve(true)));
  });
}

export async function localJson(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(1_500), redirect: "error" });
  const text = await response.text();
  if (text.length > 1_000_000) throw new Error("Unexpectedly large local readiness response");
  return { status: response.status, value: JSON.parse(text) };
}

export async function checkoutIsDeclared(port, get = localJson) {
  try {
    // Read-only nonexistent receipt probe; creates no order and loads no .env.
    const response = await get(`http://127.0.0.1:${port}/api/demo-checkout/${randomUUID()}`);
    return response.status === 404 && response.value?.error === "local demo receipt unavailable";
  } catch { return false; }
}

export async function chooseCheckout({ free = portIsFree, declared = checkoutIsDeclared } = {}) {
  const isolatedFree = await free(PORTS.checkout);
  if (!isolatedFree && await declared(PORTS.checkout)) {
    return { port: PORTS.checkout, reused: true, origin: `http://127.0.0.1:${PORTS.checkout}` };
  }
  //4311 may be an experiment-only receiver. Leave it alone if it has no checkout.
  if (!await free(4311) && await declared(4311)) {
    return { port: 4311, reused: true, origin: "http://127.0.0.1:4311" };
  }
  if (!isolatedFree) throw new Error(`Port ${PORTS.checkout} is occupied by an unverified checkout service and 4311 offers no verified checkout. No process was stopped.`);
  return { port: PORTS.checkout, reused: false, origin: `http://127.0.0.1:${PORTS.checkout}` };
}

export async function assertFreePorts(free = portIsFree) {
  // Native ownership is also checked through the shared flock, independent of PID.
  for (const port of [8766, PORTS.native, PORTS.storefront]) {
    if (!await free(port)) throw new Error(`Port ${port} is busy. Stop its owning project normally before starting the personalized store. No process was killed.`);
  }
}

export function assertNativeLockFree(python, filename, run = spawnSync) {
  const code = "import fcntl,pathlib,sys; p=pathlib.Path(sys.argv[1]); f=p.open('rb') if p.exists() else None; fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB) if f else None";
  const result = run(python, ["-c", code, filename], { env: safeEnvironment(), encoding: "utf8", timeout: 5_000, stdio: ["ignore", "pipe", "pipe"] });
  if (result.error || result.status !== 0) throw new Error("The shared native-owner lock is busy or could not be verified; no model was launched.");
}

async function requiredFile(filename, expectedSha, executable = false) {
  const info = await stat(filename).catch(() => { throw new Error(`Required local file is missing: ${filename}. No download or installation was attempted.`); });
  if (!info.isFile() || info.size === 0) throw new Error(`Required local file is empty or invalid: ${filename}`);
  if (executable) await access(filename, 1);
  const sha256 = expectedSha ? digest(await readFile(filename)) : null;
  if (expectedSha && sha256 !== expectedSha) throw new Error(`Reviewed local checkpoint checksum differs: ${filename}`);
  return { file: filename, bytes: info.size, ...(sha256 ? { sha256 } : {}) };
}

export async function buildPlan({ root = ROOT, assetRoot, session, manifest = MANIFEST } = {}) {
  root = path.resolve(root);
  await assertNoFrontendDotenv(root);
  assetRoot = path.resolve(assetRoot || (await stat(path.join(root, ".tools/pretrained-venv/bin/python")).then(() => root).catch(() => path.join(root, "../dubhacks-2026"))));
  const profile = await acceptedProfile(root, assetRoot);
  const python = path.join(assetRoot, ".tools/pretrained-venv/bin/python"); // Keep the lexical venv interpreter path.
  const nativeFile = path.join(root, ...NATIVE_MODULE.split(".")) + ".py";
  const persona = profile.personaHead.file;
  const candidate = path.join(assetRoot, "data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2");
  const rank = path.join(assetRoot, ORIGINAL_CATALOG_RUN);
  const rankExport = path.join(assetRoot, "data/ranker/pretrained/actual-store-function-v5-attempt-1/serving-head.json");
  const copy = path.join(assetRoot, "data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1");
  const features = path.join(assetRoot, "data/ranker/pretrained/store-parallel-text-v1-attempt-1/features.pt");
  const publicContract = path.join(candidate, "public-runtime-features.json");
  const manifestFile = path.resolve(root, manifest);
  const manifestBytes = await readFile(manifestFile).catch(() => { throw new Error(`Reviewed personalized integration manifest is not ready: ${manifestFile}. No older manifest was substituted.`); });
  JSON.parse(manifestBytes);
  const assets = await Promise.all([
    requiredFile(python, null, true), requiredFile(nativeFile), requiredFile(persona, profile.personaHead.sha256),
    requiredFile(path.join(path.dirname(persona), "public-demo-features.pt")),
    requiredFile(rankExport, RANK_EXPORT_SHA),
    requiredFile(path.join(candidate, "head.pt"), "074169e31bf5f6b99cc5bc35c9b2e64d82396d5225b3a67cc2cf5f613fcc4889"),
    requiredFile(features, "2624a947f715e489431abf652a16e6872677fd0d4b824a76df694e0f8fae6ee8"),
    requiredFile(publicContract, "bbd2400e0c6e2907c783f01b7c157714cae00ec5b76fc619790a939a0f9d07f3"),
    requiredFile(path.join(copy, "adapter.safetensors"), "56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8"),
    requiredFile(path.join(rank, "report.json")), requiredFile(path.join(rank, "packet.json")),
    requiredFile(path.join(rank, "RAW_FEATURES_FROZEN.json")),
    requiredFile(path.join(root, "apps/storefront/node_modules/.bin/vp"), null, true),
  ]);
  const model = path.join(assetRoot, ".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53");
  for (const [name, bytes] of [["model-00001-of-00002.safetensors", 5349771222], ["model-00002-of-00002.safetensors", 600449850]]) {
    const actual = await requiredFile(path.join(model, name));
    if (actual.bytes !== bytes) throw new Error(`Local model shard has unexpected size: ${name}`);
    assets.push(actual);
  }
  for (const name of ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]) assets.push(await requiredFile(path.join(model, name)));
  return { version: "personalized-local-launch.v1", root, assetRoot, python, nativeFile,
    lock: path.join(assetRoot, "data/ranker/pretrained/native-owner.lock"),
    session: session || path.join(root, "data/personalized-dev", `${new Date().toISOString().replaceAll(":", "-")}-${randomUUID()}`),
    manifest: { file: manifestFile, sha256: digest(manifestBytes) }, profile, assets,
    nativeArgs: nativeArguments({ assetRoot, profile, manifestFile, rankExport, candidate }),
    modelVerification: "Shard presence/size checked here; original full model/source/checkpoint integrity remains mandatory at native startup.",
    noDownloads: true, noEnvironmentFileLoaded: true };
}

export async function preflight(options = {}, dependencies = {}) {
  const free = dependencies.free || portIsFree;
  await assertFreePorts(free); // Fail before hashing assets or creating output directories.
  const plan = await (dependencies.buildPlan || buildPlan)(options);
  (dependencies.lock || assertNativeLockFree)(plan.python, plan.lock);
  const checkout = await chooseCheckout({ free, declared: dependencies.declared || checkoutIsDeclared });
  let go;
  if (!checkout.reused) {
    go = path.join(plan.assetRoot, ".tools/go/bin/go");
    await (dependencies.requiredFile || requiredFile)(go, null, true);
    await (dependencies.requiredFile || requiredFile)(path.join(plan.root, "apps/api/cmd/demo-checkout-server/main.go"));
  }
  return { ...plan, checkout, ...(go ? { go } : {}) };
}

export class OwnedChildren {
  constructor({ spawnChild = spawn, signal = (child, name) => process.platform === "win32" ? child.kill(name) : process.kill(-child.pid, name), delay = shutdownDelay } = {}) {
    this.spawnChild = spawnChild; this.signal = signal; this.delay = delay; this.children = []; this.stopping = false;
  }
  launch(name, command, args, options) {
    if (this.stopping) throw new Error("Launcher is stopping");
    const child = this.spawnChild(command, args, { ...options, detached: process.platform !== "win32" });
    const entry = { name, child, exited: false, closed: false, code: null, signal: null };
    entry.done = new Promise(resolve => {
      child.once("exit", (code, signal) => { Object.assign(entry, { exited: true, code, signal }); });
      // Wait for stdio to close before closing our evidence logs.
      child.once("close", (code, signal) => { Object.assign(entry, { exited: true, closed: true, code, signal }); resolve(entry); });
      child.once("error", error => { Object.assign(entry, { exited: true, code: 1, error: error.message }); resolve(entry); });
    });
    this.children.push(entry);
    return entry;
  }
  assertAlive() {
    const failed = this.children.find(entry => entry.exited);
    if (failed) throw new Error(`${failed.name} stopped before the personalized store was ready. Inspect its session log.`);
  }
  async stop({ nativeGraceMs = 180_000, otherGraceMs = 5_000 } = {}) {
    if (this.stopResult) return this.stopResult;
    this.stopping = true;
    this.stopResult = Promise.all(this.children.map(async entry => {
      if (!entry.exited) {
        try { this.signal(entry.child, "SIGTERM"); } catch { /* Already stopped. */ }
        const finished = await Promise.race([entry.done.then(() => true), this.delay(entry.name === "native" ? nativeGraceMs : otherGraceMs).then(() => false)]);
        if (!finished && !entry.exited) {
          entry.forced = true;
          try { this.signal(entry.child, "SIGKILL"); } catch { /* Already stopped. */ }
          await Promise.race([entry.done, this.delay(otherGraceMs)]);
        }
      } else if (!entry.closed && !entry.error) {
        await Promise.race([entry.done, this.delay(otherGraceMs)]);
      }
      return { name: entry.name, pid: entry.child.pid, code: entry.code, signal: entry.signal,
        forced: !!entry.forced, exited: entry.exited, ...(entry.error ? { error: entry.error } : {}) };
    }));
    return this.stopResult;
  }
}

export async function waitUntil(ready, { children, timeoutMs = 180_000, delay = sleep, now = Date.now } = {}) {
  const deadline = now() + timeoutMs;
  while (now() < deadline) {
    if (children?.stopping) throw new Error("Personalized store startup cancelled");
    children?.assertAlive();
    if (await ready()) { children?.assertAlive(); return; }
    await delay(200);
  }
  throw new Error("Personalized store readiness timed out; inspect the preserved session logs.");
}

export async function run(plan, children = new OwnedChildren()) {
  await mkdir(path.dirname(plan.session), { recursive: true, mode: 0o700 });
  await mkdir(plan.session, { mode: 0o700 }); // Never reuse or overwrite a previous lifetime.
  const record = (name, value) => writeFile(path.join(plan.session, name), JSON.stringify(value, null, 2) + "\n", { flag: "wx", mode: 0o600 });
  await record("launch.json", plan);
  const logs = [];
  function start(name, command, args, cwd, extra) {
    const log = createWriteStream(path.join(plan.session, `${name}.log`), { flags: "wx", mode: 0o600 }); logs.push(log);
    log.once("error", cause => { error ||= new Error(`Session log failed: ${cause.message}`); void stop(); });
    const entry = children.launch(name, command, args, { cwd, env: safeEnvironment(process.env, extra), stdio: ["ignore", "pipe", "pipe"] });
    entry.child.stdout.pipe(log, { end: false }); entry.child.stderr.pipe(log, { end: false });
    return entry;
  }
  let shuttingDown;
  const stop = () => shuttingDown ||= children.stop();
  process.once("SIGINT", stop); process.once("SIGTERM", stop);
  let error;
  try {
    // Recheck immediately before spawn; the native worker also owns the real flock for its whole lifetime.
    assertNativeLockFree(plan.python, plan.lock);
    start("native", plan.python, [...plan.nativeArgs, "--output", path.join(plan.session, "native")], plan.root, {});
    await waitUntil(async () => {
      try {
        const response = await localJson(`http://127.0.0.1:${PORTS.native}/health`);
        if (response.value?.status === "error" || response.value?.error) throw new Error("Native model startup failed; inspect native.log.");
        if (response.value?.ready === true && response.value?.startup?.persona?.checkpointSha256 !== plan.profile.personaHead.sha256) throw new Error("Native model startup failed: unexpected persona checkpoint.");
        return response.status === 200 && response.value?.ready === true;
      } catch (cause) { if (cause.message?.includes("Native model startup failed")) throw cause; return false; }
    }, { children });
    if (!plan.checkout.reused) {
      start("checkout", plan.go, ["run", "./cmd/demo-checkout-server"], path.join(plan.root, "apps/api"), {
        PORT: String(PORTS.checkout), DEMO_CHECKOUT_DIR: path.join(plan.session, "checkout-receipts"),
        GOCACHE: path.join(plan.assetRoot, ".cache/go-build"), GOMODCACHE: path.join(plan.assetRoot, ".cache/go-mod"),
        GOPROXY: "off", GOSUMDB: "off", GOTOOLCHAIN: "local" });
      await waitUntil(() => checkoutIsDeclared(PORTS.checkout), { children, timeoutMs: 60_000 });
    }
    start("storefront", path.join(plan.root, "apps/storefront/node_modules/.bin/vp"), ["dev"], path.join(plan.root, "apps/storefront"), {
      PORT: String(PORTS.storefront), VITE_PERSONALIZED_STORE: "true", DEV_API_ORIGIN: plan.checkout.origin });
    await waitUntil(async () => !await portIsFree(PORTS.storefront), { children, timeoutMs: 30_000 });
    await record("ready.json", { url: `http://127.0.0.1:${PORTS.storefront}`, native: PORTS.native, checkout: plan.checkout, at: new Date().toISOString() });
    console.log(`Personalized store ready — http://127.0.0.1:${PORTS.storefront}\nSession: ${plan.session}\nPress Ctrl+C to stop this session.`);
    await Promise.race(children.children.map(entry => entry.done));
    if (!children.stopping) throw new Error("A personalized store child stopped; all owned children will shut down.");
  } catch (cause) { error = cause; }
  finally {
    const stopped = await stop();
    if (stopped.some(entry => entry.forced || !entry.exited)) error ||= new Error("Owned child required forced shutdown; native integrity is unconfirmed. Inspect session receipts.");
    process.removeListener("SIGINT", stop); process.removeListener("SIGTERM", stop);
    for (const log of logs) log.end();
    await record("launcher-shutdown.json", { at: new Date().toISOString(), children: stopped,
      nativeIntegrityClaim: "See native/shutdown.json; launcher exit alone is not model-integrity proof.",
      ...(error ? { error: error.message } : {}) });
  }
  if (error) throw error;
}

export async function main(argv = process.argv.slice(2)) {
  const options = {}; let execute = false;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--run") execute = true;
    else if (argv[i] === "--asset-root" && argv[i + 1]) options.assetRoot = argv[++i];
    else if (argv[i] === "--help") { console.log("node scripts/dev-personalized.mjs [--run] [--asset-root LOCAL_EXISTING_REPO]\nWithout --run: preflight only. No download, install, .env load or service takeover."); return; }
    else throw new Error(`Unknown launcher argument: ${argv[i]}`);
  }
  const plan = await preflight(options);
  if (!execute) { console.log(JSON.stringify({ ...plan, execution: "preflight-only; no session directory or services created" }, null, 2)); return; }
  process.umask(0o077);
  await run(plan);
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
