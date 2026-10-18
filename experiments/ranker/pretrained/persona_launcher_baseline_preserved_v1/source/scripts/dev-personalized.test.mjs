import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { safeEnvironment, assertFreePorts, chooseCheckout, checkoutIsDeclared, assertNativeLockFree,
  preflight, OwnedChildren, waitUntil, PORTS, NATIVE_MODULE, assertNoFrontendDotenv,
  validateReviewedProfile, acceptedProfile, PROFILE, ORIGINAL_CATALOG_RUN, nativeArguments } from "./dev-personalized.mjs";

function reviewedProfile() {
  return { version: "personalized-local-profile.v1", status: "accepted", contextKind: "actual-reviewed-native-context",
    personaHead: { root: "worktree", file: "data/ranker/pretrained/reviewed-context/head.pt", sha256: "a".repeat(64) },
    personaFeatureRoot: { root: "assets", directory: ORIGINAL_CATALOG_RUN } };
}

test("default entry requires an explicit accepted quality-gated model profile, with no old weak-head fallback", async () => {
  for (const status of [undefined, "pending", "rejected", "experimental"]) {
    assert.throws(() => validateReviewedProfile({ ...reviewedProfile(), status }, "/worktree", "/main"), /quality gate/);
  }
  const old = reviewedProfile(); old.personaHead.sha256 = "12629e50562305ffe1f5e7497f02e649f10e0d14d149e5f46115bd6b5561310e";
  assert.throws(() => validateReviewedProfile(old, "/worktree", "/main"), /weak experimental head/);
  await assert.rejects(() => acceptedProfile("/worktree", "/main", async () => { throw Object.assign(new Error("not yet frozen"), { code: "ENOENT" }); }), /pending.*quality gate/);
  const profile = await acceptedProfile("/worktree", "/main", async filename => {
    assert.equal(filename, `/worktree/${PROFILE}`); return Buffer.from(JSON.stringify(reviewedProfile()));
  });
  assert.equal(profile.personaHead.file, "/worktree/data/ranker/pretrained/reviewed-context/head.pt");
  assert.equal(profile.personaFeatureRoot, `/main/${ORIGINAL_CATALOG_RUN}`);
  assert.match(profile.sha256, /^[a-f0-9]{64}$/);
});

test("personaFeatureRoot is the original16 catalog run, not the context extraction or adjacent demo cache", () => {
  for (const record of [
    { root: "assets", directory: "data/ranker/pretrained/persona-qwen-context-432-extraction" },
    { root: "assets", directory: "data/ranker/pretrained/reviewed-context/public-demo-features.pt" },
    { root: "worktree", directory: ORIGINAL_CATALOG_RUN },
  ]) {
    assert.throws(() => validateReviewedProfile({ ...reviewedProfile(), personaFeatureRoot: record }, "/worktree", "/main"), /original16-product catalog feature run/);
  }
  const profile = validateReviewedProfile(reviewedProfile(), "/worktree", "/main");
  const args = nativeArguments({ assetRoot: "/main", profile, manifestFile: "/worktree/reviewed-manifest.json",
    rankExport: "/main/rank-export.json", candidate: "/main/copy-run" });
  assert.equal(args[args.indexOf("--persona-feature-root") + 1], `/main/${ORIGINAL_CATALOG_RUN}`);
  assert.equal(args[args.indexOf("--persona-head") + 1], "/worktree/data/ranker/pretrained/reviewed-context/head.pt");
  assert.equal(args.includes("public-demo-features.pt"), false);
  assert.equal(profile.personaFeatureRoot, `/main/${ORIGINAL_CATALOG_RUN}`);
});

test("reviewed profile selects only local model artifacts and exact checkpoint digests", () => {
  for (const mutate of [p => { p.personaHead.file = "../.env"; }, p => { p.personaHead.root = "cloud"; },
    p => { p.personaHead.sha256 = "bad"; }, p => { p.personaFeatureRoot.directory = "data/ranker/pretrained/../secret"; },
    p => { p.personaHead.file = "data/ranker/pretrained/head.txt"; }]) {
    const candidate = reviewedProfile(); mutate(candidate);
    assert.throws(() => validateReviewedProfile(candidate, "/worktree", "/main"));
  }
});

test("child environment excludes inherited secrets, Vite overrides and Portless settings", () => {
  const env = safeEnvironment({ PATH: "/local/bin", HOME: "/home/example", TIGER_DATABASE_URL: "secret",
    DATABASE_URL: "secret", OPENAI_API_KEY: "secret", VITE_PERSONALIZED_STORE: "false", PORT: "4200", PORTLESS_LAN: "1" },
  { PORT: "4313", VITE_PERSONALIZED_STORE: "true" });
  assert.deepEqual(Object.keys(env).sort(), ["PATH", "HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "OMP_NUM_THREADS", "PORT", "VITE_PERSONALIZED_STORE"].sort());
  assert.equal(env.PORT, "4313"); assert.equal(env.HF_HUB_OFFLINE, "1");
});

test("frontend dotenv files are inspected only for existence and never loaded", async () => {
  const checked = [];
  await assertNoFrontendDotenv("/worktree", async filename => { checked.push(filename); throw Object.assign(new Error("missing"), { code: "ENOENT" }); });
  assert.equal(checked.length, 4);
  await assert.rejects(() => assertNoFrontendDotenv("/worktree", async () => ({})), /contents were not read/);
  await assert.rejects(() => assertNoFrontendDotenv("/worktree", async () => { throw Object.assign(new Error("permission denied"), { code: "EACCES" }); }), /permission denied/);
});

test("all fixed native/frontend ports fail closed without touching asset/model loaders", async () => {
  for (const busy of [8766, 8767, 4313]) {
    let built = false;
    await assert.rejects(() => preflight({}, { free: async port => port !== busy,
      buildPlan: async () => { built = true; throw new Error("must not run"); } }), new RegExp(`Port ${busy} is busy`));
    assert.equal(built, false);
  }
  const checked = [];
  await assertFreePorts(async port => { checked.push(port); return true; });
  assert.deepEqual(checked, [8766, 8767, 4313]);
  await assertFreePorts(async port => port !== 4200); //4200 was a process ID, not a reserved service port.
});

test("reuse only the declared read-verified checkout; never stop or commandeer listeners", async () => {
  const checked = [];
  const actual = await chooseCheckout({ free: async port => { checked.push(port); return false; }, declared: async port => port === 4311 });
  assert.deepEqual(actual, { port: 4311, reused: true, origin: "http://127.0.0.1:4311" });
  assert.deepEqual(checked, [4331, 4311]);
  await assert.rejects(() => chooseCheckout({ free: async () => false, declared: async () => false }), /unverified checkout/);
  assert.deepEqual(await chooseCheckout({ free: async () => true }), { port: 4331, reused: false, origin: "http://127.0.0.1:4331" });
  assert.equal(await checkoutIsDeclared(4331, async url => {
    assert.match(url, /^http:\/\/127\.0\.0\.1:4331\/api\/demo-checkout\/[a-f0-9-]+$/);
    return { status: 404, value: { error: "local demo receipt unavailable" } };
  }), true);
  assert.equal(await checkoutIsDeclared(4331, async () => ({ status: 200, value: { ok: true } })), false);
});

test("experiment-only4311 does not block the actual checkout on4331, which is preferred", async () => {
  const portChecks = []; const receiptReads = [];
  const result = await chooseCheckout({ free: async port => { portChecks.push(port); return false; },
    declared: port => checkoutIsDeclared(port, async () => {
      receiptReads.push(port);
      return port === 4331 ? { status: 404, value: { error: "local demo receipt unavailable" } } :
        { status: 404, value: { error: "experiment route not found" } };
    }) });
  assert.deepEqual(result, { port: 4331, reused: true, origin: "http://127.0.0.1:4331" });
  assert.deepEqual(portChecks, [4331]); assert.deepEqual(receiptReads, [4331]);
});

test("unrelated4311 with free4331 permits our standalone checkout without touching4311", async () => {
  const receiptReads = [];
  const result = await chooseCheckout({ free: async port => port === 4331,
    declared: async port => { receiptReads.push(port); return false; } });
  assert.deepEqual(result, { port: 4331, reused: false, origin: "http://127.0.0.1:4331" });
  assert.deepEqual(receiptReads, [4311]);
});

test("unknown occupied4331 can fall back to the verified normal4311 checkout", async () => {
  const receiptReads = [];
  const result = await chooseCheckout({ free: async () => false,
    declared: async port => { receiptReads.push(port); return port === 4311; } });
  assert.deepEqual(result, { port: 4311, reused: true, origin: "http://127.0.0.1:4311" });
  assert.deepEqual(receiptReads, [4331, 4311]);
});

test("both unknown occupied checkout ports fail with targeted4331 error before child/session preparation", async () => {
  const receiptReads = []; let checkoutFilesChecked = 0;
  await assert.rejects(() => preflight({}, {
    free: async port => ![4311, 4331].includes(port),
    buildPlan: async () => ({ root: "/worktree", assetRoot: "/main", python: "/python", lock: "/lock" }),
    lock: () => {}, declared: async port => { receiptReads.push(port); return false; },
    requiredFile: async () => { checkoutFilesChecked++; },
  }), /Port 4331 is occupied.*4311 offers no verified checkout.*No process was stopped/);
  assert.deepEqual(receiptReads, [4331, 4311]); assert.equal(checkoutFilesChecked, 0);
});

test("native lock probe uses lexical venv and pure stdlib; uncertainty or busy lock rejects", () => {
  const python = "/main/.tools/pretrained-venv/bin/python";
  assertNativeLockFree(python, "/main/data/ranker/pretrained/native-owner.lock", (command, args, options) => {
    assert.equal(command, python); assert.equal(args[0], "-c"); assert.match(args[1], /LOCK_NB/);
    assert.doesNotMatch(args[1], /mlx|torch|unlink|kill/); assert.equal(args[2], "/main/data/ranker/pretrained/native-owner.lock");
    assert.equal(options.timeout, 5000); assert.equal(options.env.HF_HUB_OFFLINE, "1"); return { status: 0 };
  });
  for (const result of [{ status: 1 }, { status: null }, { error: new Error("unavailable") }]) {
    assert.throws(() => assertNativeLockFree(python, "/lock", () => result), /shared native-owner lock/);
  }
});

test("preflight busy shared lock stops before checkout and never creates a session", async () => {
  let probedCheckout = false;
  await assert.rejects(() => preflight({}, { free: async () => true,
    buildPlan: async () => ({ python: "/python", lock: "/main/lock" }), lock: () => { throw new Error("busy lock"); },
    declared: async () => { probedCheckout = true; return true; } }), /busy lock/);
  assert.equal(probedCheckout, false);
});

test("offline standalone checkout is planned only when both declared ports are free", async () => {
  const files = [];
  const plan = await preflight({}, { free: async () => true, buildPlan: async () => ({ root: "/worktree", assetRoot: "/main", python: "/python", lock: "/lock" }),
    lock: () => {}, requiredFile: async file => { files.push(file); } });
  assert.equal(plan.checkout.reused, false); assert.equal(plan.go, "/main/.tools/go/bin/go");
  assert.deepEqual(files, ["/main/.tools/go/bin/go", "/worktree/apps/api/cmd/demo-checkout-server/main.go"]);
});

function child(pid) {
  const value = new EventEmitter(); value.pid = pid; return value;
}

test("lifetime signals and waits only owned child groups, with idempotent graceful cleanup", async () => {
  const signals = []; const spawned = [];
  const children = new OwnedChildren({ spawnChild: (command, args, options) => { const actual = child(100 + spawned.length); spawned.push({ actual, command, args, options }); return actual; },
    signal: (actual, signal) => { signals.push([actual.pid, signal]); actual.emit("exit", 0, signal); actual.emit("close", 0, signal); }, delay: () => new Promise(() => {}) });
  children.launch("native", "/venv/python", ["-m", NATIVE_MODULE], { cwd: "/worktree", env: {} });
  children.launch("storefront", "/vp", ["dev"], { cwd: "/worktree/apps/storefront", env: {} });
  assert.equal(spawned[0].options.detached, process.platform !== "win32");
  const first = await children.stop(); const second = await children.stop();
  assert.deepEqual(first, second); assert.deepEqual(signals, [[100, "SIGTERM"], [101, "SIGTERM"]]);
  assert.equal(first.every(row => row.exited && !row.forced), true);
  assert.throws(() => children.launch("extra", "wrong", [], {}), /stopping/);
});

test("own child timeout and spawn failure are explicit; already exited children are not signalled", async () => {
  const native = child(701); const signals = [];
  const children = new OwnedChildren({ spawnChild: () => native, signal: (actual, signal) => {
    signals.push([actual.pid, signal]); if (signal === "SIGKILL") { actual.emit("exit", null, signal); actual.emit("close", null, signal); }
  }, delay: async () => {} });
  children.launch("native", "python", [], {});
  const [result] = await children.stop();
  assert.deepEqual(signals, [[701, "SIGTERM"], [701, "SIGKILL"]]); assert.equal(result.forced, true);
  const failed = child(702); const failedChildren = new OwnedChildren({ spawnChild: () => failed, signal: () => assert.fail("exited child signalled") });
  failedChildren.launch("storefront", "vp", [], {}); failed.emit("error", new Error("cannot spawn"));
  assert.throws(() => failedChildren.assertAlive(), /storefront stopped/);
  assert.equal((await failedChildren.stop())[0].error, "cannot spawn");
});

test("normal exit waits for stdio close before resolving owned-child lifetime", async () => {
  const actual = child(703);
  const children = new OwnedChildren({ spawnChild: () => actual, signal: () => assert.fail("already exited child signalled"), delay: () => new Promise(() => {}) });
  const entry = children.launch("native", "python", [], {});
  actual.emit("exit", 0, null);
  let finished = false;
  entry.done.then(() => { finished = true; });
  await Promise.resolve(); assert.equal(finished, false);
  const stopping = children.stop(); actual.emit("close", 0, null);
  assert.equal((await stopping)[0].exited, true); assert.equal(finished, true);
});

test("readiness has bounded timeout, preserves child failure and cancellation, and checks ready child", async () => {
  let now = 0; let checks = 0;
  await waitUntil(async () => ++checks === 2, { timeoutMs: 10, now: () => now++, delay: async () => {}, children: { assertAlive() {}, stopping: false } });
  assert.equal(checks, 2);
  await assert.rejects(() => waitUntil(async () => false, { timeoutMs: 2, now: () => now++, delay: async () => {} }), /timed out/);
  await assert.rejects(() => waitUntil(async () => true, { children: { stopping: true } }), /cancelled/);
  await assert.rejects(() => waitUntil(async () => true, { children: { assertAlive() { throw new Error("native exited"); } } }), /native exited/);
  assert.deepEqual(PORTS, { storefront: 4313, native: 8767, checkout: 4331 });
});
