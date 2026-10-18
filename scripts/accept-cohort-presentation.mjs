/** Real local presentation acceptance. Default: CPU plan only. Run after ALLBROWSERIDLE. */
import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ORIGIN = "http://127.0.0.1:4310";
const DEFAULT_REPORT_CONFIG = "apps/web/src/fieldwork/presentation-report.json";
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$/;
const SHA = /^[a-f0-9]{64}$/;
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const compact = (value) => value.replace(/\s+/g, "").trim();
const slides = [
  { id: "title", label: "Storefront Lab", action: "initial navigation" },
  { id: "results", label: "Shopper results", action: "ArrowRight" },
  { id: "model", label: "The model", action: "navigation button" },
  { id: "speed", label: "Ranking comparison", action: "ArrowRight" },
  { id: "stakes", label: "Why speed matters", action: "navigation button" },
];
const SOURCE_FILES = [
  "scripts/accept-cohort-presentation.mjs", "apps/web/src/App.tsx", "apps/web/src/CohortPresentation.tsx",
  "apps/web/src/lib/presentation-results.ts", "apps/web/src/slides/Deck.tsx",
  "apps/web/src/slides/CohortResultsSlide.tsx", "apps/web/src/slides/ModelSlide.tsx",
  "apps/web/src/slides/SpeedSlide.tsx", "apps/web/src/slides/StakesSlide.tsx",
  "apps/web/src/slides/SlideFrame.tsx", "apps/web/src/styles.css", "apps/web/vite.config.ts",
  DEFAULT_REPORT_CONFIG, "apps/web/src/fieldwork/current-model.json", "apps/web/src/fieldwork/model-run.json",
  "apps/web/src/fieldwork/llm-run.json", "apps/api/internal/experimentresults/presentation.go",
];

export function parseOptions(args) {
  const options = { run: false, reportId: null, out: null };
  const seen = new Set();
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    assert(["--run", "--report", "--out"].includes(flag), `Unknown option ${flag}`);
    assert(!seen.has(flag), `Duplicate option ${flag}`); seen.add(flag);
    if (flag === "--run") options.run = true;
    else {
      const value = args[++i];
      assert(value && !value.startsWith("--"), `Missing value for ${flag}`);
      if (flag === "--report") { assert(SAFE_ID.test(value), "An exact safe immutable report ID is required"); options.reportId = value; }
      else options.out = resolve(value);
    }
  }
  if (options.run) assert(options.reportId && options.out, "--run requires exact --report ID and unique --out directory");
  return options;
}

export function acceptancePlan(options) {
  return { version: "cohort-presentation-acceptance.v1", execute: options.run, origin: ORIGIN,
    reportId: options.reportId, output: options.out, viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1,
    browser: "Installed headed Chrome through root @playwright/test; one owned isolated context",
    prerequisite: "Parent must explicitly confirm ALLBROWSERIDLE before invoking --run; this script cannot infer that state.",
    actualDataOnly: true, slides, defaultReportConfig: DEFAULT_REPORT_CONFIG,
    requiredDefaultConfigSchema: "presentation-report.v1: reportId, reportSha256, metricsSourceSha256, synthetic:true, humanConversionClaim:false, scope",
    requestGuardDisablesHttpCache: true, checks: [
      "Fresh strict-Tiger report GET; exact immutable ID, synthetic 60/120 scope and evidence digests",
      "Final default configuration must exist and bind exact selected ID, API envelope digest and metrics source digest",
      "Bare /#title must load that same selected Tiger report; unconfigured legacy default fails",
      "Visible goal, purchase and merchandise figures equal actual API metrics, including exact cents",
      "All five measured slides reached using both keyboard and navigation buttons; screenshots saved",
      "Explicit ?report=ID remains a separate checked entry path with identical figures and Tiger data",
      "Current model values match local audited projection; historical ranking scope remains explicit",
      "Unknown report actual 404 and empty report client error remain outside legacy deck",
      "No /api/runs, /api/catalog, model/personalize or write request may leave the context",
    ], noSyntheticResponses: true, noModelCalls: true, noPerformanceBenchmark: true,
    defaultBehavior: "No network, browser import/launch or output-directory creation without --run",
    example: "node scripts/accept-cohort-presentation.mjs --run --report EXACT_FINAL_REPORT_ID --out NEW_DIRECTORY" };
}

function validateReport(value, reportId) {
  assert(value?.version === "experiment-presentation.v1" && value.reportId === reportId, "Immutable report ID/version mismatch");
  assert(value.source === "tiger" && value.cloudAcknowledged === true && value.synthetic === true
    && value.humanConversionClaim === false && SHA.test(value.sha256), "Strict acknowledged synthetic Tiger data required");
  const data = value.data, metrics = data?.metrics;
  assert(data?.version === "shopper-checkout-cohort.v1" && data.fixedCases === 60 && data.fixedSessions === 120
    && metrics?.version === "shopper-synthetic-metrics.v1" && metrics.fixedCases === 60 && metrics.fixedSessions === 120
    && metrics.synthetic === true && metrics.humanConversionClaim === false, "Expected exact60-pair/120-session cohort");
  assert(data.provenance?.sourceReports?.length === 6, "Six source panels required");
  for (const source of [data.provenance.plan, data.provenance.metrics, data.provenance.serverOutcomes, ...data.provenance.sourceReports]) {
    assert(typeof source?.path === "string" && SHA.test(source.sha256), "Missing evidence source digest");
  }
  for (const arm of ["generic", "local-model"]) {
    const a = metrics.arms?.[arm]; assert(a?.assigned === 60, "Every assigned slot must remain in each arm");
    for (const f of [a.intentCorrectSuccess, a.syntheticConversion]) {
      assert(Number.isSafeInteger(f?.numerator) && f.numerator >= 0 && f.numerator <= 60 && f.denominator === 60
        && Math.abs(f.rate - f.numerator / 60) < 1e-12, "Invalid assigned outcome count/rate");
    }
    assert(a.verifiedPurchases === a.syntheticConversion.numerator && a.verifiedPurchases <= a.intentCorrectSuccess.numerator,
      "Purchase and intent-success metrics contradict one another");
    assert(a.revenue?.currency === "USD" && Number.isSafeInteger(a.revenue.totalCents) && a.revenue.totalCents >= 0
      && a.allServerConfirmedOrders?.evaluated === true
      && Number.isSafeInteger(a.allServerConfirmedOrders.merchandiseSubtotalCents)
      && a.allServerConfirmedOrders.merchandiseSubtotalCents >= a.revenue.totalCents, "Invalid merchandise metrics");
  }
  return value;
}

export function isProhibitedRequest(request) {
  const url = new URL(request.url());
  return request.method() !== "GET" && request.method() !== "HEAD"
    || /\/(?:personalize|api\/(?:runs|catalog))(?:\/|$)/.test(url.pathname);
}
async function bounded(promise, milliseconds, label) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} exceeded ${milliseconds}ms`)), milliseconds);
  })]); } finally { clearTimeout(timer); }
}

export async function runAcceptance(options) {
  assert(options.run && SAFE_ID.test(options.reportId ?? "") && options.out, "Explicit run/report/out required");
  // Non-recursive, exclusive creation: never overwrite an existing acceptance run.
  await mkdir(options.out);
  const save = (name, value) => writeFile(join(options.out, name), JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
  const hashes = async () => Object.fromEntries(await Promise.all(SOURCE_FILES.map(async (path) => [path, sha(await readFile(join(ROOT, path)))])));
  const report = { ...acceptancePlan(options), startedAt: new Date().toISOString(), status: "running",
    checks: [], requests: [], requestFailures: [], apiResponses: [], pageErrors: [], console: [], forbiddenRequests: [], screenshots: [] };
  let browser, context, page, phase = "preflight", requestIndex = 0, responseIndex = 0;
  const pending = [], requestMetadata = new WeakMap();
  const check = (name, condition, evidence = null) => {
    report.checks.push({ name, passed: Boolean(condition), evidence }); assert(condition, name);
  };
  const screenshot = async (name) => {
    const file = `${name}.png`, bytes = await page.screenshot({ path: join(options.out, file), fullPage: true, animations: "disabled", timeout: 10000 });
    report.screenshots.push({ file, sha256: sha(bytes), bytes: bytes.length, phase, fullPage: true });
  };
  const flush = () => bounded(Promise.allSettled([...pending]), 12000, "Actual API body capture");
  const apiGet = async (id, file) => {
    const url = `${ORIGIN}/api/experiment-results/presentation?reportId=${encodeURIComponent(id)}&source=tiger`;
    const response = await fetch(url, { signal: AbortSignal.timeout(15000), cache: "no-store", credentials: "omit", redirect: "error" });
    const bytes = Buffer.from(await response.arrayBuffer());
    await writeFile(join(options.out, `${file}.body.json`), bytes, { flag: "wx" });
    const receipt = { url, method: "GET", status: response.status, bytes: bytes.length, sha256: sha(bytes), file: `${file}.body.json` };
    await save(`${file}.receipt.json`, receipt); report.apiResponses.push({ ...receipt, source: "node-real-GET", phase });
    return { response, data: JSON.parse(bytes.toString("utf8")) };
  };
  try {
    report.sourceHashesBefore = await hashes(); await save("plan.json", { ...acceptancePlan(options), sourceHashes: report.sourceHashesBefore });
    const referenceResponse = await apiGet(options.reportId, "reference");
    check("Real selected Tiger endpoint returns200", referenceResponse.response.status === 200);
    const reference = validateReport(referenceResponse.data, options.reportId);
    const defaultConfig = JSON.parse(await readFile(join(ROOT, DEFAULT_REPORT_CONFIG), "utf8"));
    check("Final default config selects same immutable synthetic report", defaultConfig.version === "presentation-report.v1"
      && defaultConfig.reportId === options.reportId && defaultConfig.synthetic === true && defaultConfig.humanConversionClaim === false
      && typeof defaultConfig.scope === "string" && defaultConfig.scope.trim().length > 0, defaultConfig);
    check("Final default config digests equal real Tiger envelope and metrics", defaultConfig.reportSha256 === reference.sha256
      && defaultConfig.metricsSourceSha256 === reference.data.provenance.metrics.sha256,
      { configuredReport: defaultConfig.reportSha256, actualReport: reference.sha256,
        configuredMetrics: defaultConfig.metricsSourceSha256, actualMetrics: reference.data.provenance.metrics.sha256 });
    report.defaultConfiguration = { path: DEFAULT_REPORT_CONFIG, ...defaultConfig };
    report.reference = { reportId: reference.reportId, experimentId: reference.experimentId, sha256: reference.sha256,
      provenance: reference.data.provenance, metrics: reference.data.metrics };
    const measured = JSON.parse(await readFile(join(ROOT, "apps/web/src/fieldwork/current-model.json"), "utf8"));
    // Playwright is intentionally loaded only after --run and the real data preflight.
    const { chromium } = createRequire(join(ROOT, "package.json"))("@playwright/test");
    browser = await chromium.launch({ channel: "chrome", headless: false }); report.browserVersion = browser.version();
    context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1, serviceWorkers: "block", acceptDownloads: false });
    // Abort only prohibited traffic; never fabricate a response. Any attempt fails the run.
    await context.route("**/*", async (route) => {
      if (isProhibitedRequest(route.request())) {
        report.forbiddenRequests.push({ url: route.request().url(), method: route.request().method(), phase, blockedBeforeDelivery: true });
        await route.abort("blockedbyclient");
      } else await route.continue();
    });
    page = await context.newPage(); page.setDefaultTimeout(15000); page.setDefaultNavigationTimeout(20000);
    page.on("request", (request) => {
      const meta = { index: requestIndex++, url: request.url(), method: request.method(), resourceType: request.resourceType(), phase };
      requestMetadata.set(request, meta); report.requests.push(meta);
    });
    page.on("requestfailed", (request) => report.requestFailures.push({ ...requestMetadata.get(request), failure: request.failure() }));
    page.on("pageerror", (error) => report.pageErrors.push({ phase, message: error.message, stack: error.stack ?? null }));
    page.on("console", (message) => { if (["error", "warning"].includes(message.type())) report.console.push({ phase, type: message.type(), text: message.text() }); });
    page.on("response", (response) => {
      const url = new URL(response.url());
      if (url.origin !== ORIGIN || url.pathname !== "/api/experiment-results/presentation") return;
      const request = requestMetadata.get(response.request());
      const meta = { ...request, index: responseIndex++, requestIndex: request?.index ?? null,
        status: response.status(), source: "browser-actual-response" };
      pending.push((async () => {
        const receipt = { ...meta };
        try {
          const bytes = await response.body(); receipt.bytes = bytes.length; receipt.sha256 = sha(bytes);
          receipt.bodyFile = `browser-api-${meta.index}.body.json`;
          await writeFile(join(options.out, receipt.bodyFile), bytes, { flag: "wx" });
          try { receipt.data = JSON.parse(bytes.toString("utf8")); } catch (error) { receipt.parseError = String(error); }
        } catch (error) { receipt.captureError = String(error); }
        await save(`browser-api-${meta.index}.receipt.json`, receipt);
        report.apiResponses.push(receipt);
      })());
    });
    const main = (label) => page.getByRole("main", { name: label, exact: true });
    const verifyBrowserReport = async (entryPhase) => {
      await flush();
      const successful = report.apiResponses.filter((r) => r.source === "browser-actual-response" && r.phase === entryPhase && r.status === 200 && r.data);
      check(`${entryPhase}: actual browser received selected Tiger report`, successful.length > 0);
      for (const response of successful) {
        const requested = new URL(response.url);
        check(`${entryPhase}: browser explicitly requested selected ID from Tiger`, requested.searchParams.get("reportId") === options.reportId
          && requested.searchParams.get("source") === "tiger", response.url);
        validateReport(response.data, options.reportId);
        check(`${entryPhase}: browser API exactly matches reference metrics/provenance`, response.data.sha256 === reference.sha256
          && isDeepStrictEqual(response.data.data, reference.data), { file: response.bodyFile, sha256: response.sha256 });
      }
    };
    const readResultView = () => main("Shopper results").evaluate((element) => {
      const paragraph = Array.from(element.querySelectorAll("p")).find((p) => p.querySelector(":scope > span > span")?.textContent?.trim() === "of 60");
      const withinViewport = (node) => { if (!node) return false; const r = node.getBoundingClientRect();
        return r.width > 0 && r.height > 0 && r.top >= 0 && r.left >= 0 && r.bottom <= innerHeight && r.right <= innerWidth; };
      return { text: element.textContent, headlineVisible: withinViewport(paragraph),
        metricValuesVisible: Array.from(element.querySelectorAll("dl dd")).every(withinViewport),
        counts: Array.from(paragraph?.querySelectorAll(":scope > span") ?? [], (span) => ({
          numerator: span.firstChild?.textContent?.trim(), denominator: span.querySelector("span")?.textContent?.trim(),
        })) };
    });
    const dl = async () => page.locator("main").evaluate((element) => Array.from(element.querySelectorAll("dl > div"), (row) => ({
      label: row.querySelector("dt")?.textContent?.trim(), text: row.querySelector("dd")?.textContent?.trim(),
      values: Array.from(row.querySelectorAll("dd > span"), (span) => span.textContent?.trim()),
    })));
    phase = "bare-default-presentation";
    report.bareEntry = { requestedURL: `${ORIGIN}/#title` };
    await page.goto(report.bareEntry.requestedURL, { waitUntil: "domcontentloaded" });
    await main("Storefront Lab").waitFor({ state: "visible" });
    report.bareEntry.finalURL = page.url();
    // Even a legacy title that happens to render cannot pass without this exact real Tiger response.
    await verifyBrowserReport(phase);
    for (const [i, slide] of slides.entries()) {
      if (slide.action === "ArrowRight") await page.keyboard.press("ArrowRight");
      else if (slide.action === "navigation button") await page.getByRole("button", { name: `Slide ${i + 1}: ${slide.label}`, exact: true }).click();
      await main(slide.label).waitFor({ state: "visible" });
      check(`Slide ${slide.id} selected by ${slide.action}`, await page.getByRole("button", { name: `Slide ${i + 1}: ${slide.label}`, exact: true }).getAttribute("aria-current") === "step");
      check("Exactly five measured-slide navigation buttons", await page.getByRole("navigation", { name: "Slides" }).getByRole("button").count() === 5);
      if (slide.id === "results") {
        const view = await readResultView();
        check("Compared result figures fully within viewport", view.headlineVisible && view.metricValuesVisible, view);
        const arms = [reference.data.metrics.arms.generic, reference.data.metrics.arms["local-model"]];
        check("Visible intent-success counts and fixed denominators equal actual API", isDeepStrictEqual(view.counts,
          arms.map((arm) => ({ numerator: String(arm.intentCorrectSuccess.numerator), denominator: "of 60" }))), view);
        const definitions = await dl(), expected = [
          ["Checkouts that fit", arms.map((arm) => String(arm.verifiedPurchases))],
          ["Value of checkouts that fit", arms.map((arm) => arm.revenue.totalCents)],
          ["All accepted order value", arms.map((arm) => arm.allServerConfirmedOrders.merchandiseSubtotalCents)],
        ];
        for (const [label, values] of expected) {
          const row = definitions.find((r) => r.label === label); check(`Visible metric ${label} present`, Boolean(row));
          if (label !== "Checkouts that fit") check(`Visible ${label} contains two USD amounts`, row.values.length === 2
            && row.values.every((value) => /^\$(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?$/.test(value)), row.values);
          const actual = label === "Checkouts that fit" ? row.values : row.values.map((value) => Math.round(Number(value.replace(/[^0-9.-]/g, "")) * 100));
          check(`Visible ${label} equals exact API counts/cents`, isDeepStrictEqual(actual, values), { actual, expected: values, displayed: row.values });
        }
        check("Synthetic, abstention and merchandise scope visible", /60 paired synthetic shoppers/.test(view.text)
          && /including choosing nothing when nothing fit/.test(view.text)
          && /Server-confirmed simulated orders; merchandise value only/.test(view.text));
        report.visibleResults = { ...view, definitions };
      } else if (slide.id === "model") {
        const definitions = await dl();
        for (const [label, before, after] of [["Right product first", measured.rank.before.topOneCorrect, measured.rank.after.topOneCorrect],
          ["Right product in six", measured.rank.before.targetInTopSix, measured.rank.after.targetInTopSix]]) {
          const row = definitions.find((r) => r.label === label);
          check(`Visible current ${label} equals audited projection`, isDeepStrictEqual(row?.values, [String(before), String(after), `of ${measured.rank.cases}`]), row);
        }
        const latency = definitions.find((r) => r.label === "Fresh personalized cards, p50 / p95");
        check("Visible loaded browser p50/p95 matches audited projection", compact(latency?.text ?? "")
          === `${Math.round(measured.loadedBrowser.local.p50Ms)}/${Math.round(measured.loadedBrowser.local.p95Ms)}ms`, latency);
        const text = await main(slide.label).innerText();
        check("Browser endpoint scope remains explicit", /First three cards fully visible with decoded images/.test(text)
          && /Physical paint and cold navigation excluded/.test(text));
        report.visibleModel = { definitions, text };
      } else if (slide.id === "speed") {
        const text = await main(slide.label).innerText();
        check("Historical comparison explicitly excludes copy/images/rendering", /Historical comparison:/.test(text)
          && /Ranking only;\s*copy, images and rendering excluded/.test(text)
          && /Codex CLI turn overhead/.test(text), text);
      } else if (slide.id === "stakes") {
        check("Real shopper conversion lift not claimed", /Our own shopper sessions are synthetic/.test(await main(slide.label).innerText())
          && /not yet measured a conversion lift with real/.test(await main(slide.label).innerText()));
      }
      await screenshot(`slide-${i + 1}-${slide.id}`);
    }
    await verifyBrowserReport(phase);
    phase = "explicit-report-presentation";
    report.explicitEntry = { requestedURL: `${ORIGIN}/?report=${encodeURIComponent(options.reportId)}#results` };
    await page.goto(report.explicitEntry.requestedURL, { waitUntil: "domcontentloaded" });
    await main("Shopper results").waitFor({ state: "visible" }); report.explicitEntry.finalURL = page.url();
    await verifyBrowserReport(phase);
    const explicitView = await readResultView(), explicitDefinitions = await dl();
    check("Explicit report figures fully within viewport", explicitView.headlineVisible && explicitView.metricValuesVisible, explicitView);
    check("Explicit report shows same exact API-bound counts and values as bare default",
      isDeepStrictEqual(explicitView.counts, report.visibleResults.counts)
      && isDeepStrictEqual(explicitDefinitions, report.visibleResults.definitions), { view: explicitView, definitions: explicitDefinitions });
    check("Explicit report retains exactly five measured slides", await page.getByRole("navigation", { name: "Slides" }).getByRole("button").count() === 5);
    report.explicitEntry.visibleResults = { ...explicitView, definitions: explicitDefinitions };
    await screenshot("explicit-report-results");
    phase = "missing-report";
    const missingId = `acceptance-missing-${randomUUID()}`;
    const missing = await apiGet(missingId, "missing-reference"); check("Real missing immutable report returns404", missing.response.status === 404);
    await page.goto(`${ORIGIN}/?report=${missingId}#results`, { waitUntil: "domcontentloaded" });
    await page.getByRole("status").filter({ hasText: /unavailable \(HTTP 404\)/ }).waitFor({ state: "visible" });
    check("Missing report is truthful error without legacy deck", await page.getByRole("navigation", { name: "Slides" }).count() === 0);
    await screenshot("missing-report-status"); await flush();
    check("Browser missing-report status came from actual strict404", report.apiResponses.some((r) => r.source === "browser-actual-response"
      && r.phase === "missing-report" && r.status === 404 && new URL(r.url).searchParams.get("reportId") === missingId
      && new URL(r.url).searchParams.get("source") === "tiger"));
    phase = "empty-report";
    await page.goto(`${ORIGIN}/?report=#results`, { waitUntil: "domcontentloaded" });
    await page.getByRole("status").filter({ hasText: /safe immutable report ID/ }).waitFor({ state: "visible" });
    check("Explicit empty report is validation error without legacy deck", await page.getByRole("navigation", { name: "Slides" }).count() === 0);
    check("Empty report never fetches the presentation endpoint", !report.requests.some((r) => r.phase === phase && new URL(r.url).pathname === "/api/experiment-results/presentation"));
    await screenshot("empty-report-status"); await flush();
    check("No legacy/model/write request attempted", report.forbiddenRequests.length === 0, report.forbiddenRequests);
    const unexpectedPageErrors = report.pageErrors.filter((e) => !(e.message === "WebSocket closed without opened." && /\/@vite\/client:/.test(e.stack ?? "")));
    check("No unexpected JavaScript page error", unexpectedPageErrors.length === 0, unexpectedPageErrors);
    const unexpectedFailures = report.requestFailures.filter((r) => !(/ERR_ABORTED/.test(r.failure?.errorText ?? "")
      && new URL(r.url).pathname === "/api/experiment-results/presentation"));
    check("No unexpected request failure", unexpectedFailures.length === 0, unexpectedFailures);
    report.expectedNegativeControlConsole = report.console.filter((entry) => entry.phase === "missing-report" && /404/.test(entry.text));
    report.expectedStrictModeCancellations = report.requestFailures.filter((entry) => /ERR_ABORTED/.test(entry.failure?.errorText ?? "")
      && new URL(entry.url).pathname === "/api/experiment-results/presentation");
    const residualConsole = report.console.filter((entry) => !report.expectedNegativeControlConsole.includes(entry));
    report.status = report.pageErrors.length || residualConsole.length ? "passed-with-warnings" : "passed";
  } catch (error) {
    report.status = "failed"; report.error = { message: error.message ?? String(error), stack: error.stack ?? null };
    if (page) await screenshot("failure-state").catch((cause) => { report.failureScreenshotError = String(cause); });
  } finally {
    if (browser) await bounded(browser.close(), 10000, "Owned browser close").catch((error) => {
      report.status = "failed"; report.browserCloseError = String(error);
    });
    await flush().catch((error) => { report.status = "failed"; report.captureSettlementError = String(error); });
    try { report.sourceHashesAfter = await hashes(); report.sourcesUnchanged = isDeepStrictEqual(report.sourceHashesBefore, report.sourceHashesAfter); }
    catch (error) { report.sourcesUnchanged = false; report.sourceHashError = String(error); }
    if (!report.sourcesUnchanged) { report.status = "failed"; report.sourceChangeError = "Reviewed source bytes changed during acceptance"; }
    report.finishedAt = new Date().toISOString();
    report.scope = "Actual local presentation functional acceptance; no synthetic responses, model requests, benchmark timing, final-data fabrication or fallback promotion. Whole-dollar UI values must also match exact API cents.";
    await save("report.json", report);
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const options = parseOptions(process.argv.slice(2));
    if (!options.run) console.log(JSON.stringify(acceptancePlan(options), null, 2));
    else {
      const report = await runAcceptance(options);
      console.log(JSON.stringify({ status: report.status, out: options.out, checks: report.checks.length, error: report.error ?? null }, null, 2));
      if (report.status === "failed") process.exitCode = 1;
    }
  } catch (error) { console.error(error.message ?? String(error)); process.exitCode = 1; }
}
