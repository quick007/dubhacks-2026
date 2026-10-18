import { randomUUID } from "node:crypto";
import { appendFile, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { chromium, type Browser, type BrowserContext, type ElementHandle, type Page } from "playwright";
import { SHOPPER_ALLOWED_ORIGINS } from "./shopper-contract.js";

export type ChromeShopperAction = { type: "click"; ref: string }
  | { type: "search"; query: string } | { type: "scroll"; direction: "up" | "down" };
type Selection = { originalName: string; catalogPrice: number; selectedColor: string; quantity: 1 };
type Outcome = { kind: "inspect" | "choose-variant" | "add-to-cart" | "open-cart" | "confirm-demo-order" | "search";
  target: string; selected?: Selection; confirmationVisible: boolean };

/** A fresh, visible Chrome instance per shopper. This provider knows no catalog or gold labels. */
export class ChromeShopperBrowser {
  readonly browserVersion: string;
  readonly contextId = randomUUID();
  readonly tabId = randomUUID();
  private controls = new Map<string, ElementHandle<HTMLElement>>();
  private sequence = 0;
  private controlSequence = 0;
  private closed = false;
  private logFailure: Error | null = null;
  private checkoutEvidence: { schema: string; synthetic: true;
    post: { url: string; httpStatus: number; request: unknown; body: unknown } | null;
    get: { url: string; httpStatus: number; body: unknown } | null;
    confirmation: { visible: boolean; checkoutId: string | null; receiptId: string | null }; errors: string[] } = {
      schema: "shopper-demo-checkout-evidence.v1", synthetic: true, post: null, get: null,
      confirmation: { visible: false, checkoutId: null, receiptId: null }, errors: [] };
  private constructor(private browser: Browser, private context: BrowserContext, private page: Page,
    private artifactDirectory: string, private origin: string, private armSearch: string) {
    this.browserVersion = browser.version();
  }

  static async open(entryUrl: string, artifactDirectory: string): Promise<ChromeShopperBrowser> {
    const entry = new URL(entryUrl);
    if (!SHOPPER_ALLOWED_ORIGINS.some((o) => o === entry.origin) || entry.pathname !== "/"
      || entry.username || entry.password || entry.hash || !["", "?model=local"].includes(entry.search)) {
      throw new Error("Shopper Chrome requires a clean declared local storefront entry URL");
    }
    await mkdir(artifactDirectory, { recursive: true });
    const browser = await chromium.launch({ channel: "chrome", headless: false });
    try {
      const context = await browser.newContext({ viewport: { width: 1280, height: 960 },
        serviceWorkers: "block", acceptDownloads: false });
      await context.tracing.start({ screenshots: true, snapshots: true, sources: false });
      const page = await context.newPage();
      page.setDefaultTimeout(5000);
      const shopper = new ChromeShopperBrowser(browser, context, page, artifactDirectory, entry.origin, entry.search);
      await context.route("**/*", async (route) => {
        const request = route.request();
        const url = new URL(request.url());
        const navigation = request.isNavigationRequest();
        const allowed = !url.username && !url.password && (navigation
          ? request.frame() === page.mainFrame() && shopper.safeNavigation(url)
          : url.origin === entry.origin || url.origin === "http://127.0.0.1:8766");
        if (allowed) await route.continue();
        else { await shopper.log({ type: "blocked-request", url: request.url(), navigation }); await route.abort("blockedbyclient"); }
      });
      await context.routeWebSocket("**/*", async (socket) => {
        const url = new URL(socket.url());
        const httpOrigin = `${url.protocol === "wss:" ? "https:" : "http:"}//${url.host}`;
        if (httpOrigin === entry.origin && !url.username && !url.password) socket.connectToServer();
        else { await shopper.log({ type: "blocked-websocket", url: socket.url() }); socket.close(); }
      });
      context.on("page", (popup) => { if (popup !== page) void popup.close().catch(() => {}); });
      page.on("dialog", (dialog) => { void dialog.dismiss(); });
      page.on("download", (download) => { void download.cancel(); });
      await page.goto(entryUrl, { waitUntil: "domcontentloaded", timeout: 30000 });
      await page.getByRole("button", { name: "View cart", exact: true }).waitFor({ state: "visible" });
      await shopper.log({ type: "opened", entryUrl, browserVersion: shopper.browserVersion,
        contextId: shopper.contextId, tabId: shopper.tabId, viewport: { width: 1280, height: 960 }, headless: false });
      return shopper;
    } catch (error) { await browser.close(); throw error; }
  }

  private safeNavigation(url: URL) {
    if (url.username || url.password || url.origin !== this.origin || url.pathname !== "/" || url.search !== this.armSearch) return false;
    const parts = url.hash.slice(1).split("?");
    if (parts.length > 2 || !["", "collection", "cart", "confirmed"].includes(parts[0])) return false;
    const parameters = new URLSearchParams(parts[1] ?? "");
    const keys = [...parameters.keys()];
    return keys.every((key) => ["q", "c"].includes(key)) && new Set(keys).size === keys.length
      && (!parameters.has("c") || ["all", "bags", "clothing", "home", "food", "accessories"].includes(parameters.get("c")!));
  }

  private async log(event: unknown) {
    try { await appendFile(join(this.artifactDirectory, "browser.jsonl"), JSON.stringify({ at: new Date().toISOString(), event }) + "\n"); }
    catch (error) { this.logFailure = error instanceof Error ? error : new Error(String(error)); throw this.logFailure; }
  }

  private assertOpen() {
    if (this.closed) throw new Error("Shopper browser is closed");
    if (this.logFailure) throw this.logFailure;
    if (!this.safeNavigation(new URL(this.page.url()))) throw new Error("Storefront navigated outside assigned treatment");
  }

  private async clearControls() {
    await Promise.all([...this.controls.values()].map((handle) => handle.dispose()));
    this.controls.clear();
  }

  async observe() {
    this.assertOpen();
    await this.clearControls();
    const controls: { ref: string; role: string; name: string }[] = [];
    const handles = await this.page.locator("button, a[href], [role=button]").elementHandles();
    for (const handle of handles) {
      const item = await handle.evaluate((element) => {
        const el = element as HTMLElement;
        const r = el.getBoundingClientRect(), style = getComputedStyle(el);
        if (!r.width || !r.height || r.bottom <= 0 || r.right <= 0 || r.top >= innerHeight || r.left >= innerWidth
          || style.visibility !== "visible" || style.display === "none" || style.opacity === "0"
          || el.closest('[aria-label="Example searches"], .example-searches, .demo-controls, [inert], [aria-hidden="true"]')
          || el.matches(":disabled, [aria-disabled=true]") || el.closest("form")) return null;
        if (el.tagName === "A") {
          const url = new URL((el as HTMLAnchorElement).href);
          if (url.origin !== location.origin || url.pathname !== "/" || url.search !== location.search || el.hasAttribute("download") || el.getAttribute("target")) return null;
        }
        const name = el.getAttribute("aria-label") || el.innerText.trim() || el.getAttribute("title") || "";
        if (!name || (/order|pay|checkout|purchase|buy/i.test(name) && name !== "Place demo order")) return null;
        return { role: el.getAttribute("role") || (el.tagName === "A" ? "link" : "button"), name };
      });
      if (!item) { await handle.dispose(); continue; }
      const ref = `control-${++this.controlSequence}`;
      this.controls.set(ref, handle as ElementHandle<HTMLElement>);
      controls.push({ ref, ...item });
    }
    const state = await this.page.evaluate(() => {
      const texts: string[] = [];
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      let node: Node | null;
      while ((node = walker.nextNode())) {
        const parent = node.parentElement;
        if (!parent || parent.closest('script, style, noscript, [aria-hidden="true"], [inert], .example-searches')) continue;
        const r = parent.getBoundingClientRect(), s = getComputedStyle(parent);
        if (r.width <= 0 || r.height <= 0 || r.bottom <= 0 || r.top >= innerHeight || r.right <= 0 || r.left >= innerWidth
          || s.visibility !== "visible" || s.display === "none" || s.opacity === "0") continue;
        const value = node.textContent?.replace(/\s+/g, " ").trim();
        if (value) texts.push(value);
      }
      let readiness: unknown[] = [];
      try { const parsed: unknown = JSON.parse(document.documentElement.getAttribute("data-local-model-readiness") ?? "[]"); if (Array.isArray(parsed)) readiness = parsed; } catch { /* absent or malformed mirror */ }
      return { text: texts.join("\n"), observedAtPerformanceMs: performance.now(),
        viewport: { scrollY, height: innerHeight, documentHeight: document.documentElement.scrollHeight },
        modelStatusText: document.querySelector('.collection [role="status"]')?.textContent?.trim() ?? "",
        cartItemCount: Number(document.querySelector(".cart-count")?.textContent?.trim() ?? "0"),
        cartEmpty: !document.querySelector(".cart-count")
        || document.querySelector(".cart-count")?.textContent?.trim() === "0", readiness };
    });
    const screenshot = await this.page.screenshot({ path: join(this.artifactDirectory, `observation-${++this.sequence}.png`) });
    await this.log({ type: "observation", sequence: this.sequence, ...state, controls, url: this.page.url() });
    return { ...state, controls, url: this.page.url(), screenshotBase64: screenshot.toString("base64") };
  }

  private async cartSelection(): Promise<Selection | undefined> {
    const rows = this.page.locator(".cart-items > div");
    if (await rows.count() !== 1) return undefined;
    return rows.first().evaluate((row) => {
      const spans = row.querySelectorAll(":scope > span");
      const details = spans[0]?.querySelector("small")?.textContent?.trim() ?? "";
      const match = details.match(/^(.*?)\s*·\s*×\s*1$/);
      const name = [...(spans[0]?.childNodes ?? [])].filter((n) => n.nodeType === Node.TEXT_NODE).map((n) => n.textContent).join("").trim();
      const price = Number(spans[1]?.textContent?.trim().replace(/^\$/, ""));
      if (!match || !name || !Number.isFinite(price) || price <= 0) return undefined;
      return { originalName: name, selectedColor: match[1], catalogPrice: price, quantity: 1 as const };
    });
  }

  async act(action: ChromeShopperAction): Promise<Outcome> {
    this.assertOpen();
    let kind: Outcome["kind"] = "inspect", target = "", selected: Selection | undefined;
    if (action.type === "scroll") {
      target = action.direction;
      await this.page.mouse.wheel(0, action.direction === "down" ? 640 : -640);
      await this.page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    } else if (action.type === "search") {
      if (!action.query.trim() || action.query.length > 200) throw new Error("Invalid bounded search query");
      target = action.query; kind = "search";
      await this.page.getByRole("searchbox").fill(action.query);
      await this.page.getByRole("search").getByRole("button", { name: "Search", exact: true }).click();
    } else {
      const control = this.controls.get(action.ref);
      if (!control || !await control.isVisible() || !await control.isEnabled()) throw new Error("Unknown, stale or unavailable visible control ref");
      if (!await control.evaluate((element) => {
        const r = element.getBoundingClientRect();
        return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth
          && !element.closest('form, .example-searches, .demo-controls, [inert], [aria-hidden="true"]');
      })) throw new Error("Control is no longer in the observed viewport");
      target = await control.getAttribute("aria-label") || (await control.innerText()).trim();
      if (target === "Place demo order") {
        if (!await this.page.getByRole("heading", { name: "Your demo cart", exact: true }).isVisible()
          || !await this.page.getByText("No payment required. Nothing will be shipped.", { exact: true }).isVisible()) {
          throw new Error("Refusing checkout without the simulated cart safety notice");
        }
        selected = await this.cartSelection();
        if (!selected) throw new Error("Simulated checkout requires exactly one visible cart item with quantity one");
        kind = "confirm-demo-order";
      } else if (target === "Add to cart") {
        selected = await control.evaluate((button) => {
          const article = button.closest("article");
          const colors = article?.querySelector('.product-colors[aria-label]')?.getAttribute("aria-label") ?? "";
          const name = colors.replace(/ colors$/, "").trim();
          const color = article?.querySelector(".selected-color")?.textContent?.trim() ?? "";
          const price = Number(article?.querySelector(".product-heading > span")?.textContent?.trim().replace(/^\$/, ""));
          if (!name || !color || !Number.isFinite(price) || price <= 0) return undefined;
          return { originalName: name, selectedColor: color, catalogPrice: price, quantity: 1 as const };
        });
        if (!selected) throw new Error("Cannot identify selected product from rendered article");
        kind = "add-to-cart";
      } else if (target === "View cart") kind = "open-cart";
      else if (target.includes(" color: ")) kind = "choose-variant";
      if (kind === "confirm-demo-order") {
        const responseReady = this.page.waitForResponse((response) => response.request().method() === "POST"
          && new URL(response.url()).origin === this.origin && new URL(response.url()).pathname === "/api/demo-checkout",
        { timeout: 15000 });
        try {
          await control.click();
          const response = await responseReady;
          const bytes = await response.body();
          if (bytes.length > 64 * 1024) throw new Error("Demo checkout response exceeds evidence bound");
          const body = JSON.parse(bytes.toString("utf8"));
          this.checkoutEvidence.post = { url: response.url(), httpStatus: response.status(), request: response.request().postDataJSON(), body };
          if (response.status() !== 202 || typeof body?.receipt?.checkoutId !== "string"
            || !/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(body.receipt.checkoutId)) {
            throw new Error("Demo checkout server did not confirm an order receipt");
          }
          const url = `${this.origin}/api/demo-checkout/${body.receipt.checkoutId}`;
          const readback = await this.page.request.get(url, { maxRedirects: 0, timeout: 10000 });
          const readbackBytes = await readback.body();
          if (readbackBytes.length > 64 * 1024) throw new Error("Demo checkout readback exceeds evidence bound");
          this.checkoutEvidence.get = { url, httpStatus: readback.status(), body: JSON.parse(readbackBytes.toString("utf8")) };
          await this.page.getByRole("heading", { name: "Demo order confirmed", exact: true }).waitFor({ state: "visible", timeout: 5000 });
          this.checkoutEvidence.confirmation = await this.page.evaluate(() => ({
            visible: document.querySelector(".confirmation h1")?.textContent?.trim() === "Demo order confirmed",
            checkoutId: document.querySelector("main")?.getAttribute("data-demo-checkout-id") ?? null,
            receiptId: document.querySelector("main")?.getAttribute("data-demo-receipt-id") ?? null,
          }));
        } catch (error) {
          this.checkoutEvidence.errors.push(error instanceof Error ? error.message : String(error));
          await responseReady.catch(() => {});
          throw error;
        }
      } else await control.click();
      if (kind === "open-cart") selected = await this.cartSelection();
    }
    await this.clearControls();
    this.assertOpen();
    const confirmationVisible = await this.page.getByRole("heading", { name: "Demo order confirmed", exact: true }).isVisible();
    const outcome: Outcome = { kind, target, ...(selected ? { selected } : {}), confirmationVisible };
    await this.log({ type: "action", action, outcome, url: this.page.url() });
    return outcome;
  }

  async close() {
    if (this.closed) return;
    this.closed = true;
    let failure: unknown;
    try { await writeFile(join(this.artifactDirectory, "checkout-receipt.json"), JSON.stringify(this.checkoutEvidence, null, 2) + "\n", { flag: "wx" }); }
    catch (error) { failure = error; }
    try { await this.page.screenshot({ path: join(this.artifactDirectory, "screenshot.png") }); }
    catch (error) { failure = error; }
    try { await this.context.tracing.stop({ path: join(this.artifactDirectory, "trace.zip") }); }
    catch (error) { failure ??= error; }
    try { await this.log({ type: "closed", ...(failure ? { artifactError: String(failure) } : {}) }); }
    catch (error) { failure ??= error; }
    finally { await this.browser.close(); }
    if (failure) throw failure;
  }
}
