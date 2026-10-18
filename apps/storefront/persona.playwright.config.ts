import { defineConfig } from "@playwright/test";

// Parent-controlled existing candidate only. Never starts servers or model workers.
export default defineConfig({
  testDir: "./e2e",
  testMatch: "persona-switcher.spec.ts",
  workers: 1,
  retries: 0,
  timeout: 60_000,
  use: { baseURL: "http://127.0.0.1:4313", browserName: "chromium", channel: "chrome", headless: false,
    viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1, trace: "retain-on-failure" },
});
