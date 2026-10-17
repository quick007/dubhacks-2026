import { defineConfig } from "@playwright/test";
import { LOCAL_ORIGINS, LOCAL_PROXY_PORT } from "./packages/contracts/src/local";
export default defineConfig({
  testDir: "./tests",
  workers: 1,
  timeout: 60_000,
  use: { baseURL: LOCAL_ORIGINS.dashboard, trace: "retain-on-failure" },
  webServer: {
    command: "pnpm dev",
    // TCP readiness avoids macOS system DNS, which may not resolve *.localhost.
    port: LOCAL_PROXY_PORT,
    reuseExistingServer: false,
    timeout: 60_000,
    gracefulShutdown: { signal: "SIGTERM", timeout: 10_000 },
    env: { STOREFRONT_DATA_DIR: `../../data/smoke/${Date.now()}-${process.pid}` },
  },
});
