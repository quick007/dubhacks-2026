import { defineConfig, lazyPlugins } from "vite-plus";
import react from "@vitejs/plugin-react";
import { LOCAL_ORIGINS, LOCAL_PROXY_PORT } from "@storefront/contracts";
const demoCheckoutApi = process.env.DEV_API_ORIGIN || "http://127.0.0.1:4311";
if (!["http://127.0.0.1:4311", "http://localhost:4311", "http://127.0.0.1:4331", LOCAL_ORIGINS.api].includes(demoCheckoutApi)) {
  throw new Error("DEV_API_ORIGIN must identify the declared local API.");
}
export default defineConfig({
  plugins: lazyPlugins(() => [react()]),
  server: {
    host: "127.0.0.1",
    port: Number(process.env.PORT || 4312),
    strictPort: true,
    allowedHosts: [new URL(LOCAL_ORIGINS.storefront).hostname],
    proxy: { "/api/demo-checkout": { target: demoCheckoutApi, changeOrigin: true } },
    hmr: {
      protocol: "ws",
      host: new URL(LOCAL_ORIGINS.storefront).hostname,
      clientPort: LOCAL_PROXY_PORT,
    },
  },
});
