import { defineConfig, lazyPlugins } from "vite-plus";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { LOCAL_ORIGINS, LOCAL_PROXY_PORT } from "@storefront/contracts";

// A standalone evidence receiver can serve the measured deck without the legacy database app.
const directApi = process.env.DEV_API_ORIGIN;
if (directApi && !["http://127.0.0.1:4311", "http://localhost:4311"].includes(directApi)) {
  throw new Error("DEV_API_ORIGIN must identify the declared loopback API on port 4311.");
}

export default defineConfig({
  plugins: lazyPlugins(() => [react(), tailwindcss()]),
  server: {
    host: "127.0.0.1",
    port: Number(process.env.PORT || 4310),
    strictPort: true,
    allowedHosts: [new URL(LOCAL_ORIGINS.dashboard).hostname],
    hmr: directApi ? undefined : {
      protocol: "ws",
      host: new URL(LOCAL_ORIGINS.dashboard).hostname,
      clientPort: LOCAL_PROXY_PORT,
    },
    proxy: {
      "/api": {
        // Node's macOS resolver may not resolve *.localhost. Route by Host over loopback.
        target: directApi || `http://127.0.0.1:${LOCAL_PROXY_PORT}`,
        changeOrigin: true,
        configure(proxy) {
          if (directApi) return;
          proxy.on("proxyReq", (request) =>
            request.setHeader("host", new URL(LOCAL_ORIGINS.api).host),
          );
        },
      },
    },
  },
});
