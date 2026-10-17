import { defineConfig, lazyPlugins } from "vite-plus";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { LOCAL_ORIGINS, LOCAL_PROXY_PORT } from "@storefront/contracts";

export default defineConfig({
  plugins: lazyPlugins(() => [react(), tailwindcss()]),
  server: {
    host: "127.0.0.1",
    port: Number(process.env.PORT || 4310),
    strictPort: true,
    allowedHosts: [new URL(LOCAL_ORIGINS.dashboard).hostname],
    hmr: {
      protocol: "ws",
      host: new URL(LOCAL_ORIGINS.dashboard).hostname,
      clientPort: LOCAL_PROXY_PORT,
    },
    proxy: {
      "/api": {
        // Node's macOS resolver may not resolve *.localhost. Route by Host over loopback.
        target: `http://127.0.0.1:${LOCAL_PROXY_PORT}`,
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (request) =>
            request.setHeader("host", new URL(LOCAL_ORIGINS.api).host),
          );
        },
      },
    },
  },
});
