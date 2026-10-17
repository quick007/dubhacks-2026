import { defineConfig, lazyPlugins } from "vite-plus";
import react from "@vitejs/plugin-react";
import { LOCAL_ORIGINS, LOCAL_PROXY_PORT } from "@storefront/contracts";
export default defineConfig({
  plugins: lazyPlugins(() => [react()]),
  server: {
    host: "127.0.0.1",
    port: Number(process.env.PORT || 4312),
    strictPort: true,
    allowedHosts: [new URL(LOCAL_ORIGINS.storefront).hostname],
    hmr: {
      protocol: "ws",
      host: new URL(LOCAL_ORIGINS.storefront).hostname,
      clientPort: LOCAL_PROXY_PORT,
    },
  },
});
