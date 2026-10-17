import { defineConfig, lazyPlugins } from "vite-plus";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: lazyPlugins(() => [react(), tailwindcss()]),
  server: {
    host: "127.0.0.1",
    port: 4310,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:4311" },
  },
});
