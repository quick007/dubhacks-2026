import { defineConfig, lazyPlugins } from "vite-plus";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: lazyPlugins(() => [react()]),
  server: { host: "127.0.0.1", port: 4312, strictPort: true },
});
