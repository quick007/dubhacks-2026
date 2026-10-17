import { spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const localGo = path.join(root, ".tools/go/bin/go");
const binary = process.env.STOREFRONT_GO_BINARY || (existsSync(localGo) ? localGo : "go");
const cache = path.join(root, ".cache");
mkdirSync(cache, { recursive: true });
const child = spawn(binary, process.argv.slice(2), {
  stdio: "inherit",
  env: {
    ...process.env,
    GOCACHE: path.join(cache, "go-build"),
    GOMODCACHE: path.join(cache, "go-mod"),
  },
});
child.once("error", (error) => {
  console.error(
    `Go could not start: ${error.message}. Install Go 1.24+ or set STOREFRONT_GO_BINARY.`,
  );
  process.exitCode = 1;
});
child.once("exit", (code, signal) => {
  process.exitCode = code ?? (signal ? 1 : 0);
});
