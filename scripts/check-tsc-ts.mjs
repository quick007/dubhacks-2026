import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { readdirSync, existsSync } from "node:fs";

const root = fileURLToPath(new URL("../", import.meta.url));
const compiler = path.join(root, "node_modules/tsc-ts/bin/tsc-ts.js");
// Run the requested compiler directly: Vite+ check uses its own TypeScript Go path.
for (const group of ["packages", "apps"]) {
  for (const entry of readdirSync(path.join(root, group), { withFileTypes: true })) {
    const config = path.join(group, entry.name, "tsconfig.json");
    if (!entry.isDirectory() || !existsSync(path.join(root, config))) continue;
    console.log(`tsc-ts: ${config}`);
    const result = spawnSync(process.execPath, [compiler, "-p", config, "--noEmit"], {
      cwd: root,
      stdio: "inherit",
    });
    if (result.error) throw result.error;
    if (result.status !== 0) process.exit(result.status ?? 1);
  }
}
