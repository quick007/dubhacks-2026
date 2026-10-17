import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const environment = path.join(root, ".tools/ranker-venv");
const python = path.join(
  environment,
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);
const args = process.argv.slice(2);
const command = args.shift();
const env = {
  ...process.env,
  UV_PYTHON_DOWNLOADS: "never",
  UV_CACHE_DIR: path.join(root, ".cache/uv"),
  PIP_CACHE_DIR: path.join(root, ".cache/pip"),
  PYTHONPYCACHEPREFIX: path.join(root, ".cache/ranker/pycache"),
  OPENBLAS_NUM_THREADS: "1",
  OMP_NUM_THREADS: "1",
  VECLIB_MAXIMUM_THREADS: "1",
  MKL_NUM_THREADS: "1",
};
function run(binary, parameters) {
  const result = spawnSync(binary, parameters, { cwd: root, env, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
function available(binary) {
  return spawnSync(binary, ["--version"], { env, stdio: "ignore" }).status === 0;
}

if (command === "setup") {
  mkdirSync(path.join(root, ".tools"), { recursive: true });
  const uv = available("uv");
  const interpreter = process.env.RANKER_PYTHON || "python3";
  if (!existsSync(python)) {
    if (uv) run("uv", ["venv", "--python", interpreter, environment]);
    else run(interpreter, ["-m", "venv", environment]);
  }
  const requirements = path.join(root, "experiments/ranker/requirements.txt");
  if (uv) run("uv", ["pip", "install", "--python", python, "-r", requirements]);
  else run(python, ["-m", "pip", "install", "-r", requirements]);
  run(python, [
    "-c",
    'import sys,numpy; assert numpy.__version__ == "2.5.4"; print(f"Ranker ready: Python {sys.version.split()[0]}, NumPy {numpy.__version__}")',
  ]);
} else if (["data", "train", "evaluate", "infer", "serve", "benchmark", "test"].includes(command)) {
  if (!existsSync(python)) {
    console.error("Ranker environment missing. Run pnpm ranker:setup first.");
    process.exit(1);
  }
  const moduleArgs =
    command === "data"
      ? ["experiments.ranker.dataset.generate"]
      : command === "test"
        ? [
            "unittest",
            "experiments.ranker.dataset.test_generate",
            "experiments.ranker.model.test_common",
            "experiments.ranker.model.test_network_service",
          ]
        : ["experiments.ranker.model.cli", command];
  run(python, ["-m", ...moduleArgs, ...args]);
} else {
  console.error(
    "Usage: node scripts/ranker.mjs setup|data|train|evaluate|infer|serve|benchmark|test [options]",
  );
  process.exit(1);
}
