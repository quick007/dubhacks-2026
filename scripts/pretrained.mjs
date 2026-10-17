import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const environment = path.join(root, ".tools/pretrained-venv");
const python = path.join(
  environment,
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);
const repository = "mlx-community/Qwen3.5-9B-4bit";
const revision = "8b2b98c00a6b4d291155e4890773ca8f769aee53";
const cache = path.join(root, ".cache/huggingface/hub");
const snapshot = path.join(cache, "models--mlx-community--Qwen3.5-9B-4bit/snapshots", revision);
const args = process.argv.slice(2);
const command = args.shift();
const commands = [
  "setup",
  "data",
  "test",
  "test-mlx",
  "download",
  "verify",
  "smoke",
  "workflow",
  "infer",
  "fusion",
  "test-fusion",
  "fusion-benchmark",
];
const env = {
  ...process.env,
  UV_PYTHON_DOWNLOADS: "never",
  UV_CACHE_DIR: path.join(root, ".cache/uv"),
  PIP_CACHE_DIR: path.join(root, ".cache/pip"),
  PYTHONPYCACHEPREFIX: path.join(root, ".cache/pretrained/pycache"),
  HF_HOME: path.join(root, ".cache/huggingface"),
  HF_HUB_CACHE: cache,
  HF_HUB_DISABLE_IMPLICIT_TOKEN: "1",
  HF_XET_HIGH_PERFORMANCE: process.env.HF_XET_HIGH_PERFORMANCE || "1",
  OPENBLAS_NUM_THREADS: "1",
  OMP_NUM_THREADS: "1",
  VECLIB_MAXIMUM_THREADS: "1",
  MKL_NUM_THREADS: "1",
};
function run(binary, parameters, timeout) {
  const result = spawnSync(binary, parameters, {
    cwd: root,
    env,
    stdio: "inherit",
    timeout,
    killSignal: "SIGTERM",
  });
  if (result.error?.code === "ETIMEDOUT") {
    console.error("Pretrained workflow exceeded the 600-second total wall limit and was stopped.");
    process.exit(1);
  }
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
function available(binary) {
  return spawnSync(binary, ["--version"], { env, stdio: "ignore" }).status === 0;
}

// Hugging Face artifacts are data only. No downloaded Python or pickle weights are executed.
const artifactScript = String.raw`
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
command, repository, revision, cache, expected_snapshot = sys.argv[1:]
snapshot = Path(expected_snapshot)
if command == "download":
    from huggingface_hub import snapshot_download
    location = snapshot_download(repo_id=repository, revision=revision, cache_dir=cache,
        token=False, allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "vocab.txt", "merges.txt", "chat_template.jinja", "README.md", "LICENSE*"],
        ignore_patterns=["*.py", "*.bin", "*.pt", "*.pth", "*.pkl", "*.pickle"])
    if Path(location).resolve() != snapshot.resolve():
        raise RuntimeError("Downloaded snapshot differs from the pinned revision path")
if not snapshot.is_dir():
    raise SystemExit("Pinned snapshot is missing. Run pnpm pretrained:download first.")
unsafe = [str(p.relative_to(snapshot)) for p in snapshot.rglob("*") if p.is_file() and p.suffix in {".py", ".bin", ".pt", ".pth", ".pkl", ".pickle"}]
if unsafe:
    raise RuntimeError("Unexpected executable/pickle artifacts in data-only snapshot: " + ", ".join(unsafe))
config = json.loads((snapshot / "config.json").read_text())
quantization = config.get("quantization", config.get("quantization_config", {}))
if quantization.get("bits") != 4:
    raise RuntimeError("Expected a 4-bit quantized checkpoint")
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
shard_names = sorted(set(index["weight_map"].values()))
expected = {
    "model-00001-of-00002.safetensors": {"bytes": 5349771222, "sha256": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd"},
    "model-00002-of-00002.safetensors": {"bytes": 600449850, "sha256": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d"},
}
expected_names = sorted(expected)
if shard_names != expected_names or sorted(p.name for p in snapshot.glob("*.safetensors")) != expected_names:
    raise RuntimeError("Expected exactly the two pinned 4-bit safetensors shards")
artifacts = []
for name in shard_names:
    artifact = snapshot / name
    digest = hashlib.sha256()
    with artifact.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = {"bytes": artifact.stat().st_size, "sha256": digest.hexdigest()}
    if actual != expected[name]:
        raise RuntimeError("Artifact size/SHA-256 mismatch: " + name)
    artifacts.append({"filename": name, **actual, "expected": expected[name], "verified": True})
manifest = {"schemaVersion": 1, "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
    "repository": repository, "revision": revision, "format": "MLX 4-bit safetensors",
    "snapshotPath": str(snapshot), "quantization": quantization, "tokenUsed": False,
    "remoteCodeExecuted": False,
    "verification": "Both shard sizes and SHA-256 values match Hugging Face API LFS metadata for the pinned community repository revision.",
    "provenance": {"kind": "community MLX quantized conversion", "declaredSource": "Qwen/Qwen3.5-9B", "declaredConversionTool": "mlx-vlm 0.3.12", "originalUpstreamRevision": None,
        "limitations": ["The community model card base_model points to Qwen3.5-9B-Base while README conversion source says Qwen3.5-9B.", "Matching downloaded shard hashes verifies these published artifacts, not upstream weight identity or reproducibility of the conversion."]},
    "artifacts": artifacts}
output = Path("data/ranker/pretrained/qwen3.5-9b-4bit-artifacts.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"manifest": str(output), **manifest}, indent=2))
`;

if (!commands.includes(command) || args.includes("--help") || args.includes("-h")) {
  console.log(
    `Usage: node scripts/pretrained.mjs ${commands.join("|")} [options]\n\nsmoke/workflow/infer default to the pinned local Qwen3.5-9B-4bit snapshot, 1536 shared-prefix tokens and a 6 GiB budget.\nworkflow has a 600-second total wall limit. --model/--revision are fixed to this checkpoint.\ndata forwards generator options; smoke/workflow/infer forward experiment CLI options.\nsetup uses PRETRAINED_PYTHON or an existing python3; it never installs a system Python.\ndownload uses anonymous data-only Hugging Face access and verifies/hashes both safetensors shards.\nverify only reads the pinned local snapshot and writes the ignored artifact manifest.\nHF_XET_HIGH_PERFORMANCE defaults to 1; set it to 0 to disable.`,
  );
  if (!commands.includes(command) && !["--help", "-h"].includes(command)) process.exitCode = 1;
} else if (command === "setup") {
  const requirements = path.join(root, "experiments/ranker/pretrained/requirements.txt");
  if (!existsSync(requirements)) {
    console.error(
      "Pretrained requirements are not ready. Wait for the model module owner before setup.",
    );
    process.exit(1);
  }
  mkdirSync(path.join(root, ".tools"), { recursive: true });
  const uv = available("uv");
  const interpreter = process.env.PRETRAINED_PYTHON || "python3";
  if (!existsSync(python)) {
    if (uv) run("uv", ["venv", "--python", interpreter, environment]);
    else run(interpreter, ["-m", "venv", environment]);
  }
  if (uv) run("uv", ["pip", "install", "--python", python, "-r", requirements]);
  else run(python, ["-m", "pip", "install", "-r", requirements]);
} else {
  if (!existsSync(python)) {
    console.error("Pretrained environment missing. Run pnpm pretrained:setup first.");
    process.exit(1);
  }
  if (["download", "verify"].includes(command)) {
    if (args.length)
      throw new Error("download/verify use the fixed checkpoint and accept no options.");
    run(python, ["-c", artifactScript, command, repository, revision, cache, snapshot]);
  } else if (command === "data") {
    run(python, ["-m", "experiments.ranker.dataset_v2.generate", ...args]);
  } else if (["test", "test-mlx", "test-fusion"].includes(command)) {
    const modules =
      command === "test"
        ? [
            "experiments.ranker.dataset_v2.test_generate",
            "experiments.ranker.pretrained.test_pretrained",
          ]
        : command === "test-mlx"
          ? ["experiments.ranker.pretrained.test_mlx_compat"]
          : ["experiments.ranker.pretrained.test_fusion"];
    run(python, ["-m", "unittest", ...modules, ...args]);
  } else {
    if (args.some((arg) => /^(--model|--revision)(=|$)/.test(arg))) {
      throw new Error(
        "This wrapper fixes model/revision to the pinned 4-bit checkpoint. Use the experiment CLI directly for another explicit research configuration.",
      );
    }
    run(
      python,
      [
        "-m",
        command === "fusion-benchmark"
          ? "experiments.ranker.pretrained.benchmark"
          : command === "fusion"
            ? "experiments.ranker.pretrained.fusion"
            : "experiments.ranker.pretrained.cli",
        ...(["fusion", "fusion-benchmark"].includes(command) ? [] : [command]),
        "--model",
        snapshot,
        "--revision",
        revision,
        ...(["fusion", "fusion-benchmark"].includes(command)
          ? [
              "--heads",
              path.join(root, "data/ranker/pretrained/shared-prefix/adaptation"),
              "--max-input-tokens",
              "1536",
            ]
          : [
              "--encoding",
              "shared-prefix",
              "--run",
              path.join(root, "data/ranker/pretrained/shared-prefix"),
              "--max-tokens",
              "1536",
            ]),
        "--memory-gib",
        "6",
        ...args,
      ],
      ["workflow", "fusion-benchmark"].includes(command) ? 600_000 : undefined,
    );
  }
}
