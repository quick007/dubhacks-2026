# Run the non-promoted development demo

This separate entry starts the local personalized UI using the technically
verified existing adapter. Its quality gate failed17/32 against frozen-base20/32;
model promotion and accepted-profile status remain false. The accepted-only
launcher and profile parser are unchanged.

After root merges the prepared code and copies the required model bundle below,
run this one command from main:

```sh
node scripts/dev-personalized-development.mjs --run \
  --receipt data/ranker/pretrained/persona-development-admission-v1-attempt-1/receipt.json \
  --receipt-sha256 d774bc8e0e5dcd09bee705c4658817e07703201bd72888dc4577c94d3b12fa58
```

Omit `--run` for preflight only. Preflight contacts no running model and creates
no session directory. It refuses occupied8766/8767/4313 ports and a held original
native lock; it never takes over a process. Existing UI/model sessions must be
stopped normally by their owner before this entry can run. An explicit
`--asset-root` may select the existing main asset repository; its lexical venv
Python path is preserved. `--session` must be an absolute path under the current
code root and the process runner refuses an existing session directory.

All runtime arguments resolve to the selected main asset root and current code
root. No runtime argument falls back to worktree weights, caches or UI files.
Missing copied assets fail even if historical originals remain valid. The
historical admission/evidence paths remain explicit provenance: the verifier
checks their bytes in the preserved evidence tree, then verifies every required
Python source byte against the current code root. This is source equivalence,
not a rewritten historical quality result.

Root must copy these exact bytes from the integration worktree into the SAME
relative paths under main ignored data, preserving the originals:

- `data/ranker/pretrained/persona-choice-train-v1-attempt-1/serving.json`,
  `inference-parity.json`, `fit/adapter.safetensors`, and
  `public-features/index.json` plus all20 referenced co-located `.npz` files.
- `data/ranker/pretrained/query-intent-gate-v3-fit-attempt-1/head.pt`.
- `data/ranker/pretrained/persona-qwen-optimize-v2-attempt-1/raw-2500/head.pt`.
- `data/ranker/pretrained/persona-qwen-fit-v1-attempt-1/public-demo-features.pt`.
- The development `receipt.json` above, unchanged, for the convenient command.

The reviewed integration manifest
`experiments/ranker/pretrained/store_demo_integration_personalized_adapter_v2/manifest.json`
is a current-code artifact and must have the same reviewed16acf... bytes after
merge. Original v3 catalog features, v5 serving export/head, native copy adapter,
public48 parallel copy assets and frozen local Qwen shards already live under
main. None is downloaded or retrained by this command. All20 copied public X/H/Y
archives and the co-located certificate are hashed before spawn. The native
engine still verifies its full model/checkpoint/source/927 integrity.

The new launcher reuses the existing generic owned-process runner, port/lock and
verified checkout helpers, and exact adapter argument builder. It supplies a
separate explicit development policy instead of invoking or relabelling the
accepted-profile selector. It starts one native worker8767, verified local
checkout4331 (or reuses a declared checkout on4331/4311), and Vite4313 with the
trusted personalized-store mode. The ordinary bare UI is
`http://127.0.0.1:4313`. No browser is opened automatically.

The console and `launch.json` preserve failed quality/promotion false. Sessions
are unique under `data/personalized-development/`; logs, native request outputs,
checkout receipts and shutdown results are preserved. Ctrl+C gracefully stops
only owned children. A reused checkout is left alone. The launcher exit is not
927-integrity proof: retain the native `shutdown.json` receipt. No `.env` is read,
no new credentials or paid/network compute are requested, and Go/Hugging Face
remain offline.

Ten CPU tests use injected reads, evidence subprocess, ports, lock, checkout and
process runner. They cover source equivalence after merge, exact main-only
runtime arguments, missing-main rejection, all20 public archives, no accepted
alias, busy-worker failure before subprocesses, explicit run, and source changes
before child startup. They start no actual process, socket, native model or UI.

```sh
node --test scripts/dev-personalized-development.test.mjs
```
