# Frozen paired shopper cohort

`src/shopper-cohort.ts` runs 60 synthetic shopper pairs against the real local
storefront. Each assignment opens a fresh headed Chrome process and storage
context. The shopper uses screenshots and visible controls through the installed,
signed-in Codex CLI (`gpt-6.1-sol`, low reasoning effort). Only its assigned public
query can enter store search. The private shopper needs and independent evaluator
gold remain separate.

The frozen design has 48 purchase-feasible and 12 negative scenarios per arm.
The first five randomized pairs form a runtime pilot and remain in the main
denominator. Every session has 24 actions and 120 seconds, including model waits.
Execution is sequential. The pilot does not authorize tuning or replacement.

The current experiment is declared in
`experiments/ranker/pretrained/store_shopper_eval_v2/cohort-preregistration.json`.
Six manifest-bound packets, `store_shopper_eval_v2` through `v7`, contain the
60 scenarios. Products are known to training; these are new scenario wordings
and private preferences, not an unseen-product evaluation.

## Runtime and launch

The frozen local runtime uses storefront `127.0.0.1:4312`, model service
`127.0.0.1:8766` and the local-only demo-checkout receiver `127.0.0.1:4331`.
The storefront must proxy `/api/demo-checkout` to that receiver. See
`../api/DEMO_CHECKOUT.md` for its standalone command. There are no payments or
real purchases. A checkout is confirmed only after the server accepts an
immutable receipt and the storefront displays its identity.

Coordinate a browser/GPU window with the model owner before launching. Freeze
the actual source files, images, catalog, checkpoints, owner READY receipt,
service health identity and three PIDs in a new runtime snapshot. The runner
checks these before dispatch and checks source hashes again after each session.
A service restart or changed source requires a new experiment; do not alter a
running experiment's snapshot to make its checks pass.

From the repository root, create a plan without opening Chrome:

```sh
node --import tsx apps/runner/src/shopper-cohort.ts plan \
  --packets-json '["experiments/ranker/pretrained/store_shopper_eval_v2","experiments/ranker/pretrained/store_shopper_eval_v3","experiments/ranker/pretrained/store_shopper_eval_v4","experiments/ranker/pretrained/store_shopper_eval_v5","experiments/ranker/pretrained/store_shopper_eval_v6","experiments/ranker/pretrained/store_shopper_eval_v7"]' \
  --snapshot experiments/ranker/pretrained/store_shopper_cohort_snapshot_v2.json \
  --out /tmp/new-shopper-cohort \
  --seed 20261018 --pilot-pairs 5 --deadline 2026-10-18T13:00:00Z
```

For that plan, change `plan` to `run --resume` with the same arguments. The
dated snapshot and deadline above identify the recorded experiment; they are
not a reusable future service configuration. Preserve the plan before execution.
Model availability is checked before browser dispatch. Missing CLI/login/model
support stops execution without a scripted substitute or copied credentials.

At the deadline, future dispatch stops. A signal also stops future dispatch;
the current bounded session retains its outcome. A valid existing receipt is
never rerun. An attempt lock without a valid final receipt is uncertain evidence
and blocks automatic resume rather than silently replacing the attempt.

Each completed pair saves an immutable checkpoint. Final aggregate files live
at each `packet-N` root so their relative session paths remain valid. All 120
assigned slots remain present, including timeouts, interruptions, invalid
sidecars and unattempted assignments. Large traces and intermediate screenshots
stay local; the normalized observations, final screenshot and artifact hashes
provide a compact committed record. Raw `browser.jsonl` stays local as well.

## Final evidence and metrics

Result-only utilities under
`experiments/ranker/pretrained/store_shopper_checkout_smoke_v1` prepare an
arm-label-blinded review of every negative assignment, verify artifact integrity,
and produce a compact slide projection. Reviewers use the preregistered rubric;
the separate audited aggregate may only lower unjustified abstention credit.
The original report and session evidence are preserved.

After the audit, verify receipts from the exact six audited report files:

```sh
node --import tsx apps/runner/src/shopper-server-outcomes.ts \
  --report PATH/packet-1/aggregate-audited.json \
  --report PATH/packet-2/aggregate-audited.json \
  --report PATH/packet-3/aggregate-audited.json \
  --report PATH/packet-4/aggregate-audited.json \
  --report PATH/packet-5/aggregate-audited.json \
  --report PATH/packet-6/aggregate-audited.json --out PATH/server-outcomes.json
```

This command performs fresh GET readbacks only. It matches the captured POST,
captured GET, immutable server receipt, catalog prices and artifact hashes; it
never creates an order. Run `src/shopper-metrics.ts` with the same repeated
`--report` arguments, one matching `--packet` per report,
`--server-outcomes PATH/server-outcomes.json`, and `--out PATH/shopper-metrics.json`.
Outputs are exclusive-created and bind their exact inputs.

Primary intent success combines valid checkout and justified abstention.
Report feasible checkout separately over 48 assignments per arm. Report valid
purchase merchandise revenue and all accepted server-order merchandise revenue
separately, both divided by all 60 assignments per arm. Nonpurchases contribute
zero; hidden delivery charges remain unknown. Preferences are a rubric proxy.
Measured elapsed time excludes partial spans and labels missing observations.
Paired intervals are conditional on these hand-authored, repeated-catalog cases.
The evidence cannot establish human conversion lift, retention, satisfaction
or real revenue.

Optional preselected A/A checks and reversed-order repeats may run only after
all 120 main receipts, with matching frozen runtime and at least 25 minutes
remaining. Their permanent claim prevents reruns under another output path.
Keep these eight assignments separate and retain their original case clusters.
The runner does not automatically export any cohort to a hosted database.
