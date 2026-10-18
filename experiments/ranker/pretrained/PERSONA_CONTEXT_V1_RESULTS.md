# Public persona context candidate v1

The fixed CPU experiment completed against the actual Tiger readback export. It uses genuine frozen Qwen catalog vectors, with a separate hashed public-context encoder. It does **not** encode persona text with Qwen or adapt the backbone.

- Inputs: 160 TRAIN sessions from 80 groups, 40 family/group-heldout sessions from 20 groups. Ten named demo people excluded from fit. All 16 catalog identities were seen in TRAIN; heldout primary choices cover five identities and five authored families.
- Training: 250 full-batch AdamW updates, seed 20261018, lr .003, weight decay .0001, plus the separately predeclared 250-update one-hot catalog ablation. No selection or schedule changes from heldout outcomes.
- Actual Tiger public export SHA: `c36a54ebf54021389486f594275258852fbe460cb0933f8f9701f93c939c25f9`. Raw exported JSON is key-sorted but the public profiles compare deeply equal to the frozen authored seed.
- Model SHA: `12629e50562305ffe1f5e7497f02e649f10e0d14d149e5f46115bd6b5561310e`. Exact state reload and TRAIN prediction parity passed. All checkpoints and outputs remain in `data/ranker/pretrained/persona-context-v1-attempt-1/`.

| Fixed40 evaluation | Novel relevant first choice | Exact highest-grade first choice | NDCG@6 |
|---|---:|---:|---:|
| Context + Qwen catalog | 24/40 | 1/40 | .696 |
| One-hot catalog ablation | 27/40 | 3/40 | .671 |
| No history | 22/40 | 8/40 | .735 |
| No profile | 8/40 | 0/40 | .335 |
| Shuffled profile | 11/40 | 0/40 | .325 |
| TRAIN popularity | 16/40 | 0/40 | .373 |
| Most recent owned item | 0/40 | 0/40 | .272 |

TRAIN reaches 159/160 highest-grade first choices and 160/160 novel relevant first choices. Convergence is strong, but specificity transfers poorly. The Qwen catalog representation does not show an advantage over fixed identity codes here.

A separate controlled history diagnostic froze 20 predictions before its private judgments were parsed. Changing one purchase to the formerly recommended item reduced that item's learned rank in **0/10** pairs; the newly owned item remained first in all ten. There is no deterministic owned-item filter. Plausible named-demo recommendations alone therefore do not establish learned purchase-history complementarity. This candidate is not promoted.

The next preregistered candidate uses real frozen Qwen public-context encodings, 160 disclosed TRAIN-only ownership contrasts, and a shared ownership coefficient initialized at zero and optimized from TRAIN labels. The old40 and old history probes are development replay from this point onward. A new independent32 packet remains separate; its private judgments must be parsed only after checkpoint and all predictions/baselines are frozen. Native extraction is root-owned and requires the shared main worker lock. No native or browser execution occurred in this lane during preparation.
