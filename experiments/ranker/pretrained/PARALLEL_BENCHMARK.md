# Bounded independent-field native-batch comparison

The comparison runner is prepared but has not run. The existing manifest baseline remains preserved: 7/22 complete valid pages overall, including 7/12 evidence-source pages, with 22/22 raw baseline replay in the last comparison. The locked-evidence serialization is not promoted.

`parallel_benchmark.py` freezes all twelve existing evidence-source cases, including the fixed 1/3/6-card scale cases. Each measured arm begins before the same user-first ranking prefill. Manifest whole-page generation is compared separately with factorized sequential generation and native batch widths 2/3/6. Identical factorized prompts and per-field budgets isolate concurrency; whole-page versus factorized prompts change another factor. Total output caps remain 64/128/224. Catalog-first cache conditions are excluded because substituting their state would change the ordered context.

The runner requires an actual passing tiny native compatibility receipt at `data/ranker/pretrained/parallel-decode-v1/native-compatibility.json` before constructing the pretrained backbone. No receipt exists. A metadata-only or fabricated pass must never replace the native cache tests. The expected gate covers complete recurrent/convolution/KV forks, branch independence, unequal suffixes, sequential-versus-batch token parity, early EOS and row compaction. The six-GiB process cap remains unchanged.

Each factorized arm must include state materialization, suffix prefill, native decoding, raw assembly and unchanged strict validation in complete latency. Every generated row's exact token IDs, stop reason, prompt hash and budget are compared with sequential output. Failed and unattempted arms remain in the fixed twelve-case denominator; rejected pages cannot count as usable-page speedups. The runner stops after any native failure or memory error rather than raising the allowance. Increasing width order introduces a disclosed warmth/order bias. These are inspected synthetic cases, not human conversion evidence or an untouched generalization benchmark.

Two CPU checks pass for token/prompt/budget comparison and denominators that retain failed, unattempted and rejected pages:

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_parallel_benchmark -v
```

The model command is gated and has not been executed:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.parallel_benchmark --model /absolute/verified/pinned/snapshot
```

## Exact approval blocker

Automatic review denied the worker's edit/test operation, then denied the single retry after new parent-supplied user transcript evidence. Neither attempt executed. The rejected action was to update `parallel_decode.py`, `test_parallel_decode.py` and `PARALLEL_DECODE.md` for separate title/badge/heading rows, fixed field budgets and continuous refill, then run the CPU suite and tiny native Metal cache-parity test. The retry rejection stated:

> The action modifies and tests a substantial native parallel-decoding implementation beyond the explicitly authorized scaffold, while the available quoted user remarks do not specifically authorize these code changes or native test side effects. Do not bypass this rejection through a workaround or indirect execution.

The existing draft therefore still has coupled title/badge rows and fixed waves. Its three CPU tests pass; its native test is skipped. No native GPU run, compatibility pass, batching speed measurement or quality result is claimed. Resolve authorization for those exact edits and native test before continuing. This separate orchestration source does not implement or execute the denied refinement.
