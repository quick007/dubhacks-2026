# Clear adjudicator v2

V1 source, tests, docs and freeze are preserved. Independent source review found its inherited combined verifier could admit264 rows while the new fused DEVELOPMENT history check needs344. V2 adds an explicit all-four-model344-row check before the admission receipt and before private quality parsing. This is an evidence admission correction before any new candidate outputs; scores, labels, denominator32,20/24/28 thresholds, five strict controls, replacement3/4 and history8/8 remain unchanged.

Eight CPU tests pass, including a264-row failed admission that verifies no private label read and all16 inherited fake math/denominator tests. No actual new model predictions or quality JSON was parsed. V1 remains an unusable prelaunch draft.

```sh
OMP_NUM_THREADS=1 /Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.audit_persona_choice_clear_combined_v2 --adapter-run data/ranker/pretrained/persona-choice-clear-v2-attempt-1 --combined-run data/ranker/pretrained/NEW-COMBINED-RUN --out data/ranker/pretrained/persona-choice-clear-independent-adjudication-v2-attempt-1
```

Run only after root releases the complete new raw/fused all-model freezes. Exit2 means completed offline adjudication;1 means admission/audit failure. Offline results never imply serving promotion. Mechanical scope remains976historical+264fresh, not1240fresh gates. See preserved v1 documentation for the objective and initial-hash anchor.
