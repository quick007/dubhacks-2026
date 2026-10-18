# Persona choice attempt 1

This directory preserves 39 byte-identical JSON receipts from the first lexical
LoRA choice experiment, its query-intent head and independent audits.
`COPY_INDEX.json` records each original path, byte count and SHA-256. Original
archives and checkpoints remain intact in ignored local data directories.
Weights and numerical feature archives are not included here. The copied serving
manifest is provenance, not a portable serving bundle.

The experiment completed 1,240 feature gates and 3,584 optimizer updates. All
927 frozen backbone tensors remained unchanged, checkpoint reload and native
feature parity passed, and shutdown completed. These mechanical checks did not
establish useful personalization.

The corrected independent audit found 17/32 primary choices for both the raw
adapter and combined model. Their frozen-base comparisons scored 20/32 and
19/32 respectively. The candidate failed the unchanged quality gate and was not
promoted. The original audit's derived-case identity bookkeeping failure is
preserved alongside the corrected audit; the correction did not change labels,
thresholds or model outputs.

The inspected 32-case packet is now development data. Any subsequent fit must use
a separately frozen evaluation packet and retain this failure. Technical browser
QA of this candidate is explicitly a development demo, not an accepted model or
evidence of a relevance advantage.
