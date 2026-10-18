# Performance work and frozen evaluation boundary

The user resumed work and explicitly asked to keep reducing useful latency,
including costs below the earlier193ms backend milestone. Code and evidence are
pushed through45e2ff5; checkout and the context-only training bridge are merged
through55daca6. No deployment, paid compute or new credentials are involved.

## Completed checkpoints

The ALL48 parallel copy head remains a supervised catalog demo: fresh learned
word/EOS predictions from cached public Qwen fact features, no response-text
cache or runtime teacher lookup. All32 prior timed requests passed20 factual
fields;193.50ms was the older backend median, excluding UI/images/startup.

Balanced function training adds64 public paraphrases to32 original TRAIN rows.
The fixed80-step v5 fit used96 new native forwards plus48 verified reused
vectors. Independent replay confirms29/32 top1 and32/32 top6 versus frozen-v3
14/32 and29/32 on the same independently authored requests. All927 tensors
remain unchanged. Original reports/checkpoints are immutable; a miss-rank-only
reporting bug has separate corrected receipts. See function_v5_audit_v1/.

Responsive image transfer dropped96.47% in the matched historical browser
workload, while its cold median did not improve. A compact runtime resolver
then removed151KB minified JavaScript. Latest v5/compact navigation8/8 and
loaded collection4/4 verify raw responses, DOM fields, decoded images and
source/checkpoint identities. Loaded first3 fully visible median269.95ms;
cold/reload navigation through full text plus6 images533.10/362.40ms. These
small samples are not tail-latency or human-conversion evidence.

## Current work

1. Finish the isolated fast-path host profile:20 counterbalanced pairs on the
   four existing public browser queries. Separate ordinary vs instrumented
   requests, account observer overhead, compare exact scores/actions/copy slots
   and retain startup/shutdown integrity. Attempt1 stopped during excluded
   instrumentation warmup;40 timed arms remained unattempted and927/checkpoint
   checks passed. Attempt2 fixes delegated-tokenizer restoration only.
2. Measure generic versus local request-to-useful-content delay with20 paired
   loaded collection observations per arm. Fix AB/BA order, explicit browser
   cache eligibility, matching two-frame visibility/decode endpoint, actual
   POST/raw-body provenance and p50/p95 methodology. Generic static text must
   never be represented as20 generated fields. Preserve failed slots/outliers.
3. Freeze v11 plus the server-confirmed demo-checkout hook, health identity,
   model/export/assets/source hashes and process IDs for the60-pair shopper
   cohort. Keep weights, features, storefront and server behavior unchanged
   throughout that cohort. Root owns GPU windows; harness owns shopper results.
4. Prepare isolated optimizations from measured critical-path cost: resident
   verified catalog tensors/projections and fixed public query-prefix reuse.
   Prefix reuse has26 cached tokens and43–52 fresh suffix tokens on32 public
   controls, full KV/SSM cloning and strict numeric/ranking gates. Ten CPU tests
   pass; no native parity or speed result exists. Never claim reused response
   text as fresh generation or patch the frozen cohort worker.

The heldout pointer decoder improved0/126 to102/126 card fields but remains
unpromoted. Its24 failures concentrate in multiword-color copy gating; coverage
analysis proposes separately labeled TRAIN-only auxiliary facts and copy
alignment supervision. No augmented fit has run. The compact fresh-AR diagnostic
is prepared but deferred behind the useful fast path; its older binding must be
reviewed before any native run. DFlash remains research without installation.

## Evidence scope

The original20-session suite remains generic8/10 expected outcomes versus old
local7/10; it does not evaluate the current parallel/v5 snapshot. New cohort
conversion and revenue are synthetic, server-confirmed demo outcomes. Effort,
constraint satisfaction and speed are separate metrics, not human conversion.
Telemetry-to-consumer verification currently has34 synthetic observations and
four positive choices in two sessions; default real export contains zero rows.
Its CPU protocol vectors are not Qwen features, and no live model was retrained.
