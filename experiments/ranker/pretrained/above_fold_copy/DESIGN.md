# Prospective ten-field first-section producer

This package prepares one bounded model-only endpoint and synthetic supervision.
No model/native generation, optimizer update, UI/API edit or external integration
was performed by this lane. The separate queued 504-row fourteen-field STB fit,
its data, prompt functions and source files are unchanged.

## Layout, assembly and cache ownership

`layout.build_layout(full_plan, request, catalog)` returns a fresh observable-root
message, ten independent jobs, and public evidence. Jobs are S1/T1/B1/T2/B2/T3/B3/
C1/C2/C3 with caps 16/24/8/24/8/24/8/32/32/32, summed **208**. S1 receives all
three members and exact universal tags. T/B/C receive the same selected card's
public facts, approved-title decomposition, selected text-described image and
supported observed tags. The exact catalog identity is also explicitly bound as
`originalNameSubtitle`. No row sees another row's generated output, teacher
answer, opaque product/image ID, family metadata or hidden preference.

C is genuine same-backbone fresh generation in the prospective native runner.
The layout is not an answer generator: it creates requests only. Its independent
C rule requires one short factual sentence, current 10–180-character copy
limits, native stop, no markup/labels/multiline text, observed support in the copy
itself when available, and no literal catalog-description/category/image-text
fallback. `assemble_payload` joins the returned raw values with exact trusted
catalog/image/order/name/price/fee metadata and delegates v1 validation. It never
strips or repairs values, guesses missing C, supplies teacher text, or invents
`NONE`. Optional native-receipt checking verifies actual sampled steps including
EOS stay within each cap and agree with AR/token receipts. Native mode enables
that check; teacher grammar audits use disclosed synthetic stop only.

The native owner must begin from a pristine empty cache under the **current
adapter configuration**, then use the exact observable root. There is no ranking
prefix in the copy context. `fork_root` verifies model object, root messages,
complete tokens, token hash and unconsumed state, and delegates the existing
complete `from_state(deepcopy(entry.state))` snapshot for every cache entry.
Recurrent, convolution and KV/history state are all preserved; no arbitrary
trim, splice or adapted-prefix substitution is allowed. CPU fake-cache tests
prove nested ownership only; actual native alias/materialization/logit controls
remain mandatory. Do not infer native equivalence from CPU tests.

The prospective runner creates a new root inside each frozen/adapter context
and destroys it before changing contexts. Its controls also bind a mode receipt.
It performs no adapter updates, so no root crosses a weight configuration change.
Card-fact prefix reuse is **not implemented**: fact suffixes deliberately repeat
for T/B/C. No vendor throughput promise or measured speedup is implied.

## Frozen teacher extension and its limits

`manifest.json`, `AUDIT.json` and six JSONL files pin a strict projection of
`semantic_copy_pairs`: 36 training views/360 fields, 3 development views/30 fields,
3 test views/30 fields. Only the original first three cards are used. Catalogs,
requests, names, image choices, original full six-card fixture plans and parent
split membership are retained unchanged. `pendingPagePlan` is explicitly the
three-card validation projection; `fullSourcePagePlan` and the original requested
count remain available. This is not a changed full-page objective.

The seven visible STB teacher values are retained exactly. C targets are audited
synthetic grammar (`A {selected factual tag} daypack for everyday travel.`), using
an observed supported tag when present and otherwise an actual selected tag.
They are **supervision only**, never runtime inference text or native output.
All 42 teacher payloads pass strict current title identity/factuality/novelty,
wire/body policy and exact subtitle/catalog binding. There are 108 training C
examples but only 17 distinct C sentences; repeated copy grammar across splits
is an intentional leakage/coverage limitation. Exact non-NONE target-string
overlap is train/dev 9, train/test 11, dev/test 4. No unseen wording claim applies.

Projection retains 42 positively supported training C examples across 17 atoms;
`organizer` has no positive observed C target above fold. Sixty-six other
training C examples assert factual catalog attributes without observed support,
and their badges are required `NONE`. Only 12/18 paired families retain a shared
changed product in the first three cards. Visible paired products change T, B
and C targets across the observed preference change. These losses are disclosed,
not repaired by reordering or substituting cards. Development/test each contain
six supported and three other C examples, three required `NONE` badges, two
positive-universal headings and one empty-intersection heading.

Parent development/test fixtures were inspected during STB data preparation;
they are reused, reviewed-parent cases, not newly sealed holdouts. This lane has
performed no inference on them. Their targets cannot enter a fit, prompt search,
checkpoint selection or early stopping. Names/request combinations are disjoint
according to the original parent inventory, but this extension creates no new
independence claim or production/human validation.

`fields.*.jsonl` uses the existing field-record interface: `messages` is the
complete three-turn root/fact/request input, `targetValue` is assistant-only
supervision, and scheduler identity/cap/kind plus source case/family metadata stay
outside model input. Never serialize the entire record into a prompt. The
producer's layout module never reads these target files. The original 504-row
fit does not consume C examples and cannot claim C training coverage.

## CPU token preflight and bounded native plan

Eight CPU tests pass across `test_layout` and `test_benchmark`. Tests cover ten
jobs/caps, repeated card-fact equality, raw-value rejection, independently
responsive C, EOS-inclusive sampled caps, full nested cache snapshots, root
identity rejection, exact rendering, fixed lane counts and completed-fit gates.
The real pinned local tokenizer preflight is `TOKEN_PREFLIGHT.json`; it executed
no model calls. It checked all 360 training and 60 development/test target
contexts, actual EOS-inclusive target fit, exact serving/teacher prompts and
untruncated root/field boundaries. Maximum training teacher context is 969
tokens; maximum evaluation prompt plus cap is 986, maximum suffix 601, maximum
evaluation teacher target including EOS 15. This is token fit, not native quality.

`benchmark.py` is prospective native code; **its run action has not been called**.
The agreed fixed comparison is six reviewed parent dev/test cases × widths
1/3/6 × frozen and final paired-STB adapter, **36 arms**, all ten identical jobs
and caps. It includes positive-universal and empty-intersection cases. It
compares width-dependent outputs semantically under unchanged v1 policy; exact
cross-width token equality is not a correctness requirement.

Before any timing arms, each model mode runs fresh-root controls on the first
fixed case at actual widths 1, 3 and 6. Every width has exactly N real distinct
jobs, not a three-row job mislabeled width six. Width one uses C1; width three
uses S1/T1/C1; width six uses S1/T1/B1/C1/T2/C2. All controls use natural greedy
generation capped at seven sampled steps, chunk256, and fork/independently
rebuilt/repeat runs. Their purpose is state/distribution control, not completed
quality. Same-width sampled tokens, stops, AR counts and normalized distribution
hashes must match exactly. Source root states must remain exact. A mismatch is
fatal, saves strict failure evidence and stops before timing. Controls and their
cumulative peak are disclosed separately from the fixed quality denominator.

The adapter freeze gate accepts only a completed final 504-update report,
completed/evaluated optimizer receipts, unchanged-base proof, learned-cache
proof and exact final checkpoint hash. An incomplete/interim checkpoint is
rejected. Until that fit finishes, there is no final adapter-bound benchmark
packet. The freeze command can then pin the completed report/checkpoint alongside
source/data/runtime/tokenizer receipts for root review. Native loading freshly
verifies the pinned local model bytes; it never downloads a model. The native
active **and** peak guard is 7 GiB; controls contribute to the cumulative peak.
Root must serialize this worker after the current fit and review its budget.

Every timed attempt starts before layout/metadata/root preparation, generates
all ten fields, validates the complete payload, and serializes the exact payload
before recording ready time. Invalid views get no accepted ready-time sample.
The receipt binds payload hash and actual metadata/validated-serialized times.
No callback is needed for this first fixed three-card endpoint: all ten jobs
are required and all must finish. The old decoder's per-row EOS timestamps are
also retained. The outer decoder-call bridge is explicitly only a conservative
last-required-token lower bound because internal BatchGenerator initialization
precedes its own row clock; the accepted ready metric is measured directly
after actual validation/serialization, not inferred from that lower bound.

This experiment is copy-stage payload handoff with locked recorded decisions.
It excludes ranking and browser asset/render/paint. It does not compare dropping
below-fold fields against the old fourteen-field budget as a speedup, claim
full-page validity, establish conversion lift, or implement new full-grid
cardinality. Job wall includes load/controls/attempts after preflight and fresh
model-byte verification; those CPU preparation steps are outside timed attempts.
The reused adapter is explicitly labeled `paired-STB-adapter; no C training`.

## Distinct prospective subtitle-bound title policy v2 (design only)

The user's title freedom can be supported later by a separately versioned
`first-section-subtitle-title.v2` policy. It is **not implemented here** and
changes neither `above_fold.py/v1`, legacy validators, this initial anchored
teacher corpus, nor the queued 504-row fit.

Before considering a v2 title, validate exact product eligibility, ordered
catalog binding, compatible selected image/key, exact originalName subtitle,
price/fee metadata and exact-name query constraints. The subtitle must be joined
from the trusted selected catalog row, never supplied or guessed by the model.
A missing/stale/other-product subtitle or image binding rejects the title/view.
Once that binding is proved, permit a new descriptive title without repeating
the product-name anchor; factual selected attributes and ordinary framing may
provide the title. Generic framing is allowed only under that proven exact
subtitle binding. It must not erase an explicitly named query product.

All current raw one-line/stop, forbidden-claim, selected factual lexicon,
character-length, approved-title novelty and observed-preference support checks
remain. A supported compact card cannot get an unresponsive generic title merely
because its name is in the subtitle; it still needs compact support in the
title. Numeric claims remain restricted to current selected catalog identity
facts. Body, badge and universal-heading policies remain strict. Exact-name
queries must show the correct original name; partial/ambiguous query resolution
still needs its own public contract.

V2 would require an isolated title predicate and payload/version receipt, tests
for all trusted-binding failures and explicit v1/v2 metrics. It must not silently
bypass `identity_errors` inside legacy functions or rewrite historical pass
rates. A future benchmark must prospectively declare v2 before inference. Any
historical v2 rescore would be separately labeled policy-only diagnostic evidence,
never a newly measured model result. Root review is required before implementing
this policy; the initial ten-field producer remains strict v1.

## Commands for the owning root, after review

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.above_fold_copy.data check
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.above_fold_copy.test_layout experiments.ranker.pretrained.above_fold_copy.test_benchmark -v
# Final fit must already be completed; use a new packet destination.
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.above_fold_copy.benchmark freeze --preflight experiments/ranker/pretrained/above_fold_copy/TOKEN_PREFLIGHT.json --fit-report <completed-fit>/results.json --packet <new-packet.json>
# Separate root-authorized native launch only, after worker/budget review:
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.above_fold_copy.benchmark run --packet <frozen-packet.json> --out <new-result-directory>
```
