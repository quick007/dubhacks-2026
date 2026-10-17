# Scheduler-owned field identity quality experiment

Deterministic scheduler identity removes the observed label-generation failures, but does not establish usable fresh storefront pages. Values-only transport passes 42/42 rows versus 22/42 for generated labels. Both protocols remain 0/3 accepted pages overall and 0/2 on fresh synthetic holdouts. Complete latency remains roughly 8.3–9.0 seconds, with no demonstrated usable-page speedup.

## Protocol and unchanged content policy

The experiment changes who serializes field IDs. Each native UID remains bound before generation to its existing heading/title/badge row and fixed product/section. The model generates only one fresh raw value. The scheduler encodes exactly `trustedField + ": " + rawValue` for the existing assembler/validator. Rawvalues, sampled token IDs and finish reasons are preserved separately from canonical wire rows. These host labels are syntax, not model-generated wording or sampled tokens.

Empty/whitespace-only values, any line separator including a trailing newline, and embedded generated S/T/Bnumericlabels (case-insensitive and NFKC-detected) fail the transport contract. Nothing is stripped, guessed, defaulted to NONE or repaired after the fact. The old a7bdf19 outputs remain evaluated under their original label-generating protocol; no prior bare output was retrospectively rescued. All products/images,14fields, per-field caps totaling 224 sampled steps, width three, native 256 chunks, quantization, trained heads and strict factuality/novelty/observed-support/global-completion validation stay fixed.

`NONE` was already optional with observed support and required without support. A raw NONE value is present data, not a missing value. Prior bare NONE rows failed the required-label serialization contract; some other otherwise correct fields were also rejected by the unchanged global non-stop policy. Those mechanisms are recorded separately. A factual product tag without observed support still fails a NONE-required badge; no optional badge is forced merely to increase personalization counts.

## Frozen diagnostic and fresh synthetic holdouts

Before native inference, commit 0f632d3 froze the protocol and all inputs. One prior six-card diagnostic provides exact a7bdf19 labeled-output replay. Two additional six-card synthetic cases contain 12 new product IDs, 24 new variant IDs, new names and tag combinations absent from the old 84-product catalog. They are not old test cases renamed as holdouts. All new facts are explicitly supplied; new attributes make this a harder/different distribution. Frozen-head ranking/image utility has no new oracle labels and is not certified by generation or compatibility checks.

The no-history/no-preference case requires all six badges NONE and has packable as a true attribute of every section member. The explicit compact/durable preference case has four supported title/badge targets and twoNONE-required cards; every three-product section has an empty attribute intersection. Prefix lengths 1431/1468 fit the existing 1536 ranking guard. Every case requests the full six unique products and all fourteen fields. No holdout was dropped or used for later protocol tuning. These are fresh synthetic protocol holdouts, not a sealed independent production benchmark.

## Usable-page rate and full completion

| Case | Generated-label full time | Scheduler-identity full time | Valid pages, labels / scheduler | Protocol-valid rows, labels / scheduler | Accepted titles / badges / headings, labels → scheduler |
| --- | ---: | ---: | --- | --- | --- |
| evidence-page-scale-6 | 8644.7 ms | 8278.2 ms | 0/1 / 0/1 | 9/14 / 14/14 | 3/2/0 → 5/4/1 |
| scheduler-holdout-none-control | 8508.6 ms | 8369.2 ms | 0/1 / 0/1 | 6/14 / 14/14 | 1/0/1 → 6/0/0 |
| scheduler-holdout-explicit-preferences | 8968.9 ms | 8991.9 ms | 0/1 / 0/1 | 7/14 / 14/14 | 0/0/0 → 2/4/0 |

Fresh-holdout full-completion medians are 8738.8ms with generated labels and 8680.5ms with scheduler identity (observed difference 0.67%). These are two single attempts per protocol, not a reliable speed estimate. Neither arm has a valid-page latency sample. Overall and fresh-holdout denominators remain separate; six completed attempts are not six independent cases.

Full wall begins before public-input/ranking work and includes trained decisions, evidence-prefix preparation, complete cache forks, all fields, unchanged validation, wire encoding and response JSON serialization. It excludes one resident model load and report IO, and includes diagnostic state hashing/tokenization. Model-cold/full 12/24-card latency is unmeasured. Actual response serialization costs 0.03–0.04 ms. Sharing evidence still requires about 1.3–1.4 seconds of preparation and about 4.5–5.2 seconds of native field work, plus roughly 1.9–2.0 seconds ranking/structure. Removing labels does not make those costs disappear.

All 42 scheduler-value rows stop naturally; generated-label rows have 41 EOS stops and one length-capped row. Sampled response steps drop 353→306; text tokens excluding EOS 312→264. Host labels free some cap capacity for content, so this is a transport/payload change rather than identical generated text with a parser tweak. The slightly longer values-only instructions raise aggregate field suffix tokens 504→630 and add 15 shared-prefix tokens per case, offsetting some output work. No sampling, content templates or completed-response cache was introduced.

## Actual remaining content failures

Replay case:

```text
S1: Compact durable choice for everyday travel
S2: Simple daily choice for everyday travel
T1: Grove 23L Daypack for compact everyday travel
B1: compact
T2: Cedar 26L Daypack for compact durable travel
B2: Compact and durable
T3: Durable River 26L Daypack for travel
B3: Durable
T4: Field 14L Daypack for everyday carry
B4: NONE
T5: Ridge 23L Daypack for everyday travel
B5: NONE
T6: River 14L Daypack for everyday travel
B6: Durable Compact
```

The first heading asserts compact and durable as universal, although the section has no shared attribute. T6 does not mention its supported compact/durable preference. Five titles, four badges and one heading pass the existing lexical policy, but the page remains invalid. Correct NONE values now survive serialization.

Fresh no-preference case excerpts:

```text
S1: Water-resistant and packable daypacks
S2: Choose durable and roomy daypacks for everyday travel
T6: Meet the compact everyday travel choice
B6: packable
```

Only packable is universal; the additional heading attributes are false. Packable is a true selected-product fact but is not supported by any user preference/history, so B6 must be NONE. Five other raw NONE values serialize and validate correctly. The generic T6 line is accepted by the existing lexical validator despite omitting the selected Aster identity entirely. Its six accepted-title count is therefore not six verified identity-preserving titles. No promotion follows from that count.

Fresh explicit-preference case excerpts:

```text
S1: Discover everyday choice with roomy ventilated backpacks
S2: Durable and packable everyday choice
T3: Jasper 25L Daypack for everyday travel
T4: Moraine 32L Daypack for everyday travel
T5: Islet 21L Daypack for everyday travel
T6: Horizon 19L Daypack — compact
```

Both headings import non-universal member attributes. T3/T4/T5 omit required supported compact/durable evidence; T6 copies an approved title. Four badges correctly mention observed supported attributes and two actual NONE values remain correct, but the title and heading failures reject the page. These are factuality, responsiveness and novelty errors after serialization is fixed.

The independent audit also finds that the accepted replay heading has six whitespace words despite the 3–5 word prompt target. Existing validation does not enforce that word count or full identity presence; both gaps are disclosed rather than silently changing the baseline policy or inflating success. No observed field names another selected product, but omission of identity remains a real weakness.

## Matched native correctness and limits

The values-only common prefix passed fork/per-lane independent common-extension rebuild/repeat controls at widththree. Each control returned 16 responses (7 heading, 7 title, 2 badge including natural EOS), 48 total. Complete cache hashes match, all source states remain immutable, every observed response distribution is checked finite and its digest matches. The independent rebuild starts from the same immutable ranking state; it is not a fresh entire-prompt native 256 equivalence claim. Full raw distribution vectors were not persisted, so the independent audit verifies executed finite checks and digest receipts.

An independent CPU audit verifies frozen committed cases/source/runtime/native receipt, all 84 prompt boundaries and raw decodes, canonical row raw equality, paired selected IDs/evidence, state hashes, exact labeled diagnostic replay, response byte counts, unchanged strict quality/pageplan/audit replays and fixed denominators. Content-policy and semantic audits are separate from the state/provenance audit.

Global peak MLX:6,362,874,212bytes under the7,516,192,768-byte seven-GiB guard. Model load:2650.8ms; bounded process:66189.5ms. Executed commit`0f632d3d939a0a3dcc8f0e1d43022d974b289489`, source SHA-256`53c7a72f44285ebc501adad47832b0a079b07298042b1585417fcfdb4680dcc1`. All six attempts completed; no native or memory abort. Per-arm peaks are cumulative.

[Raw results](SCHEDULER_IDENTITY_RESULTS.json), [analysis](SCHEDULER_IDENTITY_ANALYSIS.json), [frozen cases](SCHEDULER_IDENTITY_CASES.json), [policy audit](SCHEDULER_IDENTITY_POLICY_AUDIT.md), [quality audit](SCHEDULER_IDENTITY_QUALITY_AUDIT.md), [state audit](SCHEDULER_IDENTITY_STATE_AUDIT.md), and ignored `data/ranker/pretrained/scheduler-identity-v1/` preserve the evidence. The 12/24-card workload and original manifests/failed outputs remain unchanged. Q001 is still the one manually relayed, execution-only external job; no duplicate dispatch or Gmail setup request was sent.

## Next step

Scheduler-owned identity is a viable serialization mechanism, not a factual or personalization solution. Keep it experimental and preserve the best manifest reference. The next copy experiment should directly address semantic field binding: title identity and required observed support, universal-only heading attributes and approved-title novelty. A separately declared constrained semantic decoder or trained grounding adapter would test that mechanism more directly than another label-prompt dialect. It must retain genuine new wording, every required product/field and held-out complete-page evaluation; no template/fallback can be counted as fresh generated success.

The optional CPU catalog-fusion preparation is a different representation/training lane. No model features were extracted and no project fit or ranking-speed claim was made. A new trained checkpoint, sealed full-grid ranking/image evaluation and cold/warm/fresh-user conditioning measurements remain prerequisites. Faster head scoring alone cannot be credited with complete fresh-copy latency.
