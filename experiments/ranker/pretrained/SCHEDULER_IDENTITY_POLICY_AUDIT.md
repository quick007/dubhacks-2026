# Scheduler-owned field labels and badge policy

The proposed values-only transport is compatible with the existing content policy if the scheduler fixes each field identity before generation, preserves the model's raw value and finish, and passes the canonical line through unchanged `assemble` / `validate_generated`. It is a new prospective transport experiment, not permission to repair the previous unlabeled outputs. This CPU-only audit changes no source, model, validator, output or Q001 artifact.

## Current `NONE` semantics

`evidence_plan.validate_generated` permits an optional badge to abstain even when evidence exists. Support is a prerequisite for a factual badge, not a requirement to produce one. Exact policy:

| Bound card's approved observed support | Generated badge value | Existing outcome when the entire page stops cleanly |
| --- | --- | --- |
| Nonempty | Exact `NONE` | Valid omission, `source=none`; not a generated badge |
| Nonempty | Factual supported value | Generated badge only if it meets length, lexical facts and supported-tag mention checks |
| Empty | Exact `NONE` | Required valid omission; `correctNone=true` for this control |
| Empty | Any non-`NONE` value | Rejection, even if that attribute is objectively true of the product |

`badgeDecision=GENERATE` in a supported pending plan does not override this policy: `validate_page_plan` allows the decoder's eventual badge field to have `source=none`. Lowercase `none`, an empty output, a quoted or explained `NONE`, and a host-injected default are not the same as the model's exact `NONE`. Parser whitespace normalization remains the existing validator's behavior; preserve raw value bytes rather than silently cleaning them in transport.

Generated factual badges must be 5–90 characters and contain a supported tag under existing normalization, with no forbidden or out-of-product lexical claims. The current validator does not enforce the prompt's separate 2–5-word aspiration. Do not add or remove that gate silently. `NONE` has no corresponding omission meaning for a required title or heading; those still need valid newly generated text.

CPU probes called the unchanged validator on deliberately fabricated one-card policy fixtures, explicitly **not model output**. A supported Grove card accepted both `NONE` and `compact` when finish was `stop`. A no-history/no-preferences Cedar card accepted `NONE` and rejected `compact` despite compact being a true catalog tag. Changing finish to `length` rejected every complete fixture and reduced accepted title/heading/badge counts to zero. These probes verify policy, not native quality or newly accepted pages.

## Required transport contract

Each job needs a trusted scheduler record binding native UID, immutable row ID, expected field label, selected product/image or section membership, prompt hash and cap. Derive the expected label from the frozen plan, never from model text, completion order, an inferred product name or a guessed card index. Preserve unique exact expected field order: two headings plus six title/badge pairs in the six-card workload.

For a new values-only row, keep separate evidence fields such as `rawValue`, raw sampled token IDs including EOS, stop-token ID, finish reason, AR steps, trusted label and canonical text. Canonicalization is exactly:

```text
canonicalLine = trustedField + ": " + rawValue
```

The host contributes only known protocol syntax. Every word of the displayed value must come from that new model generation; no heading, title, badge attribute or `NONE` is supplied by the scheduler. A single resident model still generates genuinely fresh values in independent autoregressive streams.

Reject the transport row before normal acceptance if:

- The raw value is empty or whitespace-only.
- It contains any line break, including a trailing/leading CR/LF or Unicode line separator. Checking only `len(splitlines()) == 1` misses some terminal line breaks; inspect separator characters explicitly.
- It includes a field label such as `S1:`, `T2:` or `B6:` anywhere in the value, including a repeated own label. Do not strip that label, pick the first line or reinterpret it as a different row. A reject-only NFKC/case-insensitive label check can catch disguised labels without changing the original value.
- Trusted row identity/label is missing, duplicated, unexpected, rebound to another native UID, or inconsistent with the frozen plan.

Retain rejected raw rows and record an explicit transport failure so the page cannot become valid merely because a malformed row was omitted. Do not emit a guessed canonical value or inject `NONE` to complete the set. The unchanged content validator still rejects markup, controls, copied approved titles, unresponsive facts, unsupported attributes, numerals and other failures; transport success is only an initial gate.

The previous shared-evidence lines such as bare `Durable`, bare `NONE` and the unlabeled second heading remain failed historical output. They cannot be passed through the new wrapper after the fact and counted as a new successful page. Native generation under a predeclared values-only prompt/transport is the experimental intervention.

## Completion and metric accounting

Preserve the native stop reason. If even one row ends by length, `assemble` must still pass global finish `length`; the unchanged validator rejects the whole page and its otherwise stopped fields. Canonical labels do not imply that the raw value completed. Do not convert a visibly plausible truncated value into a `stop`, add EOS metadata, or validate each row with an invented stop reason and sum isolated eligibility as official accepted fields.

Maintain six unique selected products/images, all fourteen required rows and the 224 aggregate sampled-token cap. Host-owned labels consume no sampled model tokens, so values-only changes effective payload capacity relative to a label-generating baseline even when nominal caps are unchanged. Disclose that transport factor and report raw sampled steps separately from canonical text length/tokenization. Do not describe a resulting gain as only cache reuse or improved factual reasoning.

Report complete valid pages, actual accepted new titles/badges/headings, individually correct raw `NONE` diagnostics, raw truncations, transport failures, canonical output, total timing and memory. `correctNone=false` can result from unrelated global truncation or a malformed other no-support badge; it does not prove every raw `NONE` was wrong. A valid optional omission is not an accepted generated badge. All failed/unattempted cases stay in fixed denominators.

## Review of two new six-card synthetic holdouts

The proposed new cases are appropriate prospective generation holdouts only if frozen before native generation. This is a plan review; no new holdout artifact's disjointness or hash has been certified here. Required gates:

1. Generate new case/request IDs, product IDs, image-variant IDs and catalog facts with a recorded fixed seed. Check disjoint product/variant IDs against the old catalog and prior diagnostics. Also compare canonical product/variant fact/text signatures: renaming an old SKU or relabeling the old six-card test is not a genuinely new case. Existing attribute vocabulary may overlap; do not claim unseen-attribute generalization.
2. Ensure each case has six eligible unique products, two real sections and all fourteen pending fields. Both approved image options and compatibility facts must be explicit. Derive observed support only from current explicit preferences or cutoff-valid referenced saves; unknown/weak history supplies no positive personalization evidence.
3. Include complementary factual controls: at least one empty universal section intersection (headings must remain generic), at least one positive shared attribute section, supported optional badges and cards requiring `NONE`. This tests both false universal claims and over-eager personalization without supplying any prewritten heading/title/badge answer.
4. Freeze input JSON, generator source/seed, plan schema, trusted row mapping, selected-ID policy, prompts, budgets, model/head/tokenizer/runtime receipts, strict validator and the exact run order before the first native result. Preserve prompt/input hashes in each result. New cases must not be selected or discarded after seeing model output.
5. Freeze the implementation using development cases first. Do not alter prompts, samplers, caps, transport rules or fusion weights in response to either holdout, then rerun it as though still untouched. A later change needs a separately named experiment and fresh prospective evidence. CPU schema/fact checks before generation are allowed; optimizing against native holdout responses is not.

Both cases remain synthetic and authored for this experiment. Their factual copy/identity/protocol can be evaluated against known public catalog evidence, but trained ranking desirability is **unlabeled** unless an independent ranking target is separately created and frozen. Do not equate filling six of six eligible products, preserving the baseline scores, or validating image compatibility with learned ranking/image preference quality. Any old-case development result and the two new holdouts need separate denominators; the two holdouts are two cases per arm, not twelve cases and not repeated generations counted as extra examples.

The protocol can remove a known labeling burden while preserving raw creative values. It does not by itself solve copied approved titles, false section facts or stale product evidence. Those remain subject to unchanged strict validation and independent raw-text review, with no template fallback or silent reduction of the requested grid.
