# Bounded generation-failure diagnosis

This increment inspects all ten saved catalog-cache attempts and all 24 evidence-plan arms (23 failed pages and one fulfilled page). It preserves every previous output/checkpoint. [GENERATION_FAILURE_DIAGNOSIS.json](GENERATION_FAILURE_DIAGNOSIS.json) records overlapping mechanism labels and actual raw outputs; it does not change acceptance. An independent read-only agent inspected the same outputs against the public catalog and observable profile, with six representative examples reviewed in detail. No private persona truth or new training is used.

## Established mechanisms

All ten cache outputs have the wrong field set. Eight hit their exact 64/128-token limit; the two six-card outputs stop at 175 tokens but use Markdown and `Card N:` prose instead of the required field protocol. The cache prompt says “all S1/S2 section headings” even for one section, followed by “T2/B2, etc.” without a finite key list. Both prompt families leave Card N→TN/BN binding implicit. Planner control word `GENERATE` is copied as badge text in one cache and four evidence outputs. These are protocol/control-to-content failures, not proof that the catalog cache is corrupted.

Genuine errors also occur: a changed-profile request selects River 26L while T1 names a different public catalog product, Field 14L. Six-card headings assert tags not shared by every product in their section. These are output-to-plan binding and factual scope errors; a fact existing somewhere in the catalog does not make it true for the selected item/whole section.

Other failures are distinct. A copied approved title fails novelty even if factual. A badge can be factual for the selected product but unsupported as an observed personal reason. The restrictive vocabulary rejects safe generic framing such as `use` and plural `options`; those are conservative lexical false negatives in the reviewed examples, not new material/performance claims. Blank separator lines count as unexpected lines under the current strict protocol. The validator, vocabulary, novelty and truncation policies remain unchanged in this experiment; no fields are rescued or rewritten after decoding.

Generation settings are greedy temperature 0, 64/128/224 tokens for 1/3/6 cards, zero retries, and the same frozen 9B decoder/decision heads. Eight cache truncations are consistent with generating extra fields, not simply a shortage of tokens for the intended three-line one-card response. Cold/warm and return-A equality, distinct B inputs/outputs, and cache invalidation tests do not establish stale/corrupt cache. They do not prove broad serving correctness either.

## Actual input/output examples

| Observable input and selected facts                                                                                                   | Actual output excerpt                                                                                                   | Diagnosis                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| One card, Grove 23L: water-resistant + compact. Explicit durable/compact; the selected item supports compact.                         | `S1: Compact Everyday Travel Choice` then extra `S2`, `T2`, `T3`; `T1: Grove 23L Daypack — Compact`; `B1: Compact`.     | Early title/badge are factual and newly phrased/supported, but the response overruns the one-card protocol and truncates. Global truncation rejection remains in force. |
| Profile B: lightweight/roomy, no history. Structured selection is River 26L: durable + organizer.                                     | `T1: Field 14L Daypack — lightweight`; `B1: lightweight roomy`.                                                         | Public catalog facts are attached to the wrong selected product. The ranking error and decoder binding error are separate.                                              |
| Six-card cache page, first section includes River 26L: durable + organizer. Second section includes Field/Ridge: lightweight + roomy. | `## Compact Daily Travel Packs`; later `## Compact Durable Everyday Packs`; correctly named individual `Card N:` prose. | Names all six products, but wrong protocol and genuinely false section-wide attribute scope. It does not repeat one item across all six cards.                          |
| Evidence-plan Grove 23L: water-resistant + compact; supported observed attribute compact.                                             | `T1: Compact Grove Daypack for Daily Use`; `B1: water-resistant`.                                                       | Title's `use` is a conservative lexical rejection. Badge is product-factual but not supported as this user's observed reason.                                           |
| Meadow 17L: water-resistant + organizer; explicit water-resistant/roomy, no history.                                                  | `T1: Water-Resistant Meadow Daypack`; `B1: Organizer`.                                                                  | New title passes; badge is factual but lacks observed preference support.                                                                                               |
| No-history/no-preference control, Cedar 26L: durable + compact.                                                                       | `S1: Compact Durable Everyday Options`; `T1: Cedar 26L Daypack`; `B1: NONE`.                                            | NONE is correct; title is copied rather than new; `options` is outside the conservative framing lexicon.                                                                |

## One intervention, fixed comparison

The sole intervention is an exact output manifest: finite ordered labels, explicit per-field product/section bindings, supported badge evidence as content constraints rather than a `GENERATE` enum, and “stop after the last key.” It supplies a schema, not finished titles or headings. The same model must generate genuine new wording.

The fixed roster includes **all ten original cache conditions plus all twelve original plan-arm cases**, regardless of outcome. Every case receives fresh baseline and manifest runs in alternating arm order. Requests, catalog versions, trained product/image decisions, prefix layout, cold/warm condition, greedy settings and 64/128/224-token caps are identical within each pair. Full response timing includes ranking prefill, cache build/materialization where applicable, fresh decoding, all rejected fields, and validation. The same strict validator scores both. All-token-boundary preflight passed: maximum prefix 1446 tokens, maximum manifest suffix 716.

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_generation_diagnosis -v
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.generation_diagnosis diagnose
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.generation_diagnosis benchmark --model /absolute/verified/pinned/snapshot
```

Five offline protocol/guard/reporting tests pass. The actual paired comparison completed; [GENERATION_MANIFEST_RESULTS.json](GENERATION_MANIFEST_RESULTS.json) preserves measured results and raw generated examples. All **22/22 baseline raw replays exactly matched** the earlier saved outputs, and every selected product/image ID matched its saved plan. No model, head, factual validator, novelty criterion or decoding cap was changed.

| Population             | Fully valid pages: baseline → manifest | Accepted NEW titles | Full attempt median: baseline → manifest |
| ---------------------- | -------------------------------------- | ------------------- | ---------------------------------------- |
| 10 cache conditions    | 0/10 → 0/10                            | 0/24 → 18/24        | 6420 → 4989 ms                           |
| 12 evidence-plan cases | 1/12 → **7/12** (58.3%)                | 6/19 → **16/19**    | 3195 → 3199 ms                           |
| All fixed cases        | 1/22 → 7/22                            | 6/43 → 34/43        | 4987 → 3690 ms                           |

The manifest produced the exact required field order in **22/22** cases, versus 9/22 baseline, and reduced non-stop/truncated responses from 8/22 to zero. Control enum `GENERATE` no longer appears as output. This supports a protocol/binding contribution to the failure mechanism. It does not fix every binding, grounding or heading error. The pooled population includes repeated cache controls and mixed page sizes; its median is not production traffic or independent-user evidence. Cache-attempt speed gains include avoiding extra-field overrun; **zero cache pages were valid**, so none is a successful fast page.

Full evidence-plan size-specific measurements were:

| Cards | Baseline full attempt | Manifest full attempt | Manifest complete page                                |
| ----- | --------------------: | --------------------: | ----------------------------------------------------- |
| 1     |               3730 ms |               3801 ms | Valid                                                 |
| 3     |               5182 ms |           **5390 ms** | Valid                                                 |
| 6     |               8420 ms |               8923 ms | Rejected headings; six valid new card titles retained |

These are single runs per size. The seven fully valid manifest responses took 2901–5390 ms (median 3098 ms, conditional on successful cases). The evidence-path full-attempt median did not improve, and the near-traditional-page latency target remains unmet. Six-card section headings still make claims not true for every bound product. Peak MLX allocation remained 6,344,673,924 bytes under the 6,442,450,944-byte cap; no OOM or parallel heavy runs occurred.

### Actual after-intervention examples

For Grove 23L (water-resistant/compact, observable supported tag compact), the earlier rejected `Compact Grove Daypack for Daily Use` and unsupported-as-personal-reason `water-resistant` badge became:

```text
S1: Compact water-resistant everyday travel choice
T1: Grove 23L Daypack for daily compact needs
B1: compact
```

All fields passed the unchanged checks in **3061 ms**, versus a 2957 ms baseline attempt that failed. The title is genuinely new, not an approved-title lookup or template substitution.

The no-history/no-preference control generated:

```text
S1: Everyday travel backpacks
T1: Cedar 26L Daypack for daily needs
B1: NONE
```

This whole response passed in **2901 ms** without asserting invented personal intent. The three-card evidence case also passed with newly generated `Grove 23L Compact Daily Bag`, `Cedar 26L Durable Travel Pack`, and `River 26L Durable Organizer Pack`, under the generic factual heading `Everyday travel companion daypacks`.

Failures remain visible. In the catalog-first one-card path, `Grove 23L Daypack for short trips` is novel but rejects **both** lexical `short` and omission of the required explicit `compact` attribute. In changed-profile B, the manifest still emits `Field 14L Daypack lightweight` for structured River 26L (durable/organizer): a real binding failure, rejected. Six-card headings such as `Compact durable daypacks` are rejected when a section's products share no such universal tags. These are not rescued into successes.

### Review and limits

The independent agent confirmed fixed case reconstruction (all ten observed profile hashes match), unchanged heads/settings/validator, cold/warm parity in both arms, full timing, and product/image identity assertions. Diagnostic counts were refined **after generation without changing acceptance or latency**: omitted title support is recorded separately, field order is checked explicitly, NONE is treated as a control rather than rejected prose, and numeric/brand flags against the generic section proxy are marked context-review signals. Those flags are not confirmed hallucination counts; a one-product heading naming its actual River 26L item can be factually true yet rejected by the conservative generic section proxy. The detailed original benchmark remains preserved alongside `benchmark-reviewed.json` and evidence hashes.

Validity means the unchanged implemented validator, not every natural-language instruction: novelty checks the selected product's approved titles, and prompt word-count suggestions are not all enforced by that validator. Factuality vocabulary was not enlarged, no rejected wording was rewritten into acceptance, and no truncation or false section claims were ignored.

This is a one-intervention result on cases already inspected during diagnosis, with repeated controls, not untouched generalization or a human conversion result. Product ranking uses the same previously regressed experimental checkpoint; this prompt result does not repair ranking or establish learned section grouping. No additional training or broad sweep was performed. The experiment stops here.
