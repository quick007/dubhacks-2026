# Factorized output audit: saved eleven-attempt run

This CPU-only audit independently inspected `PARALLEL_BENCHMARK_RESULTS.json`, the matching ignored `parallel-case-plan.json`, `parallel_decode.py`, `full_page_reliability.py`, `generation_diagnosis.py`, `evidence_plan.py` and `fusion.py`. It does not load a model, run native/Metal work, alter acceptance, rewrite generated text, or promote an experimental arm. The benchmark JSON SHA-256 at audit time is `8d92a398f6426fc9b9ee755954262bffba694fd3cfc8f4e2d5e9dc1ec098f1f8`.

## Observed counts and cost

The saved factorized run contains three distinct one-card cases and eleven executed attempts: three sequential, three width two, three width three and two width six. Each fixed arm's denominator remains twelve cases. All multi-card cases are unattempted. Width six has at most three available fields here; it supplies no six-stream or six-card measurement.

| Saved diagnostic | Actual count |
| --- | ---: |
| Complete pages accepted by unchanged validator | 0/11 executed; 0/12 fixed cases in each arm |
| Accepted headings/titles/generated badges in assembled pages | 0/0/0 |
| Headings ending by length at 16 tokens | 11/11 |
| Titles ending by stop | 11/11 |
| Titles independently satisfying lexical, novelty and supported-tag checks | 8/11 instances; two distinct titles |
| Stopped raw `NONE` badges | 8/11 instances |
| Badges ending by length at 12 tokens | 3/11 |

Independent title eligibility is a diagnostic computed from the saved row's own stop, factual lexicon, normalized approved-title difference and required observed tag. It is not official page acceptance or a counterfactual valid page. The other three titles copy an approved title. `NONE` is allowed even when support exists, and required when no support exists; all eight stopped `NONE` rows satisfy that policy individually.

| Arm | Emitted tokens, excluding EOS | Sum of saved full-attempt wall times |
| --- | ---: | ---: |
| Sequential | 106 | 11,995.7 ms |
| Width two | 106 | 10,933.2 ms |
| Width three | 106 | 9,843.0 ms |
| Width six, two attempts | 65 | 6,459.7 ms |
| Total | 383 | 39,231.5 ms |

These wall times sum the benchmark's per-arm `fullCompletionMs`; they exclude resident model load, diagnostic scripts and orchestration gaps. They are rejected-attempt costs and establish no usable-page speedup. The saved benchmark's global MLX peak is 6,284,001,495 bytes against the unchanged 6,442,450,944-byte limit; it is not an independently sampled audit measurement.

## Actual examples and causes

Case `evidence-r-476a18e5ac18fd7f6d`, sequential, 4,008.8 ms:

```text
S1: River 14L Daypack meets durable and compact needs within
T1: Grove 23L Daypack — compact
B1: NONE
```

The selected product is Grove 23L, with water-resistant/compact tags and observed compact support. The heading names another catalog product, asserts unsupported durable and truncates. The title is factual and different from the selected approved titles; its own row stops. The badge's own row stops with permitted `NONE`.

Case `evidence-r-b5beff326b517231b1`, sequential, 3,791.0 ms:

```text
S1: a along an and are as backpack backpacks bag bags bring carry
T1: Cedar 26L Daypack — compact durable
B1: NONE
```

The heading copies the beginning of the alphabetically sorted framing-word list and exhausts its cap. Its vocabulary is permitted, but it is incoherent copy. The title's own row stops and passes the independent lexical/novelty checks. There is no badge support; raw `NONE` is correct. The reported assembled-page `correctNone=false` is caused by the global non-stop finish, rather than a raw badge-policy failure.

Case `evidence-r-e1f73b869dc810bc15`, sequential, 4,195.9 ms:

```text
S1: a backpacks choice for a daily trip with water-resistant and organizer
T1: Meadow 17L Daypack — water-resistant
B1: B1: Meadow 17L Day
```

The heading uses the selected universal tags but truncates. The title's own row stops but copies an approved title. The badge repeats its label and truncates; width three instead emits `B1: B1: B1: B1:`. This is a real content/protocol failure independent of the batch-shape token difference. The broad row regex recognizes the first `B1` as a label even when another label appears in its value; `field_audit` records fused fields and the lexical validator rejects the embedded label/number. The assembly concatenates saved rows unchanged; it introduces none of these repetitions.

The section prompt contains member indexes/tags and universal tags, but omits selected member names. The reused ranking prefix contains other catalog products, so the heading lacks a local identity anchor even though the card prompts carry one. This is a plausible cause of misbinding, not a causal result established by a paired prompt test. The long sorted allowed-word list and instruction to produce a "short heading" without an explicit word count are also plausible contributors to list copying and truncation. Raising the cap alone would not correct the already wrong first-case identity/attribute or the second-case incoherence.

## Acceptance limitations

`assemble` reduces all row finish reasons to one page finish. One unfinished heading makes `global_ok=false` in `validate_generated`, rejecting otherwise stopped factual titles and `NONE` badges. This is the existing strict fail-closed page policy and must stay unchanged for comparisons. Report per-row stop/protocol/factual diagnostics separately so zero accepted titles is not mistaken for eleven factually wrong title generations.

The strict validator is conservative about vocabulary and facts, but it is not a complete semantic-quality measure. It enforces 5–90 characters per generated field rather than the prompted 4–8 title words, and it does not require an explicit selected-product name. Normalized approved-title inequality is its novelty test; repetition or rearrangement can satisfy it. Heading coherence and grammar are not validated.

A deliberately fabricated CPU-only probe on the first saved case produced the following result from the unchanged validator. This is **not model output**, not a new accepted experimental page, and not included in any counts above:

```text
S1: compact compact compact
T1: compact compact compact
B1: NONE
```

The returned quality is `allFieldsFulfilled=true`, `newTitleCount=1`, `generatedHeadingCount=1`, `titleObservedAlignmentRate=1.0`, with no rejections. The three-word title lacks the selected identity and both fields are repetitive. A semantic width comparison must therefore retain strict acceptance and additionally report title word count, identity binding, repetition/coherence review and raw examples. No validator relaxation is justified by this probe.

## One bounded next experiment

After the matched-shape native-state control actually passes, run one predeclared **heading-prompt-only intervention** against the preserved factorized prompt. Use the same fixed twelve cases, selected IDs, tokenizer, model, greedy sampler, row order, per-row budgets and total 64/128/224 caps. Keep title/badge prompts, decode/cache code and strict validator unchanged. Replace only the heading instruction/data serialization: state 3–5 words; provide explicit selected member names as binding evidence while prohibiting names/numbers in the heading; list shared tags as facts rather than a sorted exhaustive framing vocabulary; require a single fresh phrase and stop. Do not supply a heading to copy.

First measure both prompts at width one to isolate output design, retaining all raw rows and failures. If the intervention does not yield complete valid pages, stop the quality experiment instead of interpreting faster rejected widths as success. If it does, measure the identical intervention prompts at widths two/three/six, retaining exact-token mismatch evidence alongside independent semantic diagnostics and complete-page latency. Report title word counts and identity binding as diagnostics without silently changing the official baseline metric; review incoherent lexical passes explicitly. This is a fixed experiment on already inspected synthetic cases, not an untouched generalization evaluation. Badge repetition and copied approved titles remain separate observed failures requiring a later experiment if they persist.
