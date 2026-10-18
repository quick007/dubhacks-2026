# Bounded heading prompt diagnosis

This experiment is prepared for the parent to schedule after its native controls. No native run is claimed by this protocol. The helper constructs no model. It receives the parent's already loaded `backbone` and trained `heads`.

Installed mlx-lm 0.32.2 `GenerationBatch.next` counts every sampled token against `max_tokens`, including field-label tokens. It sets `length` when the cap is reached, then overrides the reason to `stop` if that same token matches the configured stop sequence. Existing `decode_rows` supplies only tokenizer EOS singleton sequences. A newline is not a native stop. Thus saved 16-token/`length` headings did not sample EOS at the boundary; there is no demonstrated EOS-at-cap accounting bug. This conclusion comes from source inspection, not a new native execution.

`heading_rows(..., compact=True)` changes only heading messages: 3–5 words, selected member-name anchors as data, a prohibition on names/numbers in the heading, only shared tags as attributes, and no exhaustive alphabetically sorted framing vocabulary. It supplies no example heading or prewritten answer. Card prompts, row order, all per-field and total caps, tokenizer EOS stopping, cache cloning, native decoder, raw assembly and strict validator stay unchanged. Heading word count/name/number diagnostics are descriptive and never rescue acceptance.

The paired diagnosis is bounded to the three already executed one-card case IDs in `DIAGNOSIS_CASE_IDS`, selected without reference to quality. Use width one to isolate prompt design. Run baseline first on cases zero/two, compact first on case one, disclosing order/warmth bias. Do not tune the variant after inspecting outcomes. Preserve every raw row/token/finish, selected ID, prefix/prompt hash, accepted field/page counts, timings and memory. Stop after native/memory/selection failures; do not raise the memory cap. These are previously inspected synthetic cases, not a held-out quality estimate.

Parent API:

```python
from experiments.ranker.pretrained.heading_experiment import diagnosis_cases, heading_arm

# saved_cases is the existing parallel-case-plan.json["cases"].
for index, case in enumerate(diagnosis_cases(saved_cases)):
    order = (False, True) if index % 2 == 0 else (True, False)
    for compact in order:
        result = heading_arm(backbone, heads, case["request"], case["catalog"],
                             case["cards"], compact=compact, width=1)
        # Parent persists immediately, verifies selections, checks failures,
        # and releases/clears native caches between arms using its existing policy.
```

`heading_arm` starts full timing before fresh ranking prefill and returns the same core row/quality/timing fields as `factorized_arm`, plus `experiment` and `headingDiagnostics`. It catches runtime failures with actual elapsed time and `budgetRaised=False`; invalid width/card scope raises before using the backbone. `diagnosis_cases` requires all three fixed IDs and rejects duplicate IDs. The parent must not launch concurrent GPU workers.

CPU verification:

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_heading_experiment -v
```

If compact headings improve but a copied title or malformed badge still prevents page acceptance, report that outcome and the independent failure. Do not infer usable-page latency gains from faster rejected attempts. Any later widths comparison is a separate parent-owned measured experiment; this helper intentionally rejects widths above one.

## Actual native paired result and independent audit

The parent subsequently executed all six predeclared native arms, alternating prompt order as specified, with no abort. The saved result is `HEADING_EXPERIMENT_RESULTS.json` (SHA-256 `23184575af9773b4812e8f915c903a594f0afcf2c28a9b93e364bdf0c331f97c`), backed by ignored `data/ranker/pretrained/heading-diagnosis-v1/benchmark.json` and `case-plan.json`. Native prefill step size was 256, following the parent's matched-shape controls; this differs from the historical 512-step run. Both baseline and compact arms were freshly measured at 256, so only their within-pair heading messages change. The reviewer performed CPU inspection of actual saved fields and token arrays; it did not launch another native run.

| Fixed case, abbreviated | Baseline / compact valid page | Baseline / compact emitted tokens | Baseline / compact full completion |
| --- | --- | --- | --- |
| Grove, `r-476a18e5ac18fd7f6d` | No / yes | 32 / 24 | 4,085.7 / 4,000.1 ms |
| Cedar, `r-b5beff326b517231b1` | No / yes | 33 / 24 | 3,934.4 / 3,537.4 ms |
| Meadow, `r-e1f73b869dc810bc15` | No / no | 41 / 33 | 4,411.9 / 4,042.2 ms |

Strict complete-page acceptance changed from 0/3 to 2/3 on these inspected cases. Official accepted heading/title/generated-badge counts are 0/0/0 for baseline and 2/2/0 for compact. Both accepted compact pages have `NONE` badges, which are valid omissions rather than generated badges. All three compact headings independently stop on EOS within the original sixteen-token cap: eight/seven/eight emitted tokens, including labels and excluding EOS. All three satisfy the prompted 3–5 words by whitespace count (four/four/three), contain no product names or numerals, and use attributes shared by their selected members.

Actual compact outputs:

```text
S1: Compact water-resistant travel bag
T1: Grove 23L Daypack — compact
B1: NONE
```

Grove 23L has compact/water-resistant tags and compact support. The heading is coherent and factual, the title preserves the selected identity and is different from its approved titles, and omitting the supported optional badge is permitted.

```text
S1: Compact and durable pack
T1: Cedar 26L Daypack — compact durable
B1: NONE
```

Cedar 26L has compact/durable tags. Its heading is coherent and factual; the title preserves the selected identity and is new under the existing normalized-title check. There is no badge support, so `NONE` is required and correct.

```text
S1: Water-resistant organizer daypack
T1: Meadow 17L Daypack — water-resistant
B1: B1: Meadow 17L Day
```

Meadow 17L has water-resistant/organizer tags. Its compact heading is coherent and factual and stops on EOS, but the unchanged title copies an approved title and the unchanged badge repeats its label and truncates at twelve tokens. Strict global rejection correctly leaves this page invalid and masks the otherwise sound heading in its accepted-field count. No wording was repaired or retried.

Independent CPU checks of every actual title/badge pair confirmed equal messages, budgets, raw sampled token IDs (including EOS), finish reasons, and recorded AR-step counts. The intervention therefore changes heading generation without an observed change to either card field. The first two compact titles also meet the prompted 4–8 words (four and five by whitespace, omitting the em dash separator); this is a review observation, not a new validator gate. No lexical false positive was found in the three compact paired outputs. The general validator limitations demonstrated in `FACTORIZED_OUTPUT_AUDIT.md` still apply.

Baseline totals are 106 emitted tokens / 111 sampled AR steps and 12,431.9 ms summed full-arm latency; compact totals are 81 / 89 and 11,579.7 ms. Full-attempt medians are 4,085.7 versus 4,000.1 ms. The two accepted compact pages complete in 4,000.1 and 3,537.4 ms (median 3,768.8 ms). These compare different heading prompts at sequential width one; they establish no concurrency speedup, and baseline has no accepted page for a paired usable-page latency ratio. The report's measured overall `totalWallMs` is 27,216.8 ms, including overhead beyond the 24,011.7 ms sum of arm timers. Peak MLX allocation is 6,133,848,518 bytes against the unchanged 6,442,450,944-byte cap.

The heading truncation mechanism improved in this bounded experiment. Remaining badge-protocol and title-novelty failures persist. The parent's subsequent twelve-case semantic width comparison is a separate experiment; its in-progress outputs do not change the six-arm result or provide a completed quality estimate here.
