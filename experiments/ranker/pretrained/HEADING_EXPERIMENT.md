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
