# Shared-evidence output quality audit

Independent CPU-only inspection of the completed native experiment finds **zero complete accepted pages in every arm**. The cached shared evidence produces the same raw sampled tokens and strict quality as repeating the shared evidence in each row. All six selected product/image decisions, fourteen requested field rows and the 224-token aggregate cap are preserved. Removing truncation did not resolve required labels, title novelty or universal heading facts.

Source: `SHARED_EVIDENCE_RESULTS.json`, SHA-256 `72de2e0cf460f3e1f61ee37af70bdcf8ee1f4504062a78031b6f3be14da56f64`, with matching ignored files under `data/ranker/pretrained/shared-evidence-v1/`. This is one previously inspected six-card case at width three/native prefill 256 and a separately declared seven-GiB guard. It is not a twelve-/twenty-four-card result or a held-out quality estimate. The reviewer performed no native inference, output repair, source change or Q001 action.

## Official acceptance and raw completion

| Arm | Complete accepted pages | Accepted new titles / generated badges / headings | Stopped / length-capped rows | Text tokens / sampled steps including EOS | Full completion |
| --- | --- | --- | --- | --- | ---: |
| Baseline independent prompts | 0/1 | 0 / 0 / 0 | 10 / 4 | 132 / 142 | 11,875.3 ms |
| Shared evidence repeated per row | 0/1 | 3 / 2 / 0 | 14 / 0 | 110 / 124 | 25,778.3 ms |
| Shared evidence cached once | 0/1 | 3 / 2 / 0 | 14 / 0 | 110 / 124 | 8,699.8 ms |
| Cached evidence with trace instrumentation | 0/1 | 3 / 2 / 0 | 14 / 0 | 110 / 124 | 8,970.4 ms |

These accepted field counts come from unchanged page validation on rejected pages. They do not mean that three cards or any complete page is usable. The trace arm is an instrumentation control, not another independent case. The denominator is one fixed case per arm and does not merge into earlier twelve-/two-case experiments.

CPU replay of `assemble` with the saved rows/plan/request/catalog exactly reproduced all four saved quality dictionaries. Baseline/repeated/cached selected product/image pairs are identical, each has fourteen rows whose caps sum to 224, and every saved source-state before/after hash is unchanged. The repeated and cached arms have identical per-row raw sampled token IDs and finish reasons, including EOS. The strict assembly retains raw rows unchanged; no missing labels or `NONE` values were injected by the reviewer.

## Baseline failures

The two baseline headings assert unsupported section-wide attributes:

```text
S1: Durable Compact Travel Packs
S2: Three Compact Travel Options
```

Section one contains Grove water-resistant/compact, Cedar durable/compact and River durable/organizer. Section two contains Field lightweight/roomy, Ridge lightweight/roomy and River durable/compact. Both universal tag intersections are empty. Durable/compact in section one and compact in section two are substantive factual errors, not merely conservative lexical rejection. The spelled count `Three` is additionally outside the strict framing vocabulary, although the member count itself is true.

Baseline badges B2/B3/B4/B6 end by length at eight tokens with identity-only values:

```text
B2: Cedar 26L
B3: River 26L
B4: Field 14L
B6: River 14L
```

They fail completion and supported-attribute requirements. B4 has no observed support and should be `NONE`. B1/B5 are individually stopped `NONE`, but the global non-stop page finish also rejects otherwise sound fields; `correctNone=false` must not be read as proving raw B5 is wrong.

The baseline title `T5: Ridge 23L Daypack — durable everyday` names its selected product but asserts unsupported durable. Selected Ridge tags are lightweight/roomy. This is a card-level fact-binding defect. The other five baseline title rows stop, preserve selected identities and have no lexical fact errors; their isolated properties do not count as accepted fields under the globally rejected page.

## Shared repeated/cached failures and improvements

Both shared arms emit the following exact raw assembly:

```text
S1: Compact daily choice for everyday travel
Simple daily choice for everyday travel
T1: Grove 23L Daypack — compact choice
B1: compact
T2: Cedar 26L Daypack — durable and compact
Durable and compact
T3: River 26L Daypack — durable
Durable
T4: Field 14L Daypack — lightweight
NONE
T5: Ridge 23L Daypack — lightweight
NONE
T6: River 14L Daypack — durable and compact
B6: Durable Compact
```

All fourteen rows stop on EOS within their original caps. However, five rows omit their required label: heading two and badges two, three, four and five. Only nine of fourteen rows satisfy their one-line label protocol. The bare `NONE` values are not valid B4/B5 lines. Reconstructing those labels would change the measured output and is not permitted.

The first heading is fluent but still falsely asserts compact for every section-one member; River 26L lacks compact. The second heading's value is coherent generic framing with no unsupported attribute, but it lacks `S2:`. Thus neither heading is accepted. These are different mechanisms and should not be collapsed into a single factuality diagnosis.

Every shared title preserves the intended selected product identity, and all six are lexically factual for their bound products. No observed shared title names another product. T5 now says lightweight, correcting the baseline's unsupported durable. Nevertheless, three titles copy approved catalog variants exactly under the unchanged normalization rule:

```text
T3: River 26L Daypack — durable
T4: Field 14L Daypack — lightweight
T5: Ridge 23L Daypack — lightweight
```

Those are genuine novelty failures. T1/T2/T6 remain factual new title wording and are the three officially accepted title fields. Title success must not be inflated by calling the three copied variants newly generated wording merely because the decoder emitted them.

The values of shared badges one/two/three/six describe supported observed attributes. B1/B6 retain correct labels and count as the two accepted generated badges. Badges two/three omit labels. Badges four/five correctly choose the value `NONE` for unsupported personalization, but omit labels and therefore fail protocol. There is no unsupported private-intent or new delivery/promotion claim in these shared badge values. This content-level observation does not rescue the four malformed badge fields or the page.

No accepted lexical false positive was observed among the shared arm's three titles and two badges. The validator still has previously documented gaps in general semantic coherence, explicit identity requirements and word-count enforcement; this audit does not turn lexical acceptance into broad semantic proof.

## Scope and implication

Peak MLX allocation in the completed report is 6,366,069,178 bytes against the unchanged experiment guard of 7,516,192,768 bytes, with no abort. Cached evidence is faster than this rejected baseline attempt, and far faster than repeating the larger shared packet fourteen times; neither observation establishes usable-page acceleration because every page fails strict validation. The original fourteen-field/six-card workload and raw failures remain intact.

The next quality work should preserve the real content gains while restoring exact row-label compliance and genuine title novelty, and prevent a section heading from importing a non-universal member attribute. Current common evidence contains all cards; caching it correctly does not enforce semantic binding. Any new prompt/protocol change needs its own fixed comparison rather than silently repairing these outputs or replacing required fields with templates. Full twelve-/twenty-four-card acceptance remains unmeasured.
