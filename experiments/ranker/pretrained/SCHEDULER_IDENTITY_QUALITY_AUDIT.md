# Scheduler-owned identity: completed quality audit

Scheduler-owned field labels resolve the observed transport failures but do not produce a complete valid page. Both protocols accept **0/3 pages overall** and **0/2 fresh synthetic holdout pages**. The six-card/fourteen-field workload and 224 aggregate sampled-token cap remain intact. No missing values were supplied by the host, no old output was repaired, and no holdout was removed.

This independent CPU review inspected `SCHEDULER_IDENTITY_RESULTS.json` (SHA-256 `4b3a653bb3a026f4d175e6fc1ebcba272456487aa3637afab5d38129cb451b3a`), its ignored `scheduler-identity-v1` per-arm files and frozen `SCHEDULER_IDENTITY_CASES.json` (SHA-256 `41f3ce458a6bf355fda7e41a6bd62da0661c70d9e3e8c59943ed94d48e47c9b6`). The source receipt is `53c7a72f44285ebc501adad47832b0a079b07298042b1585417fcfdb4680dcc1`. The review ran no native model, changed no prompt/source/output, and performed no Q001 action.

## Official metrics and transport evidence

| Case | Protocol | Valid complete pages | Accepted new titles / generated badges / headings | Text tokens / sampled steps | Full completion including serialization |
| --- | --- | --- | --- | --- | ---: |
| Inspected six-card diagnostic | Generated labels | 0/1 | 3 / 2 / 0 | 110 / 124 | 8,644.7 ms |
| Inspected six-card diagnostic | Scheduler identity | 0/1 | 5 / 4 / 1 | 89 / 103 | 8,278.2 ms |
| Fresh no-history `NONE` control | Generated labels | 0/1 | 1 / 0 / 1 | 91 / 105 | 8,508.6 ms |
| Fresh no-history `NONE` control | Scheduler identity | 0/1 | 6 / 0 / 0 | 84 / 98 | 8,369.2 ms |
| Fresh explicit-preference control | Generated labels | 0/1 | 0 / 0 / 0 | 111 / 124 | 8,968.9 ms |
| Fresh explicit-preference control | Scheduler identity | 0/1 | 2 / 4 / 0 | 91 / 105 | 8,991.9 ms |

Accepted field totals occur on rejected pages; they must not be interpreted as usable cards or valid pages. Overall accepted fields are 4 titles / 2 badges / 1 heading for generated labels versus 13 / 8 / 1 for scheduler identity. On the two fresh cases alone they are 1 / 0 / 1 versus 8 / 4 / 0. The one reused diagnostic case and two new holdouts remain separate evidence. Ranking preference quality has no new labels.

All forty-two scheduler value rows stop on EOS and pass the values-only transport gate. Generated-label rows have twenty-two correct isolated row labels out of forty-two, forty-one EOS stops and one length-capped heading. The scheduler experiment therefore establishes 42/42 transport success for these new value generations, rather than improved model generation of labels. Host syntax changes effective value-token capacity even with the same nominal caps; canonical labels are not sampled model tokens.

CPU replay of unchanged `assemble` or `assemble_values`, as appropriate, exactly reproduced all six quality dictionaries. Each arm retains six unique selected products/images, fourteen rows and caps summing to 224. Selected product/image IDs match within every pair. All raw values are retained separately from canonical host-labeled lines. No complete-page latency improvement is established because every page is rejected.

## Inspected diagnostic case

Representative scheduler values and their canonical lines:

```text
S1: Compact durable choice for everyday travel
S2: Simple daily choice for everyday travel
T3: Durable River 26L Daypack for travel
T6: River 14L Daypack for everyday travel
B6: Durable Compact
```

Both section tag intersections are empty. S1 falsely treats compact/durable as universal; Grove lacks durable and River 26L lacks compact. S2 is coherent generic wording and is accepted. T3 is factual, preserves River 26L identity and is genuinely different from the approved titles, improving on the prior copied variant. T6 is factual generic wording and preserves River 14L identity, but omits its required compact/durable observed emphasis; it is correctly rejected. B6 has actual observed support and is accepted.

The other five scheduler titles pass the existing facts/novelty/support checks. Four supported badges are generated; the two unsupported cards actually generate `NONE`, with correct scheduler labels and `correctNone=true`. No wrong-product substitution was found. The remaining two rejection mechanisms are substantive heading factuality and title responsiveness, rather than label omission or truncation.

## Fresh holdout: all products packable, no badge support

The selected order is Flint 30L, Delta 16L, Coral 27L, Briar 22L, Ember 24L and Aster 18L. Each has packable plus a different secondary tag; both sections' only universal attribute is packable. There is no history or stated attribute preference, so every badge requires `NONE`.

Actual scheduler examples:

```text
S1: Water-resistant and packable daypacks
S2: Choose durable and roomy daypacks for everyday travel
T1: Flint water-resistant packable daypack
T6: Meet the compact everyday travel choice
B6: packable
```

S1 falsely generalizes Flint's water-resistant tag to Delta/Coral. S2 combines Briar's durable and Ember's roomy even though neither applies to all Briar/Ember/Aster members. Both headings are correctly rejected. B6 is objectively true of Aster, but has **no observed support** and violates the required `NONE` policy; rejecting it is not a fact error in the catalog. The other five badges generate exact `NONE` and are individually policy-correct.

All six titles are accepted by the unchanged validator. They are new under its normalized-title comparison and contain no unsupported lexical attribute. However, T6 has no selected product name or capacity at all. Its generic compact wording passes the lexical validator despite the identity-preservation goal. T1 retains only the unique Flint name and omits the 30L capacity/full catalog identity. This is an independent identity diagnostic; the strict page still fails on headings and B6. No title names a different product, but a missing identity should not be described as strong product binding.

The generated-label baseline has individually correct raw `NONE` values for all six cards, but all six omit their required labels. Its second heading also omits its label and makes a false durable claim; one title omits its label. Other titles copy approved variants. The scheduler protocol fixes labels prospectively but introduces the B6 unsupported badge decision in this paired generation, so it is not a uniform improvement across all quality dimensions.

## Fresh holdout: explicit compact/durable preferences

The selected order is Kestrel 15L (roomy/ventilated), Larch 28L (lightweight/adjustable), Jasper 25L (compact/foldable), Moraine 32L (durable/packable), Islet 21L (durable/reflective) and Horizon 19L (compact/padded). Both section intersections are empty. The first two cards lack preference support and require `NONE`; the last four have supported compact or durable evidence.

Actual scheduler examples:

```text
S1: Discover everyday choice with roomy ventilated backpacks
S2: Durable and packable everyday choice
T3: Jasper 25L Daypack for everyday travel
T4: Moraine 32L Daypack for everyday travel
T5: Islet 21L Daypack for everyday travel
T6: Horizon 19L Daypack — compact
```

S1 generalizes Kestrel's roomy/ventilated facts to Larch/Jasper. S2 generalizes Moraine's durable/packable facts to Islet/Horizon. Both are substantive false universal claims. T3/T4/T5 preserve the correct identities and are factual/new, but omit their required observed compact/durable emphasis and are correctly rejected. T6 mentions supported compact but exactly copies an approved variant, so fails novelty. Kestrel/Larch titles are factual new wording with correct identities and no unsupported required emphasis; those are the two accepted title fields.

All six badges obey policy in this scheduler arm: actual `NONE` for Kestrel/Larch, compact for Jasper/Horizon and durable for Moraine/Islet. `correctNone=true` and four generated badges are therefore appropriate. Supported `NONE` would also have been allowed; support does not force generation. The labeled baseline's Islet `NONE` is a permitted value, although its missing label and the globally truncated first heading prevent acceptance.

## Word count, identity and holdout boundaries

The existing validator enforces 5–90 characters and factual vocabulary, not all prompt constraints. For example, the accepted diagnostic heading `Simple daily choice for everyday travel` has six whitespace-separated words, exceeding the shared prompt's 3–5-word heading goal. The generic accepted Aster title demonstrates that selected identity is not explicitly required. These are real acceptance-policy limitations and should remain visible independently of complete-page acceptance. They do not justify lowering fact/novelty/support checks or dropping a holdout.

CPU inspection verified that both fresh catalogs' product IDs, variant IDs and product titles are disjoint from the original eighty-four-product catalog, and each fresh tag set is new relative to that catalog. Their new attributes include packable, padded, reflective, foldable, ventilated and adjustable. The two fresh cases are frozen before their native generation and are not old test records with changed IDs. They are authored synthetic protocol holdouts, not an independently sealed production benchmark. The frozen trained heads' ranking quality on the new identities/attributes is unlabeled; correct full cardinality or compatible image IDs do not establish preference quality.

All six attempts completed with no abort. Peak MLX allocation is 6,362,874,212 bytes under the declared 7,516,192,768-byte guard. Repeated-case, order/warmth and single-attempt limitations apply; no twelve-/twenty-four-card run occurred. The next quality work should address universal-attribute leakage and required title emphasis/novelty using development evidence, without tuning and rerunning these two holdouts as though untouched. Scheduler-owned identity solves a protocol responsibility; it does not solve semantic content binding.
