# Independent locked-heading audit

The completed locked-evidence arm has **4/25 accepted headings**: 4/12 cache headings and 0/13 evidence headings. Zero does not describe the overall arm.

Reproduce from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.locked_heading_audit
```

Only this module and its JSON/Markdown outputs are written. No GPU/model loading, generation, training, validator/acceptance edits, core experiment edits or commits occur.

## Mutually exclusive results

| Classification | Heading fields |
| --- | ---: |
| Accepted supported wording | 4 |
| Single-product generic/numeric proxy false positive | 10 |
| Genuine universal/member-binding failure | 9 |
| Other lexical or semantic-policy rejection | 2 |

All count, population, native-stop, saved intersection and original-acceptance assertions pass. All 22 preserved manifest baseline outputs replay exactly. The 25-field exclusive classification contains no double counting.

## Exact examples and counterexamples

| Case / field | Raw heading | Actual members and tags | Classification |
| --- | --- | --- | --- |
| 0 / S1 | The bag is a compact, water-resistant choice for daily needs. | Grove 23L Daypack [water-resistant, compact] | accepted_supported |
| 1 / S1 | The bag is a compact, water-resistant choice for daily needs. | Grove 23L Daypack [water-resistant, compact] | accepted_supported |
| 2 / S1 | The bag is a compact, water-resistant choice for daily needs. | Grove 23L Daypack [water-resistant, compact] | accepted_supported |
| 3 / S1 | The River 26L Daypack is a durable, organizer backpack. | River 26L Daypack [durable, organizer] | single_product_proxy_false_positive |
| 4 / S1 | The bag is a compact, water-resistant choice for daily needs. | Grove 23L Daypack [water-resistant, compact] | accepted_supported |
| 5 / S1 | The River 26L Daypack is a durable and compact choice for everyday travel. | River 26L Daypack [durable, organizer] | genuine_universal_or_section_binding_failure |
| 6 / S1 | The selected items are a durable, compact choice for daily needs. | Grove 23L Daypack [water-resistant, compact]; Cedar 26L Daypack [durable, compact]; River 26L Daypack [durable, organizer] | genuine_universal_or_section_binding_failure |
| 7 / S1 | The selected items are a durable, compact choice for daily needs. | Grove 23L Daypack [water-resistant, compact]; Cedar 26L Daypack [durable, compact]; River 26L Daypack [durable, organizer] | genuine_universal_or_section_binding_failure |
| 8 / S1 | The Grove 23L Daypack is a compact choice for short trips. | Grove 23L Daypack [water-resistant, compact]; Cedar 26L Daypack [durable, compact]; River 26L Daypack [durable, organizer] | genuine_universal_or_section_binding_failure |
| 8 / S2 | The Cedar 26L Daypack is a durable and compact option. | Field 14L Daypack [lightweight, roomy]; Ridge 23L Daypack [lightweight, roomy]; River 14L Daypack [durable, compact] | genuine_universal_or_section_binding_failure |
| 9 / S1 | The Grove 23L Daypack is a compact choice for short trips. | Grove 23L Daypack [water-resistant, compact]; Cedar 26L Daypack [durable, compact]; River 26L Daypack [durable, organizer] | genuine_universal_or_section_binding_failure |
| 9 / S2 | The Cedar 26L Daypack is a durable and compact option. | Field 14L Daypack [lightweight, roomy]; Ridge 23L Daypack [lightweight, roomy]; River 14L Daypack [durable, compact] | genuine_universal_or_section_binding_failure |
| 10 / S1 | The Grove 23L Daypack is a compact choice for everyday needs. | Grove 23L Daypack [water-resistant, compact] | single_product_proxy_false_positive |
| 11 / S1 | The Cedar 26L Daypack is a durable and compact choice for everyday travel. | Cedar 26L Daypack [durable, compact] | single_product_proxy_false_positive |
| 12 / S1 | The Meadow 17L Daypack is a water-resistant choice for daily needs. | Meadow 17L Daypack [water-resistant, organizer] | single_product_proxy_false_positive |
| 13 / S1 | The Summit 23L Daypack is a roomy and water-resistant choice for your daily needs. | Summit 23L Daypack [water-resistant, roomy] | single_product_proxy_false_positive |
| 14 / S1 | The Field 20L Daypack is a water-resistant option for daily needs. | Field 20L Daypack [water-resistant, roomy] | single_product_proxy_false_positive |
| 15 / S1 | The Field 20L Daypack is a practical choice for everyday needs. | Field 20L Daypack [water-resistant, roomy] | single_product_proxy_false_positive |
| 16 / S1 | The Cedar 26L Daypack is a practical choice for carrying your laptop and other essentials. | Cedar 26L Daypack [laptop-sleeve, roomy] | other_lexical_or_semantic_policy_rejection |
| 17 / S1 | The Pine 23L Daypack is a durable and compact choice for everyday needs. | Pine 23L Daypack [durable, compact] | single_product_proxy_false_positive |
| 18 / S1 | The Cedar 26L Daypack is a durable and compact choice for everyday travel. | Cedar 26L Daypack [durable, compact] | single_product_proxy_false_positive |
| 19 / S1 | The Grove 23L Daypack is a compact choice for everyday needs. | Grove 23L Daypack [water-resistant, compact] | single_product_proxy_false_positive |
| 20 / S1 | This choice meets needs of everyday travel and short trips. | Grove 23L Daypack [water-resistant, compact]; Cedar 26L Daypack [durable, compact]; River 26L Daypack [durable, organizer] | other_lexical_or_semantic_policy_rejection |
| 21 / S1 | A daily choice for water-resistant and compact needs. | Grove 23L Daypack [water-resistant, compact]; Cedar 26L Daypack [durable, compact]; River 26L Daypack [durable, organizer] | genuine_universal_or_section_binding_failure |
| 21 / S2 | A practical option for lightweight and roomy travel. | Field 14L Daypack [lightweight, roomy]; Ridge 23L Daypack [lightweight, roomy]; River 14L Daypack [durable, compact] | genuine_universal_or_section_binding_failure |

## What the failures mean

**Proxy false positives:** “The River 26L Daypack is a durable, organizer backpack.” (case 3) is factual for the selected River 26L, but the generic section proxy rejects `River/26L`. Similarly, all nine single-product evidence proxy cases retain correct selected identities/attributes but are rejected for brand/capacity tokens. These are not established fabricated product facts; all remain rejected under the unchanged strict policy.

**Universal/member failures:** River 26L lacks compact (case 5). The three-card section has Grove 23L without durable and River 26L without compact, so “durable, compact” is not universal (cases 6/7). In cases 8/9, S2 says “The Cedar 26L Daypack is a durable and compact option.” but section 2 contains Field 14L, Ridge 23L and River 14L, not Cedar 26L. That is wrong section identity/binding; Cedar’s isolated attribute facts are catalog-true. The case-21 multi-card headings also assert attributes not shared by every section member.

**Other lexical/policy cases:** case 16 says “The Cedar 26L Daypack is a practical choice for carrying your laptop and other essentials.” Its actual-product lexical check rejects `other/essentials`; the generic proxy additionally rejects brand/capacity. Case 20 says “This choice meets needs of everyday travel and short trips.” and is rejected for `meets/short`, without a detected unsupported catalog-tag claim. These diagnoses do not authorize broader wording or establish universal semantic safety.

**Style conflict:** the locked serialization emphasizes selected identity and factual tags globally. Many headings consequently become full product-name sentences, while the existing generic heading proxy intentionally has no product identity/capacity vocabulary. Missing or malformed output and native truncation do not explain this run: all 22 locked responses stop normally and preserve exact field order.

The JSON includes exact complete raw generations, case/section/product IDs, actual member tags, claimed-tag intersections and missing-member counterexamples, named-identity evidence, original provenance, proxy-versus-actual lexical checks and SHA-256 hashes. Counts describe this fixed synthetic run only; they imply neither human conversion lift nor generalization.
