# Independent section-claim audit

CPU-only results for all 22 preserved prior manifest responses. No model/GPU was loaded; no validator, acceptance rule, generated output, or current comparison source was modified.

Run: `.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.section_claim_audit`

## Findings

- 25 section fields audited across 10 cache and 12 evidence responses.
- Six cache responses contain true universal tag-claim failures. This is not explained by conservative generic-word rejection.
- The six-card evidence response adds another true universal failure response.
- Two single-product evidence responses are generic/numeric section-proxy false positives: the actual selected Field 20L product accepts the wording, while the generic section proxy drops its brand/capacity.
- All assertions passed: five independent synthetic checks and three preserved-population checks. Existing strict acceptance remains unchanged.

## Actual failing/proxy examples

| Case / field | Actual heading | Member facts | Finding |
| --- | --- | --- | --- |
| cache-3-changed-profile-B / S1 | Lightweight roomy backpacks | River 26L Daypack: durable, organizer | Not universal: lightweight, roomy |
| cache-5-changed-catalog-unavailable / S1 | Daily durable compact daypacks | River 26L Daypack: durable, organizer | Not universal: compact |
| cache-6-cold-catalog / S1 | Everyday durable compact daypacks | Grove 23L Daypack: water-resistant, compact; Cedar 26L Daypack: durable, compact; River 26L Daypack: durable, organizer | Not universal: compact, durable |
| cache-7-warm-catalog / S1 | Everyday durable compact daypacks | Grove 23L Daypack: water-resistant, compact; Cedar 26L Daypack: durable, compact; River 26L Daypack: durable, organizer | Not universal: compact, durable |
| cache-8-cold-catalog / S1 | Compact durable daypacks | Grove 23L Daypack: water-resistant, compact; Cedar 26L Daypack: durable, compact; River 26L Daypack: durable, organizer | Not universal: compact, durable |
| cache-8-cold-catalog / S2 | Lightweight roomy daypacks | Field 14L Daypack: lightweight, roomy; Ridge 23L Daypack: lightweight, roomy; River 14L Daypack: durable, compact | Not universal: lightweight, roomy |
| cache-9-warm-catalog / S1 | Compact durable daypacks | Grove 23L Daypack: water-resistant, compact; Cedar 26L Daypack: durable, compact; River 26L Daypack: durable, organizer | Not universal: compact, durable |
| cache-9-warm-catalog / S2 | Lightweight roomy daypacks | Field 14L Daypack: lightweight, roomy; Ridge 23L Daypack: lightweight, roomy; River 14L Daypack: durable, compact | Not universal: lightweight, roomy |
| evidence-r-36f617c0fe5c7f6164 / S1 | Field 20L Daypack | Field 20L Daypack: water-resistant, roomy | Proxy false positive; actual selected product lexical check passes. |
| evidence-r-4171cae29da747fcef / S1 | Field 20L Daypack is a water-resistant roomy bag. | Field 20L Daypack: water-resistant, roomy | Proxy false positive; actual selected product lexical check passes. |
| evidence-page-scale-6 / S1 | Everyday compact water-resistant choices | Grove 23L Daypack: water-resistant, compact; Cedar 26L Daypack: durable, compact; River 26L Daypack: durable, organizer | Not universal: compact, water-resistant |
| evidence-page-scale-6 / S2 | Durable compact daily travel options | Field 14L Daypack: lightweight, roomy; Ridge 23L Daypack: lightweight, roomy; River 14L Daypack: durable, compact | Not universal: compact, durable |

## Interpretation

Grove 23L is compact but not durable; River 26L is durable/organizer but not compact. Their shared section has an empty tag intersection, so “compact durable” is a genuine universal-claim error. Field 14L and Ridge 23L are lightweight/roomy, but their section also contains River 14L (durable/compact), making “lightweight roomy” false for the whole section.

Changed-profile B still selects River 26L even though the observable preferences are lightweight/roomy. A heading claiming lightweight/roomy for that selected River is incorrect. Product locking can improve output faithfulness without repairing recommendation selection.

The two Field 20L headings are factual single-product identity/attribute statements: “Field 20L Daypack” and “Field 20L Daypack is a water-resistant roomy bag.” They remain rejected under the existing generic section proxy. This audit labels the mechanism without relaxing acceptance.

“Options” in the six-card evidence heading is additionally an unknown framing token; that lexical issue does not erase the separate true compact/durable universal-claim error. Missing/unknown-format fields and unknown framing are separate categories in the JSON.

The proposed next prompt should expose the UNIVERSAL intersection explicitly; empty intersections allow generic framing only. Selected identity/capacity and true product facts must remain distinct from the narrower observable preference/history evidence used for badges. No outcome from that next prompt is included here.

The JSON contains every section’s actual product IDs/tags, detected claimed tags, missing-member counterexamples, original provenance, proxy errors, actual-product checks and source hashes. Tag phrase matching is a bounded diagnostic, not unrestricted semantic factuality proof.
