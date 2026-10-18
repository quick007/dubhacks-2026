# Speed/quality frontier packet: independent audit

The corrected [v1 packet](store_speed_quality_frontier_v1/manifest.json) passes this bounded CPU and public-source audit: **32 distinct queries, two author-written primary-function labels for each of the actual 16 catalog products**. All 32 labels reproduce the selected product's public name, category and search terms exactly. The request/catalog hashes match the manifest. No native model output, timing or quality outcome exists in this packet.

The [JSON audit](STORE_SPEED_QUALITY_FRONTIER_V1_AUDIT.json) records every query, label, source pointer, semantic rationale, parser result and source hash. Audited public-request SHA-256: `ce26a7fdf13a39e9db80f5fb1745013fc3c1d009a280bbdf6869cfbf35e98701`; manifest: `63dd7072a974dcdf3f19d9894e68e01e16397133c7c53a4b5931efc907adcb10`; catalog: `dea9866bb96dc82cec729875c8e680197d223573cbd5d0a07aab583a39c74468`.

## Two issues corrected before execution

| Case | Original problem | Final wording and status |
| --- | --- | --- |
| `frontier-01-1` | “a small everyday backpack” implies size not established by *Everyday daypack*. “Pack a smaller plan” is a slogan, not capacity evidence. | “an everyday backpack”; primary target remains reasonable without the unsupported size attribute. |
| `frontier-03-1` | “a waterproof jacket with a hood” is factually supported for the jacket, but the existing literal-waterproof parser leaves only one eligible product, contradicting six cards/20 fields. | “a hooded rain jacket”; supported by the public jacket name/category, with all 16 products eligible. |

The parent applied these fixes and preserved the original packet in [draft_01](store_speed_quality_frontier_draft_01/manifest.json). The auditor did not edit either fixture.

I executed the existing `checked_request`, `validate_request`, `guard`, `rank`, `guarded_order` and `exact_name_order` definitions through AST extraction of their exact CPU code. Sixteen zero scores exercise deterministic eligibility without encoding, fitting or model imports. **Every final request has all 16 products eligible**, no budget, hard requirement, additional guard rejection or whole-name override. Every row declares six cards/20 fields. Thus all 32 support that fixed workload; this does not guarantee that a future model returns six valid cards or that all six are relevant alternatives. The original waterproof draft independently produces one eligible item under the same code.

## Support and ambiguity

All 16 product classes have a reasonable primary-function target after correction. Notebook/writing follows *Pocket field notes* and public notebook image metadata; mug/drinking is an ordinary product function. Materials, hood, cordless lamp, bamboo lid, cushioned headphones, low-top sneakers, soft cotton crewneck and striped towel are explicit public names/categories. Water, groceries, reading, food, music, walking, travel, sun, cables and swimming also have public search-term support. “Cozy” is a declared throw search term. No query establishes thermal protection, brightness, absorbency, airline allowance, UV rating, comfort measurement, dimensions or capacity. “Bedside reading light” is intended placement/use; it must not become an eye-safety or brightness claim. “Acetate frames” is an ordinary reading of *Round acetate sunglasses*, not independent material testing.

These are author judgments about the best primary product class, not exclusive truths. A daypack, tote and duffel can share carrying uses. Broad intended uses and close paraphrases make this a useful known-catalog retrieval check, but not broad preference composition or six independently relevant recommendations. Copy factuality must be scored against each selected product's own facts, independently of a top1 label.

Three final queries exactly repeat existing v3 TRAIN preferences: `frontier-03-1` → `train-03-1`, `frontier-05-1` → `train-05-0`, and `frontier-15-2` → `train-15-0`. Other queries are close public-function paraphrases. Keep the **curated synthetic development** designation; “32 unseen queries,” sealed evaluation or independent human preference truth would be unsupported.

## Inference boundary and limits

All 32 `request` objects contain only `query`; no hidden persona, gold ID, label or evaluator field appears inside them. Offline labels occupy a separate file. Future inference must receive **only `row.request`**. The outer `caseId` embeds the target catalog index (`frontier-00`, etc.); passing the whole public row or case ID into the model would expose a label pattern. Scheduling metadata and offline relevance labels must stay outside the model prompt. This audit verifies static separation, not a future runner's call path.

No further mandatory wording change was found in the corrected packet. Freeze these hashes before native execution and check request-only projection in the runner. This packet supplies no budget, negative preference, color preference, multi-need, stock, size, abstention or hard-constraint evidence. It binds the pinned actual-store catalog bytes; it does not perform a new live browser export or inspect image pixels. The separate shopper evaluation packet remains untouched.
