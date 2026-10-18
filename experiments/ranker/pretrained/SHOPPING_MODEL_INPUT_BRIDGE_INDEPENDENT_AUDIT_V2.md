# Shopping model-input bridge independent recheck

**Pass for the bounded context-only consumer contract.** Both v1 correctness findings are fixed in bridge source SHA `ad18dfac964ed36d5aeb8e572aea96d8713b639bf924a137b4086ae2b36143f2`. The original changes-requested v1 audit and all earlier packets remain preserved. No API files were edited by this reviewer.

The mapping path now calls actual `store_function_adaptation.load_snapshot`, which delegates to the independently pinned raw export and exact ordered catalog projection. Reordering external mapping, model inputs or raw products is rejected, including when their per-file receipt hashes are rewritten. Thus the chosen public ID cannot silently acquire a different class index. Exact boolean guards at both preparation and serialized readback reject truthy strings, integers, null and containers as synthetic permission.

All eight Python CPU tests independently pass in 15.807 s, including those regressions, all sixteen target mappings, target-independent requests, session-split isolation and serialized tamper rejection. Fifty additional lightweight checks verify the twelve source pins, both complete packet inventories and manifests, saved consumer results, input hash and four synthetic choices from two sessions. The producer's five Node choice-builder tests were reported passed; this review did not rerun them.

The reviewer independently loaded and consumed both preserved v2 packets through the actual `store_function_adaptation.fit_inputs` path:

| Packet | Accepted observations | Independent consumer result |
| --- | ---: | --- |
| production-real default | 0 | `skipped-no-training-rows`; no fabricated sample |
| explicit synthetic | 4, from 2 sessions | query `[4,4096]`, catalog `[16,4096]`, targets `[4]`; four TRAIN request loads |

Both recomputed results exactly match their saved receipts. The separate split fixture contains one TRAIN, one validation and one test session; only the TRAIN request and sixteen public catalog rows are loaded, with heldout access guarded by an exception. The synthetic v2 packet itself contains four TRAIN choices and no heldout rows; it does not alone demonstrate a real validation cohort.

Requests remain fixed public context text plus observed mode, ordering and shipping visibility. Chosen product/variant, candidate IDs/positions, session/event identity and model identity remain outside encoder text. Chosen product is a separate positive label, candidates are preserved exposure provenance, and no negative label is inferred. No shopper query was collected, so these rows do not supervise semantic shopper intent or establish a preference or purchase.

The 4096-dimensional vectors are explicit deterministic fake CPU protocol fixtures. No Qwen/native extraction, optimizer, training, prediction-quality evaluation, v5 driver modification, serving promotion or deployment occurred. A future training path still needs a frozen real cohort, native feature provenance, an appropriate observational objective, exposure and repeated-action handling, session weighting, and heldout evaluation. The source/file hashes verify frozen local evidence; they do not independently authenticate the upstream cloud export.

Evidence: `SHOPPING_MODEL_INPUT_BRIDGE_INDEPENDENT_AUDIT_V2.json`, `apps/api/shopping-model-input.verification.json`, and preserved `data/shopping-context-{default,synthetic}-v2` packets.

```sh
.tools/pretrained-venv/bin/python -B -m unittest discover -s apps/api/scripts -p test_shopping_model_input.py
.tools/pretrained-venv/bin/python -B apps/api/scripts/shopping_model_input.py --check-packet data/shopping-context-default-v2
.tools/pretrained-venv/bin/python -B apps/api/scripts/shopping_model_input.py --check-packet data/shopping-context-synthetic-v2
```
