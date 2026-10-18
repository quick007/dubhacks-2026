# Shopping model-input bridge review

Changes requested for the reviewed source SHA `b0ed42001fd5e1090a7318c438b65451de29782c48b187aae34c9d2883c106a4`. Six CPU tests pass (0.413 s), but two fail-closed correctness gaps need owner fixes. No API files were edited and no native encoder, optimizer, training or serving action was run.

1. `catalogs()` assigns target indices from unchecked external-mapping order. Swapping mapping entries 0 and 1 is accepted while the intake map stays unchanged: mug becomes target index 1, whose public catalog label is Everyday daypack. Current source hashes record those altered bytes rather than independently verifying their identity binding. Require the authoritative frozen projection/snapshot and test rejection of a mapping permutation. The actual present catalog and mapping do match the independent model pins; current all-16 target labels are correct.
2. `prepare(..., allow_synthetic='false')` accepts a synthetic row because the string is truthy. The CLI supplies proper booleans, but `load_packet` trusts a serialized manifest's flag without checking its type. Require an exact boolean and reject a rehashed packet with a non-boolean flag. This preserves the requirement that synthetic inclusion needs an explicit flag.

The central consumer boundary is sound. Requests contain fixed context text plus observed mode/order/shipping only. Chosen product, variant, candidate list, model identity and event/session identity stay outside encoded request text. Chosen product becomes a separate label; candidates remain exposure provenance, with no negative labels inferred. Every session's split is independently recalculated from its ID, preserving TRAIN/validation/test isolation.

`cpu_consumer_check` calls the actual `store_function_adaptation.fit_inputs`, with explicit fake 4096-dimensional CPU vectors derived from public context/catalog text. Its store rejects any validation/test request access; the test exercises one row in each split and observes only one TRAIN request load plus the 16 catalog rows. Empty accepted TRAIN data skips the consumer instead of fabricating a sample. This is a protocol check, not native feature, optimizer, training-quality or objective-suitability evidence. The v5 fixed-count driver and serving head are untouched.

The bridge correctly states that no shopper query was collected. These observations can support context-conditioned choice statistics; they do not supervise semantic shopper intent. The manifest's false query/training/negative-label claims are checked on readback, and packet-file hashes, derivation and source pins are regenerated. Independent production-row machine evidence remains pending from the API owner.

Run the existing tests with:

```sh
.tools/pretrained-venv/bin/python -B -m unittest discover -s apps/api/scripts -p test_shopping_model_input.py
```

Machine findings, source hashes and exact reproductions are in `SHOPPING_MODEL_INPUT_BRIDGE_INDEPENDENT_AUDIT_V1.json`. Recheck the two owner corrections before accepting a final consumer packet.
