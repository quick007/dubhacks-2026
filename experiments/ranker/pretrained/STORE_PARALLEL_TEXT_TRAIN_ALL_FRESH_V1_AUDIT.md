# Known-catalog parallel text: fresh-query audit

The successful second fresh benchmark has **32/32 complete factual first3 and full6 payloads**, with six cards and20 generated fields each. Independent CPU review finds all640 main fields and480 additional style-counterfactual fields readable and source-correct. Median backend first3 readiness is **193.43 ms**; full6 is **193.50 ms**. This is a fully supervised known-catalog demo, not heldout or browser quality evidence.

Source: `data/ranker/pretrained/store-parallel-text-train-all-fresh-v1-attempt-2/`. The [JSON audit](STORE_PARALLEL_TEXT_TRAIN_ALL_FRESH_V1_AUDIT.json) contains every main raw field, style output, source fact, receipt hash, guard check and ranking miss. The head is the independently audited [320-step TRAIN-ALL candidate](STORE_PARALLEL_TEXT_TRAIN_ALL_DEMO_V1_AUDIT.md), SHA-256 `074169e31bf5f6b99cc5bc35c9b2e64d82396d5225b3a67cc2cf5f613fcc4889`.

All192 main titles retain the correct product identity/color, all192 badges state the selected color and all192 short bodies state its correct color and item price. The64 generic headings make no unsupported shared-use claim. No wrong price, wrong color, added benefit, meaningful repetition or incoherent caption was found. These judgments accept useful concise captions and do not impose the older long-copy grammar. Every accepted raw string equals its serialized visible value; no repair or teacher replacement occurred. Slot-to-text reconstruction and all32 request hashes were verified.

Eight counterfactual sets preserve the same public card facts while changing the explicit style among plain, browse and compact. All480 resulting fields remain factual, and every set changes all20 values between plain and each other style. This demonstrates the trained style control, not open-ended shopper-intent prose. The audited generator uses public cached Qwen features and learned word-slot weights with a107-word vocabulary; canonical metadata remains separate. All48 product/color bindings and all three styles are TRAIN-supervised.

Offline primary-function relevance is **30/32 top1;31/32 target present in six**. The two misses remain:

- “a notebook for writing ideas” ranks the organizer pouch first and omits field notes from six.
- “a cotton blanket to use as a throw” ranks the cotton towel first; the throw is third.

The relevance labels remain outside inference. These32 curated development queries include three exact prior ranking-TRAIN wording overlaps. Correct copy does not make every below-first card an equally relevant alternative, and this packet does not establish broad preference composition.

All eight separate deterministic guard probes pass: exclusive exact-name abstention, inclusive exact-name/budget ordering, unknown stock, unknown size, declared waterproof only, contradictory colors as an expected safe rejection, typed inclusive budget and a nondefault Cream variant. These checks validate bounded guard behavior, not learned constraint reasoning. Unknown stock/size and impossible joint colors produce no invented actions. The stored shutdown integrity receipt reports all927 frozen tensors and the copy checkpoint unchanged; this audit did not reexecute or rehash model tensors.

The timing receipts include fresh Qwen query encoding/ranking, hard guards, cached-public-feature head generation, validation, source freshness and JSON serialization. They exclude startup, offline feature extraction, fitting, network, React and image transfer. All20 fields are generated in one pass before first3/full6 serialization. Thus193.50 ms is observed backend completion for this checkpoint; it is not physical paint, browser above-fold readiness or shopper conversion.

The failed first fresh attempt remains separately preserved. Its32 main requests passed, but the overall run failed when a contradictory-color guard raised an uncaught exception, after five guard receipts. The second attempt explicitly records that safe rejection and completes all eight probes. Neither the first attempt's status nor the original v1 heldout-color failure is rewritten. This audit performed no native call, GPU work, serving promotion or Git action.
