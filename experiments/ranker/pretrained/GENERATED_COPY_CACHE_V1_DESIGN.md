# Native-generated catalog copy cache candidate

This CPU-prepared candidate reuses previously generated public-fact copy. It has
not run native warming, changed live serving, or demonstrated a latency result.
It offers a plausible way to remove multi-second copy decoding from warm requests;
fresh query ranking and complete browser readiness still need measurement.

`store_generated_copy_cache_v1.py` exposes `producer_spec`, `warm_plan`,
`GeneratedCopyCache.record_generation`, `GeneratedCopyCache.resolve` and
`personalize_with_cache`. The wrapper times fresh learned query ranking, existing
public hard guards, actual action binding, lookup, native-receipt/factual checks
and full UTF8 serialization. Its timing excludes startup, images and browser paint.

Copy in the current service uses a neutral browsing request and bound public
product/color facts. It does not depend on the shopper query. The same field-user
prompt can therefore be reused across learned orderings if exact prompt equality
is verified. The helper changes only scheduler labels when moving a cached card;
text, sampled tokens, EOS, original prompt, original field IDs and provenance are
retained. This does not support reuse of genuinely shopper-conditioned wording.

Keys include public source facts, external action mapping, exact common/field
messages, sampled caps, adapter/checkpoint/model/tokenizer versions, native width,
temperature and producer/validator source hashes. A changed namespace is required
for changed facts, prices, variants, checkpoints, prompts or validator code. Source
fingerprints are checked at use; the resident rank session remains responsible for
validating current storefront source/export bindings. No hidden persona is read.

The fixed warm plan covers all 48 real product/color pairs in eight six-card
requests. Every request keeps all 20 fields and the 416-token cap, for 160 fields
and a maximum 3328 sampled tokens overall. Two generic section headings come from
the first native attempt; duplicate headings from later requests are archived.
No result is chosen because it passed a quality gate. Failed fields—including the
known lamp-body problem—remain cached as original raw output and are withheld as
null on every response. The current unchanged practical v1 factual gate is used;
prospective v2 usefulness labels do not replace its failure reasons.

Every hit discloses `copySource: generated-earlier`, generation timestamp, age,
hit/miss and original cold-generation cost. It never claims freshly generated
request copy. Original page-generation costs appear on each referring entry:
deduplicate by attempt SHA before totaling them. Per-request cold generation cost
is separately zero on a pure hit, rather than being hidden in a warm latency claim.

A missing card or heading invokes the supplied native generator for the complete
current page. A six-card miss still generates all 20 fields. There is no template,
two-card shortcut, missing-body omission or silent fallback. Without a generator,
`NativeGenerationRequired` stops with explicit missing keys. The callback must
return current-request generation timestamps, exact checkpoint/tokenizer/width
provenance, a same-width control digest and raw rows. Earlier archives cannot be
passed as fresh miss generation. Existing hits are not overwritten or repaired by
new attempts. Original native attempts and all rejected fields remain on disk.

Overnight native work remains owned by root and isolated from all other GPU jobs:

1. Verify the completed adapter, current 16-product/48-color export, frozen model
   tensors and memory headroom. Keep the initial 7 GiB guard; inspect actual pressure
   before any separate decision to change it.
2. Freeze this candidate and the exact warm plan. Prove fork/rebuild/repeat equality
   at the chosen native width, then execute the eight complete warm requests once.
   Preserve cold startup, all cold-generation costs, token counts, peak memory,
   raw rows, EOS receipts, timestamps and factual rejection reasons.
3. Freeze a public frontier before looking at candidate results: all 16 existing
   actual-store evaluation paraphrases, all 10 shopper public observations, and
   eight guard probes for inclusive/exclusive prices, stock/size abstention,
   exact-name eligibility and declared colors. Report overlapping queries
   separately instead of pretending 34 independent unseen personas.
4. Compare uncached fresh copy and warm cached copy on the same full useful
   first-three/full-six payloads. Keep warm hits, cold misses and all invalid or
   withheld-field results distinct. A cached speedup does not establish better
   ordering, useful accepted copy, new wording per request, or unseen catalog skill.
5. Measure browser commit plus two-frame opportunities separately using the existing
   local readiness instrumentation. Report image readiness and physical paint as
   unmeasured. Do not call rank-only, cache-only, or server-only timing full UI latency.

The target is low hundreds of milliseconds for useful personalized content under
declared warm-cache conditions. No measurement in this preparation establishes
that target; query encoding may still dominate it.
