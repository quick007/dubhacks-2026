# Resident actual-store ranking and action seam

Completed24 native B1 query forwards on one resident frozen Qwen backbone.
All16 development-replay score vectors and product orderings match the completed
actual-store trained run exactly. Four additional requests match frozen scores
and actual product/color/image actions exactly while a loaded failed-daypack
copy adapter is explicitly disabled for query encoding. All927 base language
tensor hashes remain unchanged.

Median freshquery→serialized ranked real-store actions: 174.942ms
for the16 replayrequests. The method includes before/after store-source freshness,
whole-query tokenization/encoding,16 verified warm catalog loads, learned scores,
hard guards, exact-namepolicy, real variant/image actionbinding andserialization.
It excludes generatedcopy/images/browserpaint and is a different endpoint/run
from the earlier203.702ms extraction/replay benchmark. Do not infer a speedup.

Four native policychecks pass: exact whole-query Trail mug is first; a$23 item
budget excludes that$24 mug; Cobalt binds the actual Cobalt variant/image; unknown
stock returns0products with6disclosed shortfalls. Exact-name override and image/
color binding are deterministic, not learned. Conflicting color requirements
fail closed. No padding or nonexistent stock/size/delivery facts are supplied.

This is a model-side resident adapter ready for an app owner to consume; the live
React UI/API has not been changed and no browserpaint is measured. No copy mode
is promoted. `store_rank_session.py` rejects an active undisclosed copy adapter
rather than contaminating frozen-base catalog features. Five targeted CPUtests
pass. Native wall15.152s; load2.028s;
MLX peak5,277,654,182bytes.
