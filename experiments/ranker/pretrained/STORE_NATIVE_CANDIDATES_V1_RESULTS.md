# Native decoder and generated-earlier cache candidates

These are local synthetic development experiments on the actual 16-product
catalog. All original runs, weights, feature archives and failed outputs remain
under `data/ranker/pretrained/`. The [lossless text archive](store_native_candidate_results/v1/)
contains per-file checksums and raw receipts; weights are excluded from Git.

| Condition | Fixed timed workload | Median full serialized attempt | Complete title/body pages |
|---|---:|---:|---:|
| Fresh adapted AR, width10, fresh root, explicit8GiB | 3 six-card/20-field queries | 6402.946ms | 0/3 |
| Fresh adapted AR, width20, immutable common root, explicit8GiB | Same3, AB/BA/AB order | 4571.540ms | 0/3 |
| Generated-earlier native cache, repaired verification run | 32 six-card/20-field queries | 219.627ms | 0/32 |

The AR ratio is an attempt-time improvement, not useful-page speedup. Both arms
use fresh token generation. The candidate changes both width and prompt-state
reuse, so this comparison cannot isolate either contribution. Same-width
fork/rebuild/repeat controls passed separately at widths10 and20; cache miss/hit
and A→other queries→A token equality passed. Different batch shapes produced
some different tokens, which remain in the raw receipts. Candidate first-three
attempt median was4384.983ms versus5860.355ms; useful-readiness timestamps remain
null. All927 frozen tensors and the copy adapter were unchanged at shutdown.
Peak MLX allocation was7,484,641,720bytes under the explicit8GiB budget. The
memory increase followed fresh OS headroom checks before and after loading;
no application was closed and the ordinary7GiB constructor was preserved.

The earlier width20/7GiB probe completed one ordinary request in4577.697ms,
then failed during synchronized profiling at7,571,165,592bytes, above its
7,516,192,768byte guard. The two remaining ordinary slots and two remaining
profile slots were unattempted. This failed comparison is preserved separately;
it does not establish a completed width20 profile or authorize a hidden budget
increase.

The cache's original eight actual native warming requests covered all48 catalog
product/color bindings and took5411–6439ms each. Original invalid fields were
cached alongside valid fields without retries or selection. The first32-hit
run had a194.784ms median but failed later on a nested process-alarm conflict
while proving a cold miss. Its original unscored relevance placeholder is not
zero ranking accuracy. A separate repair reused85 verified immutable files,
retained the original failure, and completed all32 hits, seven guard probes and
a genuine new native miss followed by a verified hit. Its median was219.627ms,
maximum236.010ms; all32 were below300ms. First-three title/body passed9/32;
full-six title/body and all20 fields passed0/32. Offline top-one relevance was
30/32 after predictions were frozen. This is fast access to earlier generated
text with the producer's existing quality failures, not fresh request-time
copy generation. The service DTO visibly distinguishes fresh text, earlier
text, mixed work and zero-generation abstentions.

The separate catalog-trained parallel text candidate is the useful fast path
under integration evaluation. Its all48 supervised binding scope must accompany
its timing; the earlier heldout decoder failures remain separate evidence.
None of these measurements includes browser rendering, image readiness,
physical paint, human purchases or conversion effects. Startup, controls,
warming, shutdown and archives are outside the per-request medians.
