# Image selection and payload checkpoint

The responsive implementation is pushed in `86a4b51`. All87 original rasters
remain byte-identical. Full-size WebP alternatives total13,245,460bytes versus
44,840,476 original bytes;409 lower-resolution variants cover five declared
widths. Gallery cropping, color binding and original-format error fallback are
preserved. The runtime geometry JSON is3,125bytes; an isolated minified resolver
build is4,703bytes versus156,042bytes when importing the full audit metadata.
Full hash/dimension/ICC manifests remain available for offline verification.

The [independent responsive browser audit](CANDIDATE_BROWSER_PROBE_V2_RESPONSIVE_INDEPENDENT_AUDIT_V1.md)
records median product-image transfer21,269,174→749,804bytes (96.47% less).
Whole-resource development-server transfer is still29.51MB, so the image number
is not a whole-page payload claim. Cold content-plus-six-decoded-image median
497.8→523.4ms did not improve; reload348.85→322.6ms did. All48 observed primary
images preserve source identity and clipping at inferred DPR1. No high-DPR,
physical-paint or learned-vision result is claimed. Later compact-import and
balanced-ranker measurements use separate receipts.

After the initial fetch described below, upstream advanced to9c401a9. Canonical
mergeea33cf7 retained teammate API/UI work and every preserved local file; final
telemetry integration5a81487 is included. The API mock fixture moved to16
fieldwork personas, while the actual storefront products/colors/image source
bytes remained identical. The model binding records that auxiliary-source
transition explicitly and passes no fixture persona truth into inference.

The following initial inventory and vision assessment remain applicable.

The safe canonical fetch/pull on 2026-10-18 found main already current at
`ba2b942c83e9e710414b290442acbf0bcd394bd1`. The fetched design branch 88227ca
was 50 commits behind main with no unique commits; vision branch e28a381 was 95
behind with no unique commits. No feature branch merge, reset or stash was
needed. All local model and teammate changes were preserved. The pictures on
those branches were already present in main; the model's actual public catalog
remains 16 products and 48 declared product/color bindings. The frozen import
contains 81 referenced image assets and 195 gallery actions. There are 87 raster
files in the public products folder, totaling 44,840,476bytes.

There is currently **no learned visual ranking or pixel reasoning**. The Qwen
product text encoder reads name/category/itemPrice/searchTerms and available
color names. It does not encode images or image embeddings. The model action
binder chooses the first declared satisfying color and its first declared image.
The React gallery separately chooses a view from deterministic context metadata,
with a declared fallback, and maps the selected color to its gallery sheet/tile.
A shopper can select another color or gallery view. These operations inspect
metadata lists and perform no additional native model forward. The learned
image-selection count per request is zero.

Actual headed Chrome evidence in navigation probe attempt2 shows a payload
problem despite fast loopback timings. Four new contexts fetch 29–30 image URLs
and 20.82–21.30MB encoded image bodies. Same-context reloads transfer only 8.7–9KB
of image response headers, with zero recorded encoded bodies. Those conditions
must remain separate: a warmed browser hides the cold image cost. Some source
PNG sheets are 1.97–2.84MB each; thumbnails can request the same full gallery
sheets. All images remain lazy, but browser proximity heuristics still request
many below-fold assets. No image removal is justified by these data alone.

The initial bounded plan was render-only WebP derivatives at identical dimensions,
with original bytes and public source identity retained. This preserves sheet
cropping, gallery/color binding and teammate imagery. Record source/derivative
hashes, sizes and visual checks, then measure fresh-context bytes and first3/full6
readiness again. The renderer must disclose the derivative mapping in its new
integration receipt. The old16-catalog timing is not evidence for a changed
product catalog; any actual product/fact change requires separate model binding
and evaluation.

For future learned image choice, the lowest request-time cost is to encode
image features offline, reuse the existing fresh shopper-query representation,
and batch-score only the eligible product's real gallery candidates with a
small trained compatibility head. The image encoder and alignment quality are
not implemented or verified here. Pairwise user choices can provide useful
training evidence when both images were actually shown and comparable. A click
or unviewed alternative alone is not a clean aesthetic preference label.

All-pairs comparison needs n(n-1)/2 pairs; an elimination tournament needs n-1
comparisons across adaptive rounds. Neither automatically reduces model work
relative to one batched score per candidate, and both add computation to the
current zero-vision-forward path. No speculative pairwise vision benchmark was
run without a defined preference target, supported visual encoder or labels.
The immediate measured opportunity is the 21 MB image transfer, while ranking
quality and the already fast text path remain primary work.

The user has explicitly endorsed precomputing static catalog/image features and
keeping resident warm caches. That matches the existing16 catalog vectors and48
public fact vectors: fresh query encoding and fresh neural text predictions are
measured separately from setup/feature extraction. Model revision, checkpoint,
source-fact and asset identities must invalidate their relevant cache entries;
a new image should not silently inherit another image's features. A rendered
compressed derivative can map to its original content identity with both hashes
recorded, rather than pretending compression creates a new learned preference.

The pinned model configuration includes a vision tower and processor files,
but the installed MLX-LM0.32.2 Qwen3.5 loader explicitly removes `vision_tower`
and `model.visual` weights in `sanitize`. The current FrozenBackbone declares
vision unused. `mlx_vlm` and Pillow are not installed in that environment.
Therefore configuration support alone is not evidence that the running service
can extract pixels. No visual feature extraction or pairwise image model result
is claimed. A future isolated visual-cache implementation needs a verified
image processor/runtime, source-aligned features and real preference labels;
its setup cost and invalidation receipts belong outside warm-request latency.
