# Shared evidence state audit

Independent CPU audit confirms that the shared-prefix controls passed, the retained six-card baseline replayed exactly, and repeated/cached shared layouts supplied identical full token inputs and produced identical raw outputs. Cached/plain tracing also preserved raw outputs, validation and scheduling. All four attempts remain invalid complete pages. No native/model work was launched by this audit; only this report was created.

## Scope and provenance

Executed commit: `a368fe36d4ba3535dc29ed9152b8e1ad80bb6f12`. Executed `shared_evidence_experiment.py` SHA-256: `2f02403b87fe64658b28de947ee7d1b928f471398cd666b5dfc607742a4e59a3`. The source on disk matches its case-plan and final receipt. Runtime metadata matches installed MLX 0.32.4 / MLX-LM 0.32.2. The executed native compatibility receipt also matches the current runtime and shared decoder/test source hashes.

All four published evidence hashes were recomputed. Published results without the added evidence map equal ignored `data/ranker/pretrained/shared-evidence-v1/report.json` exactly. Each saved arm was replayed through unchanged raw assembly and strict validation, with identical raw text and quality results. All six selected product/image choices, fourteen unique field rows and the 224-step output budget were preserved.

| Evidence | SHA-256 |
| --- | --- |
| `SHARED_EVIDENCE_RESULTS.json` at audit time | `72de2e0cf460f3e1f61ee37af70bdcf8ee1f4504062a78031b6f3be14da56f64` |
| Raw `report.json` | `74bb1ea22b660f3be3f279386c5440e00317624a2095cf183de630efecd26b06` |
| `case-plan.json` | `7f0b42d49b877acadfb5f248ec86550a6143f9cbc10dd335a0e4cc7cf4304be1` |
| `input-analysis.json` | `5bebc80f14507e38dd3415c044ad9e9ebf2f2a46944c85b711954a848a2504c7` |
| Frozen `BATCH_STAGE_RESULTS.json` baseline | `751db3762e0d842a7ec7b99cce0c2d5ef1bded9f89efaf6bb938207d143cf77e` |

## State and response controls

Fork, independent-common-rebuild and repeated-fork controls all completed. Each generated seven heading responses, seven title responses and five badge responses, with the badge ending naturally at EOS 248046. There are 57 sampled responses and 57 corresponding distribution hashes across the three controls. Every row's response count, `arSteps` and distribution-hash count agree. EOS IDs and stop-token accounting agree.

All three independently rebuilt lanes have complete-cache hashes equal to the pristine shared state. This reconstructs the common evidence extension separately for each lane from the same reference ranking state; it is not three independently recomputed ranking prefills. The pristine evidence hash remains `c8ab02b59bf64e96dcc7501e2de793d2283367d9fd3957fe79fdbdeab2fd372c` before and after every control. Raw tokens, finish reasons and every corresponding float32 response-distribution hash match across the three controls.

The executed observer converts each native response distribution to host float32 and explicitly rejects any nonfinite value before hashing it. All three receipts record `allObservedDistributionsFinite: true`. This audit verified those source checks, flags, counts and exact distribution hashes. Full probability arrays were not saved, so finiteness and hash contents cannot be independently recomputed from raw vectors here. These controls establish the tested fork/rebuild/repeat behavior, not universal native cache correctness.

## Input boundaries and full attempts

The ranking prefix is 1,446 tokens. The common evidence user turn adds 950 tokens once, producing a 2,396-token complete prefix without an assistant-generation marker. Fourteen minimal field suffixes total 504 tokens. Complete token hashes and lengths match row by row between repeated and cached shared layouts, despite their different cache boundaries and prefill segmentation. A separate CPU-only load of the pinned local tokenizer, with downloads and remote code disabled, reconstructed all fourteen full-token hashes and exact ranking/common/field boundaries independently.

| Layout | Aggregate suffix work after ranking prefix | Full completion wall | Output tokens excluding EOS | Sampled responses including EOS | Complete valid pages |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained baseline | 3,437 tokens | 11,875.29 ms | 132 | 142 | 0/1 |
| Shared evidence repeated per row | 13,804 tokens | 25,778.34 ms | 110 | 124 | 0/1 |
| Shared evidence cached once | 950 + 504 = 1,454 tokens | 8,699.84 ms | 110 | 124 | 0/1 |
| Shared cached, instrumented | 1,454 tokens | 8,970.37 ms | 110 | 124 | 0/1 |

The fresh retained baseline's raw tokens, finish reasons and strict quality equal the frozen six-card `plain256` baseline. All rank/evidence source hashes remain unchanged in every full attempt. Repeated and cached shared outputs match exactly across all fourteen rows; their full input-token hashes also match. The cached trace pair additionally preserves admissions, finish order, response rounds, live width and row steps.

The shared layout reduces suffix token work by 57.70% relative to the old row prompts. Its cached attempt completed 26.74% earlier than the retained baseline in this single inspected pair, including 1,348.36 ms of common evidence preparation. Compared with the same full-input shared layout repeated per row, caching completed 66.25% earlier with identical raw outputs. Order, pressure and observation overhead remain uncontrolled; baseline-to-shared also changes prompt layout and generated wording. These are measured attempt times, not a general serving-speed or usable-page claim.

## Timing accounting

Cached-trace native exclusive intervals sum to 5,103.93 ms. Recomputing all category totals from event exclusives gives the same sum as disjoint root scheduler/insertion inclusive intervals, with no negative exclusives or parent/child double counting. The enclosing native suffix/decode timer is 5,105.75 ms, leaving 1.82 ms outside traced roots.

Within the trace, minimal suffix prefill takes 1,913.76 ms, startup/first-sample forwards 523.85 ms and active decode/lookahead 2,578.75 ms. Common evidence preparation takes a separate 1,370.21 ms outside that native trace. Every one of fourteen UIDs has exactly `N+1` native forward appearances for `N` sampled responses: 138 total forward lane appearances for 124 responses. The terminal native lookahead was retained.

To reconstruct full wall time, add structured work, common preparation when present, the inclusive continuous-decode window and assembly/validation, then retain the residual. Those phase sums are 11,410.40 / 25,378.56 / 8,279.02 / 8,493.92 ms for baseline / repeated / cached / cached-trace. Residuals are 464.89 / 399.78 / 420.82 / 476.45 ms, respectively. Residuals include complete-state hashing/transfers, layout preparation, full-input retokenization/hashing and orchestration; the profiler does not separately attribute every item. Native stage totals are nested within continuous decoding and must not be added again.

Process wall time was 69,300.87 ms including loading, controls and all attempts. Peak MLX allocation was 6,366,069,178 bytes under the declared seven-GiB allowance of 7,516,192,768 bytes. This separate allowance does not replace any preserved failed six-GiB result. Cached arms retain a larger KV history; runtime memory observations and state hashing are part of the recorded experiment.

## Quality and remaining work

All fourteen shared rows stop naturally, compared with four length-limited baseline rows, but the shared page still has missing field labels, extra/unparseable lines and factual/heading failures. Examples include unlabeled `Simple daily choice for everyday travel`, `Durable and compact` and `NONE` lines. Global validation remains false in all four attempts. Three titles and two badges receive individual strict acceptance in shared arms; those counts do not rescue complete-page acceptance.

The evidence supports sharing repeated instructions/evidence as a performance direction while preserving complete hybrid state. The next quality experiment should address field protocol and per-item/section factual binding with the shared prefix retained, using unchanged validators and raw-output inspection. Twelve/twenty-four-card quality, latency, ranking coverage and memory remain unmeasured; this experiment contains one inspected six-card case. No model, runtime, validator, UI or external integration was changed by this audit.
