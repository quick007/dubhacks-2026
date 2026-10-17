# Retained manifest profile

The retained manifest spends about 1.90 seconds on the shared 1,446-token ranking prefix and 1.73/3.33/6.73 seconds generating one/three/six-card copy. Decision heads take 0.14–0.31 ms, complete-cache handoff needs no fork, and validation/serialization take only milliseconds. The measured generation region dominates as page size grows.

All six native arms completed: three unchanged baselines and three synchronized profiles. Every pair matched raw sampled token IDs, text, stop reason, strict quality, selected product/image IDs and response byte count. The baselines also replayed historical manifest text. These are three inspected diagnostic cases, one pair each, with baseline/profile order alternating; they are not a new twelve-case quality result or a speedup claim.

## Paired timings

| Cards | Baseline complete | Profile complete | Shared prefix prefill + pooled readout | Native generation | Heads | Strict validation | JSON serialization | Valid page |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 3739.7 ms | 3658.4 ms | 1895.8 ms | 1733.2 ms | 0.308 ms | 1.015 ms | 0.069 ms | yes |
| 3 | 5190.0 ms | 5258.1 ms | 1897.8 ms | 3332.5 ms | 0.234 ms | 0.908 ms | 0.047 ms | yes |
| 6 | 8190.5 ms | 8667.5 ms | 1906.5 ms | 6733.8 ms | 0.136 ms | 1.273 ms | 0.090 ms | no |

Input construction/tokenization is 10.1–10.6 ms; pooled-feature host transfer and finite checking 9.5–11.2 ms; empty-cache/input materialization 0.07–0.08 ms; state metadata 0.09–0.18 ms; deterministic action/section projection and structural validation together under 0.4 ms; suffix boundary/tokenization 1.7–4.4 ms. Strict copy validation remains unchanged. Section grouping remains deterministic, trained heads choose products/images, and fresh wording comes from the same frozen backbone. Only text catalog descriptions are supplied.

## Native generation and line delivery

| Cards | Setup before first progress | Suffix prefill | Last prompt token + dispatched lookahead | After prompt progress | Suffix tokens | Text tokens / sampled steps incl. EOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 56.0 ms | 409.0 ms | 85.5 ms | 1182.7 ms | 275 | 29 / 30 |
| 3 | 62.4 ms | 575.6 ms | 78.7 ms | 2615.8 ms | 405 | 65 / 66 |
| 6 | 31.8 ms | 847.6 ms | 80.2 ms | 5774.2 ms | 625 | 139 / 140 |

The four time columns sum exactly to native generation wall time. Native callback(total) occurs after dispatching one-token lookahead, so that boundary is not a pure prompt-only timer. All streams ended with sampled EOS 248046. The report stores every raw token, response fragment and timestamp. Per-field windows describe delivered lines in one serial manifest stream, not independently attributable GPU time; final EOS/generator completion adds 41.6/39.6/44.7 ms after the last text delivery, already included in post-prompt time.

Profile-minus-baseline totals are −81.3/+68.1/+477.0 ms. They include synchronization/observation, order and machine-pressure variation; the unchanged baseline also builds an evidence capsule and metadata omitted by the profiler. They cannot be labeled pure instrumentation overhead or improvement. Native prefix pooling evaluates only pooled readouts plus every recurrent/convolution/KV state, never unused full logits. The same complete cache object passes to generation without trim, splice or reset.

## Quality and raw examples

One- and three-card pages remain valid. The six-card attempt preserves six accepted new titles, four generated badges and two correct NONE values, but both headings fail factual intersection validation. No validator was relaxed.

Accepted one-card output:

```text
S1: Compact water-resistant everyday travel choice
T1: Grove 23L Daypack for daily compact needs
B1: compact
```

Rejected six-card headings (all raw output remains in the result JSON):

```text
S1: Everyday compact water-resistant choices
S2: Durable compact daily travel options
```

## Cost diagnosis and next experiment

[Independent cost audit](DECOMPOSED_COST_AUDIT.md) shows factorized suffix payload grows from 668 to 1,734 to 3,437 tokens, versus manifest 275/405/625. It has no established usable-page win. Explicit Python deep copies (0.4–4.3 ms) and strict assembly (0.5–3.9 ms) cannot explain multi-second excess. That excess is inside native admission/prefill/decode/state operations. Existing factorized256 versus manifest512 prefill chunk sizes are also a disclosed confound.

The next local performance experiment should instrument the existing native BatchGenerator prompt/admission versus active-token decode boundaries on the same fixed scale cases at width three, with complete-state controls and unchanged prompts, chunk size and validation. It should then run a separately declared 256/512 chunk comparison under the same seven-GiB guard if both controls pass. This identifies whether repeated short suffix processing or active-state batching is the larger target before expanding width or changing prompts. The independent external Q001 packet addresses badge-prompt quality separately and makes no serving-latency claim.

[MTP audit](NATIVE_MTP_COMPATIBILITY.md) finds no switch for this pinned local artifact/runtime: MTP tensors are absent, the model loader strips MTP keys, and separate-draft speculation requires trimmable state that this hybrid cache lacks. No MTP artifact or runtime was downloaded or installed.

## Evidence, memory and limits

Resident model load 1744.1 ms; bounded total wall 36997.9 ms. Peak MLX 6,344,671,500 bytes under explicit seven-GiB guard 7,516,192,768. This peak also fits six GiB, but this run used seven. RSS 2,310,897,664 bytes overlaps MLX allocations and must not be added to them. Available memory before/after: 8,652,275,712/5,696,503,808 bytes. Physical RAM 25,769,803,776 bytes. Timings are resident end-to-end work excluding model loading and report IO.

Executed source SHA-256 `e6b0358309ac3d43c43257e85ce7e500d40e5fc584e395c6c493c4592b9d8539` from HEAD `3b02c4f9bfe42268162d94a5c91cae8c94d6fa1c` plus the uncommitted profiler. [Raw result](MANIFEST_PROFILE_RESULTS.json), [additive analysis](MANIFEST_PROFILE_ANALYSIS.json), and ignored `data/ranker/pretrained/manifest-profile-v1/` preserve case plans, per-arm files, native events and hashes. Two CPU delivery-accounting tests and native pair gates passed; an independent agent replayed all six artifacts on CPU.

The executed profiler source is retained unchanged. Its failure paths have two known robustness limits: an exception during post-arm synchronization/cleanup could prevent a final fixed-denominator report, and an exception inside the profiled arm can lose its local partial events. Neither path occurred in this six-arm run. Harden these paths before extending the profiler; do not infer failed attempts succeeded.
