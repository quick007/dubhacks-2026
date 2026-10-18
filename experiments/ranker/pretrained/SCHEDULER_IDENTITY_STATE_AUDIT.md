# Scheduler identity state and evidence audit

CPU-only independent audit of the completed run at `0f632d3d939a0a3dcc8f0e1d43022d974b289489`. No model or native execution, source changes, output repair, validator changes or Q001 edits were performed. The saved evidence supports correct cache reuse and faithful prospective values-only transport for this bounded comparison. Neither protocol produced a valid complete page.

## Provenance and frozen inputs

Verified the run source and frozen case file byte-for-byte against the execution commit. Also verified committed bytes for the shared-evidence implementation, decoder/native test, validator, fusion, page-plan and heading sources. Installed `mlx 0.32.4` / `mlx-lm 0.32.2` match the report; the existing actual, non-skipped native compatibility receipt passes its runtime/source gate. No new native test was launched by this audit.

| Artifact | SHA-256 |
| --- | --- |
| Executed scheduler source | `53c7a72f44285ebc501adad47832b0a079b07298042b1585417fcfdb4680dcc1` |
| Frozen cases | `41f3ce458a6bf355fda7e41a6bd62da0661c70d9e3e8c59943ed94d48e47c9b6` |
| Saved public results | `4b3a653bb3a026f4d175e6fc1ebcba272456487aa3637afab5d38129cb451b3a` |
| Raw run report | `aea0919213935b7b1bfe644e45fe65779af908716928d41b6088f8e548d36675` |
| Native controls | `faf8c2811ce7f24f205d46887ce80a725408dee821bdd9f0427544ee61830f05` |
| Run case plan | `287446c4b5666c3d6fbb5443e1ec3f8b31798d982ab629257ea8fde8a3ba2aa7` |

All declared evidence, validator, full-grid and historical-baseline hashes match the actual files. The raw report equals the public report with its evidence map removed; each of the six separately saved arm files equals its report record.

The roster is one previously inspected diagnostic plus two fresh synthetic cases frozen before this protocol generation. The new cases contain twelve product IDs and twenty-four variant IDs disjoint from the old catalog and each other, new identity names and tag combinations relative to that old catalog, no history, six eligible candidates each, and no ranking-quality oracle labels. The NONE case has no stated preferences; the other has explicit compact/durable preferences. These are prospective synthetic protocol probes, with a separate two-case denominator, rather than independent production-generalization evidence.

## Complete-state controls

The values-only layout was used for all three control modes: fork, independently rebuilt common evidence per lane, and repeat. All three modes completed. Their pristine source hash, before and after, is:

`4f0e09eb5be7f96343bca6835eb1f8b12c702269c42fcdf8559a548f1a144f8b`.

All three independently rebuilt common-evidence lanes match this hash. They rebuild the common extension from the same original ranking state; they do not independently rebuild ranking prefill. Repeat reruns the cached state, rather than uncached full-prompt execution.

Independently compared every control row's metadata, prompt hash, suffix length, cap, raw sampled tokens, text tokens, stop token, finish, AR steps and distribution digests. All match exactly. Each mode has sixteen responses: heading seven including EOS, title seven ending by length under its diagnostic seven-step cap, and badge two including EOS. All forty-eight persisted distribution digests match across corresponding observations, and each digest count equals the corresponding raw-response count. The executed observer checks every float32 response vector for finiteness before hashing and records success. Actual vectors are not persisted, so their finiteness cannot be independently recalculated from the digests alone.

## Prompt, transport and completion checks

Loaded only the pinned local tokenizer, with local files required and remote code disabled. Reconstructed all eighty-four full prompt token sequences from the frozen requests/catalogs, original ranking template, protocol-specific common turn and field turn. Every rank-prefix hash, common-message hash, complete-prefix length, per-field prompt hash and suffix length matches. Labeled rows also match their persisted full-input hashes. Every saved raw text is the exact tokenizer decoding of its saved text-token IDs. This CPU reconstruction does not execute the backbone.

| Case | Rank tokens | Labeled common / field suffix total | Values common / field suffix total |
| --- | ---: | ---: | ---: |
| Old diagnostic | 1,446 | 950 / 504 | 965 / 630 |
| Fresh NONE | 1,431 | 948 / 504 | 963 / 630 |
| Fresh preferences | 1,468 | 955 / 504 | 970 / 630 |

Every arm has six unique selected products, fourteen rows in exact `S1,S2,T1,B1,...,T6,B6` order, width three, prefill chunk 256 and aggregate sampled cap 224. Paired arms have identical product/image selections, support/evidence capsules, ranking-prefix hashes and ranking-state hashes. Each immutable shared source is unchanged before/after generation.

For every values row, canonical text is exactly `trustedField + ": " + rawValue`; `rawValue` equals the original saved model text byte-for-byte. Recomputed wire rows, transport failures, global finish, strict assembly, page plan, field audit and quality match all six saved arms. No old unlabeled row was reused as a values result. All forty-two values rows stop naturally and have no transport rejection. The labeled preferences arm has one length-truncated heading; its global failure and zero official accepted fields are retained. Raw-token length, AR-step count, cap and EOS accounting agree for all eighty-four rows.

The old labeled diagnostic exactly replays the retained shared-cached raw tokens, finishes and quality. New prompts and canonical host labels change the transport and effective generated payload capacity: labels consume no sampled tokens in the values arm. This is not a controlled attribution to cache reuse alone.

## Measured results and limits

| Case / protocol | Accepted titles / badges / headings | Stop / length | Text tokens / AR steps | External full ms | Valid page |
| --- | ---: | ---: | ---: | ---: | --- |
| Diagnostic, labeled | 3 / 2 / 0 | 14 / 0 | 110 / 124 | 8,644.73 | No |
| Diagnostic, values | 5 / 4 / 1 | 14 / 0 | 89 / 103 | 8,278.18 | No |
| Fresh NONE, labeled | 1 / 0 / 1 | 14 / 0 | 91 / 105 | 8,508.63 | No |
| Fresh NONE, values | 6 / 0 / 0 | 14 / 0 | 84 / 98 | 8,369.18 | No |
| Fresh preferences, labeled | 0 / 0 / 0 | 13 / 1 | 111 / 124 | 8,968.93 | No |
| Fresh preferences, values | 2 / 4 / 0 | 14 / 0 | 91 / 105 | 8,991.87 | No |

These are unchanged validator counts. A `NONE` omission is not an accepted generated badge. Independently recomputed both summaries: zero valid pages out of three per protocol overall and zero out of two fresh cases per protocol; six completed attempts, zero failed/unattempted arms. There is no valid-page latency improvement to claim.

Concrete remaining limits are visible in untouched output. Diagnostic values heading `Compact durable choice for everyday travel` fails section-universal evidence, while title `River 14L Daypack for everyday travel` misses its required observed attribute. Fresh NONE values include `B6: packable` despite requiring `NONE`. Its `T6: Meet the compact everyday travel choice` is accepted by the current lexical validator but omits the selected Aster identity entirely. Thus six accepted titles there do not establish six identity-preserving titles. The existing validator also does not enforce the prompt's title/heading word ranges; those diagnostics and independent semantic review remain necessary.

Disjoint measured structured/common-preparation/decode/assembly sums are below each internal full completion by 372.16–499.03 ms; host hashing/layout and other work explain a residual rather than overlapping stage attribution. External completion exceeds internal completion by 1.10–2.84 ms and includes response serialization; independently re-serialized response byte counts match. Total process wall time is 66,189.55 ms including resident load and controls. Observed MLX peak is 6,362,874,212 bytes, below the declared adjustable seven-GiB process guard of 7,516,192,768 bytes. It is allocator telemetry, not total system memory or a hard environmental quota.

One paired attempt per case is insufficient for p95 or a reliable latency-effect estimate. Values-only removes the observed label failures, but factuality, identity and preference responsiveness still block every page. The full 12/24-card grids were not run, and no reduced field set or repaired output is counted as success.
