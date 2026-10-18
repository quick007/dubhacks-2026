# Full-grid scaling estimate: twelve and twenty-four cards

Twelve- and twenty-four-card grids have not been run. Rough two-point projections of the current implementation put completion in the tens of seconds, before accounting for the larger candidate context needed to select a full grid. These are planning estimates, not throughput guarantees or evidence of accepted pages. Every requested product/image decision and fresh title/badge/heading remains in scope; no smaller grid or template substitution is assumed.

## Measured inputs and estimated latency

The inputs below are the actual fixed three-/six-card attempts in `PARALLEL_SEMANTIC_SCALE_7G_RESULTS.json` (SHA-256 `ce9d171ad3f973bd4c93891cd08f2e58d25604c5a4f201dd944cd4bc2a72fddf`). Both used the same six-candidate, 1,446-token ranking prefix, one resident pinned nine-billion-parameter model, and a separately declared seven-GiB process guard. Independent generation uses compact heading prompts and native prefill 256; manifest uses its original prompt and native prefill 512.

For each arm, fit `T(N) = a + b*N` through its two measured full-completion times, with `b = (T6-T3)/3` and `a = T3-3*b`. Full completion includes ranking, complete cache handling, suffix processing, generation and strict validation. Resident model load is excluded from each attempt.

| Path | Actual three cards | Actual six cards | Fitted intercept a | Fitted seconds/card b | Estimated twelve cards | Estimated twenty-four cards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Manifest | 5.132 s | 8.565 s | 1.699 s | 1.144 | 15.4 s | 29.2 s |
| Independent width three | 6.389 s | 11.904 s | 0.874 s | 1.838 | 22.9 s | 45.0 s |
| Independent width six | 5.894 s | 11.035 s | 0.754 s | 1.713 | 21.3 s | 41.9 s |

The intercept is an algebraic fit, not measured startup cost or a physical lower bound. Actual ranking alone takes about 1.9–2.1 seconds in this run; timing/order noise and the change from one to two sections affect the fitted slope. There are only two inspected points, one attempt per cell, and no variance estimate or confidence interval. Linear scaling is an explicit assumption, not an observed stable throughput law. Twelve cards are twice the largest measurement; twenty-four cards are four times it.

The fit holds candidate/ranking cost, average wording length, selected evidence, section density, runtime scheduling and effective token cost approximately constant. A real twelve-/twenty-four-card request needs at least twelve/twenty-four eligible unique products, while these measurements rank only six candidates. Larger ranking contexts, longer attention histories, more two-digit labels, different facts, memory pressure and new numerical shapes can make the actual result slower or fail outright. No predicted speedup is assigned to an unimplemented architecture.

Measured quality is separate: manifest accepted the three-card page and rejected the six-card page (1/2 in this separate supplement). Every independent three-/six-card page was rejected (0/2 per width). There are **zero actual twelve-/twenty-four-card attempts and zero evaluated full-grid acceptance outcomes**. This report does not extrapolate page pass probability or merge the two-case supplement into the main twelve-case report.

## Work and current implementation limits

Assuming the intended deterministic grouping remains three products per section, twelve cards require four generated headings, twelve generated titles and twelve badge decisions/outputs: **28 field rows**. Twenty-four cards require eight headings, twenty-four titles and twenty-four badge decisions/outputs: **56 field rows**. Badge `NONE` remains an allowed generated outcome where policy permits it; it is not a reason to omit the field from the job.

Actual independent suffix payload grows from 1,734 tokens at three cards to 3,437 at six. A purely illustrative fit `S(N)=31+567.667*N` gives about 6,843 aggregate suffix tokens at twelve cards and 13,655 at twenty-four. This is the sum across independent lanes, not one context window. Manifest suffixes are 405/625 tokens; their two-point fit `185+73.333*N` gives roughly 1,065/1,945 tokens. These are estimated prompt payloads, not measured tokenizer counts for a full-grid packet.

The current code cannot execute those grids unchanged:

- `make_page_plan` and `validate_page_plan` accept only one/three/six cards. Its grouping implementation explicitly uses `cards[:3]` and `cards[3:]`; adding count enums alone would incorrectly make a large second section rather than four/eight sections.
- `parallel_decode.BUDGETS` and whole-page budgets cover only one/three/six cards. Extending the existing six-card per-field caps of 24 title, eight badge and sixteen heading tokens would imply aggregate caps of 448/896 for twelve/twenty-four cards, but those are proposed arithmetic extensions, not configured or quality-validated budgets.
- The shared ranking prefix rejects lengths over 1,536 tokens. Six candidates already use 1,446 in the scale fixture. Full-grid selection needs a larger eligible pool and an explicitly supported context plan; silently dropping candidates or truncating their facts would change the decision task.
- Appended generation suffixes reject lengths over 768 tokens. The illustrative whole-manifest full-grid suffix estimates exceed that guard. Independent per-field suffixes currently stay around 144–267 tokens, but their aggregate processing grows with every field.

The complete-grid contract must expose all requested unique product/image choices, all sections and every required copy field, or explicitly fail. A partial first screen, repeated product, reduced candidate roster, prewritten heading or approved-title fallback would not satisfy this estimate's scope.

## Memory: no defensible twelve-/twenty-four-card extrapolation

The seven-GiB supplement's observed global MLX peak is **6,940,200,480 bytes** against **7,516,192,768 bytes**, leaving about 549 MiB of guard headroom in that run. Per-arm `mlxPeakBytes` values are cumulative within the shared process: repeated identical peaks do not show equal standalone memory requirements. They must not be fit as a per-card memory slope. The prior six-GiB three-card width-six failure remains a failure at its original guard.

Continuous refill lazily materializes at most the active width's full cache copies, so 28/56 pending jobs do not imply 28/56 simultaneous model copies or full native caches. There is still one resident model. However, each active lane carries complete hybrid recurrent/convolution/KV state; the KV/history and native prefill work depend on actual context length. Expanding the ranking prefix can increase both memory and time even with fixed width. The existing peak does not prove that twelve/twenty-four cards fit seven GiB. Any new grid needs measured peak checks, actual full-count denominators and preservation of failed/unattempted work.

## Strongest evidence-backed architecture experiment

Keep the full-grid decision pass, then separate **generation context** from the all-candidate ranking history. Use the same resident frozen backbone for genuine copy generation:

1. Rank the complete eligible candidate set and select/order every requested product/image with the trained heads. Retain the deterministic three-card grouping and its honest provenance. Supporting the larger decision context is explicit work; changing the representation would require its own head-compatibility evaluation.
2. Prepare a small generation-specific public request context. For each selected card, prefill its identity, exact selected facts, approved-title exclusions and supported observed tags once; take a complete native snapshot and fork it into independent title and badge suffixes. The model still generates new wording autoregressively in each field. No generated answer is supplied in advance.
3. Generate each section heading independently from its public framing and universal tag intersection, excluding non-universal member attributes from the heading's local evidence. Keep all four/eight heading jobs. Admit card/heading jobs through bounded continuous refill and discard finished private states; never trim or splice hybrid caches.

This targets the measured bottleneck while preserving independent fields. The current title/badge pair repeats a roughly 250–267-token binding payload twice and carries the entire ranking history in both lanes. Cache-copy/materialization itself costs only about 0.4–4.3 ms and strict assembly about 0.5–3.9 ms in the audited cases, so optimizing those Python operations cannot remove multi-second gaps. Sharing selected-card evidence before the split can reduce repeated prefill; short local generation histories can reduce irrelevant candidate contamination and active state traffic. Those are testable hypotheses, not a claimed quality or latency result.

The architecture must keep strict fact binding and novelty checks. Existing failures include identity-copy/truncated badges, false universal section attributes and Ridge's unsupported `durable` title. Compact prompts did not solve them. A bounded comparison should first verify exact complete-cache isolation/refill, then measure the same full twelve-/twenty-four-card requests with all fields and failures retained. Model-generated titles remain new rather than copied approved variants, and grouping remains deterministic rather than claimed learned. No storefront/UI change is required for this model experiment.
