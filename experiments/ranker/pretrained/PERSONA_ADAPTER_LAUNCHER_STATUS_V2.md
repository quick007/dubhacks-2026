# Adapter launcher preparation update

The launcher work described as future work in the frozen
[HTTP V2 preparation document](./PERSONA_ADAPTER_HTTP_PROBE_V2.md) has now been
implemented by its owner in
[`personalized-adapter-profile.mjs`](../../../scripts/personalized-adapter-profile.mjs)
and [`dev-personalized.mjs`](../../../scripts/dev-personalized.mjs).
The original HTTP preparation document and receipt remain historical snapshots;
this addendum preserves their recorded bytes and checksums.

The new explicit profile version is `personalized-local-adapter-profile.v2`, with
kind `lexical-choice-lora-query-gate.v2`. The parser requires status `accepted`
and a separate root-owned `persona-adapter-live-acceptance.v1` receipt. No actual
accepted profile has been published by this preparation, and no model, service
or browser has been launched by this lane. The original context-head entry is
preserved; there is no failed-head or arbitrary-module fallback.

The profile binds seven artifact references: serving manifest, query-intent head,
color head, color cache, original v5 rank export, public48 copy manifest and
integration manifest. It retains the original16-product v3 catalog feature root
and exact query TRAIN manifest. References use an explicit assets/worktree root,
relative file and SHA; serving arrays and sanitized inference certificate stay
co-located with their index/manifest. The helper verifies the cache/checkpoint
inventory and current relative source inventory before spawning anything.

Seven measured evidence references are mandatory: `offlineAudit`,
`admissionBeforeGold`, `queryMechanicalAudit`, `queryCombinationReplay`,
`httpValidation`, `browserValidation` and `shutdown`. A separate `acceptance`
reference binds all seven artifact hashes, all seven evidence hashes and current
source bytes, with explicit browser/fresh-copy/shutdown/source-equivalence flags.
The root's separately frozen320-query regression policy also binds the actual
full320 HTTP roster and all320 same-guard v5 top-choice/abstention agreements;
these are behavior-preservation checks, not new relevance truth.

The adapter branch passes the exact `store_persona_adapter_engine_v2` CLI:
serving manifest+SHA, original catalog-feature root, color head/cache, query gate
head+SHA and TRAIN manifestSHA, public copy candidate, integration manifest,
v5 export+SHA, original asset root and unique output. It does not pass old
`--persona-head` or `--port` flags; this engine fixes8767/width20/7GiB. Existing
shared-lock/free-port checks, verified checkout reuse, unique session outputs,
owned-child cleanup and offline/no-secret behavior remain in the launcher.
Full schema and commands are documented in
[`scripts/dev-personalized.md`](../../../scripts/dev-personalized.md).

The launcher owner reports26 CPU tests passing; these fake fixture checks are
preparation evidence only. Actual acceptance remains blocked. The original
quality auditorV1 admitted the saved artifacts but failed after opening gold
because of derived-ID bookkeeping. A separate corrected auditorV2 preserves the
same quality gate, public packet, labels, models and frozen predictions, and is
under independent review. Its version/source must be explicitly admitted only
after the actual outcome and root direction. The frozen HTTP V2 selection and
current launcher audit contract must not alias the corrected auditor to V1 or
rewrite their source checksums. No such compatibility change is made here.
