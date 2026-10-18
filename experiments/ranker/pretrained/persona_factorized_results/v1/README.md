# Factorized persona serving measurement

V4 is an explicit technical development option, with the same first failed choice
checkpoint and different context conditioning. It has no independent model-quality
promotion result. Original V2/V3 functions, receipts and checkpoints remain.

| Actual browser workload | Fixed valid count | p50 | p95 |
|---|---:|---:|---:|
| Loaded, pre-scrolled profile switch, three fully visible cards + decoded images | 10/10 | 335.5ms | 649.0ms |
| Same switches, six-card copy + decoded images | 10/10 | 351.2ms | 649.0ms |
| Ten distinct typed queries, six-card copy + decoded images | 10/10 | 413.5ms | 463.4ms |
| Ten typed queries, three fully visible cards | 0/10 | — | — |

One initial navigation completed six-card decoded copy in1.681s on the navigation
clock (839.9ms from native fetch), but had no useful above-fold card endpoint.
Search scroll and the initial hero leave copy below the fold; initial navigation
shows zero complete useful cards. Six decoded cards do not imply all six visible.
Twenty fields are generated; the current UI renders nineteen strings. Timers
exclude startup/browser launch/physical paint. Nested component medians do not sum.

Before READY, ten neutral public contexts cost18.753s:5.129s color encoding plus
13.624s rank feature capture,10 base forwards and37 rank-body forwards. All23
measured browser requests performed a fresh actual-query B1 forward, zero rank-body
forwards, a fresh learned projection/color choice and fresh neural word predictions.
No scores, rankings or response words are cached. Known-profile fresh-query p50
was222.1ms for switches and277.9ms for typed queries; server-full p50 was300.3ms
and360.5ms respectively. Complete useful-content timing is the browser table above.

A separate HTTP-only new history cost2063.5ms; it is not a browser sample. The
separately archived root authored-background probe cost1976.8ms, then a distinct
query on that same context completed270.1ms. Unknown context costs remain real.
The budget1/empty and budget40/Graphite-one-item checks pass and remain outside
six-card latency. The single-item response withheld one unsafe badge.

All13 primary protocol requests, both functional arms, ten new-query protocols and
one new-history HTTP smoke passed; source/startup snapshots stayed identical.
Graceful shutdown verified all927 frozen tensors, copy/rank/cache/source assets
and adapter restoration. Peak5,727,066,390bytes stayed under the fixed7GiB guard.

Independent public audit found factual copy and138 source/image bindings valid.
Ownership repeats remain in four profiles, three within the first three cards;
notebook/organizer/lunchbox occupy30/60 profile slots. Lena's palette miss persists.
Grace's lunchbox moved from fourth to third and Rosa's towel entered fifth. All
ten first choices stayed unchanged versus V3; lexical vectors changed. This is
not a quality-improvement claim. See the independent public audit alongside these
files and the separately preserved failed independent choice fits.

COPY_INDEX.json records17 exact evidence/screenshot copies. browser-derived-summary.json
is a labeled subset of the unchanged39.2MB raw report, with its original local
path/hash. The full raw report, native receipts and all large numerical archives
remain locally preserved; no weights or large feature arrays were copied here.
