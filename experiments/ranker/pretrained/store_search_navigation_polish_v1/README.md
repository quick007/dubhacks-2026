# Submitted search navigation correction

This was a real visibility bug. Completed results sat below the hero: at1280×960,
the first title began around886.5px and Add controls ended1112.5px. After the
route commits, a small use-route.ts effect scrolls to the existing collection.
The after screenshot shows complete cards and Add controls ending781.5px.
No layout, image, model, request-schema, score, weight or checkpoint changed.

Ten fixed fresh queries on the same running native91663 passed protocol and
factual checks both times. Visible-three results changed0/10→10/10. After-fix
fetch-to-visible3 p50/p95 is332.4/395.4ms; decoded full6 is349.4/413.6ms. Every
request, score vector, selected order/variant and generated-copy section matches
exactly. All20 requests still execute a fresh query B1 forward. This is a search
visibility correction, not evidence of faster model inference or better relevance.
The before-run missing-visibility waits cause different between-request idle
intervals; do not turn timing differences into an inference speedup claim.

report.json contains source hashes, actual geometry and exact parity checks.
Both unchanged full raw reports remain at their recorded local paths/hashes.
Screenshots are exact copies. The initial no-query hero and all quality caveats
remain unchanged. A separate fresh five-slide presentation acceptance passes50
checks, and the storefront passes76 tests plus typecheck/build.
