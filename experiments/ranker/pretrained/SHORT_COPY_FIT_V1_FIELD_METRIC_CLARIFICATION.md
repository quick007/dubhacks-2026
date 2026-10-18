# Short-copy field metric clarification

The V2 audit's “Bodies” / `acceptedBodies` metric already requires every body-specific section check to pass. For each local C field, it excludes a row if any section rejection starts with `Cn:` or `core:Cn:`. This includes `body-independent-observed-support-missing`, factual or lexical errors, unfinished generation, and sampled-cap/EOS receipt errors. It does not use `copyChecks.copyAccepted` alone.

Across six cases and both sections (36 body fields per mode/width):

| Mode | Width | Lexical `copyChecks.copyAccepted` | All body-specific section checks accepted |
|---|---:|---:|---:|
| prior-STB | 3 | 13/36 | 13/36 |
| prior-STB | 6 | 14/36 | 14/36 |
| new-C | 3 | 20/36 | 13/36 |
| new-C | 6 | 20/36 | 13/36 |

New-C has 17 observed-support-missing rejections at each width; several overlap other body failures. Seven lexically accepted bodies at each width fail additional section checks and are excluded from the V2 count. No V2 counts changed.

Titles, headings and visible badges use the unchanged strict core validator's individual accepted-field counters. Exact NONE is counted only with no corresponding B-field section rejection. Field-level acceptance does not establish a valid payload: all 24 three-card and all 24 six-card attempts remain invalid, with zero valid-result latency samples.
