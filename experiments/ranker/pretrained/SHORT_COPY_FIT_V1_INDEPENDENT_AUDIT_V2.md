# Independent short-copy fit audit V2

V2 corrects the first audit's natural-EOS-only assumption: six rows honestly reached their sampled cap. Their length receipts are valid evidence and remain strict quality failures. Original native evidence and V1 audit are preserved.

Status: passed; 9488 CPU checks, 0 issues.

All 240 exact nonzero-B row gates and all 240 actual shuffled optimizer updates were verified. The saved A/B checkpoint matches its reload dtype, shapes and parameter bytes. 480 raw field rows were independently decoded and reassembled through the unchanged strict validators.

Actual compressed feature archives and metadata: 1,844,548,247 bytes. Native work wall: 972.864 seconds; peak MLX: 6,626,986,154 bytes.

| Split/mode/width | Valid first3 | Valid full6 | Titles | Bodies | Headings | Visible badges | Exact NONE |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev/prior-STB/width-3 | 0/3 | 0/3 | 16/18 | 8/18 | 2/6 | 9 | 6 |
| dev/prior-STB/width-6 | 0/3 | 0/3 | 15/18 | 8/18 | 1/6 | 9 | 6 |
| dev/new-C/width-3 | 0/3 | 0/3 | 12/18 | 5/18 | 2/6 | 12 | 6 |
| dev/new-C/width-6 | 0/3 | 0/3 | 12/18 | 5/18 | 2/6 | 12 | 6 |
| test/prior-STB/width-3 | 0/3 | 0/3 | 18/18 | 5/18 | 3/6 | 12 | 6 |
| test/prior-STB/width-6 | 0/3 | 0/3 | 18/18 | 6/18 | 3/6 | 12 | 6 |
| test/new-C/width-3 | 0/3 | 0/3 | 15/18 | 8/18 | 3/6 | 12 | 6 |
| test/new-C/width-6 | 0/3 | 0/3 | 13/18 | 8/18 | 4/6 | 12 | 6 |

| Split/mode/width | Attempt first3 median ms | Attempt full6 median ms | Valid first/full latency samples |
|---|---:|---:|---:|
| dev/prior-STB/width-3 | 7139.42 | 11826.09 | 0/0 |
| dev/prior-STB/width-6 | 9420.89 | 10624.35 | 0/0 |
| dev/new-C/width-3 | 7015.09 | 11617.87 | 0/0 |
| dev/new-C/width-6 | 7786.63 | 10203.37 | 0/0 |
| test/prior-STB/width-3 | 7431.66 | 12561.85 | 0/0 |
| test/prior-STB/width-6 | 9896.87 | 11579.93 | 0/0 |
| test/new-C/width-3 | 6928.23 | 12167.20 | 0/0 |
| test/new-C/width-6 | 8322.82 | 11216.99 | 0/0 |

Invalid payloads have no usable-result latency sample. Full per-arm timings, raw examples, fixed denominators and hashes are in the JSON audit.

- Runtime frozen927-tensor equality is authenticated producer evidence; this CPU audit does not execute a second native tensor hash job.
- Existing dev/test catalogs were inspected; quality is copy-only under locked selections, not sealed generalization or integrated ranking.
- Readiness measures actual serialized model payload, not browser paint.
- Native work wall starts at run() entry; Python/module startup is excluded.

Raw new-C examples:

dev width 3, case short-copy-69788a1e31832d3c1f: first3 7015.09 ms; full6 11617.87 ms.
- S1: 'Every daypack choice for everyday travel.'
- T1: 'Walinor 13L Daypack for adjustable travel'
- C1: 'A compact daypack for everyday travel.'
- C2: 'A compact daypack for everyday travel.'
- C3: 'A compact daypack for everyday travel.'
Rejections: core:S1:unsupported-or-missing-heading; C1:attribute_not_in_selected_product_tags:compact; C1:outside_factual_or_generic_lexicon:compact; C2:attribute_not_in_selected_product_tags:compact; C2:outside_factual_or_generic_lexicon:compact; C1:body-independent-observed-support-missing; C2:body-independent-observed-support-missing; C3:body-independent-observed-support-missing

test width 3, case short-copy-18bebe450c31e1f0de: first3 6928.23 ms; full6 11098.64 ms.
- S1: 'Everyday travel choices'
- T1: 'Zelilan 31L Daypack for roomy travel'
- C1: 'A compact daypack for everyday travel.'
- C2: 'A compressible daypack for everyday travel.'
- C3: 'A structured daypack for everyday travel.'
Rejections: C1:body-independent-observed-support-missing; C2:body-independent-observed-support-missing; C3:body-independent-observed-support-missing
