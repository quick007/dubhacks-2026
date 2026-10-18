# Paired synthetic copy data audit

The CPU audit passes all 42 teacher pages under the current unchanged strict
validator and identity policy. This establishes grammar/fact-policy consistency
of targets only. It does not establish native stopping, learned quality,
ranking/image accuracy, human quality, latency or a production success rate.

| Split | Six-card requests | Individual fields | Unique products | Identity-valid teacher titles | Supported badges | Required NONE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 36 | 504 | 72 reused | 216 | 68 | 148 |
| Fresh development | 3 | 42 | 18 new | 18 | 12 | 6 |
| Fresh test | 3 | 42 | 18 new | 18 | 12 | 6 |

Each case has two three-card sections, fourteen independent fields and summed
cap 224: heading 16, title 24, badge 8. Every case requires at least one actual
`NONE` (fresh cases require two). Train headings cover 34 positive universal
intersections and 38 empty intersections; each fresh split covers three of each.
All heading targets are 3–5 words; titles preserve full selected catalog
identity, include the observed supported tag when present, are novel versus
approved titles, and satisfy the requested 4–8 words. Badges use only observed
support or actual `NONE`. There is no history or hidden persona truth.

## Counterfactual and attribute coverage

There are 18 families, with 21 shared-product comparisons in total. Within each
family, catalog values and product/image order are identical in both requests.
Only the public task text and singleton stated preference change; request IDs
are bookkeeping, outside rendered evidence. The target title and supported
badge change on every designated shared product. The current CPU audit and
tests verify this; generated counterfactual behavior has not been measured.

All twelve old training fixture catalogs are used. The union is exactly the
original 72 products, including every description, tag, size, price and image
variant. Canonical catalog hashes are stored for each family. No old catalog
entry or fact triple is edited. Fresh products are used only in fresh dev/test.

| Positive attribute | Title examples | Matching supported badge examples |
| --- | ---: | ---: |
| adjustable | 4 | 4 |
| compact | 5 | 5 |
| compressible | 4 | 4 |
| durable | 2 | 2 |
| expandable | 4 | 4 |
| foldable | 2 | 2 |
| lightweight | 4 | 4 |
| mesh | 6 | 6 |
| organizer | 2 | 2 |
| packable | 6 | 6 |
| padded | 2 | 2 |
| quilted | 4 | 4 |
| reflective | 3 | 3 |
| rolltop | 3 | 3 |
| roomy | 3 | 3 |
| structured | 5 | 5 |
| ventilated | 3 | 3 |
| water-resistant | 6 | 6 |

Coverage is unequal and still small. The new package removes the original
absence of positive compact/foldable/quilted targets; it is not a balanced or
exhaustive language curriculum. Singleton training intentionally isolates the
observable conditioning signal, while fresh requests require two preferences.
No supported-`NONE` positive targets, history, broad product categories,
adversarial input or real user phrasing are added.

## Overlap, disjointness and provenance

Repeated exposure is intentional. Thirty-six training requests contain 33
distinct rendered public-evidence contexts and 462 distinct field prompts;
three repeated singleton/catalog contexts add 42 duplicate field instances to
the fixed 504 updates. Repeated prompts have exactly one consistent target.
There are 152 distinct non-`NONE` training target strings, and 56 of those appear
in more than one paired family. Headings do not change within a pair because
the products/universal facts do not change; unsupported title/badge targets may
also repeat between families.

Short heading and badge grammar is shared across splits intentionally. The
normalized non-`NONE` target-string intersections are train/dev 8, train/test 8,
dev/test 2; these are generic headings or attribute badge phrases, with no
shared complete title targets using held-out names. `NONE` is also shared policy
syntax. This is a held-out request/product/combination evaluation, not an unseen
wording claim. The teacher grammar is never concatenated into inference input
as an answer inventory.

`exclusions.json` lists exact hashed sources: original semantic-copy train/dev/
test cases, scheduler diagnostic and two fresh scheduler fixtures, the v2 public
catalog, and all v2 public JSONL splits/diagnostics. Fresh names, product/image
IDs, three-tag combinations and two-tag preference sets are checked against
this inventory. The manifest makes this precise boundary explicit; it does not
claim disjointness from uninspected external data. The original inspected
semantic-copy dev/test cases remain reference-only replay files and separate
denominators.

## Exact interface and reproduction

Files `cases.{train,dev,test}.jsonl` and `fields.{train,dev,test}.jsonl` retain the
previous teacher package schema. Cases contain `request`, `catalog`,
`pendingPagePlan`, `publicEvidence`, fourteen `targets`, and bookkeeping. Fields
contain `caseId`, `personaFamilyId`, trusted `field`, `rowId`, `kind`, one-based
`cardIndex` or `sectionIndex`, `maxTokens`, `messages`, `messagesSha256`,
`targetValue`, and source metadata. `messages` is the complete two-turn common
evidence plus independent field request; only those messages are model input.
`targetValue` is assistant-only supervision, one raw value without S/T/B labels.
Never tokenize the entire JSON record or expose target/family/pair metadata.
The misleadingly inherited `personaFamilyId` key is bookkeeping for an observed
pair family, never a latent persona feature.

`pairs.jsonl` maps the two case IDs and singleton preferences to their original
fixture, immutable catalog hash and shared products whose title/badge must
change. `inspected_replays.json` pins only old case-file references and IDs.
`manifest.json` pins all generated payloads, generator, original inputs and
validator snapshots. `AUDIT.json` stores the original grammar audit.

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.semantic_copy_pairs.generator check
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.semantic_copy_pairs.test_data -v
```

The generator's `write` mode refuses existing outputs. `check` regenerates in
memory and compares every frozen artifact exactly. Seven CPU tests cover
counterfactual context changes, positive atom coverage, strict page validation,
required `NONE`, fresh two-preference support, exact reproduction, contradictory
duplicate targets, mutation detection and inspected replay separation. No native,
model, tokenizer or training calls were performed. Target-plus-EOS cap fit and
complete tokenizer context fit remain required before the owning lane trains.
