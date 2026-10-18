# Branch-local copy independent audit

inspected old dev/test replay; copy stage only; original saved counts plus additional stricter raw-row accounting.

Run status `completed`; all twelve arms recorded: True. Raw values were reassembled with the locked packet plan/request/catalog under the original policy; every saved quality record matches exactly. Original fields and a separate stricter raw-wire account follow; no saved outputs or pass rates are rewritten.

| Inspected split / arm | Original pages / 3 | Strict pages / 3 | Titles original→strict / 18 | Badges incl NONE original→strict / 18 | Headings original→strict / 6 | Malformed rows / originally counted | Copy median all ms | Copy median valid ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev / frozen | 0 | 0 | 7→7 | 18→18 | 2→2 | 0 / 0 | 8939.6 | unavailable |
| dev / global-copy-adapter | 0 | 0 | 15→15 | 18→18 | 4→4 | 0 / 0 | 8938.6 | unavailable |
| test / frozen | 0 | 0 | 4→4 | 18→18 | 1→1 | 0 / 0 | 9248.8 | unavailable |
| test / global-copy-adapter | 0 | 0 | 11→11 | 18→15 | 4→4 | 3 / 3 | 9282.7 | unavailable |

Run wall 129949.8 ms; load 1984.4 ms. Copy timing includes fresh root, repeated fact suffixes, generation and validation, and excludes ranking. Do not compare these numbers directly with the old global end-to-end times. Complete token/AR, memory, input inventory, original policy, raw fields/facts and stricter rejections are retained in the companion JSON.

## Malformed rows counted by original field provenance

- test / copy-acc8677838207254 / global-copy-adapter / B1: raw "compact\n"; stop=stop, tokens=2/8; original-counted=True, strict-counted=False; wire=["multiline-or-line-separator"]; facts={"selectedIdentity": "Herilis 24L Daypack", "productId": "copy-p-564bf3c0048c9938", "imageCaptionId": "copy-v-c0cc8f7516483bb0", "tags": ["compact", "foldable", "ventilated"], "observedSupport": ["compact"], "approvedTitles": ["Herilis 24L Daypack", "Herilis 24L Daypack", "Herilis 24L Daypack — compact"]}.
- test / copy-acc8677838207254 / global-copy-adapter / B2: raw "compact\n"; stop=stop, tokens=2/8; original-counted=True, strict-counted=False; wire=["multiline-or-line-separator"]; facts={"selectedIdentity": "Evidor 38L Daypack", "productId": "copy-p-bb9e12bd0af44fa5", "imageCaptionId": "copy-v-1ab2f356a328f0df", "tags": ["compact", "durable", "organizer"], "observedSupport": ["compact"], "approvedTitles": ["Evidor 38L Daypack", "Evidor 38L Daypack", "Evidor 38L Daypack — compact"]}.
- test / copy-acc8677838207254 / global-copy-adapter / B3: raw "compact\nmesh\npackable"; stop=stop, tokens=6/8; original-counted=True, strict-counted=False; wire=["multiline-or-line-separator"]; facts={"selectedIdentity": "Soriren 31L Daypack", "productId": "copy-p-ab738f255a5c2bed", "imageCaptionId": "copy-v-9b3852e2d664f9b6", "tags": ["compact", "mesh", "packable"], "observedSupport": ["compact"], "approvedTitles": ["Soriren 31L Daypack", "Soriren 31L Daypack", "Soriren 31L Daypack — compact"]}.

## Suggested prospective codec correction

Keep original raw values and wire errors as immutable evidence. After ordinary validation, reject the entire trusted field whose raw value is empty, multiline or contains a generated field label: set its returned field text to null and provenance to rejected, and recompute generated counts and accepted alignment from that rejected plan. Never retain a malformed row’s first parsed line as an accepted field. Do not strip/repair/retry text, invent NONE or rewrite old reports. Keep the existing global finish rule. This is a proposed fix, not a codec or validator edit in this audit.

Regression coverage should include first-line-valid malformed heading/title/badge, supported and required NONE rows, embedded field labels and Unicode line separators; assert raw bytes preserved, affected provenance rejected, generated counts excluded and page false.

## Limits

- No new training; global-copy adapter checkpoint is reused under a different copy layout.
- All six requests were previously inspected; this is not fresh generalization evidence.
- Copy-stage times exclude ranking and are not comparable with direct end-to-end request completion.
- Original counts preserve the first-line parsing/provenance defect; stricter counts do not rewrite them.
- Global unfinished-output conservatism is retained in both accounts.
- Strict raw checks still do not establish human usefulness or fluency.
- Cumulative native peak includes controls/previous arms, not isolated arm memory.

## Independent content and accounting findings

This layout and reused global-copy adapter are not promoted. All twelve arms
finish, but there are zero accepted pages in every inspected split/arm cell,
under both accounts. The stricter account preserves every old rejection and
removes only three incorrectly counted generated badge fields. Development
titles are frozen 7/18 versus adapter 15/18, test titles 4/18 versus 11/18;
headings are 2/6 versus 4/6 on development and 1/6 versus 4/6 on test. Original
test adapter badges are 18/18 including nine `NONE`; the strict account is
15/18, comprising six generated badges and nine accepted `NONE`. The saved
generated-badge count remains nine in the original result and original column.

The three malformed rows occur together in adapter test case
`copy-acc8677838207254`:

| Field | Exact JSON raw value | Output tokens / cap | AR steps | Finish | Original generated provenance | Strict raw-wire acceptance |
| --- | --- | ---: | ---: | --- | --- | --- |
| B1 | `"compact\n"` | 2 / 8 | 3 | stop | accepted | rejected |
| B2 | `"compact\n"` | 2 / 8 | 3 | stop | accepted | rejected |
| B3 | `"compact\nmesh\npackable"` | 6 / 8 | 7 | stop | accepted | rejected |

These are protocol errors, not truncation or unsupported badge attributes.
Their selected products respectively have compact/foldable/ventilated,
compact/durable/organizer, and compact/mesh/packable catalog tags, with observed
`compact` support. The canonical parser accepts each first `compact` line;
`encode_value_rows` correctly retains the newline and rejects the page. The
returned page plan nevertheless retains generated badge text, and its generated
badge count includes all three. The old analysis helper similarly trusts the
generated field source for title/heading/non-`NONE` badge acceptance. The
separate strict account in this audit requires both original acceptance and
full raw wire/fact/identity/novelty/support checks. No value is trimmed, joined,
repaired or substituted, and the policy pass rates are not rewritten.

The new layout resolves neither universal-heading truth nor preference binding.
Adapter development `Roomy travel choice` and `Reflective travel choice` apply
to sections with empty universal-tag intersections. Adapter test `Quilted
travel daypacks` and `Quilted travel choice` likewise apply a member attribute
to all three cards incorrectly. In the compact test case, `Herilis 24L Daypack
for everyday travel` preserves identity and is novel but omits observed compact
support; `Evidor 38L Daypack for organized travel` also omits compact and uses
the unapproved lexical paraphrase `organized`. On the foldable test case,
`Evivia 36L Daypack for water-resistant travel` adds an attribute absent from
that selected product (actual tags adjustable/foldable/reflective), while
Darodor and Jorilis generic titles omit foldable. Accepted-and-aligned supported
titles are development 7/9 in each arm and test frozen 4/9 versus adapter 2/9.
Preserving identity alone is insufficient for personalized copy quality.

All 168 rows terminate with stop; only the three badge rows above violate the
raw value protocol. Every required raw `NONE` is correct: nine per split/arm,
and the original policy reports correct-NONE on all three pages in each cell.
There is no global length failure in this run. The strict accounting defect is
therefore observable without changing the global truncation rule.

Copy-stage medians are 8.940 versus 8.939 seconds on inspected development and
9.249 versus 9.283 seconds on inspected test (frozen versus reused adapter).
There are no valid-only timing samples. Across all twelve arms, output totals
are 1,091 tokens and 1,259 AR steps. The run takes 129.950 seconds including
controls and load; cumulative native peak is 6,208,584,982 bytes, below the
7-GiB guard. No usable-page latency reduction is established. These are copy
stage measurements with locked recorded decisions, not direct-request ranking
and completion timings. The old direct/global end-to-end results must keep
their separate timing scope and denominators.

The CPU analysis ran only after final native status was completed and verified
exact saved-policy equality for all twelve arms. Input/result/source hashes
were identical before and after replay, and native benchmark, packet and layout
hash receipts matched current sources at audit time. Five CPU regression tests
passed. Original reports and raw artifacts were read only. A prospective codec
fix changes field provenance/counting, so replay of the old saved-quality record
must use these pinned original sources; this helper deliberately rejects a
saved/current policy mismatch instead of silently rewriting evidence.
