# Independent semantic-factor audit

Read-only CPU audit passed at execution commit `7328a10b95f3bacb4dc6a419f2fbb8f382e037e2`: **192 saved model score tensors**, **48 recomputed nonlearned oracle derivative tensors**, and **480 raw/guarded readouts**. These are 240 score tensors total; there are four learned/control arms plus one oracle, not six arms. All pair metrics, 24 atom diagnostic blocks and 168 independently counted per-atom confusion tables match. Source/checkpoint hashes and exact receipts are in the adjacent JSON.

The 28-query/42-unique-variant TRAIN means and public target receipts recompute exactly. All factor checkpoints contain those frozen means; all 20 held-out queries and 15 used held-out products are excluded. The copied image checkpoint is byte-identical to the previous centered checkpoint. Existing native source reports retain their hashes.

| Split | Full product | Prior | Query permutation | Random | Full product pairs both correct |
|---|---:|---:|---:|---:|---:|
| Train |28/28|4/28|7/28|6/28|14/14|
| Validation |8/8|2/8|1/8|2/8|4/4|
| Test |12/12|2/12|4/12|6/12|6/6|

The public seven-atom distillation method transfers on these already inspected paired splits. This is stronger product evidence than the rank-four centered scorer, but it remains synthetic grammar supervision. Image behavior is inherited, with no new image fit. No complete page, title/badge, grouping, human preference or conversion result follows. The reported 0.8917-second wall time covers cached CPU work only.

The source and test contract gates all ranking labels and held-out vector reads behind checkpoint freeze; the manifest agrees. There is no syscall trace, so file-access history is not independently reconstructible beyond those contracts/receipts. No fitting or native execution occurred in this audit.
