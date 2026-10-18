# Memory and model assessment after corrected-data v2

The M5 has 25,769,803,776 bytes (24 GiB) unified memory. A post-run read-only
sample reported 9,210,249,216 bytes available (8.58 GiB), 8,350,007,296 bytes swap
occupied (7.78 GiB), and `memory_pressure -Q` system-wide free percentage 57%.
The pressure tool's free percentage and psutil's available-memory fraction are
different counters. Swap occupancy is a point-in-time observation, not proof
that this run caused swapping; no before-run swap baseline was captured.

The completed corrected-data native comparison used 192 B1 encodes, peaked at
4,706,802,514 bytes MLX allocation (4.38 GiB), and completed under its declared
6 GiB process guard. It did not hit a memory limit. That guard is an experiment
setting, not a user-imposed hardware cap. Increasing it alone would not make this
workload smarter or faster. Future larger batches may use a new explicitly sized
guard based on then-current headroom, with available-memory/peak checks; previous
bounded experiments and their guard receipts remain unchanged.

No apps were closed: memory was not the bottleneck in this run, and no reliable
long-idle evidence was needed or collected. Chrome, Codex, Discord, Messages,
project processes and unsaved work were preserved. Low CPU usage alone would
not justify closing an app. No paid compute, new credentials, installations or
weight downloads were used.

## Installed and potential models

Only the pinned Qwen3.5-9B four-bit snapshot has complete local weight shards:
two shards totaling 5,950,221,072 bytes. The cached Qwen3-8B directory has config
and tokenizer metadata but zero safetensors shards, so it is not an available
local model. It is also not an established quality upgrade.

Two possible future candidates are the official
[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) and
[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
The former has 27B language parameters, 64 layers and hidden dimension 5120;
the latter has 35B total/3B active parameters, 40 layers and dimension 2048.
Their published benchmark results motivate testing, but do not establish better
storefront copy or preference ranking on our data. Dense 27B likely increases
per-token work; MoE active count reduces compute without removing the need to
store the total experts. Neither has a measured local latency result here.

The following are planning estimates, not measured artifact allocations:

| Candidate | Four-bit parameter packing minimum | Practical implication |
| --- | ---: | --- |
| 27B | 13.5 GB / 12.57 GiB | Add quantization scales, unquantized tensors/vision, activations, cache, allocator and host memory; allow roughly 16–20 GiB process headroom before testing. |
| 35B-A3B | 17.5 GB / 16.30 GiB | All experts still need residency or slower streaming; a practical allowance around 19–23+ GiB leaves inadequate room for the active desktop. |

For scale, the published 27B architecture has 16 full-attention layers with
4 KV heads of dimension 256. At 4096 tokens and BF16 KV, those caches alone are
256 MiB per lane: `16 * 4096 * 4 * 256 * 2(K,V) * 2bytes`. Its 48 linear layers
also carry recurrent/convolution state; assuming FP32 48-head 128x128 recurrent
state adds approximately 144 MiB per lane before convolution state. These are
architecture-based estimates, not a validated replacement artifact/cache layout.
Six lanes multiply state storage. Native prefill/adapter differentiation can
add substantial temporary allocations beyond the saved cache.

The current small CPU factor fits have 57,358 parameters per full head; optimizer
state is small. Full-model training is different: even a simplified 12-byte-per-
parameter weight/gradient/Adam budget for 27B is about 324 GB before activations,
so only frozen-backbone heads or small adapters are realistic here. A larger
frozen model still needs separately measured adapter activations and native
cache correctness. Existing 4096-dimensional heads, means and LoRA shapes cannot
be reused blindly with 5120/2048-dimensional candidates.

With 8.58 GiB currently available and substantial existing swap use, neither
candidate is a safe immediate load. A larger-model experiment would require an
explicit compatible quantized artifact, a substantial multi-GB download, pinned
hashes/runtime/tokenizer, new feature/checkpoint dimensions, freshly measured
memory headroom and a matched quality/latency baseline. No such setup was started.

## Decision for the next model step

Corrected balanced supervision plus fact-focused native inputs achieved 21/21
on this v2 test ranking component, while full-text inputs achieved 7/21. The
shared 9B backbone therefore handled this controlled generalization task when
its inputs and data were corrected. This does not prove adequate real-store
copy quality. First transfer the representation to the actual 16-product public
catalog and validate new tasks, image-action binding and genuine generated copy.
Revisit a larger model if those measured errors persist after aligned inputs;
do not attribute distribution/input failures to parameter count alone.
