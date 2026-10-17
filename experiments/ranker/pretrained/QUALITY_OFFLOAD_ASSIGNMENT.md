# Independent Mac quality lane assignment

The parent has delivered the external Mac agent its task prompt. That agent exclusively owns `experiments/ranker/pretrained/quality_offload/` and its own result directory. The local performance lane must not edit that directory or duplicate its native generation work.

The external assignment is baseline versus field-specific prompts and stopping/token-budget interventions, held-out quality/factuality, the same MLX Qwen3.5-9B, raw outputs/metrics, and separately owned commits pushed to main. Shared batching/native-diagnosis/main-benchmark and UI files are outside that lane. These are separate Mac workers, so its independent quality runs can overlap local CPU work without contaminating this Mac's measured GPU timings. Final serving-latency conclusions still need the target Mac.

Local performance work is complete; see [measured report](PARALLEL_SEMANTIC.md), [independent audit](PARALLEL_SEMANTIC_AUDIT.md), and [memory-guard provenance](MEMORY_GUARD_PROVENANCE.json). The compact generator is not promoted. The original manifest remains best; repeated badge/identity text, nonuniversal headings and selected-title factuality are known failures for the quality lane to preserve in its baseline.

Before the second machine was clarified, this lane prepared CPU-only diagnostic inputs/scoring utilities. They were never used for new inference or remote access and are archived locally under ignored `data/ranker/offload-generation/prepared-handoff-before-external-mac-assignment/`; they are not a new shared quality implementation or claimed held-out result. No CUDA/Spark port was attempted. The twelve existing evidence cases are inspected diagnostics and must not be relabeled held-out.

Coordinate main changes by fetching/rebasing only the lane's own clean local commits; preserve all teammate commits and uncommitted files. Do not reset, overwrite shared experiments, or launch replacement chats.
