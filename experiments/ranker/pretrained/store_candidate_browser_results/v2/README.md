# Compact browser evidence

Five original runs are preserved, including failed navigation protocol and failed
collection response capture attempts. `artifacts-text.tar.gz` retains every
original JSON, text and saved source file byte for byte. `inventory.json` lists
every original file, including omitted PNG paths, sizes and SHA-256 hashes.
Three representative PNGs retain their exact original bytes under `screenshots/`.

The original directories and the earlier `v1` archive remain unchanged. This is
an additional archive, not a move. `manifest.json` records exact source bytes
available by their original plan hashes; it does not claim a complete source
dependency graph. Raw reports and audits retain all original limitations.

Verify archive checksums, lossless decompression, all original file hashes and
unchanged `v1` files from the repository root:

```sh
python3 experiments/ranker/pretrained/store_candidate_browser_results/v2/build_archive.py --verify
```

Inspect the tar without modifying originals:

```sh
tar -tzf experiments/ranker/pretrained/store_candidate_browser_results/v2/artifacts-text.tar.gz
```
