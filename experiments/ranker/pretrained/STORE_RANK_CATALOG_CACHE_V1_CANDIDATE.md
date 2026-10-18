This is an unpromoted CPU candidate. The profile measured4.3750ms median exclusive catalog loading, against170.6221ms fresh backbone encoding. Reusing deserialized public catalog tensors and their unchanged head projection may save a few milliseconds. No measured latency improvement is claimed.

`store_rank_catalog_cache_v1.py` provides `make_cached_catalog_v5_factory(export_path, export_sha256)` with the existing optional rank-session constructor interface. It loads the exact original16 catalog rows through the existing validated loader, then retains private tensor copies and the original head's complete16-row catalog probabilities. Every request still fully SHA256-verifies all32 original tensor/receipt files before and after execution. Existing source validation, v5 export/head checks, fresh native query encoding, fresh query projection, hard guards and action binding remain active. The result adds explicit catalog-cache provenance.

The candidate caches no query features, generated words or responses. Cached values are returned as owned copies. Unknown catalog rows, changed files, reentrant access, changed private tensors and nonidentical complete catalog projection inputs fail closed. Original checkpoint bytes and head arithmetic remain unchanged. The additional verification and copying may consume the potential gain; this requires a paired native comparison before any promotion.

The small CPU fixtures exercise exact projection parity, all32 file-tamper paths, fresh query projection call counts, clone ownership, rejection of changed inputs and cleanup after exceptions. They do not establish native parity or speed. They are intentionally separate from live serving code.

After the browser measurement window is released, CPU checks can run with:

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_store_rank_catalog_cache_v1
```

No service entry, source manifest or model weights have been changed. Native integration and benchmarking require a new explicit worker configuration under the existing owner lock.
