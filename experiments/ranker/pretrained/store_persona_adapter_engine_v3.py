"""Explicit measurement entry: one-time public context encoding and fresh B1 queries.

No accepted model profile, source promotion, native execution or launch by import.
V2 remains byte-exact; this candidate retains its ranking and copy computation.
"""
import argparse
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading

from . import store_persona_adapter_engine_v2 as previous
from . import store_persona_adapter_contract_v2 as contract
from .store_persona_adapter_runtime_v3 import AdapterPersonaRuntime

VERSION = 'persona-adapter-engine.v3-measurement'


def engine_class(*, runtime_factory=AdapterPersonaRuntime, **dependencies):
    # Each generated class has its runtime policy fixed at construction, before
    # base initialization attaches the resident model and becomes ready.
    def construct(*, force_fresh_query=True, preencode_public_profiles=True,
                  color_cache_entries=32, **kwargs):
        def configured_runtime(*args, **runtime_kwargs):
            return runtime_factory(*args, force_fresh_query=force_fresh_query,
                preencode_public_profiles=preencode_public_profiles,
                color_cache_entries=color_cache_entries, **runtime_kwargs)
        configured_base = previous.engine_class(runtime_factory=configured_runtime, **dependencies)
        class Engine(configured_base):
            def __init__(self, **engine_kwargs):
                super().__init__(**engine_kwargs)
                try:
                    self.adapter_engine_sources[str(Path(__file__))] = contract.sha(__file__)
                    self.startup['persona']['measurementEngine'] = VERSION
                    self.startup['persona']['featurePreencoding'] = self.adapter_runtime.preencoding
                    self.startup['persona']['forceFreshQuery'] = force_fresh_query
                    self.startup['persona']['wholePromptRankCacheIncludesQuery'] = True
                    self.startup['persona']['noResponseWarmupOrTextCache'] = True
                    self._assert_candidate()
                    (self.output / 'startup.json').write_text(json.dumps(self.startup, indent=2) + '\n')
                except BaseException as error:
                    try: self.close()
                    except BaseException as cleanup_error: error.add_note('V3 metadata cleanup: ' + str(cleanup_error))
                    raise
        return Engine(**kwargs)
    return construct


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    for name in ('asset-root', 'output', 'serving-manifest', 'catalog-feature-root', 'color-head', 'color-cache',
                 'query-intent-head', 'candidate-run', 'integration-manifest', 'rank-head-export'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('serving-manifest-sha256', 'query-intent-head-sha256', 'rank-head-export-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--query-intent-source-manifest-sha256')
    parser.add_argument('--force-fresh-query', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--preencode-public-profiles', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--color-cache-entries', type=int, default=32)
    args = parser.parse_args(argv); root = Path(__file__).resolve().parents[3]
    if not args.output.is_absolute() or not args.output.resolve().is_relative_to(root) or args.output.exists():
        parser.error('Unique absolute output inside current code root required')
    if not args.asset_root.is_absolute(): parser.error('Explicit absolute original asset root required')
    if not 1 <= args.color_cache_entries <= 128: parser.error('Color cache capacity must be 1..128')
    if args.preencode_public_profiles and args.color_cache_entries < 10:
        parser.error('Preencoding ten profiles needs at least ten cache entries')
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    os.chdir(args.asset_root)
    from .store_demo_service import ModelWorker
    from .store_rank_session_v5 import make_v5_rank_session_factory
    klass = engine_class()
    server = ThreadingHTTPServer(('127.0.0.1', 8767), previous.handler_for(None))
    worker = ModelWorker(lambda: klass(mode='parallel', serving_manifest=args.serving_manifest,
        serving_sha=args.serving_manifest_sha256, catalog_feature_root=args.catalog_feature_root,
        color_head=args.color_head, color_cache=args.color_cache, query_intent_head=args.query_intent_head,
        query_intent_sha=args.query_intent_head_sha256, query_intent_source_manifest_sha=args.query_intent_source_manifest_sha256,
        candidate_run=args.candidate_run, integration_manifest=args.integration_manifest, output=args.output, width=20,
        force_fresh_query=args.force_fresh_query, preencode_public_profiles=args.preencode_public_profiles,
        color_cache_entries=args.color_cache_entries,
        rank_session_factory=make_v5_rank_session_factory(args.rank_head_export, args.rank_head_export_sha256)))
    server.RequestHandlerClass = previous.handler_for(worker); server.daemon_threads = True
    def stop(*_): threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    try: server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close(): raise RuntimeError('Measurement worker teardown integrity unconfirmed')


if __name__ == '__main__': main()
