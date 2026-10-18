"""Explicit serving-only factorized profile experiment; unchanged V2/V3 entries."""
import argparse
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
from . import store_persona_adapter_engine_v2 as transport
from . import store_persona_adapter_engine_v3 as previous
from . import store_persona_adapter_contract_v2 as contract
from .store_persona_adapter_runtime_v4 import AdapterPersonaRuntime, POLICY

VERSION = 'persona-adapter-engine.v4-factorized-profile'


def engine_class(*, runtime_factory=AdapterPersonaRuntime, **dependencies):
    base = previous.engine_class(runtime_factory=runtime_factory, **dependencies)
    def construct(**kwargs):
        value = base(force_fresh_query=True, preencode_public_profiles=True, color_cache_entries=32, **kwargs)
        try:
            value.adapter_engine_sources[str(Path(__file__))] = contract.sha(__file__)
            value.startup['persona']['measurementEngine'] = VERSION
            value.startup['persona']['wholePromptRankCacheIncludesQuery'] = False
            value.startup['persona']['contextConditioning'] = {'version': POLICY,
                'query': 'Browse the store', 'itemBudget': None, 'hardRequirements': [],
                'publicPersonaAndObservedHistoryPreservedExactly': True,
                'actualRequestUsedForFreshQueryHeadAndHardGuards': True,
                'differentFromV2V3': True, 'modelPromotion': False}
            value._assert_candidate()
            (value.output / 'startup.json').write_text(json.dumps(value.startup, indent=2) + '\n')
            return value
        except BaseException as error:
            try: value.close()
            except BaseException as cleanup_error: error.add_note('V4 metadata cleanup: ' + str(cleanup_error))
            raise
    return construct


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    for name in ('asset-root', 'output', 'serving-manifest', 'catalog-feature-root', 'color-head', 'color-cache',
                 'query-intent-head', 'candidate-run', 'integration-manifest', 'rank-head-export'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('serving-manifest-sha256', 'query-intent-head-sha256', 'rank-head-export-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--query-intent-source-manifest-sha256')
    args = parser.parse_args(argv); root = Path(__file__).resolve().parents[3]
    if not args.output.is_absolute() or not args.output.resolve().is_relative_to(root) or args.output.exists():
        parser.error('Unique absolute output inside current code root required')
    if not args.asset_root.is_absolute(): parser.error('Explicit absolute original asset root required')
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    os.chdir(args.asset_root)
    from .store_demo_service import ModelWorker
    from .store_rank_session_v5 import make_v5_rank_session_factory
    klass = engine_class(); server = ThreadingHTTPServer(('127.0.0.1', 8767), transport.handler_for(None))
    worker = ModelWorker(lambda: klass(mode='parallel', serving_manifest=args.serving_manifest,
        serving_sha=args.serving_manifest_sha256, catalog_feature_root=args.catalog_feature_root,
        color_head=args.color_head, color_cache=args.color_cache, query_intent_head=args.query_intent_head,
        query_intent_sha=args.query_intent_head_sha256, query_intent_source_manifest_sha=args.query_intent_source_manifest_sha256,
        candidate_run=args.candidate_run, integration_manifest=args.integration_manifest, output=args.output, width=20,
        rank_session_factory=make_v5_rank_session_factory(args.rank_head_export, args.rank_head_export_sha256)))
    server.RequestHandlerClass = transport.handler_for(worker); server.daemon_threads = True
    def stop(*_): threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    try: server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close(): raise RuntimeError('Factorized worker teardown integrity unconfirmed')


if __name__ == '__main__': main()
