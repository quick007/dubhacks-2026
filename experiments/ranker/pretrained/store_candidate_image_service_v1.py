"""Explicit same-catalog compressed-image serving entry; legacy CLI unchanged."""
from pathlib import Path
import argparse
from http.server import ThreadingHTTPServer
import json
import signal
import threading
from .store_candidate_engine_v1 import CandidateDemoEngine
from .store_demo_image_binding_v1 import ImageDeliveryCatalogBinding, ROOT, sha
from .store_demo_service import ModelWorker, handler_for
from .store_demo_engine import ResidentDemoEngine


def cleanup_failed_start(engine, error):
    """Retain a failure receipt and tear down a partially constructed engine."""
    receipt = {'status': 'initialization-failed', 'errorType': type(error).__name__,
        'error': str(error), 'frozenHashCleanupConfirmed': False}
    try:
        if hasattr(engine, 'bb') and hasattr(engine, 'before_hashes'):
            closed = ResidentDemoEngine.close(engine)
            receipt['shutdown'] = closed
            receipt['frozenHashCleanupConfirmed'] = closed.get('frozen927HashesUnchanged') is True
        elif hasattr(engine, 'stream_context'):
            engine.stream_context.__exit__(type(error), error, error.__traceback__)
            receipt['cleanupScope'] = 'stream exited; backbone construction did not reach initial frozen hash receipt'
        else:
            receipt['cleanupScope'] = 'failure before native stream/model ownership'
    except BaseException as cleanup_error:
        receipt['cleanupError'] = {'type': type(cleanup_error).__name__, 'message': str(cleanup_error)}
        if hasattr(engine, 'stream_context') and not getattr(engine, '_native_stream_exited', False):
            try:
                engine.stream_context.__exit__(type(error), error, error.__traceback__)
                receipt['fallbackStreamExitConfirmed'] = True
            except BaseException as stream_error:
                receipt['streamExitError'] = str(stream_error)
            finally:
                engine._native_stream_exited = True
    output = getattr(engine, 'output', None)
    if output is not None:
        try:
            (output / 'initialization-failure.json').write_text(json.dumps(receipt, indent=2) + '\n')
        except OSError:
            pass
    return receipt


class ImageCandidateDemoEngine(CandidateDemoEngine):
    def __init__(self, **kwargs):
        sources = ('experiments/ranker/pretrained/store_candidate_image_service_v1.py',
            'experiments/ranker/pretrained/store_demo_image_binding_v1.py',
            'experiments/ranker/pretrained/store_rank_session_v5.py',
            'scripts/prepare-product-webp.mjs', 'scripts/prepare-product-responsive-webp.mjs',
            'scripts/prepare-product-runtime-projection.mjs')
        self.image_sources = {name: sha(ROOT / name) for name in sources}
        try:
            super().__init__(catalog_binding_factory=ImageDeliveryCatalogBinding, **kwargs)
            self.startup['imageDelivery'] = self.binding.manifest['imageDelivery']
            self.startup['imageDeliveryManifestSha256'] = self.binding.manifest_sha
            self.startup['imageServingSourcesSha256'] = self.image_sources
            (self.output / 'startup.json').write_text(json.dumps(self.startup, indent=2) + '\n')
        except BaseException as error:
            cleanup_failed_start(self, error)
            raise

    def _assert_candidate(self):
        super()._assert_candidate()
        if any(sha(ROOT / name) != expected for name, expected in self.image_sources.items()):
            raise ValueError('Image serving source changed since startup')
        self.binding.assert_current_sources(self.session.snapshot['receipt'])
        if hasattr(self.session, 'integrity_receipt'):
            self.session.integrity_receipt()

    def close(self):
        result = super().close()
        result['imageServingSourcesUnchanged'] = True
        if hasattr(self.session, 'integrity_receipt'):
            result.update(self.session.integrity_receipt())
        (self.output / 'shutdown.json').write_text(json.dumps(result, indent=2) + '\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--integration-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--rank-head-export', type=Path)
    parser.add_argument('--rank-head-export-sha256')
    args = parser.parse_args()
    if bool(args.rank_head_export) != bool(args.rank_head_export_sha256):
        parser.error('Both explicit rank head export and reviewed checksum are required')
    rank_options = {}
    if args.rank_head_export:
        from .store_rank_session_v5 import make_v5_rank_session_factory
        rank_options['rank_session_factory'] = make_v5_rank_session_factory(
            args.rank_head_export, args.rank_head_export_sha256)
    if args.output.exists():
        raise FileExistsError('Preserve prior service lifetimes')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(None))
    worker = ModelWorker(lambda: ImageCandidateDemoEngine(mode='parallel', candidate_run=args.candidate_run,
        integration_manifest=args.integration_manifest, output=args.output, width=10, **rank_options))
    server.RequestHandlerClass = handler_for(worker); server.daemon_threads = True
    def stop(_sig, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    print(json.dumps({'event': 'image-candidate-listening', 'port': args.port}), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close():
            raise RuntimeError('Native shutdown integrity unconfirmed')


if __name__ == '__main__':
    main()
