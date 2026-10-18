"""Local opt-in NDJSON storefront demo, one resident GPU-owning worker.

Run from repository root: python -m experiments.ranker.pretrained.store_demo_service
No model download, paid compute, deployment or automatic training occurs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import select
import signal
import socket
import threading
import time
import uuid

from .store_demo_engine import (DEFAULT_COPY, DEFAULT_OUTPUT, DEFAULT_RANK,
    RequestCancelled, ResidentDemoEngine, checked_request)

ALLOWED_ORIGINS = frozenset({'http://shop.dubhacks.localhost:1356',
    'http://localhost:4312', 'http://127.0.0.1:4312',
    'http://localhost:5173', 'http://127.0.0.1:5173'})
MAX_REQUEST_BYTES = 16384


class ModelWorker:
    """Thread affinity + lifetime native-owner lock; four bounded queued jobs."""
    def __init__(self, engine_factory):
        self.engine_factory = engine_factory; self.jobs = queue.Queue(maxsize=4)
        self.state_lock = threading.Lock(); self.state = 'starting'; self.error = None
        self.startup = None; self.stopping = threading.Event(); self.active_cancel = None
        self.thread = threading.Thread(target=self.run, name='resident-native-model', daemon=True)
        self.thread.start()

    def health(self):
        with self.state_lock:
            return {'status': self.state, 'ready': self.state == 'ready', 'error': self.error,
                'queuedRequests': self.jobs.qsize(), 'startup': self.startup}

    def run(self):
        from .catalog_fusion.generalization_v2 import native_owner_lock
        engine = None
        try:
            with native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), 'root'):
                engine = self.engine_factory()
                with self.state_lock:
                    self.state = 'ready'; self.startup = engine.startup
                while not self.stopping.is_set():
                    try:
                        job = self.jobs.get(timeout=.25)
                    except queue.Empty:
                        continue
                    value, request_id, started, events, cancel, future = job
                    self.active_cancel = cancel
                    try:
                        if cancel.is_set():
                            raise RequestCancelled('Queued request already cancelled')
                        engine.personalize(value, request_id, started, events.put,
                            lambda: cancel.is_set() or self.stopping.is_set())
                        future.set_result(True)
                    except BaseException as error:
                        # Raw failure evidence is already archived by the engine.
                        # A Future must not retain native cache-bearing frames
                        # after a cancelled browser request has disconnected.
                        error.__traceback__ = None
                        error.__context__ = None
                        error.__cause__ = None
                        future.set_exception(error)
                    finally:
                        events.put(None); self.active_cancel = None; self.jobs.task_done()
                engine.close(); engine = None
                with self.state_lock:
                    self.state = 'stopped'
        except BaseException as error:
            with self.state_lock:
                self.state = 'error'; self.error = f'{type(error).__name__}: {error}'
            if engine is not None:
                try:
                    engine.close()
                except BaseException:
                    pass
        finally:
            while True:
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    break
                job[-1].set_exception(RuntimeError('Resident worker stopped'))
                job[3].put(None); self.jobs.task_done()

    def submit(self, value, request_id, started):
        if not self.health()['ready']:
            raise RuntimeError('Resident model is not ready; inspect /health')
        events = queue.Queue(); cancel = threading.Event(); future = Future()
        self.jobs.put_nowait((value, request_id, started, events, cancel, future))
        return events, cancel, future

    def close(self):
        self.stopping.set()
        if self.active_cancel is not None:
            self.active_cancel.set()
        self.thread.join(timeout=150)
        return not self.thread.is_alive()


def handler_for(worker):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            # No shopper query or generated content is printed to terminal.
            print(json.dumps({'http': fmt % args}), flush=True)

        def origin_allowed(self):
            return self.headers.get('Origin') in (None, *ALLOWED_ORIGINS)

        def cors(self):
            origin = self.headers.get('Origin')
            if origin in ALLOWED_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Private-Network', 'true')

        def respond(self, status, value):
            encoded = (json.dumps(value, allow_nan=False) + '\n').encode()
            self.send_response(status); self.cors()
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(encoded))); self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(encoded)

        def do_OPTIONS(self):
            if not self.origin_allowed():
                return self.respond(403, {'error': 'Origin is not an allowed local storefront'})
            self.send_response(204); self.cors(); self.send_header('Content-Length', '0'); self.end_headers()

        def do_GET(self):
            if not self.origin_allowed():
                return self.respond(403, {'error': 'Origin is not an allowed local storefront'})
            if self.path != '/health':
                return self.respond(404, {'error': 'Unknown endpoint'})
            health = worker.health()
            self.respond(200 if health['ready'] else 503, health)

        def disconnected(self):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable:
                    return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b''
            except (ConnectionError, OSError):
                return True
            return False

        def do_POST(self):
            started = time.perf_counter()
            if not self.origin_allowed():
                return self.respond(403, {'error': 'Origin is not an allowed local storefront'})
            if self.path != '/personalize':
                return self.respond(404, {'error': 'Unknown endpoint'})
            request_id = str(uuid.uuid4())
            if self.headers.get('Transfer-Encoding'):
                return self.respond(400, {'error': 'Bounded Content-Length request required'})
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                if not 1 <= size <= MAX_REQUEST_BYTES:
                    return self.respond(413, {'error': 'Request must be 1–16384 bytes'})
                self.connection.settimeout(10)
                data = self.rfile.read(size)
                if len(data) != size:
                    raise ValueError('Incomplete request body')
                value = json.loads(data)
                checked_request(value)
            except (ValueError, UnicodeError, TimeoutError) as error:
                return self.respond(400, {'error': str(error)})
            try:
                events, cancel, future = worker.submit(value, request_id, started)
            except queue.Full:
                return self.respond(429, {'error': 'Local model queue is full'})
            except RuntimeError as error:
                return self.respond(503, {'error': str(error), 'health': worker.health()})
            self.connection.settimeout(10)
            self.send_response(200); self.cors()
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-store'); self.send_header('Connection', 'close')
            self.send_header('X-Request-Id', request_id); self.end_headers()
            self.close_connection = True
            try:
                while True:
                    if self.disconnected():
                        cancel.set(); break
                    try:
                        event = events.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if event is None:
                        error = future.exception()
                        if error is not None and not isinstance(error, RequestCancelled):
                            event = {'type': 'error', 'requestId': request_id,
                                'message': f'{type(error).__name__}: {error}'}
                            self.wfile.write((json.dumps(event) + '\n').encode()); self.wfile.flush()
                        break
                    self.wfile.write((json.dumps(event, allow_nan=False) + '\n').encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionError, OSError, TimeoutError):
                cancel.set()
            finally:
                cancel.set()
    return Handler


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--width', type=int, choices=(6, 10, 20), default=10)
    parser.add_argument('--rank-run', type=Path, default=DEFAULT_RANK)
    parser.add_argument('--copy-run', type=Path, default=DEFAULT_COPY)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--integration-manifest', type=Path)
    args = parser.parse_args()
    worker = ModelWorker(lambda: ResidentDemoEngine(rank_run=args.rank_run, copy_run=args.copy_run,
        output=args.output, width=args.width, integration_manifest=args.integration_manifest))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(worker))
    server.daemon_threads = True
    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    print(json.dumps({'event': 'local-demo-listening', 'url': f'http://127.0.0.1:{args.port}',
        'nativeWorker': 'starting', 'width': args.width}), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close():
            raise RuntimeError('Native worker did not stop; no shutdown integrity receipt confirmed')


if __name__ == '__main__':
    main()
