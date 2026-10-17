"""Preloaded loopback decision service with a bounded single-worker deadline gate."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import time
from .common import heuristic_scores, rank_result


class DeadlineGate:
    def __init__(self, predictor, deadline_ms=5):
        self.predictor = predictor
        self.deadline_ms = deadline_ms
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ranker")
        self.lock = threading.Lock()

    def predict(self, request, fallback):
        if self.predictor is None:
            return fallback(), "model_missing"
        if not self.lock.acquire(blocking=False):
            return fallback(), "busy"
        try:
            future = self.executor.submit(self.predictor, request)
        except Exception:
            self.lock.release()
            return fallback(), "model_error"
        future.add_done_callback(lambda _: self.lock.release())
        try:
            return future.result(timeout=self.deadline_ms / 1000), None
        except TimeoutError:
            # Running NumPy work cannot safely be killed. It retains the gate until
            # completion, so subsequent requests fall back rather than queueing.
            return fallback(), "deadline"
        except Exception:
            return fallback(), "model_error"

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class RankingService:
    def __init__(self, catalog, encoder, model, deadline_ms=5):
        self.catalog = catalog
        self.encoder = encoder
        self.model_loaded = model is not None
        self.gate = DeadlineGate(lambda request: model.predict(encoder.encode(request)).tolist(), deadline_ms) if model else DeadlineGate(None, deadline_ms)

    def warmup(self, request):
        if self.gate.predictor is not None:
            # Run in the inference worker before opening the listening socket.
            self.gate.executor.submit(self.gate.predictor, request).result(timeout=1)

    def rank(self, request, mode="model"):
        start = time.perf_counter()
        ids = request.get("candidateProductIds", [])
        if len(ids) != 6 or len(set(ids)) != 6 or any(pid not in self.catalog for pid in ids):
            raise ValueError("Exactly six distinct known product candidates are required.")
        fallback = lambda: heuristic_scores(request, self.catalog)
        scores, reason = (fallback(), None) if mode == "default" else self.gate.predict(request, fallback)
        return rank_result(request, self.catalog, scores, "heuristic" if mode == "default" or reason else "parallel-mlp-adapted", guarded=True,
                           latency_ms=(time.perf_counter() - start) * 1000,
                           warnings=["Fallback: " + reason] if reason else [])


def make_server(service, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/presentation-options":
                self.respond({"synthetic": True, "policy": "deferred", "products": [{"productId": pid, "approvedVariants": product.get("presentationVariants", [])} for pid, product in service.catalog.items()]})
                return
            if self.path != "/health":
                self.send_error(404)
                return
            self.respond({"ready": True, "synthetic": True, "preloaded": service.model_loaded})

        def do_POST(self):
            if self.path != "/rank":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            if length > 65536 or length <= 0:
                self.send_error(413)
                return
            try:
                payload = json.loads(self.rfile.read(length))
                self.respond(service.rank(payload["request"], payload.get("mode", "model")))
            except (ValueError, KeyError, TypeError) as error:
                self.respond({"error": str(error)}, 400)

        def respond(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass
    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 1
    return server
