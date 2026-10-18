"""Narrow local browser-origin adapter; model and inference bodies are unchanged."""

VERSION = 'persona-local-portless-http.v1'
STOREFRONT_ORIGIN = 'http://shop.dubhacks.localhost:1356'
ALLOWED_ORIGINS = frozenset({
    'http://localhost:4313', 'http://127.0.0.1:4313', STOREFRONT_ORIGIN,
})


def handler_for(worker, base_handler_factory):
    class Handler(base_handler_factory(worker)):
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
    return Handler
