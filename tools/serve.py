#!/usr/bin/env python3
"""Static server for the dev loop, with caching turned off.

`python3 -m http.server` sends no cache headers, so browsers cache ES modules
and `.wasm` aggressively — which means a rebuild appears to do nothing and you
debug a stale binary. That is a genuinely expensive mistake: the symptom is
"the function I just added does not exist", which reads like a build failure.

Also sets the correct MIME type for .wasm, so streaming instantiation works.

    python3 tools/serve.py [port]
"""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class NoCacheHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # quiet


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    handler = partial(NoCacheHandler, directory=str(ROOT))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {ROOT} at http://127.0.0.1:{port}/web/index.html (no-cache)")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
