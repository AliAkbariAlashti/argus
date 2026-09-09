"""Expose a host-loopback Ollama daemon to the Argus Docker network."""
import http.client
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def log_message(self, *_args):
        return

    def _forward(self, method):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in ("host", "connection")
        }
        connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=300)
        try:
            connection.request(method, self.path, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in ("connection", "transfer-encoding"):
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as exc:  # noqa: BLE001
            if not self.wfile.closed:
                try:
                    self.send_error(502, str(exc))
                except Exception:
                    pass
        finally:
            connection.close()


bind = os.environ.get("ARGUS_OLLAMA_RELAY_BIND", "172.28.0.1")
ThreadingHTTPServer((bind, 11435), RelayHandler).serve_forever()
