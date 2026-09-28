"""Real loopback HTTP/TLS checks; no model credentials or external services."""

from __future__ import annotations

import json
import shutil
import socket
import socketserver
import ssl
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from mindmemos_sdk.compression import CompressionConfig, CompressionError, compress_memory


@pytest.fixture(autouse=True)
def direct_loopback(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is needed to generate a temporary loopback TLS certificate")
    directory = tmp_path_factory.mktemp("compression-tls")
    cert, key = directory / "cert.pem", directory / "key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    return cert, key


@contextmanager
def upstream(*, certificate=None, incomplete=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(
                {
                    "choices": [{"message": {"content": "verified summary"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body) + (10 if incomplete else 0)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if certificate is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*certificate)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"{'https' if certificate else 'http'}://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def compress(base_url, timeout=2):
    return compress_memory(
        task="task",
        memory=[],
        messages=[{"role": "assistant", "content": "raw"}],
        mode="full",
        config=CompressionConfig(api_key="offline", base_url=base_url, timeout_seconds=timeout),
    )


def test_https_still_verifies_and_accepts_trusted_certificate(certificate, monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", str(certificate[0]))
    with upstream(certificate=certificate) as url:
        assert compress(url).summary == "verified summary"


def test_https_rejects_untrusted_certificate(certificate, monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    with upstream(certificate=certificate) as url:
        with pytest.raises(CompressionError) as caught:
            compress(url)
        assert caught.value.code == "transport_error"
        assert not caught.value.request_dispatched


def test_http_premature_eof_cannot_commit_valid_but_incomplete_envelope():
    with upstream(incomplete=True) as url:
        with pytest.raises(CompressionError) as caught:
            compress(url)
        assert caught.value.code == "transport_error"
        assert caught.value.usage == {"prompt_tokens": 12}


def test_dns_and_tls_share_one_deadline(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class StallTLS(socketserver.BaseRequestHandler):
        def handle(self):
            entered.set()
            release.wait(2)

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), StallTLS)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    resolve = socket.getaddrinfo

    def slow_dns(*args, **kwargs):
        time.sleep(0.35)
        return resolve(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow_dns)
    started = time.monotonic()
    try:
        with pytest.raises(CompressionError) as caught:
            compress(f"https://127.0.0.1:{server.server_address[1]}/v1", timeout=0.5)
        elapsed = time.monotonic() - started
        assert entered.is_set()
        assert caught.value.code == "timeout" and not caught.value.request_dispatched
        assert elapsed < 0.75  # DNS and TLS must share the 0.5-second deadline.
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
