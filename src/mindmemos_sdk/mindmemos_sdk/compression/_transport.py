"""Per-call deadlines for OpenAI-compatible HTTP transport.

No retries or redirects.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from http.client import HTTPConnection, HTTPSConnection
from urllib.request import HTTPHandler, HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from .models import CompressionError

# getaddrinfo has no portable cancellation API. Bound resolver work so repeated
# timeouts cannot create an unlimited number of lingering resolver threads.
_RESOLVER_SLOTS = threading.BoundedSemaphore(8)


def _resolve(address, control):
    if not _RESOLVER_SLOTS.acquire(timeout=control.remaining()):
        control.check()
        raise CompressionError("compressor timed out", code="timeout")
    ready = threading.Event()
    result = {}

    def resolve():
        try:
            control.check()
            result["addresses"] = socket.getaddrinfo(*address, 0, socket.SOCK_STREAM)
        except Exception as exc:
            result["error"] = exc
        finally:
            _RESOLVER_SLOTS.release()
            ready.set()

    worker = threading.Thread(target=resolve, name="mindmemos-compression-dns", daemon=True)
    try:
        worker.start()
    except BaseException:
        _RESOLVER_SLOTS.release()
        raise
    if not ready.wait(control.remaining()):
        raise CompressionError("compressor timed out", code="timeout")
    control.check()
    if "error" in result:
        raise result["error"]
    return result["addresses"]


def _connect(address, source_address, control):
    last_error = None
    for family, kind, protocol, _name, endpoint in _resolve(address, control):
        control.check()
        sock = socket.socket(family, kind, protocol)
        try:
            # Register before connect, so even an in-progress connect shares the
            # same deadline as DNS, TLS, headers and body reads.
            control.register(sock)
            sock.settimeout(control.remaining())
            if source_address:
                sock.bind(source_address)
            sock.connect(endpoint)
            control.check()
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            control.unregister(sock)
            control.check()
        except BaseException:
            sock.close()
            control.unregister(sock)
            raise
    raise last_error or OSError("upstream resolution returned no addresses")


def _interrupt(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@contextmanager
def _deadline(seconds):
    """Interrupt established I/O even when an upstream slowly trickles headers."""
    sockets, lock = [], threading.Lock()
    expired = threading.Event()
    end = time.monotonic() + seconds

    def expire():
        with lock:
            expired.set()
            for sock in sockets:
                _interrupt(sock)

    class Control:
        request_dispatched = False

        def check(self):
            if expired.is_set() or time.monotonic() >= end:
                raise CompressionError("compressor timed out", code="timeout")

        def remaining(self):
            self.check()
            return max(0.001, end - time.monotonic())

        def register(self, sock):
            if sock is not None:
                with lock:
                    if expired.is_set():
                        _interrupt(sock)
                    else:
                        sockets.append(sock)
                self.check()

        def unregister(self, sock):
            with lock:
                if sock in sockets:
                    sockets.remove(sock)

        def sending(self):
            self.check()
            # Sending may have started: a timeout does not imply zero cost.
            self.request_dispatched = True

    timer = threading.Timer(max(0, seconds), expire)
    timer.daemon = True
    timer.start()
    control = Control()
    try:
        control.check()
        yield control
        control.check()
    except Exception:
        control.check()
        raise
    finally:
        timer.cancel()
        timer.join()
        sockets.clear()


def _handlers(control):
    class ConnectionMixin:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._create_connection = lambda address, timeout, source_address: _connect(
                address, source_address, control
            )

        def connect(self):
            control.check()
            try:
                HTTPConnection.connect(self)
                if isinstance(self, HTTPSConnection):
                    # Register the SSL socket before the handshake: wrap_socket
                    # transfers ownership of the TCP socket's file descriptor.
                    self.sock = self._context.wrap_socket(
                        self.sock, server_hostname=self._tunnel_host or self.host, do_handshake_on_connect=False
                    )
                    control.register(self.sock)
                    self.sock.settimeout(control.remaining())
                    self.sock.do_handshake()
                control.check()
            except BaseException:
                self.close()
                raise

        def send(self, data):
            control.check()
            if self.sock is None:
                self.connect()
            control.sending()
            return super().send(data)

    class HTTP(ConnectionMixin, HTTPConnection):
        pass

    class HTTPS(ConnectionMixin, HTTPSConnection):
        pass

    class DeadlineHTTPHandler(HTTPHandler):
        def http_open(self, req):
            return self.do_open(HTTP, req)

    class DeadlineHTTPSHandler(HTTPSHandler):
        def https_open(self, req):
            return self.do_open(HTTPS, req, context=self._context)

    return DeadlineHTTPHandler(), DeadlineHTTPSHandler()


def _open_upstream(url, body, headers, timeout, control):
    request = Request(url, data=body, headers=headers, method="POST")
    return build_opener(_NoRedirect(), *_handlers(control)).open(request, timeout=timeout)
