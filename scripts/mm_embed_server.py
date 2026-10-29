"""Dev-only multimodal embedding API: a local SigLIP2 model behind the OpenAI-style /v1/embeddings contract.

Implements the ``openai`` protocol of MindMemOS ``mm_embed_model_router``:

    POST /v1/embeddings  {"model": ..., "input": ["text", {"image": "data:...;base64,..."|"https://..."}, {"text": "..."}],
                          "dimensions": 768}
    ->   {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [...]}], "model": ...}

Texts and images land in SigLIP2's shared space (L2-normalized). Texts are lowercased and padded
to 64 tokens, as SigLIP2 was trained. For development and benchmarks only (production supplies its
own multimodal embedding API); not part of the MindMemOS package. torch and transformers are not
project dependencies, so run it with them added on the fly:

    uv run --no-project --with torch --with transformers --with pillow python scripts/mm_embed_server.py

The model (~1.5 GB) is downloaded from Hugging Face on first start. Server config
(requests route through litellm, which strips the provider prefix on the wire):
``mm_embed_model_router.endpoints: [{model: openai/siglip2-base-patch16-384,
api_base: http://127.0.0.1:8090/v1, dimensions: 768}]``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "google/siglip2-base-patch16-384"
MAX_TEXT_TOKENS = 64
BATCH = 32


class Siglip2:
    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.lock = threading.Lock()
        self.dimensions = int(
            self.model.config.text_config.projection_size
            if hasattr(self.model.config.text_config, "projection_size")
            else self.model.config.text_config.hidden_size
        )

    def _normalize(self, features):
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return features / features.norm(dim=-1, keepdim=True)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            batch = [t.lower() for t in texts[start : start + BATCH]]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding="max_length", max_length=MAX_TEXT_TOKENS, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
            with self.lock, self.torch.no_grad():
                features = self._normalize(self.model.get_text_features(**inputs))
            out.extend(features.float().cpu().tolist())
        return out

    def embed_images(self, images: list) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(images), BATCH):
            inputs = self.processor(images=images[start : start + BATCH], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k == "pixel_values"}
            with self.lock, self.torch.no_grad():
                features = self._normalize(self.model.get_image_features(**inputs))
            out.extend(features.float().cpu().tolist())
        return out


def _load_image(ref: str):
    from PIL import Image

    if ref.startswith("data:"):
        data = base64.b64decode(ref.split(",", 1)[1])
    elif ref.startswith(("http://", "https://")):
        with urllib.request.urlopen(ref, timeout=20) as resp:  # noqa: S310 - dev tool
            data = resp.read()
    else:
        data = base64.b64decode(ref)
    return Image.open(io.BytesIO(data)).convert("RGB")


def make_handler(encoder: Siglip2, served_name: str, max_concurrent: int = 4):
    # ThreadingHTTPServer spawns a thread per connection; without a cap a burst of requests
    # (each decoding images in memory before queueing on the model lock) grows it unbounded.
    # Over the cap the server answers 503 + Retry-After instead of falling behind.
    limit = threading.Semaphore(max(1, max_concurrent))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet default access log
            pass

        def _send(self, status: int, payload: dict, extra_headers: dict | None = None) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, str(value))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                return self._send(200, {"status": "ok", "model": served_name, "dimensions": encoder.dimensions})
            if self.path == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": served_name, "object": "model"}]})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path not in ("/v1/embeddings", "/embeddings"):
                return self._send(404, {"error": {"message": "not found"}})
            if not limit.acquire(blocking=False):
                return self._send(
                    503,
                    {"error": {"message": f"overloaded: {max(1, max_concurrent)} requests already running", "type": "overloaded"}},
                    {"Retry-After": "2"},
                )
            try:
                self._embed()
            finally:
                limit.release()

        def _embed(self) -> None:
            started = time.perf_counter()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                items = body.get("input")
                items = [items] if isinstance(items, (str, dict)) else list(items or [])
                dims = body.get("dimensions")
                if dims is not None and int(dims) != encoder.dimensions:
                    return self._send(400, {"error": {"message": f"dimensions must be {encoder.dimensions}"}})
                texts, images, slots = [], [], []
                for item in items:
                    if isinstance(item, str):
                        slots.append(("t", len(texts)))
                        texts.append(item)
                    elif isinstance(item, dict) and "image" in item:
                        slots.append(("i", len(images)))
                        images.append(_load_image(str(item["image"])))
                    elif isinstance(item, dict) and "text" in item:
                        slots.append(("t", len(texts)))
                        texts.append(str(item["text"]))
                    else:
                        return self._send(400, {"error": {"message": "input items must be strings, {text} or {image}"}})
            except Exception as exc:  # noqa: BLE001 - report bad input to the caller
                return self._send(400, {"error": {"message": f"bad request: {exc}"}})
            try:
                text_vecs = encoder.embed_texts(texts) if texts else []
                image_vecs = encoder.embed_images(images) if images else []
            except Exception as exc:  # noqa: BLE001
                return self._send(500, {"error": {"message": f"inference failed: {exc}"}})
            data = [
                {"object": "embedding", "index": i, "embedding": (text_vecs if kind == "t" else image_vecs)[pos]}
                for i, (kind, pos) in enumerate(slots)
            ]
            self._send(
                200,
                {
                    "object": "list",
                    "data": data,
                    "model": served_name,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )

    return Handler


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--served-name", default="siglip2-base-patch16-384")
    parser.add_argument("--device", default=os.environ.get("MM_EMBED_DEVICE", "auto"), help="auto, cuda, mps or cpu")
    parser.add_argument(
        "--max-concurrent", type=int, default=4, help="embedding requests served at once; further ones get 503"
    )
    args = parser.parse_args()
    device = _device(args.device)
    encoder = Siglip2(args.model, device)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(encoder, args.served_name, args.max_concurrent)
    )
    print(
        f"mm embed server: {args.served_name} ({encoder.dimensions}d, {device}, "
        f"max {max(1, args.max_concurrent)} concurrent) on {args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
