"""Simplified multimodal agentic retrieval (vanilla, multimodal mode enhanced, ``search_strategy=agentic``).

A short ReAct loop over four tools, distilled from MHG's six-tool retriever: search text
memories, search images (by text or by the question's image, results shown to the model), view
images, and submit. The agent starts from the fast-path results and may add facts and images it
finds; what it submits goes first, the fast-path results follow, so recall never drops below the
fast path. Any failure falls back to the fast-path results.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ....components.multimodal import ImageLoader
from ....components.multimodal.image_loader import image_data_uri
from ....config import MultimodalConfig, get_config
from ....logging import get_logger
from ....typing import MemoryRequestContext, MemorySearchItem, SearchPipelineInput

logger = get_logger(__name__)

AGENT_SYSTEM = """You retrieve evidence from a long-term memory of a user's past conversations with an assistant. A separate model will answer the QUESTION from the evidence you submit; you do not answer it.

Memory holds short FACTS distilled from the conversations (each with its date), VISUAL FACTS read from the images shared in them, and the IMAGES themselves (each with its date and a description).

Tools (reply with exactly ONE JSON object per turn, nothing else):
- {"tool": "search_memories", "args": {"query": "..."}} -> facts and visual facts matching a short description of what you need.
- {"tool": "search_images", "args": {"query": "..."}} or {"tool": "search_images", "args": {"by_question_image": true}} -> images matching a description of what they show, or matching the image attached to the question; the top images are shown to you.
- {"tool": "view_images", "args": {"images": ["I1", "I2"]}} -> look at images you have seen listed.
- {"tool": "submit", "args": {"facts": ["F1", "F4"], "images": ["I2"]}} -> finish with the facts and images the answer needs, most important first.

Guidance:
- Start from the CURRENT EVIDENCE below; search only for what is missing (another time, another item, the visual detail, the earliest or latest occurrence).
- For questions about what something looked like (colors, text, layout, counts in a picture), find and cite the image.
- For ordering questions (first, last, before, after, how many times), collect every occurrence with its date.
- Use short, specific queries. Do not repeat a query. Submit as soon as you have enough."""

_JSON = re.compile(r"\{.*\}", re.DOTALL)
VISUAL_FACTS_PER_SEARCH = 6


class MultimodalAgent:
    """Runs the retrieval loop for one search request."""

    def __init__(self, engine: Any, llm_client: Any, cfg: MultimodalConfig) -> None:
        self._engine = engine
        self._llm = llm_client
        self._cfg = cfg
        self._facts: dict[str, MemorySearchItem] = {}
        self._images: dict[str, MemorySearchItem] = {}

    async def refine(
        self, inp: SearchPipelineInput, context: MemoryRequestContext, fast: list[MemorySearchItem]
    ) -> list[MemorySearchItem]:
        self._project_id = context.project_id
        texts = [item for item in fast if item.memory_type != "image"]
        images = [item for item in fast if item.memory_type == "image"]
        for item in texts:
            self._fact_handle(item)
        for item in images:
            self._image_handle(item)
        question_images = await self._question_images(inp, context)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": self._opening(inp, texts, images, question_images)},
        ]
        submitted: dict[str, list[str]] | None = None
        for step in range(self._cfg.agentic_max_steps + 1):
            if step == self._cfg.agentic_max_steps:
                messages.append({"role": "user", "content": "Tool budget used up. Reply with submit now."})
            response = await self._llm.chat(task="multimodal.search.agentic", messages=messages)
            reply = response.content or ""
            action = _parse(reply)
            messages.append({"role": "assistant", "content": reply})
            if action is None:
                messages.append({"role": "user", "content": "Reply with exactly one JSON tool call."})
                continue
            tool, args = action.get("tool"), action.get("args") or {}
            if tool == "submit":
                submitted = {"facts": list(args.get("facts") or []), "images": list(args.get("images") or [])}
                break
            observation = await self._run_tool(tool, args, inp, context)
            messages.append({"role": "user", "content": observation})
        if submitted is None:
            return fast
        return self._result(submitted, texts, images)

    async def _run_tool(
        self, tool: Any, args: dict[str, Any], inp: SearchPipelineInput, context: MemoryRequestContext
    ) -> list[dict[str, Any]] | str:
        if tool == "search_memories" and args.get("query"):
            query = str(args["query"])
            search = inp.model_copy(update={"query": query, "top_k": 8})
            found = await self._engine.search_candidates(search, context)
            # Visual facts are not in text memory; they are searched through the images.
            visual = await self._engine.visual_facts_for(search, context, self._cfg, limit=VISUAL_FACTS_PER_SEARCH)
            lines = [self._fact_line(self._fact_handle(item), item) for item in found[:8]]
            visual_lines = [self._fact_line(self._fact_handle(item), item) for item in visual]
            return (
                "Facts:\n"
                + ("\n".join(lines) if lines else "(none)")
                + "\nVisual facts:\n"
                + ("\n".join(visual_lines) if visual_lines else "(none)")
            )
        if tool == "search_images":
            by_image = bool(args.get("by_question_image"))
            query = (
                inp if by_image else inp.model_copy(update={"query": str(args.get("query") or ""), "query_images": []})
            )
            if not by_image and not query.query.strip():
                return "search_images needs a query or by_question_image."
            from .engine import _to_search_item

            found = await self._engine.image_hits(query, context, self._cfg)
            hits = found.by_query_image if by_image else found.ranked
            items = [_to_search_item(hit) for hit in hits[:4]]
            return await self._show([self._image_handle(item) for item in items], header="Images found")
        if tool == "view_images":
            handles = [str(h) for h in (args.get("images") or []) if str(h) in self._images][:3]
            return await self._show(handles, header="Requested images") if handles else "Unknown image handles."
        return "Unknown tool. Use search_memories, search_images, view_images or submit."

    def _opening(
        self,
        inp: SearchPipelineInput,
        texts: list[MemorySearchItem],
        images: list[MemorySearchItem],
        question_images: list[str],
    ) -> list[dict[str, Any]]:
        handles = {id(item): handle for handle, item in self._facts.items()}
        image_handles = {item.metadata.get("blob_id"): handle for handle, item in self._images.items()}
        lines = [f"QUESTION: {inp.query}", f"question image attached: {bool(question_images)}", "", "CURRENT EVIDENCE"]
        lines += [self._fact_line(handles[id(item)], item) for item in texts] or ["(no facts)"]
        lines += [self._image_line(image_handles[item.metadata.get("blob_id")], item) for item in images]
        parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
        for uri in question_images:
            parts += [
                {"type": "text", "text": "Image attached to the question:"},
                {"type": "image_url", "image_url": {"url": uri, "detail": "low"}},
            ]
        return parts

    async def _show(self, handles: list[str], *, header: str) -> list[dict[str, Any]] | str:
        if not handles:
            return f"{header}: (none)"
        store = self._engine.blobs()
        segments: list[tuple[list[dict[str, Any]], tuple[Any, bytes] | None]] = []
        for handle in handles:
            item = self._images[handle]
            line = [{"type": "text", "text": self._image_line(handle, item)}]
            blob_id = item.metadata.get("blob_id")
            found = await store.get(self._project_id, blob_id) if blob_id else None
            segments.append((line, found))
        # b64-encoding megapixel images is CPU work: encode every image of this view in one hop
        encoded = await asyncio.to_thread(
            lambda: [
                image_data_uri(data, blob.mime_type)
                for _, found in segments
                if found is not None
                for blob, data in (found,)
            ]
        )
        uris = iter(encoded)
        parts: list[dict[str, Any]] = [{"type": "text", "text": f"{header}:"}]
        for line, found in segments:
            parts += line
            if found is not None:
                parts.append({"type": "image_url", "image_url": {"url": next(uris), "detail": "low"}})
        return parts

    async def _question_images(self, inp: SearchPipelineInput, context: MemoryRequestContext) -> list[str]:
        if not inp.query_images:
            return []
        loader = ImageLoader(get_config().blob_store, self._engine.blobs())
        loaded = [await loader.load(ref, namespace=context.project_id) for ref in inp.query_images[:2]]
        return await asyncio.to_thread(lambda: [image_data_uri(img.data, img.mime_type) for img in loaded])

    def _fact_handle(self, item: MemorySearchItem) -> str:
        for handle, existing in self._facts.items():
            if existing.id == item.id:
                return handle
        handle = f"F{len(self._facts) + 1}"
        self._facts[handle] = item
        return handle

    def _image_handle(self, item: MemorySearchItem) -> str:
        for handle, existing in self._images.items():
            if existing.metadata.get("blob_id") == item.metadata.get("blob_id"):
                return handle
        handle = f"I{len(self._images) + 1}"
        self._images[handle] = item
        return handle

    @staticmethod
    def _fact_line(handle: str, item: MemorySearchItem) -> str:
        return f"{handle} ({item.event_time or item.source_timestamp or 'unknown date'}) {item.memory}"

    @staticmethod
    def _image_line(handle: str, item: MemorySearchItem) -> str:
        # The item text: its description and the visual facts recorded about it.
        return f"{handle} ({item.event_time or 'unknown date'}) {item.memory.removeprefix('[image] ')[:600]}"

    def _result(
        self, submitted: dict[str, list[str]], texts: list[MemorySearchItem], images: list[MemorySearchItem]
    ) -> list[MemorySearchItem]:
        chosen_facts = [self._facts[h] for h in submitted["facts"] if h in self._facts]
        chosen_images = [self._images[h] for h in submitted["images"] if h in self._images]
        facts = _unique([*chosen_facts, *texts], key=lambda item: item.id)
        image_items = _unique([*chosen_images, *images], key=lambda item: item.metadata.get("blob_id"))
        return [*facts, *image_items[: max(self._cfg.evidence_images, len(chosen_images))]]


def _parse(reply: str) -> dict[str, Any] | None:
    match = _JSON.search(reply or "")
    if not match:
        return None
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _unique(items: list[MemorySearchItem], *, key) -> list[MemorySearchItem]:
    seen: set[Any] = set()
    result = []
    for item in items:
        k = key(item)
        if k in seen:
            continue
        seen.add(k)
        result.append(item)
    return result
