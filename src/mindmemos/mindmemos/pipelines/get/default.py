"""Default get-memory pipeline implementation."""

from __future__ import annotations

from ...typing import GetPipelineInput, GetPipelineResult, MemoryRequestContext
from ..base import MemoryDbPipelineMixin
from ..memory_db import MemoryCatalog
from ..registry import register


@register(type="get", name="default_get")
class DefaultGetPipeline(MemoryDbPipelineMixin):
    """Default unscored get pipeline backed by the memory catalog."""

    async def get(self, inp: GetPipelineInput, context: MemoryRequestContext) -> GetPipelineResult:
        """Return active memories matching the request filter."""

        return await MemoryCatalog(reader=self.db_reader).get(inp, context)
