from typing import Protocol

from ...typing import (
    GetPipelineInput,
    GetPipelineResult,
    MemoryListPipelineInput,
    MemoryListPipelineResult,
    MemoryRequestContext,
    MemoryScrollPipelineInput,
    MemoryScrollPipelineResult,
)


class GetPipeline(Protocol):
    async def get(self, inp: GetPipelineInput, context: MemoryRequestContext) -> GetPipelineResult:
        """Fetch memories by id or request filters.

        Args:
            inp: Get request payload.
            context: Tenant, project, and actor context for hard isolation.

        Returns:
            The hydrated memory results.
        """

    async def list(self, inp: MemoryListPipelineInput, context: MemoryRequestContext) -> MemoryListPipelineResult:
        """List memories with page/page_size metadata."""

    async def scroll(self, inp: MemoryScrollPipelineInput, context: MemoryRequestContext) -> MemoryScrollPipelineResult:
        """Scroll memories using an opaque storage cursor."""
