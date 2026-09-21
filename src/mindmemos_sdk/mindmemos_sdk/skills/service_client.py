"""Client for the standalone Skill FastAPI service, without automatic retries."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from ..config.connections import SkillConnectionConfig


class SkillServiceClient:
    """Read Skill client settings and record one attempt per algorithm operation.

    This client targets standalone Skill FastAPI, not the deployed cloud Skill API.
    The server owns project identity; requests do not configure a client project.
    """

    def __init__(self, home: Path, *, config: SkillConnectionConfig | None = None) -> None:
        self.config = config or SkillConnectionConfig.from_home(home)
        self.experiment_dir = self.config.experiment_directory(home)
        headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=self.config.timeout_seconds,
            transport=httpx.AsyncHTTPTransport(retries=0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def train(self, request: dict[str, Any]) -> Any:
        """Submit exactly one training attempt and store its output and log."""
        return await self._run("auto-train", request)

    async def evolve(self, request: dict[str, Any]) -> Any:
        """Submit exactly one evolution attempt and store its output and log."""
        return await self._run("auto-evolve", request)

    async def _run(self, operation: str, request: dict[str, Any]) -> Any:
        if "project_id" in request:
            raise ValueError("Project identity is assigned by the Skill service")
        directory = self.experiment_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex}"
        directory.mkdir(parents=True, mode=0o700)
        log = directory / "experiment.log"
        log.write_text(f"operation={operation}\nstatus=running\nattempt=1\n", encoding="utf-8")
        try:
            response = await self._client.post(f"skills/{operation}", json=request)
            response.raise_for_status()
            result = response.json()
            (directory / "output.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except BaseException as exc:
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"status=failed_or_unknown\nerror_type={type(exc).__name__}\nretry=false\n")
            raise
        with log.open("a", encoding="utf-8") as stream:
            stream.write("status=response_received\nretry=false\n")
        return result
