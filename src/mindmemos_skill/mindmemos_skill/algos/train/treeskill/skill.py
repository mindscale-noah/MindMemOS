"""TreeSkill's algorithm-owned task-time Skill implementation."""

import json
from pathlib import Path
from typing import Any, Self

from pydantic import JsonValue

from ....agents import OpenAIChatAgent
from ....envs.typing import Task
from ....logging import get_logger
from ....skill import Skill, SkillContent, SkillSnapshot
from .config import TreeSkillConfig
from .env_adapter import TreeSkillEnvAdapter
from .tree import parse_tree, render_selection
from .typing import RoutingResult, SkillNode, TreeSkillRoutingContext


def _prompt_node(node: SkillNode) -> dict:
    """Expose only local node content and topology, avoiding duplicated subtrees."""
    return {
        "id": node.id,
        "level": node.level,
        "heading": node.heading,
        "local_content": node.local_content,
        "children": [_prompt_node(child) for child in node.children],
    }


def _selected_ids(text: str) -> list[str]:
    """Read the reference router's JSON selection, including fenced responses."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Router response does not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    ids = payload.get("selected_subtree_ids", payload.get("selected_node_ids"))
    if not isinstance(ids, list) or any(not isinstance(node_id, str) for node_id in ids):
        raise ValueError("Router must return a list of subtree ID strings")
    return ids


class TreeSkill(Skill):
    """Select task-relevant Markdown subtrees while retaining auxiliary files."""

    def __init__(
        self,
        config: TreeSkillConfig,
        *,
        source: SkillContent,
        router: OpenAIChatAgent | None = None,
    ) -> None:
        """Own source content and defer default router construction until routing.

        Args:
            config: Fixed model settings recorded in Skill snapshots.
            source: Complete source Markdown, metadata, and auxiliary bytes.
            router: Optional agent with exactly the recorded router configuration.
        """
        super().__init__(source)
        self.config = config.model_copy(deep=True)
        self.env_adapter = TreeSkillEnvAdapter.build(self.config.env_adapter.class_name, self.config.env_adapter.config)
        if router is not None and router.config != config.router:
            raise ValueError("Router config must match TreeSkill config")
        self.router = router

    @classmethod
    def from_path(cls, path: str | Path, *, config: TreeSkillConfig, router: OpenAIChatAgent | None = None) -> Self:
        """Load tree content with explicitly supplied routing settings.

        Args:
            path: Skill directory or SKILL.md entrypoint.
            config: Router settings owned by this algorithm.
            router: Optional borrowed router matching the settings.
        """
        source = Skill.from_path(path)
        return cls(config, source=source.source, router=router)

    def snapshot(self) -> SkillSnapshot:
        """Record routing settings alongside the source content."""
        snapshot = super().snapshot()
        snapshot.implementation = "treeskill"
        snapshot.config_schema_version = 2
        snapshot.config = self.config.model_dump(mode="json")
        return snapshot

    def with_content(self, content: str, *, name: str | None = None, description: str | None = None) -> Self:
        """Replace content while keeping routing settings isolated.

        Args:
            content: Replacement source document.
            name: Explicit name for nonstandard content.
            description: Explicit description for nonstandard content.
        """
        candidate = super().with_content(content, name=name, description=description)
        candidate.config = self.config.model_copy(deep=True)
        return candidate

    @classmethod
    def from_snapshot(cls, snapshot: SkillSnapshot, *, router: OpenAIChatAgent | None = None) -> Self:
        """Restore this algorithm explicitly from its source snapshot.

        Args:
            snapshot: TreeSkill source and routing settings.
            router: Optional borrowed router matching the stored settings.
        """
        source = Skill.from_snapshot(snapshot)
        if snapshot.implementation != "treeskill" or snapshot.config_schema_version not in (1, 2):
            raise ValueError("Unsupported TreeSkill snapshot implementation or schema version")
        recorded_config = dict(snapshot.config)
        if snapshot.config_schema_version == 1 and "env_adapter" not in recorded_config:
            # Historical stable-ID routing always used the spreadsheet request protocol.
            if recorded_config.get("reference_routing", False):
                recorded_config["env_adapter"] = {"class_name": "spreadsheetbench_text_action"}
            else:
                recorded_config["env_adapter"] = {"class_name": "legacy"}

        recorded_adapter = recorded_config.get("env_adapter")
        if isinstance(recorded_adapter, dict) and recorded_adapter.get("class_name") == "spreadsheetbench_reference":
            # Restore historical artifacts without registering the removed environment name.
            recorded_config["env_adapter"] = {**recorded_adapter, "class_name": "spreadsheetbench_text_action"}

        config = TreeSkillConfig.model_validate(recorded_config)
        if router is not None:
            if router.config.model_dump() != config.router.model_dump():
                raise ValueError("Injected router must match the recorded settings")
            config.router = router.config.model_copy(deep=True)
        return cls(
            config,
            source=source.source,
            router=router,
        )

    def assemble(self, task: Task) -> Self:
        """Return a complete Skill copy with its task-selected subtree content.

        Args:
            task: Public task input; private reward data is never supplied.
        """
        routed = self._route(task).content
        if routed == self.content:
            return self.with_content(routed)
        candidate = super().assemble(task)
        candidate._source = SkillContent(
            content=routed,
            resources=self.resources,
            name=self.source.name,
            description=self.source.description,
            standard_format=False,
        )
        candidate.config = self.config.model_copy(deep=True)
        return candidate

    def _route(self, task: Task) -> RoutingResult:
        """Route once internally, preserving evidence and propagating provider errors."""
        source = self.content
        if self.config.reference_routing:
            import asyncio

            from .routing import TreeSkillRouter
            from .typing import ChatResponse

            router_agent = self.router if self.router is not None else OpenAIChatAgent(self.config.router)

            class RouterModel:
                async def chat(self, task: str, messages: list[dict[str, JsonValue]], **kwargs: Any) -> ChatResponse:
                    """Chat.

                    Args:
                        task: Stage audit label or public task supplied by the caller.
                        messages: Explicit messages for this operation.
                    """
                    response = await asyncio.to_thread(router_agent.run, messages=messages)
                    if not isinstance(response.output, str):
                        raise TypeError("Router must return text")
                    self.response = response
                    return ChatResponse(content=response.output)

            model = RouterModel()
            model.response = None
            routed = asyncio.run(
                TreeSkillRouter(chat_model=model, env_adapter=self.env_adapter).route(
                    skill=self,
                    task=task,
                    tree_metadata=self.config.tree_metadata or None,
                )
            )
            return RoutingResult(
                content=routed.skill_content,
                selected_subtree_ids=list(routed.content_node_ids),
                ancestor_ids=list(routed.ancestor_node_ids),
                fallback_reason=routed.fallback_reason if routed.fallback_used else None,
                response=model.response,
            )
        tree = parse_tree(source)
        if not tree.roots:
            return RoutingResult(content=source, fallback_reason="no_heading_tree")
        messages = self.env_adapter.build_routing_messages(
            TreeSkillRoutingContext(task=task, tree=[_prompt_node(root) for root in tree.roots])
        )
        router = self.router if self.router is not None else OpenAIChatAgent(self.config.router)
        response = router.run(messages=messages)
        if not isinstance(response.output, str):
            raise TypeError("TreeSkill router must return text containing its JSON selection")
        try:
            selected = _selected_ids(response.output)
            text, roots, ancestors = render_selection(tree, selected)
        except ValueError as exc:
            get_logger(__name__).warning("TreeSkill used the full source content: %s", exc)
            return RoutingResult(content=source, response=response, fallback_reason=str(exc))
        return RoutingResult(
            content=text,
            selected_subtree_ids=roots,
            ancestor_ids=ancestors,
            response=response,
        )
