"""TreeSkill's algorithm-owned task-time Skill implementation."""

from pathlib import Path
from typing import Self

from ....agents import OpenAIChatAgent
from ....common.json import extract_json_object
from ....envs.typing import Task
from ....logging import get_logger
from ....skill import Skill, SkillContent, SkillSnapshot
from .config import TreeSkillConfig
from .construction.markdown import (
    TreeMetadataError,
    ancestor_closure,
    collapse_ancestor_descendant_selections,
    compile_tree_metadata,
    parse_skill_markdown,
    parse_tree_with_metadata,
    render_selected_subtrees,
    tree_prompt_payload,
)
from .env_adapter import TreeSkillEnvAdapter
from .typing import RoutingResult, TreeSkillRoutingContext, _RoutePayload


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
        if not self.config.tree_metadata:
            self.config.tree_metadata = compile_tree_metadata(parse_skill_markdown(self.content))
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
        snapshot.config_schema_version = 3
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
        if content != self.content:
            candidate.config.tree_metadata = compile_tree_metadata(parse_skill_markdown(content))
        return candidate

    @classmethod
    def from_snapshot(cls, snapshot: SkillSnapshot, *, router: OpenAIChatAgent | None = None) -> Self:
        """Restore this algorithm explicitly from its source snapshot.

        Args:
            snapshot: TreeSkill source and routing settings.
            router: Optional borrowed router matching the stored settings.
        """
        source = Skill.from_snapshot(snapshot)
        if snapshot.implementation != "treeskill" or snapshot.config_schema_version not in (1, 2, 3):
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

        recorded_config.pop("reference_routing", None)
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

    async def assemble(self, task: Task) -> Self:
        """Return a complete Skill copy with its task-selected subtree content.

        Args:
            task: Public task input; private reward data is never supplied.
        """
        routed = (await self.route(task)).content
        if routed == self.content:
            return self.with_content(routed)
        candidate = await super().assemble(task)
        candidate._source = SkillContent(
            content=routed,
            resources=self.resources,
            name=self.source.name,
            description=self.source.description,
            standard_format=False,
        )
        candidate.config = self.config.model_copy(deep=True)
        return candidate

    async def route(self, task: Task) -> RoutingResult:
        """Select source subtrees and return routing evidence without persisting it.

        Args:
            task: Public task inputs; never private reward data.

        Returns:
            Selected content and the original model response when a call was needed.

        Model or parsing failures use the full Skill, matching the reference router.
        """
        source = self.content
        try:
            tree = parse_tree_with_metadata(source, self.config.tree_metadata)
        except TreeMetadataError as exc:
            return RoutingResult(content=source, fallback_reason=f"metadata_validation_failed: {exc}")
        if not tree.nodes:
            return RoutingResult(content=source, fallback_reason="skill_tree_has_no_routable_nodes")
        messages = self.env_adapter.build_routing_messages(
            TreeSkillRoutingContext(task=task, tree=tree_prompt_payload(tree))
        )
        router = self.router if self.router is not None else OpenAIChatAgent(self.config.router)
        response = None
        try:
            response = await router.run(messages=messages)
            if not isinstance(response.output, str):
                raise TypeError("TreeSkill router must return text containing its JSON selection")
            payload = _RoutePayload.model_validate(extract_json_object(response.output, repair=False))
            unknown = set(payload.selected_subtree_ids) - set(tree.node_by_id)
            if unknown:
                raise ValueError(f"router returned unknown node ids: {sorted(unknown)}")
            selected = collapse_ancestor_descendant_selections(tree, payload.selected_subtree_ids)
            ancestors = ancestor_closure(tree, selected)
            content = render_selected_subtrees(tree, selected) if selected else ""
        except Exception as exc:
            get_logger(__name__).warning(f"skill.routing.fallback | source=full_content | reason={exc}")
            return RoutingResult(
                content=source, response=response, fallback_reason=f"router_failed: {type(exc).__name__}: {exc}"
            )
        return RoutingResult(
            content=content,
            selected_subtree_ids=list(selected),
            ancestor_ids=list(ancestors),
            response=response,
        )
