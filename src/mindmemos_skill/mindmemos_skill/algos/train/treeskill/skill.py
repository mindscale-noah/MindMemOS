"""TreeSkill's algorithm-owned task-time Skill implementation."""

import json
from pathlib import Path
from typing import Self

from ....agents import OpenAIChatAgent
from ....envs.typing import Task
from ....logging import get_logger
from ....skill import Skill, SkillBundle, SkillResources, SkillSnapshot
from .config import TreeSkillConfig
from .prompts import ROUTING_SYSTEM_PROMPT
from .tree import parse_tree, render_selection
from .typing import RoutingResult, SkillNode


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
        bundle: SkillBundle,
        resources: SkillResources | None = None,
        router: OpenAIChatAgent | None = None,
    ):
        """Own source content and defer default router construction until routing.

        Args:
            config: Fixed model settings recorded in Skill snapshots.
            bundle: Complete source Markdown, never the previous routed output.
            resources: Unchanged references, scripts, and assets.
            router: Optional agent with exactly the recorded router configuration.
        """
        super().__init__(bundle=bundle, resources=resources)
        self.config = config.model_copy(deep=True)
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
        return cls(config, bundle=source.bundle, resources=source.resources, router=router)

    def snapshot(self) -> SkillSnapshot:
        """Record routing settings alongside the source content."""
        snapshot = super().snapshot()
        snapshot.implementation = "treeskill"
        snapshot.config_schema_version = 1
        snapshot.config = self.config.model_dump(mode="json")
        return snapshot

    def with_bundle(self, bundle: SkillBundle) -> Self:
        """Replace content while keeping routing settings isolated.

        Args:
            bundle: Replacement source content for the candidate.
        """
        candidate = super().with_bundle(bundle)
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
        if snapshot.implementation != "treeskill" or snapshot.config_schema_version != 1:
            raise ValueError("Unsupported TreeSkill snapshot implementation or schema version")
        return cls(
            TreeSkillConfig.model_validate(snapshot.config),
            bundle=source.bundle,
            resources=source.resources,
            router=router,
        )

    def assemble(self, task: Task) -> Self:
        """Return a complete Skill copy with its task-selected subtree bundle.

        Args:
            task: Public task input; private reward data is never supplied.
        """
        return self.with_bundle(self._route(task).bundle)

    def _route(self, task: Task) -> RoutingResult:
        """Route once internally, preserving evidence and propagating provider errors."""
        source = self.bundle
        tree = parse_tree(source.files["SKILL.md"])
        if not tree.roots:
            return RoutingResult(bundle=source, fallback_reason="no_heading_tree")
        payload = {"task": task.model_dump(mode="json"), "skill_tree": [_prompt_node(root) for root in tree.roots]}
        router = self.router if self.router is not None else OpenAIChatAgent(self.config.router)
        response = router.run(
            system=ROUTING_SYSTEM_PROMPT,
            user="Route skill subtrees for this task.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2),
        )
        if not isinstance(response.output, str):
            raise TypeError("TreeSkill router must return text containing its JSON selection")
        try:
            selected = _selected_ids(response.output)
            text, roots, ancestors = render_selection(tree, selected)
        except ValueError as exc:
            get_logger(__name__).warning("TreeSkill used the full source bundle: %s", exc)
            return RoutingResult(bundle=source, response=response, fallback_reason=str(exc))
        return RoutingResult(
            bundle=SkillBundle(files={"SKILL.md": text}),
            selected_subtree_ids=roots,
            ancestor_ids=ancestors,
            response=response,
        )
