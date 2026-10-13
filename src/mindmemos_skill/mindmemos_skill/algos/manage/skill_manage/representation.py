"""Derive persistable joint features and deterministic embedding text from transient SSL."""

import re
from collections import deque
from pathlib import PurePosixPath

from ....persistence.db_services.skill_management.typing import SkillGraphNodeRecord
from ....scene.typing import ProvidedSceneRecord, SceneInput, SemanticDependency
from .parsing import extract_frontmatter, extract_markdown_section, normalize_string_list
from .typing import DerivedNodeMetadata

SCRIPT_SUFFIXES = frozenset({".py", ".sh", ".js", ".ts", ".mjs", ".cjs", ".rb", ".r", ".ps1"})
"""Executable resource suffixes represented as paths, without reading or running their contents."""


def _script_path(value):
    """Accept only portable relative script references, never remote URLs or server paths."""
    value = value.strip().removeprefix("./")
    path = PurePosixPath(value)
    return (
        value
        if (
            value
            and not path.is_absolute()
            and ".." not in path.parts
            and ":" not in value
            and "\\" not in value
            and path.suffix.lower() in SCRIPT_SUFFIXES
        )
        else ""
    )


def _dependencies(dependencies):
    """Classify typed dependencies while preserving free-form document requirements."""
    local, external = [], []
    for dependency in dependencies:
        if isinstance(dependency, str):
            external.append(dependency)
        elif dependency.version.casefold() == "local":
            if path := _script_path(dependency.name):
                local.append(path)
        else:
            external.append(f"{dependency.name} ({dependency.version})" if dependency.version else dependency.name)
    return local, external


def _document_dependencies(value) -> list[SemanticDependency | str]:
    """Normalize declarations without applying SSL name limits to ordinary document text."""
    if isinstance(value, dict):
        if isinstance(value.get("name"), str):
            version = value.get("version") or ""
            return [SemanticDependency(name=value["name"], version=str(version))]
        dependencies: list[SemanticDependency | str] = []
        for name, version in value.items():
            if isinstance(version, (dict, list, tuple, set)):
                dependencies.extend(_document_dependencies(version))
            else:
                dependencies.append(SemanticDependency(name=str(name), version=str(version or "")))
        return dependencies
    if isinstance(value, (list, tuple, set)):
        return [dependency for item in value for dependency in _document_dependencies(item)]
    return normalize_string_list(value)


def document_features(skill, parsed, *, member_id, family_id):
    """Derive retrieval features solely from registered resources and the exact document.

    Args:
        skill: Registered exact-version Skill; source_path is never inspected.
        parsed: Existing document parser output, including independent runtime declarations.
        member_id: Collection owner identity.
        family_id: Canonical stored family identity.

    Returns:
        Derived metadata and clean tooling without local dependency declaration fragments.
    """
    document = skill.content
    frontmatter, body = extract_frontmatter(document, version_id=member_id)
    scripts = [path for path in sorted(skill.resources) if _script_path(path)]
    declared = normalize_string_list(frontmatter.get("script_entrypoints"))
    declared += extract_markdown_section(body, ["Scripts", "Script Entrypoints"])
    declared += re.findall(r"(?<![\w/.:\\])(?:\./)?(?:[\w.-]+/)*[\w.-]+\.(?:py|sh|js|ts|mjs|cjs|rb|r|ps1)\b", body)
    scripts = _merge(scripts, [path for item in declared if (path := _script_path(item.strip("`")))])
    dependency_values = [
        frontmatter.get("dependencies"),
        frontmatter.get("libraries"),
        parsed["metadata"].get("libraries"),
        extract_markdown_section(body, ["Dependencies", "Libraries"]),
    ]
    dependencies = [dependency for value in dependency_values for dependency in _document_dependencies(value)]
    local, declared_external = _dependencies(dependencies)
    scripts = _merge(scripts, local)
    external = []
    for dependency in declared_external:
        if path := _script_path(dependency.strip("`")):
            scripts = _merge(scripts, [path])
        else:
            external.append(dependency)
    signature = frontmatter.get("intent_signature", parsed["metadata"].get("intent_signature", ""))
    metadata = DerivedNodeMetadata(
        collection_skill_id=member_id,
        family_id=family_id,
        intent_signature=_bounded(signature.strip(), 1000) if isinstance(signature, str) else "",
        local_scripts=tuple(scripts),
        external_dependencies=tuple(_merge([], external)),
        compatibility=tuple(parsed["compatibility"]),
        allowed_tools=tuple(parsed["allowed_tools"]),
    )
    declaration_fragments = {item for value in dependency_values for item in normalize_string_list(value)}
    tooling = _merge([item for item in parsed["tooling"] if item not in declaration_fragments], external)
    return metadata, tooling


def merge_skill_semantics(node: SkillGraphNodeRecord, record: ProvidedSceneRecord) -> SkillGraphNodeRecord:
    """Merge normalized semantics while retaining document names and rule-derived features.

    Args:
        node: Rule-parsed Skill node with its exact storage identity.
        record: Validated in-memory SSL for this member.

    Returns:
        A new node with unioned I/O/tooling and explicit intent metadata.
    """
    metadata = DerivedNodeMetadata.model_validate(node.metadata)
    local, external = _dependencies(record.skill.dependencies)
    metadata = metadata.model_copy(
        update={
            "intent": record.skill.goal,
            "intent_signature": record.skill.intent_signature or metadata.intent_signature,
            "local_scripts": tuple(_merge(metadata.local_scripts, local)),
            "external_dependencies": tuple(_merge(metadata.external_dependencies, external)),
        }
    )
    return node.model_copy(
        update={
            "inputs": _merge(node.inputs, record.skill.inputs),
            "outputs": _merge(node.outputs, record.skill.outputs),
            "tooling": _merge(node.tooling, (*record.skill.tooling, *external)),
            "domain_tags": _merge(node.domain_tags, record.skill.tags),
            "metadata": metadata.model_dump(mode="json"),
        }
    )


def _merge(left, right):
    """Preserve first occurrence order while dropping blank and duplicate values."""
    merged = {}
    for value in (*left, *right):
        if value.strip():
            merged.setdefault(value.strip().casefold(), value.strip())
    return list(merged.values())


def scene_flow_text(scene: SceneInput, record: ProvidedSceneRecord) -> str:
    """Render bounded actions and conditional outcomes without serializing step identities.

    Args:
        scene: Scene whose owned step flow is rendered.
        record: Validated transient record containing all owned actions.

    Returns:
        At most 800 characters of semantic flow, with explicit truncation when needed.
    """
    steps = {step.logic_step_id: step for step in record.logic_steps}
    pending = deque([scene.entry_logic_step_id])
    visited = set()
    parts = []
    while pending:
        key = pending.popleft()
        if key in visited:
            continue
        visited.add(key)
        step = steps[key]
        outcomes = []
        for rule in step.next_step_rules:
            terminal = rule.target.startswith("YIELD_")
            target = rule.target if terminal else _bounded(steps[rule.target].action, 120)
            outcomes.append(f"if {rule.condition} => {target}" if rule.condition else f"=> {target}")
            if not terminal:
                pending.append(rule.target)
        parts.append(f"{_bounded(step.action, 120)} {'; '.join(outcomes)}")
    return _bounded(" | ".join(parts), 800)


def _bounded(value, limit):
    """Mark semantic text truncated to a fixed character budget."""
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def scene_features(scene: SceneInput, record: ProvidedSceneRecord, *, member_id: str, family_id: str) -> dict:
    """Derive allowlisted node fields and entry/exit semantics from transient input.

    Args:
        scene: Normalized Scene in the record.
        record: Owning Skill's validated transient SSL.
        member_id: Collection key for ownership, independent of display names.
        family_id: Canonical family resolved by storage.

    Returns:
        Persistable feature fields without logic steps or raw SSL.
    """
    names = {item.scene_id: item.name for item in record.scenes}
    entries = [
        f"{source.name}: {rule.condition or 'unconditional'}"
        for source in record.scenes
        for rule in source.next_scene_rules
        if rule.target == scene.scene_id
    ]
    exits = [
        f"{rule.condition} => {names.get(rule.target, rule.target)}"
        if rule.condition
        else names.get(rule.target, rule.target)
        for rule in scene.next_scene_rules
    ]
    local, external = _dependencies(scene.dependencies)
    metadata = DerivedNodeMetadata(
        collection_skill_id=member_id,
        family_id=family_id,
        scene_id=scene.scene_id,
        scene_type=scene.scene_type,
        intent=scene.goal,
        intent_signature=scene.intent_signature,
        local_scripts=tuple(local),
        external_dependencies=tuple(external),
        is_entry=scene.scene_id == record.entry_scene_id,
        flow_text=scene_flow_text(scene, record),
        entry_conditions=tuple(_merge(scene.entry_conditions, entries)),
        exit_conditions=tuple(_merge(scene.exit_conditions, exits)),
    )
    return {
        "name": scene.name,
        "description": scene.goal,
        "one_line_capability": scene.name,
        "inputs": _merge([], scene.inputs),
        "outputs": _merge([], scene.outputs),
        "tooling": _merge(scene.tooling, external),
        "domain_tags": _merge([], scene.tags),
        "metadata": metadata.model_dump(mode="json"),
    }


def joint_embedding_fields(node: SkillGraphNodeRecord) -> list[tuple[str, str]]:
    """Render only validated derived fields, equally before publication and after reload.

    Args:
        node: Joint Skill or Scene record with allowlisted metadata.

    Returns:
        Tagged non-core fields for the versioned node embedding template.
    """
    metadata = DerivedNodeMetadata.model_validate(node.metadata)
    return [
        ("INTENT", metadata.intent),
        ("INTENT_SIGNATURE", metadata.intent_signature),
        ("SCRIPT_ENTRYPOINTS", "\n".join(metadata.local_scripts)),
        ("EXTERNAL_DEPENDENCIES", "\n".join(metadata.external_dependencies)),
        ("COMPATIBILITY", "\n".join(metadata.compatibility)),
        ("ALLOWED_TOOLS", "\n".join(metadata.allowed_tools)),
        ("SCENE_TYPE", metadata.scene_type),
        ("ENTRY", "Skill entry" if metadata.is_entry else ""),
        ("ENTRY_CONDITIONS", "\n".join(metadata.entry_conditions)),
        ("EXIT_CONDITIONS", "\n".join(metadata.exit_conditions)),
        ("FLOW", metadata.flow_text),
    ]


def linking_query_text(node: SkillGraphNodeRecord) -> str:
    """Render a structured linking query distinct from the full node embedding text.

    Args:
        node: Final joint node whose potential collaborators are sought.

    Returns:
        Source-grounded goal, artifact and tool fields, without document snippets.
    """
    metadata = DerivedNodeMetadata.model_validate(node.metadata)
    fields = [
        ("GOAL", metadata.intent or node.one_line_capability or node.description),
        ("INTENT_SIGNATURE", metadata.intent_signature),
        ("INPUTS", "; ".join(node.inputs)),
        ("OUTPUTS", "; ".join(node.outputs)),
        ("TOOLS", "; ".join(node.tooling)),
        ("SCRIPTS", "; ".join(metadata.local_scripts)),
        ("CONSTRAINTS", "; ".join((*metadata.compatibility, *metadata.allowed_tools))),
    ]
    return "\n".join(f"[{tag}] {value}" for tag, value in fields if value) or f"[GOAL] {node.name}"
