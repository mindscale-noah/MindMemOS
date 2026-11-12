"""Resolve observed Skill loads without executing their commands."""

import json
import os
import re
import shlex
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

from ..errors import MindMemOSSDKError
from .client import SkillClient
from .content import content_hash, parse_frontmatter
from .typing import ListRequest, SkillLoadEvidence, SkillReadEvidence, SkillResolveRequest, VersionResolveRequest

_VERSION_ID = re.compile(r"(?:skv-|v_)[A-Za-z0-9_-]+")
_VERSION_TOKEN = re.compile(r"(?<![\w-])(?:skv-|v_)[A-Za-z0-9_-]+(?![\w-])")
_HASH = re.compile(r"[0-9a-f]{64}")
_COMMAND_SEPARATORS = {";", "&&", "||", "|", "&"}


def identify_loaded_versions(client: SkillClient, evidence: Sequence[SkillLoadEvidence]) -> list[str]:
    """Identify exact versions from ordered, successful host observations.

    Args:
        client: SDK Skill client scoped to the reporting project.
        evidence: Observed shell results and local source reads in execution order.

    Returns:
        Resolved version IDs in first-load order, without duplicates.
    """
    exported: dict[Path, str] = {}
    loaded: dict[str, None] = {}
    for item in evidence:
        if isinstance(item, SkillReadEvidence):
            path = Path(item.path)
            if path.name != "SKILL.md":
                continue
            digest = content_hash(item.content)
            exported_version = exported.get(path)
            version = _version(client, digest=digest, version_id=exported_version) if exported_version else None
            version = version or _local_version(client, item.content, digest)
            if version:
                loaded[version] = None
            continue
        if not item.content.strip():
            continue
        action, reference = _load_command(item.command)
        if action not in {"show", "export"}:
            continue
        results = list(_shell_results(item.content, action))
        if action == "export":
            for result in results:
                directory = result.get("directory")
                exported_version = _version_id(result.get("skill_version_id"))
                if isinstance(directory, str) and Path(directory).is_absolute() and exported_version:
                    exported[Path(directory) / "SKILL.md"] = exported_version
            continue
        if action == "show" and not results:
            results = [{}]
        for result in results:
            version = _shown_version(client, reference if len(results) == 1 else None, result, item.content)
            if version:
                loaded[version] = None
    return list(loaded)


def _local_version(client: SkillClient, content: str, digest: str) -> str | None:
    try:
        metadata = parse_frontmatter(content)
    except ValueError:
        return None
    name = metadata["name"]
    family = _selection(client, name).get("skill_id")
    if isinstance(family, str):
        version = _version(client, digest=digest, family=family)
        if version:
            return version

    # Registration can use an alias that differs from the source name.
    try:
        page = client.list(ListRequest(filters={"name": name}, limit=2))
    except MindMemOSSDKError:
        return None
    items = page.get("items", [])
    if not isinstance(items, list) or len(items) != 1 or page.get("next_cursor"):
        return None
    match = items[0]
    if not isinstance(match, dict) or match.get("name") != name:
        return None
    named_family = match.get("skill_id")
    if isinstance(named_family, str) and named_family != family:
        return _version(client, digest=digest, family=named_family)
    return None


def _shown_version(client: SkillClient, reference: str | None, result: dict, output: str) -> str | None:
    source = result.get("content")
    if not isinstance(source, str) or not source.strip():
        source = output if output.startswith(("---\n", "---\r\n")) else None
    digest = content_hash(source) if source else _digest(result.get("content_hash"))
    bundle_hash = _digest(result.get("skill_hash"))
    version_id = _version_id(result.get("skill_version_id"))
    if not result and not source:
        # A wrapped result may expose only the immutable ID. Multiple IDs are ambiguous.
        tokens = set(_VERSION_TOKEN.findall(output))
        version_id = next(iter(tokens)) if len(tokens) == 1 else None
    if version_id:
        return _exact_version(client, version_id, digest, bundle_hash)
    if reference:
        if _version_id(reference):
            return _exact_version(client, reference, digest, bundle_hash)
        # An alias or family can change after the observed load. Resolve it only
        # to narrow a lookup constrained by the observed source or bundle hash.
        if not digest and not bundle_hash:
            return None
        selected = _selection(client, reference)
        family = selected.get("skill_id")
        if isinstance(family, str):
            return _version(client, digest=digest, bundle_hash=bundle_hash, family=family)
    if digest or bundle_hash:
        return _version(client, digest=digest, bundle_hash=bundle_hash)
    return None


def _selection(client: SkillClient, reference: str) -> dict:
    try:
        return client.skill_resolve(SkillResolveRequest(skill_ref=reference))
    except MindMemOSSDKError:
        return {}


def _exact_version(client: SkillClient, reference: str, digest: str | None, bundle_hash: str | None) -> str | None:
    version = _version(client, digest=digest, bundle_hash=bundle_hash, version_id=reference)
    if version:
        return version
    # Some host output abbreviates the immutable ID. A name or alias with the
    # same prefix must never turn that evidence into another Skill version.
    selected = _version_id(_selection(client, reference).get("skill_version_id"))
    if selected and selected != reference and selected.startswith(reference):
        return _version(client, digest=digest, bundle_hash=bundle_hash, version_id=selected)
    return None


def _version(
    client: SkillClient,
    *,
    digest: str | None = None,
    bundle_hash: str | None = None,
    family: str | None = None,
    version_id: str | None = None,
) -> str | None:
    try:
        result = client.version_resolve(
            VersionResolveRequest(
                content_hash=digest, skill_hash=bundle_hash, skill_id=family, skill_version_id=version_id
            )
        )
    except MindMemOSSDKError:
        return None
    if (digest and result.get("content_hash") != digest) or (bundle_hash and result.get("skill_hash") != bundle_hash):
        return None
    return _version_id(result.get("skill_version_id"))


def _version_id(value: object) -> str | None:
    return value if isinstance(value, str) and _VERSION_ID.fullmatch(value) else None


def _digest(value: object) -> str | None:
    return value if isinstance(value, str) and _HASH.fullmatch(value) else None


def _load_command(command: str) -> tuple[str | None, str | None]:
    """Recognize CLI loads in a shell command, including quoted executable paths."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    if os.name == "nt":
        lexer.escape = ""
    try:
        words = list(lexer)
    except ValueError:
        return None, None
    matches = []
    start = 0
    for index, word in enumerate([*words, ";"]):
        if word.strip("\n") not in _COMMAND_SEPARATORS | {""}:
            continue
        segment = words[start:index]
        start = index + 1
        while segment and (segment[0] in {"command", "env"} or re.match(r"^\w+=", segment[0])):
            segment = segment[1:]
        if segment[:2] == ["uv", "run"]:
            segment = segment[2:]
            while segment and segment[0] in {"--project", "--directory", "--python"}:
                segment = segment[2:]
            if segment[:1] == ["--"]:
                segment = segment[1:]
        if len(segment) < 3 or segment[1:3] not in (["skill", "show"], ["skill", "export"]):
            continue
        if {"-h", "--help"}.intersection(segment[3:]):
            continue
        executable = PurePosixPath(segment[0].replace("\\", "/")).name.lower()
        if executable in {"mindmemos", "mindmemos.exe"}:
            reference = segment[3] if len(segment) > 3 else None
            if reference and (reference.startswith("-") or any(char in reference for char in "$`")):
                reference = None
            # Weak evidence from a command reference alone requires the load to
            # be the final command and not be skipped by a preceding fallback.
            if any(token.strip("\n") not in {"", ";"} for token in words[index:]) or "||" in words[:start]:
                reference = None
            matches.append((segment[2], reference))
    if len(matches) == 1:
        return matches[0]
    if matches and all(action == "show" for action, _ in matches):
        # Several recognized shows may share stdout, but no command reference
        # can identify an individual result. Require evidence in each result.
        return "show", None
    return None, None


def _shell_results(output: str, action: str | None) -> Iterator[dict]:
    """Recover complete CLI JSON objects when a host wraps stdout."""
    decoder = json.JSONDecoder()
    end = 0
    for start in re.finditer(r"\{", output):
        if start.start() < end:
            continue
        try:
            value, end = decoder.raw_decode(output, start.start())
        except ValueError:
            continue
        if action == "export":
            if "directory" in value and "skill_version_id" in value:
                yield value
            continue
        version = value.get("version")
        if action == "show" and isinstance(version, dict):
            yield version
        elif action == "show" and "skill_version_id" in value:
            yield value
