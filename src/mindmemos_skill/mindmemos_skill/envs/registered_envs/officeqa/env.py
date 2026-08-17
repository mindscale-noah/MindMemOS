"""SkillOpt-compatible OfficeQA local-document environment."""

from __future__ import annotations

import asyncio
import fnmatch
import html
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from pydantic import Field

from ....agents.base import Agent
from ....agents.react.tool import Tool
from ....registry import ComponentType, register
from ....typing import EnvConfig, Reward, Skill, SkillInjectionMode, Task, Trajectory
from ...base import BaseEnv, EnvRolloutContext, PreparedRollout
from .evaluator import evaluate
from .search import (
    DEFAULT_SERPAPI_API_KEY_ENV,
    DEFAULT_SERPAPI_BASE_URL_ENV,
    DEFAULT_SERPAPI_PROTOCOL_ENV,
    custom_search,
)

_MAX_READ_CHARS = 4000
_MAX_GREP_MATCHES = 20
_MAX_GLOB_MATCHES = 50
_MAX_ORACLE_PAGE_CHARS = 24000
_MAX_ORACLE_CONTEXT_CHARS = 80000
_FINAL_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_SEARCH_RE = re.compile(r"<search_queries>(.*?)</search_queries>", re.IGNORECASE | re.DOTALL)
_OFFLINE_SEARCH_MODE = "offline"
_CUSTOM_SEARCH_MODE = "custom_search"

_SYSTEM_PROMPT = """You are an expert OfficeQA agent working over local Treasury bulletin text files.

{skill_section}## Rules
1. Use only the provided local document tools to inspect candidate files.
2. Narrow to the most relevant file before reading long passages.
3. Prefer short targeted searches, then small reads around matching evidence.
4. Do not invent values that are not grounded in the retrieved text.
5. When the question requires arithmetic, compute only after extracting the exact operands.
6. If you have enough evidence, return the final answer inside <answer>...</answer>.

## Tool Use
Use the provided function tools directly when you need them. Prefer searching and small reads before answering. Do not ask the user for permission to use tools; just call the tools.

## Final Answer Format
When you are ready to answer, emit the final answer inside <answer>...</answer> and do not request another tool.
"""


def _extract_answer(text: str) -> str:
    match = _FINAL_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if value is not None and str(value).strip() else []


def _page_number(source_doc: str) -> int | None:
    query = parse_qs(urlparse(source_doc).query)
    for key in ("page", "pagenum", "page_id"):
        for raw_value in query.get(key, []):
            try:
                return int(raw_value)
            except ValueError:
                continue
    match = re.search(r"(?:[?&]|^)page=(\d+)", source_doc)
    if match:
        return int(match.group(1))
    return None


def _resolve_candidate_files(source_files: list[str], docs_roots: list[Path]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for root in docs_roots:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if source_files and filename not in source_files:
                    continue
                path = str(Path(dirpath, filename).resolve())
                if path in seen:
                    continue
                seen.add(path)
                resolved.append(path)
    return resolved


def _normalize_search_mode(search_mode: str | None) -> str:
    normalized = str(search_mode or _OFFLINE_SEARCH_MODE).strip().lower()
    return _CUSTOM_SEARCH_MODE if normalized in {"custom", _CUSTOM_SEARCH_MODE} else _OFFLINE_SEARCH_MODE


def _build_system(
    skill_content: str,
    *,
    search_mode: str = _OFFLINE_SEARCH_MODE,
    max_turns: int = 24,
    max_queries_per_turn: int = 4,
) -> str:
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    if _normalize_search_mode(search_mode) == _CUSTOM_SEARCH_MODE:
        return (
            "You are an expert OfficeQA research assistant. Solve the question using the provided oracle parsed "
            "OfficeQA page(s) and evidence returned by the controller-managed custom search loop.\n\n"
            "Search protocol:\n"
            f"- You have at most {max_turns} model rounds total.\n"
            f"- On any non-final round, you may either return `<search_queries>[\"query 1\", \"query 2\"]</search_queries>` "
            f"with up to {max_queries_per_turn} queries, or return `<answer>...</answer>` if you are ready.\n"
            "- If you request search, do not include an answer in the same response.\n"
            "- On the final round, you must return `<answer>...</answer>` and must not request more search.\n"
            "- Base your answer on the returned evidence, reconcile conflicting snippets carefully, and stay concise.\n\n"
            + skill_section
            + "Return the final answer inside <answer>...</answer> when you are ready."
        )
    return _SYSTEM_PROMPT.format(skill_section=skill_section)


def _build_round_instruction(*, turn: int, max_turns: int, max_queries_per_turn: int) -> str:
    if turn >= max_turns:
        return (
            "## Round Policy\n"
            f"This is the final round ({turn}/{max_turns}). You must return `<answer>...</answer>` now. "
            "Do not output `<search_queries>`."
        )
    remaining_rounds = max_turns - turn
    return (
        "## Round Policy\n"
        f"This is round {turn}/{max_turns}. "
        f"You may either return `<answer>...</answer>` now, or request up to {max_queries_per_turn} search queries "
        f"inside `<search_queries>...</search_queries>`. "
        f"After this response, at most {remaining_rounds} model rounds remain."
    )


def _extract_search_queries(text: str) -> list[str]:
    match = _SEARCH_RE.search(text or "")
    if not match or not (raw := match.group(1).strip()):
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    queries: list[str] = []
    if isinstance(parsed, dict):
        for key in ("queries", "search_queries", "query"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                queries = [value.strip()]
                break
            if isinstance(value, list):
                queries = [str(item).strip() for item in value if str(item).strip()]
                break
    elif isinstance(parsed, list):
        queries = [str(item).strip() for item in parsed if str(item).strip()]
    elif isinstance(parsed, str) and parsed.strip():
        queries = [parsed.strip()]
    if not queries:
        queries = [line for line in (line.strip(" -*\t\r\n\"'") for line in raw.splitlines()) if line]
    if len(queries) == 1:
        split_queries = [part.strip(" \"'") for part in re.split(r"[;,]", queries[0]) if part.strip(" \"'")]
        if len(split_queries) > 1:
            queries = split_queries
    return list(dict.fromkeys(query for query in queries if query))


def _single_skill_content(skills: Sequence[Skill]) -> str:
    contents = [skill.content.strip() for skill in skills if skill.content.strip()]
    if len(contents) > 1:
        raise ValueError("SkillOpt-compatible OfficeQA rollouts accept at most one non-empty Skill")
    return contents[0] if contents else ""


def _locate_parsed_json(source_file: str, docs_roots: list[Path]) -> Path | None:
    source = Path(source_file)
    name = f"{source.stem}.json"
    for root in docs_roots:
        for candidate in (root / "jsons" / name, root.parent / "jsons" / name):
            if candidate.is_file():
                return candidate
    return None


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif normalized == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def _render_content(content: str) -> str:
    text = content.strip()
    if not text:
        return ""
    if "<table" not in text.lower():
        return re.sub(r"\n{3,}", "\n\n", html.unescape(text).replace("\r\n", "\n")).strip()
    parser = _TableParser()
    parser.feed(text)
    if not parser.rows:
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"(?is)<[^>]+>", " ", text))).strip()
    width = max(len(row) for row in parser.rows)
    rows = [row + [""] * (width - len(row)) for row in parser.rows]
    escaped = [[cell.replace("|", "\\|") for cell in row] for row in rows]
    lines = [
        "| " + " | ".join(escaped[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in escaped[1:])
    return "\n".join(lines)


@lru_cache(maxsize=256)
def _load_parsed_elements(path: str) -> tuple[dict[str, Any], ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    document = raw.get("document") if isinstance(raw, dict) else None
    elements = document.get("elements") if isinstance(document, dict) else None
    return tuple(item for item in elements or [] if isinstance(item, dict))


def _render_page(path: Path, page_number: int) -> str:
    sections = []
    for element in _load_parsed_elements(str(path)):
        boxes = element.get("bbox")
        page_ids = {
            int(box["page_id"])
            for box in boxes or []
            if isinstance(box, dict) and str(box.get("page_id", "")).isdigit()
        }
        if page_number not in page_ids:
            continue
        content = element.get("content")
        if isinstance(content, str) and (rendered := _render_content(content)):
            sections.append(rendered)
    return "\n\n".join(sections)


def _build_oracle_context(
    task: Task,
    docs_roots: list[Path],
    *,
    evidence_note: str = (
        "Treat it as primary document evidence and combine it with local document tool evidence when useful."
    ),
) -> str:
    source_files = _as_list(task.metadata.get("source_files"))
    source_docs = _as_list(task.metadata.get("source_docs"))
    blocks: list[str] = []
    total = 0
    seen_pages: set[tuple[str, int]] = set()
    for index, source_doc in enumerate(source_docs):
        page_number = _page_number(source_doc)
        if page_number is None or not source_files:
            continue
        if index < len(source_files):
            source_file = source_files[index]
        elif len(source_files) == 1:
            source_file = source_files[0]
        else:
            continue
        json_path = _locate_parsed_json(source_file, docs_roots)
        if json_path is None:
            continue
        page_key = (str(json_path), page_number)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        page = _render_page(json_path, page_number)
        if not page:
            continue
        if len(page) > _MAX_ORACLE_PAGE_CHARS:
            omitted = len(page) - _MAX_ORACLE_PAGE_CHARS
            page = page[:_MAX_ORACLE_PAGE_CHARS].rstrip() + f"\n\n[... {omitted} characters omitted from this parsed page ...]"
        block = f"### {source_file} page {page_number}\nSource URL: {source_doc}\n\n{page}"
        if total + len(block) > _MAX_ORACLE_CONTEXT_CHARS:
            remaining = _MAX_ORACLE_CONTEXT_CHARS - total
            if remaining <= 0:
                break
            block = block[:remaining].rstrip() + "\n\n[... oracle parsed page context truncated ...]"
            blocks.append(block)
            break
        blocks.append(block)
        total += len(block)
    if not blocks:
        return ""
    return (
        "The following content is pre-parsed from the oracle OfficeQA source page(s). "
        f"{evidence_note.strip()}\n\n"
        + "\n\n".join(blocks)
    )


class OfficeQAEnvConfig(EnvConfig):
    max_turns: int = Field(default=24, ge=1)
    docs_root: str
    search_mode: Literal["offline", "custom_search"] = _OFFLINE_SEARCH_MODE
    max_queries_per_turn: int = Field(default=4, ge=1)
    search_api_base_url_env: str = DEFAULT_SERPAPI_BASE_URL_ENV
    search_api_key_env: str = DEFAULT_SERPAPI_API_KEY_ENV
    search_api_protocol_env: str = DEFAULT_SERPAPI_PROTOCOL_ENV
    search_max_num_results: int = Field(default=4, ge=1)
    search_timeout_seconds: int = Field(default=20, ge=1)


@register(type=ComponentType.ENV, name="officeqa")
class OfficeQAEnv(BaseEnv[OfficeQAEnvConfig]):
    """Answer OfficeQA questions through candidate-scoped local text tools."""

    config_type = OfficeQAEnvConfig

    async def _prepare(
        self,
        *,
        task: Task,
        skills: Sequence[Skill],
        context: EnvRolloutContext,
    ) -> PreparedRollout:
        prepared = await super()._prepare(task=task, skills=skills, context=context)
        if self.config.search_mode == _CUSTOM_SEARCH_MODE:
            if not os.environ.get(self.config.search_api_base_url_env, "").strip():
                raise ValueError(
                    f"OfficeQA custom_search requires API base URL env var {self.config.search_api_base_url_env}"
                )
            if not os.environ.get(self.config.search_api_key_env, "").strip():
                raise ValueError(
                    f"OfficeQA custom_search requires API key env var {self.config.search_api_key_env}"
                )
            protocol = os.environ.get(self.config.search_api_protocol_env, "").strip().lower()
            if protocol not in {"serper", "yibu"}:
                raise ValueError(
                    f"OfficeQA custom_search requires {self.config.search_api_protocol_env}=serper or yibu"
                )
        docs_roots = self._docs_roots()
        source_files = [Path(str(item)).name for item in task.metadata.get("source_files", [])]
        candidate_files = _resolve_candidate_files(source_files, docs_roots)
        tools = OfficeQATools(
            docs_roots=docs_roots,
            allowed_files=[Path(path).name for path in candidate_files],
        ).as_tools()
        oracle_context = _build_oracle_context(
            task,
            docs_roots,
            evidence_note=(
                "Treat it as primary document evidence and combine it with custom web search results when useful."
                if self.config.search_mode == _CUSTOM_SEARCH_MODE
                else "Treat it as primary document evidence and combine it with local document tool evidence when useful."
            ),
        )
        prepared.agent_request.options["skill_injection_mode"] = SkillInjectionMode.SYSTEM_PROMPT.value
        messages = self._messages(
            task,
            skill_content=_single_skill_content(skills),
            candidate_files=candidate_files,
            oracle_context=oracle_context,
            search_mode=self.config.search_mode,
            max_turns=self.config.max_turns,
            max_queries_per_turn=self.config.max_queries_per_turn,
        )
        if task.system_prompt:
            messages[0] = {
                **messages[0],
                "content": f"{messages[0]['content'].rstrip()}\n\n{task.system_prompt.strip()}",
            }
        prepared.runtime_state = {
            "messages": messages,
            "initial_messages": [],
            "tools": tools,
            "skill_runtime": {},
            "error": None,
            "finished": False,
            "turns": 0,
            "prediction": "",
            "evaluation": {},
            "oracle_context_chars": len(oracle_context),
            "search_mode": self.config.search_mode,
        }
        return prepared

    async def _execute(self, *, agent: Agent[Any], prepared: PreparedRollout) -> Trajectory:
        state = prepared.runtime_state
        messages = list(state["messages"])
        started_at = time.time()
        error: str | None = None
        finished = False
        turns = 0
        try:
            async with agent.on_skill_runtime_task(
                prepared.agent_request,
                mode=SkillInjectionMode.SYSTEM_PROMPT,
            ) as injection:
                # OfficeQA embeds the raw Skill in its benchmark prompt to match SkillOpt exactly.
                state["initial_messages"] = list(messages)
                tools: list[Tool] = [*state["tools"], *injection.tools]
                tools_by_name = {tool.name: tool for tool in tools}
                if len(tools_by_name) != len(tools):
                    raise ValueError("Skill Runtime tool conflicts with an OfficeQA tool")
                schemas = (
                    []
                    if self.config.search_mode == _CUSTOM_SEARCH_MODE
                    else [tool.to_openai_schema() for tool in tools]
                )
                for turns in range(1, self.config.max_turns + 1):
                    response = await agent.respond(prepared.agent_request, messages, tools=schemas)
                    assistant = response.to_assistant_message()
                    messages.append(assistant)
                    if self.config.search_mode == _CUSTOM_SEARCH_MODE:
                        content = assistant.get("content") if isinstance(assistant.get("content"), str) else ""
                        if _FINAL_RE.search(content):
                            finished = True
                            break
                        if turns >= self.config.max_turns:
                            error = f"Final round ({self.config.max_turns}) ended without <answer>...</answer>"
                            break
                        queries = _extract_search_queries(content)[: self.config.max_queries_per_turn]
                        if not queries:
                            error = "Model neither produced search queries nor a final answer"
                            break
                        results = await self._custom_search_round(queries)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"## Search Results Round {turns}\n{results}\n\n"
                                    + _build_round_instruction(
                                        turn=turns + 1,
                                        max_turns=self.config.max_turns,
                                        max_queries_per_turn=self.config.max_queries_per_turn,
                                    )
                                    + "\n\nFollow the round policy above exactly."
                                ),
                            }
                        )
                        continue
                    calls = assistant.get("tool_calls") or []
                    if not calls:
                        finished = True
                        break
                    for call in calls:
                        messages.extend(await self._call_tool(call, tools_by_name))
                state["skill_runtime"] = injection.metadata.get("skill_runtime", {})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ended_at = time.time()
        prediction = _extract_answer(self._last_assistant_content(messages))
        state.update(
            {
                "messages": messages,
                "error": error,
                "finished": finished,
                "turns": turns,
                "prediction": prediction,
            }
        )
        self._write_artifacts(prepared)
        return agent.build_trajectory(
            request=prepared.agent_request,
            messages=messages,
            started_at=started_at,
            ended_at=ended_at,
            n_turn=turns,
            is_success=error is None,
            error_info=error,
            metadata={
                "finished": finished,
                "turns": turns,
                "error": error,
                "prediction": prediction,
                "oracle_parsed_pages_chars": state["oracle_context_chars"],
                "search_mode": state["search_mode"],
                "skill_runtime": state["skill_runtime"],
            },
        )

    async def _evaluate(self, *, trajectory: Trajectory, prepared: PreparedRollout) -> Reward:
        del trajectory
        state = prepared.runtime_state
        gold = str(prepared.agent_request.task.metadata.get("ground_truth") or "")
        prediction = str(state["prediction"])
        evaluation = evaluate(prediction, gold)
        state["evaluation"] = evaluation
        self._write_artifacts(prepared)
        return Reward(
            score=float(evaluation["em"]),
            detail=(
                f"official_score={float(evaluation['em']):.1f}, "
                f"evaluator_commit={evaluation['evaluator_commit']}, tolerance={evaluation['tolerance']}"
            ),
            metadata=evaluation,
        )

    def _docs_roots(self) -> list[Path]:
        candidates = [Path(part).expanduser() for part in self.config.docs_root.split(os.pathsep) if part]
        roots: list[Path] = []
        for candidate in candidates:
            transformed = candidate / "transformed"
            path = transformed if transformed.is_dir() else candidate
            if path.is_dir():
                roots.append(path.resolve())
        if not roots:
            raise FileNotFoundError(f"OfficeQA docs directory not found: {self.config.docs_root}")
        return roots

    @staticmethod
    def _messages(
        task: Task,
        *,
        skill_content: str = "",
        candidate_files: Sequence[str] = (),
        oracle_context: str = "",
        search_mode: str = _OFFLINE_SEARCH_MODE,
        max_turns: int = 24,
        max_queries_per_turn: int = 4,
    ) -> list[dict[str, Any]]:
        source_docs = task.metadata.get("source_docs") or []
        candidate_block = "\n".join(f"- {item}" for item in candidate_files[:20]) or "- none resolved"
        user_parts = [f"## Question\n{task.instruction}"]
        if oracle_context:
            user_parts.append(f"## Oracle Parsed Pages\n{oracle_context}")
        normalized_search_mode = _normalize_search_mode(search_mode)
        if normalized_search_mode == _OFFLINE_SEARCH_MODE:
            user_parts.append(f"## Candidate Files\n{candidate_block}")
        if source_docs:
            user_parts.append("## Source Hints\n" + "\n".join(f"- {item}" for item in source_docs))
        if normalized_search_mode == _CUSTOM_SEARCH_MODE:
            source_files = task.metadata.get("source_files") or []
            if source_files:
                user_parts.append("## File Hints\n" + "\n".join(f"- {item}" for item in source_files))
            user_parts.append(
                _build_round_instruction(
                    turn=1,
                    max_turns=max_turns,
                    max_queries_per_turn=max_queries_per_turn,
                )
            )
            user_parts.append(
                "## Output Format\n"
                "If you need more evidence, return only `<search_queries>[...]</search_queries>`.\n"
                "If you are ready to answer, return only `<answer>...</answer>`."
            )
            user_parts.append(
                "Use only the provided oracle parsed pages and controller-provided custom search evidence. "
                "Do not rely on any built-in web search capability."
            )
        return [
            {
                "role": "system",
                "content": _build_system(
                    skill_content,
                    search_mode=normalized_search_mode,
                    max_turns=max_turns,
                    max_queries_per_turn=max_queries_per_turn,
                ),
            },
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    async def _custom_search_round(self, queries: Sequence[str]) -> str:
        blocks: list[str] = []
        for index, query in enumerate(queries, start=1):
            try:
                result = await asyncio.to_thread(
                    custom_search,
                    query,
                    base_url=os.environ[self.config.search_api_base_url_env],
                    protocol=os.environ[self.config.search_api_protocol_env],
                    api_key_env=self.config.search_api_key_env,
                    max_num_results=self.config.search_max_num_results,
                    timeout=self.config.search_timeout_seconds,
                )
            except Exception as exc:
                result = f"Query: {query}\n\n[search error: {exc}]"
            blocks.append(f"## Query {index}\n{result}")
        return "\n\n".join(blocks)

    @staticmethod
    async def _call_tool(call: Any, tools: Mapping[str, Tool]) -> list[dict[str, Any]]:
        call = call if isinstance(call, Mapping) else {}
        function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
        name = str(function.get("name") or "")
        call_id = str(call.get("id") or "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError:
            arguments = {"__raw__": raw_arguments}
        tool = tools.get(name)
        if tool is None:
            content = f"Error: unknown tool {name!r}"
        else:
            try:
                result = await tool.call(arguments or {})
                content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                content = f"Error: {type(exc).__name__}: {exc}"
        return [{"role": "tool", "tool_call_id": call_id, "name": name, "content": content}]

    @staticmethod
    def _last_assistant_content(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                content = message.get("content")
                return content if isinstance(content, str) else ""
        return ""

    @staticmethod
    def _write_artifacts(prepared: PreparedRollout) -> None:
        if prepared.environment.running_dir is None:
            return
        workspace = Path(prepared.environment.running_dir)
        state = prepared.runtime_state
        (workspace / "conversation.json").write_text(
            json.dumps(state["messages"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if state["evaluation"]:
            (workspace / "prediction.json").write_text(
                json.dumps(state["evaluation"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


class OfficeQATools:
    def __init__(self, *, docs_roots: list[Path], allowed_files: list[str]) -> None:
        self.docs_roots = docs_roots
        self.allowed_files = set(allowed_files)

    def _path(self, value: str) -> Path | None:
        raw = Path(value).expanduser()
        candidates = [raw] if raw.is_absolute() else [root / raw for root in self.docs_roots]
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.is_file():
                continue
            if not any(resolved == root or root in resolved.parents for root in self.docs_roots):
                continue
            if self.allowed_files and resolved.name not in self.allowed_files:
                continue
            return resolved
        return None

    def glob(self, pattern: str = "*") -> str:
        matches: list[str] = []
        for root in self.docs_roots:
            for path in root.rglob("*"):
                if not path.is_file() or (self.allowed_files and path.name not in self.allowed_files):
                    continue
                relative = str(path.relative_to(root))
                if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern):
                    matches.append(str(path))
                if len(matches) >= _MAX_GLOB_MATCHES:
                    return "\n".join(matches)
        return "\n".join(matches) if matches else "[no matches]"

    def read(self, path: str, start: int = 1, limit: int = 80) -> str:
        target = self._path(path)
        if target is None:
            return f"[read error: path not allowed: {path}]"
        start = max(int(start), 1)
        limit = max(int(limit), 1)
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        excerpt = "".join(lines[start - 1 : start - 1 + limit])
        return excerpt[:_MAX_READ_CHARS] or "[empty file]"

    def grep(self, pattern: str, path: str) -> str:
        target = self._path(path)
        normalized = str(pattern or "").lower()
        if target is None or not normalized:
            return "[grep error: missing pattern or path not allowed]"
        matches = []
        for index, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            if normalized in line.lower():
                matches.append(f"{index}: {line}")
            if len(matches) >= _MAX_GREP_MATCHES:
                break
        return "\n".join(matches) if matches else "[no matches]"

    def as_tools(self) -> list[Tool]:
        return [
            Tool(
                "glob",
                "Find candidate local OfficeQA document files by filename or relative-path glob pattern.",
                self.glob,
                {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
            ),
            Tool(
                "read",
                "Read a local OfficeQA text document excerpt by path and line window.",
                self.read,
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                "grep",
                "Search a local OfficeQA text document for a literal pattern and return matching lines.",
                self.grep,
                {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern", "path"],
                },
            ),
        ]


__all__ = ["OfficeQAEnv", "OfficeQAEnvConfig", "OfficeQATools"]
