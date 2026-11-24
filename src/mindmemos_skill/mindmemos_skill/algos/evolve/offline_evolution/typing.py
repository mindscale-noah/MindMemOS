"""Stage file contracts, gate reports, run state and the evolution result."""

import hashlib
import shutil
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from ....skill import Skill
from ....skill.typing import SkillSnapshot
from ..typing import SkillTrajectory
from .config import EvidenceGateConfig, EvolutionOptions, OfflineEvolutionConfig
from .constants import HOST_DIR

IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$"
"""Pattern for change ids: no spaces, colons or path separators."""

FINDING_ID = r"^[^\s:]+$"
"""Pattern for finding and case ids: non-empty text without whitespace or colons."""

InputPath = Annotated[str, Field(min_length=1), AfterValidator(lambda path: path.replace("\\", "/"))]
"""Input-relative path reported by a script; Windows separators from os.path are normalized to `/`."""

StageName = Literal["evidence", "evidence_gate", "change", "change_gate", "publish"]


class TokenUsage(BaseModel):
    """Tokens consumed by agent sessions, as reported by the backend."""

    input: int = 0
    """Uncached input tokens."""

    cache_write: int = 0
    """Input tokens written to the prompt cache."""

    cache_read: int = 0
    """Input tokens read from the prompt cache."""

    output: int = 0
    """Output tokens, including reasoning."""

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Sum two usages field by field.

        Args:
            other: Successfully returned usage to add to this total.

        Returns:
            New cumulative usage without changing either operand.
        """
        return TokenUsage(
            input=self.input + other.input,
            cache_write=self.cache_write + other.cache_write,
            cache_read=self.cache_read + other.cache_read,
            output=self.output + other.output,
        )


class _AgentFile(BaseModel):
    """Base for files written by agents; extra keys are ignored, required keys are enforced."""

    model_config = ConfigDict(extra="ignore")
    """Agents may add notes; only declared fields are read."""


class EvidenceManifest(_AgentFile):
    """Content of collection.json: the single collection entry point."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported collection options."""

    collector: str = Field(min_length=1)
    """Workspace-relative script producing the unified findings output."""


class FailedInput(_AgentFile):
    """A trajectory file the script could not parse."""

    path: InputPath
    """Input-relative path, e.g. trajectories/task/run.json."""

    reason: str = Field(min_length=1)
    """Why parsing failed."""


Issue = Literal["loading", "content", "requirement", "uncovered", "contrast"]
"""Analysis labels, not permissions for editing particular parts of a skill:
- loading: skill discovery or invocation relevant to the observed work;
- content: instructions, interfaces, or bundled resources relevant to the observed work;
- requirement: an explicit task or user requirement;
- uncovered: a clear need no skill covers;
- contrast: runs of the same kind of task taking different approaches with different results."""


class Finding(_AgentFile):
    """One located fact; its quote must occur verbatim in the cited input file."""

    id: str = Field(pattern=FINDING_ID)
    """Id unique within the collection."""

    path: InputPath
    """Input-relative path of the cited file, under trajectories/ or skills/."""

    quote: str = Field(min_length=1)
    """Text copied from the cited file; the escaped form of a JSON string is also accepted."""

    value: JsonValue = None
    """Value the script extracted for this fact, e.g. a command, count or category."""

    note: str = ""
    """Observed relationship, what it supports and any limits or unknown conditions."""

    issue: Issue | None = None
    """What the finding calls for; None for context that needs no answer."""

    skill: str | None = None
    """Skill the finding concerns; None only for uncovered needs and context."""

    case: str | None = Field(default=None, pattern=FINDING_ID)
    """Optional id grouping related comparison quotes; no outcome or causal claim is inferred."""

    @model_validator(mode="after")
    def _consistent(self) -> "Finding":
        """Uncovered needs concern no existing skill; other issue labels name the skill concerned."""
        if self.issue == "uncovered" and self.skill is not None:
            raise ValueError("an uncovered finding concerns no existing skill; set skill to null")
        if self.issue not in (None, "uncovered") and not self.skill:
            raise ValueError(f"a {self.issue} finding needs the skill it concerns")
        return self


class ScriptOutput(_AgentFile):
    """Unified JSON the collector writes to its --out file."""

    processed: list[InputPath]
    """Input-relative trajectory paths the script parsed."""

    failed: list[FailedInput] = Field(default_factory=list)
    """Trajectory files the script could not parse."""

    summary: dict[str, JsonValue] = Field(default_factory=dict)
    """Optional author-supplied statistics, always displayed as unverified."""

    findings: list[JsonValue] = Field(default_factory=list)
    """Located observations, validated one by one as `Finding` so a malformed entry
    only invalidates itself."""

    notes: str | list[str] = ""
    """Free text from the script's author; shown to the change stage as unverified."""


class SkillQuote(_AgentFile):
    """A passage of the original skills that written content comes from."""

    path: str = Field(min_length=1)
    """File relative to the skills directory."""

    quote: str = Field(min_length=1)
    """Text copied verbatim from that file."""


class Change(_AgentFile):
    """One skill modification claimed in changes.json."""

    id: str = Field(pattern=IDENTIFIER)
    """Change id unique within changes.json."""

    files: list[str] = Field(min_length=1)
    """Paths relative to the skills directory that this change adds, edits or deletes."""

    findings: list[str] = Field(min_length=1)
    """Accepted collection finding ids supporting the change; their skill attribution bounds existing-skill edits."""

    skill_quotes: list[SkillQuote] = Field(default_factory=list)
    """Passages of the original skills that the written content comes from."""

    rationale: str = Field(min_length=1)
    """Why the change addresses the cited findings."""


class Dropped(_AgentFile):
    """A problem the change agent examined and decided not to address."""

    findings: list[str] = Field(default_factory=list)
    """Finding ids examined."""

    reason: str = Field(min_length=1)
    """Why no change was made."""


class Acknowledgement(_AgentFile):
    """The change agent's answer to a gate warning it decided not to resolve."""

    warning: str = Field(min_length=1)
    """Key of the warning, as given by the gate."""

    reason: str = Field(min_length=1)
    """Why the edit is right as it is."""


class ChangeSet(_AgentFile):
    """Content of changes.json."""

    changes: list[Change] = Field(default_factory=list)
    """Claimed modifications; an empty list means no update."""

    dropped: list[Dropped] = Field(default_factory=list)
    """Examined problems left unchanged."""

    acknowledged: list[Acknowledgement] = Field(default_factory=list)
    """Gate warnings kept on purpose, with reasons."""


class FindingRef(BaseModel):
    """What the change gate needs to know about an accepted finding."""

    path: str
    """Input file the finding cites."""

    quote: str
    """Verified input quote, available for inspecting the change's stated basis."""

    issue: Issue | None = None
    """What the finding calls for."""

    skill: str | None = None
    """Skill the finding concerns."""

    case: str | None = None
    """Optional comparison group used to account for related findings together."""


class InvalidFinding(BaseModel):
    """A finding that failed verification."""

    id: str
    """Finding id as written by the script."""

    path: str
    """Cited path."""

    problem: str
    """Why the finding was rejected."""

    quote: str
    """Beginning of the cited quote."""


class EvidenceReport(BaseModel):
    """Structural verification of one unified collection; it does not grade task success."""

    attempt: int
    """Attempt number, starting at 1."""

    trajectory_files: int
    """Number of trajectory input files."""

    status: Literal["accepted", "rejected"] = "rejected"
    """Whether the collection's coverage, finding structure and quotes pass verification."""

    errors: list[str] = Field(default_factory=list)
    """Collection, coverage or finding-verification errors."""

    processed: int = 0
    """Number of trajectory files reported as processed."""

    failed: list[FailedInput] = Field(default_factory=list)
    """Inputs reported as unparsed, with reasons."""

    findings: list[Finding] = Field(default_factory=list)
    """Located findings with verified quotes and known skill attributions."""

    invalid_findings: list[InvalidFinding] = Field(default_factory=list)
    """Malformed or untraceable findings excluded from acceptance."""

    renamed_ids: int = 0
    """Repeated finding ids given a unique suffix."""

    summary: dict[str, JsonValue] = Field(default_factory=dict)
    """Optional statistics from the collector; not independently verified."""

    notes: str | list[str] = ""
    """Author explanations, not independently verified."""

    def accepted_findings(self) -> dict[str, FindingRef]:
        """Freeze the citable facts only when the collection passes the gate."""
        if self.status != "accepted":
            return {}
        return {
            finding.id: FindingRef(**finding.model_dump(include=set(FindingRef.model_fields)))
            for finding in self.findings
        }


class PreflightContext(BaseModel):
    """Public gate inputs for preview, carrying no Agent credentials or private run state."""

    model_config = ConfigDict(extra="forbid")
    """Require the current preview contract and reject undeclared data."""

    gate: EvidenceGateConfig
    """Collector execution settings without any model or connection fields."""

    options: EvolutionOptions
    """Directory-name mutation permissions used by both preview and formal gates."""

    inputs_digest: str
    """Expected immutable input snapshot identity."""

    accepted_findings: dict[str, FindingRef] = Field(default_factory=dict)
    """Already-public accepted findings; formal acceptance always uses private run state."""


class ChangeDecision(Change):
    """A claimed change with the gate's decision."""

    status: Literal["accepted", "rejected"]
    """accepted: the change is part of the candidate skills."""

    reasons: list[str] = Field(default_factory=list)
    """Why the change was rejected; empty when accepted."""


WarningKind = Literal[
    "description_changed",
    "large_rewrite",
    "orphaned_resource",
    "copied_from_trajectories",
    "version_condition",
    "long_skill",
]
"""What a change-gate warning is about."""


class GateWarning(BaseModel):
    """An accepted edit that needs attention; warnings never revert edits."""

    key: str
    """Stable identity `<kind>:<file>:<subject>`, used to acknowledge the warning."""

    kind: WarningKind
    """Category of the warning."""

    file: str
    """Skill-relative file the warning concerns."""

    message: str
    """Explanation shown to the change agent and in the gate report."""

    acknowledged: str | None = None
    """The change agent's reason for keeping the edit; None when unanswered."""

    @property
    def actionable(self) -> bool:
        """Whether the change agent is asked to revise or acknowledge it; description edits only inform."""
        return self.kind != "description_changed"


class ChangeReport(BaseModel):
    """Result of the change gate."""

    changes: list[ChangeDecision] = Field(default_factory=list)
    """Decision per claimed change."""

    dropped: list[Dropped] = Field(default_factory=list)
    """Examined problems the agent left unchanged."""

    reverted_unclaimed: list[str] = Field(default_factory=list)
    """Changed files that no change claimed; they were reverted."""

    skill_problems: dict[str, list[str]] = Field(default_factory=dict)
    """Skill directory name to format problems that caused its changes to revert."""

    accepted_files: list[str] = Field(default_factory=list)
    """Skill-relative files in the candidate that differ from the baseline."""

    warnings: list[GateWarning] = Field(default_factory=list)
    """Accepted edits that need attention."""

    unaddressed_findings: list[str] = Field(default_factory=list)
    """Accepted findings with an issue that no change cites, directly or through their case, and
    `dropped` does not list."""


class SessionRecord(BaseModel):
    """Agent session bookkeeping of one stage."""

    session_id: str | None = None
    """Latest session id, used to resume after interruption or for gate feedback."""

    sessions: int = 0
    """Number of invocations, also numbering the log files."""

    tokens: TokenUsage = Field(default_factory=TokenUsage)
    """Tokens consumed by the stage's sessions."""

    error: str | None = None
    """Error of the last invocation; None after a successful one."""

    pending_feedback: str | None = None
    """Gate feedback saved before invoking the session; cleared only after the session succeeds.
    An interrupted feedback session resumes before another gate attempt, without spending another retry."""

    skipped: str | None = None
    """Why the stage did not run an agent; None when it ran."""


class EvidenceGateRecord(BaseModel):
    """Summary of the evidence gate."""

    attempts: int = 0
    """Gate attempts so far."""

    accepted: bool | None = None
    """Whether the latest collection passed structural verification; None before the first gate."""

    errors: list[str] = Field(default_factory=list)
    """File-level errors in the latest attempt."""

    accepted_findings: dict[str, FindingRef] = Field(default_factory=dict)
    """Accepted findings in the latest attempt, by collection finding id. The change gate
    reads them from here rather than from evidence/report.json, which agents can reach."""


class ChangeGateRecord(BaseModel):
    """Summary of the change gate."""

    accepted: list[str] = Field(default_factory=list)
    """Accepted change ids."""

    rejected: list[str] = Field(default_factory=list)
    """Rejected change ids."""

    accepted_files: list[str] = Field(default_factory=list)
    """Candidate files that differ from the baseline."""

    warnings: list[GateWarning] = Field(default_factory=list)
    """Warnings of the latest gate run."""

    unaddressed_findings: list[str] = Field(default_factory=list)
    """Accepted findings left neither cited nor dropped in the latest gate run."""

    feedback_rounds: int = 0
    """Times the change session was resumed with an invalid changes.json, rejected changes, unclaimed
    files, unaddressed findings or unanswered warnings."""

    skipped: str | None = None
    """Why the gate did not run; None when it ran."""

    changes_error: str | None = None
    """Last reason changes.json could not be read; None when it was valid."""


class RunState(BaseModel):
    """Persistent state of one run, stored as state.json."""

    config: OfflineEvolutionConfig
    """Settings fixed when the run was created."""

    options: EvolutionOptions = Field(default_factory=EvolutionOptions)
    """Input directory names permitted to change and independent creation permission."""

    inputs_digest: str
    """Digest of inputs/ at creation; checked around sessions and at stage boundaries, including resume."""

    completed: list[StageName] = Field(default_factory=list)
    """Stages finished, in order."""

    evidence: SessionRecord = Field(default_factory=SessionRecord)
    """Evidence agent sessions."""

    evidence_gate: EvidenceGateRecord = Field(default_factory=EvidenceGateRecord)
    """Evidence gate summary."""

    change: SessionRecord = Field(default_factory=SessionRecord)
    """Change agent sessions."""

    change_gate: ChangeGateRecord = Field(default_factory=ChangeGateRecord)
    """Change gate summary."""

    outcome: Literal["updated", "unchanged"] | None = None
    """Final outcome; None until the publish stage completes."""


class EvolutionResult(BaseModel):
    """What a completed run returns to its caller."""

    run_dir: Path
    """Run directory holding inputs, reports, logs and the output skills."""

    outcome: Literal["updated", "unchanged"]
    """updated: gated changes were applied; unchanged: the output equals the input skills."""

    skills: Path
    """Skills to use next: `<run>/skills`, also written when unchanged so chained runs use one path."""

    state: RunState
    """Full run state for inspection."""


class RunLayout(BaseModel):
    """Paths of one run; inputs are copied once and are read-only afterwards."""

    model_config = ConfigDict(frozen=True)
    """Layouts are immutable path maps."""

    root: Path
    """Run directory, stored as an absolute path."""

    @field_validator("root")
    @classmethod
    def _absolute(cls, root: Path) -> Path:
        """Resolve the run directory: scripts and sessions run in other working directories."""
        return root.resolve()

    @property
    def host(self) -> Path:
        """Host-only: state, logs and the Claude configuration; denied to every agent session."""
        return self.root / HOST_DIR

    @property
    def state_path(self) -> Path:
        """Host-owned atomic run-state file."""
        return self.host / "state.json"

    @property
    def inputs(self) -> Path:
        """Frozen readable source and trajectory root."""
        return self.root / "inputs"

    @property
    def input_skills(self) -> Path:
        """Complete frozen input Skill collection."""
        return self.inputs / "skills"

    @property
    def input_trajectories(self) -> Path:
        """Complete frozen trajectory files."""
        return self.inputs / "trajectories"

    @property
    def evidence_workspace(self) -> Path:
        """Writable by the evidence agent."""
        return self.root / "workspace" / "evidence"

    @property
    def change_workspace(self) -> Path:
        """Writable by the change agent."""
        return self.root / "workspace" / "change"

    @property
    def edited_skills(self) -> Path:
        """Agent-writable candidate Skill copies."""
        return self.change_workspace / "skills"

    @property
    def evidence(self) -> Path:
        """Host-owned: frozen scripts, script outputs and the evidence report."""
        return self.root / "evidence"

    @property
    def candidate(self) -> Path:
        """Host-owned: gated skills and the change report."""
        return self.root / "candidate"

    @property
    def candidate_skills(self) -> Path:
        """Host-assembled Skills surviving the change gate."""
        return self.candidate / "skills"

    @property
    def output_skills(self) -> Path:
        """Host-owned: the skills to use next, written when the run completes."""
        return self.root / "skills"

    @property
    def logs(self) -> Path:
        """Host-owned native event logs for recovery."""
        return self.host / "logs"

    @property
    def preflight_path(self) -> Path:
        """Readable preview context, kept outside the private Host directory."""
        return self.root / "preflight.json"

    def snapshot(self, skills: Path, trajectories: Path) -> None:
        """Copy the skill collection and the trajectories into inputs/ unchanged.

        Args:
            skills: Directory of skill directories, or one skill directory containing SKILL.md.
            trajectories: One trajectory file or a directory of trajectories in any format.

        Raises:
            ValueError: Inputs are missing, contain no skill or trajectory, or inputs/ already exists.
        """
        if self.inputs.exists():
            raise ValueError(f"{self.inputs} already exists")
        if not skills.is_dir():
            raise ValueError(f"skills must be a directory: {skills}")
        if not trajectories.exists():
            raise ValueError(f"trajectories not found: {trajectories}")
        if (skills / "SKILL.md").is_file():
            shutil.copytree(skills, self.input_skills / skills.name)
        elif any((child / "SKILL.md").is_file() for child in skills.iterdir()):
            shutil.copytree(skills, self.input_skills)
        else:
            raise ValueError(f"no skill directory with SKILL.md under {skills}")
        if trajectories.is_file():
            self.input_trajectories.mkdir(parents=True)
            shutil.copy2(trajectories, self.input_trajectories / trajectories.name)
        else:
            shutil.copytree(trajectories, self.input_trajectories)
        if not self.trajectory_files():
            raise ValueError(f"no trajectory files under {trajectories}")

    def trajectory_files(self) -> list[str]:
        """Input-relative posix paths of every trajectory file."""
        return sorted(
            path.relative_to(self.inputs).as_posix() for path in self.input_trajectories.rglob("*") if path.is_file()
        )

    def inputs_digest(self) -> str:
        """Digest of every input path and byte; checked around sessions and stage transitions.

        Agents' shell access is not sandboxed, so a session could alter inputs despite
        its write scope; the digest makes such a run fail instead of silently continuing.
        """
        digest = hashlib.sha256()
        for path in sorted(p for p in self.inputs.rglob("*") if p.is_file()):
            digest.update(path.relative_to(self.inputs).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()


class HistoricalTrajectory(SkillTrajectory):
    """Carry original persisted JSON events without requiring message dictionaries."""

    trajectory: list[JsonValue]
    """Recorded events in their original order, including scalar and list values."""


class EvolutionInput(BaseModel):
    """Validated in-memory inputs for one algorithm execution."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    """Freeze call bindings while accepting runtime Skills as detached inputs."""

    skills: dict[str, Skill]
    """All readable input IDs and their complete source Skills."""

    trajectories: list[SerializeAsAny[SkillTrajectory]]
    """Detached historical records selected before algorithm execution."""

    work_dir: Path
    """Implementation-owned directory; the common boundary creates no files."""

    options: EvolutionOptions = Field(default_factory=EvolutionOptions)
    """Mutation permissions independent of the readable input collection."""

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        """Require a nonempty input collection and a valid editable subset."""
        if not self.skills or any(not key.strip() or key != key.strip() for key in self.skills):
            raise ValueError("Evolution requires nonempty, stable input Skill IDs")
        if not self.trajectories:
            raise ValueError("Evolution requires at least one trajectory")
        self.options.allowed_ids(self.skills)
        return self


class SkillEvolutionResult(BaseModel):
    """Complete input candidates and new Skills, without registered creation identities."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    """Reject undeclared fields and preserve concrete runtime Skills."""

    skills: dict[str, Skill]
    """Surviving input IDs and complete candidates, including unchanged members."""

    created_skills: dict[str, Skill] = Field(default_factory=dict)
    """Stable run-local creation keys and complete new Skill candidates."""

    def validate_against(self, inputs: dict[str, SkillSnapshot], options: EvolutionOptions) -> None:
        """Check complete coverage, immutable members, and creation permissions.

        Args:
            inputs: Original snapshots captured before algorithm execution.
            options: Validated permissions of the same call.

        Raises:
            ValueError: Coverage, identity, or mutation permissions are violated.
        """
        if self.skills.keys() != inputs.keys():
            raise ValueError("Evolution must retain every input Skill")
        if self.created_skills.keys() & inputs.keys():
            raise ValueError("Creation keys must not replace input identities")
        if any(not key.strip() for key in self.created_skills):
            raise ValueError("Creation keys must be nonempty")
        if self.created_skills and not options.allow_create_skills:
            raise ValueError("This evolution call does not allow new Skills")
        allowed = options.allowed_ids(inputs)
        for key, original in inputs.items():
            if key not in allowed:
                if self.skills[key].snapshot().model_dump() != original.model_dump():
                    raise ValueError(f"Evolution modified read-only Skill {key!r}")

    def snapshot(self) -> "EvolutionSnapshot":
        """Capture portable candidates for recovery and publication.

        Returns:
            Validated snapshots, local creation keys, and deletion claims.
        """
        return EvolutionSnapshot(
            skills={key: value.snapshot() for key, value in self.skills.items()},
            created_skills={key: value.snapshot() for key, value in self.created_skills.items()},
        )


class EvolutionSnapshot(BaseModel):
    """Portable candidate set preserved before any repository mutation."""

    model_config = ConfigDict(extra="forbid")
    """Reject missing or ambiguous candidate fields."""

    skills: dict[str, SkillSnapshot]
    """Complete surviving candidates keyed by original input identity."""

    created_skills: dict[str, SkillSnapshot] = Field(default_factory=dict)
    """Complete new candidates keyed by stable run-local identity."""
