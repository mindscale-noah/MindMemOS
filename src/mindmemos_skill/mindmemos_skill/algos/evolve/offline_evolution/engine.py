"""Offline skill evolution: evidence agent → evidence gate → change agent → change gate → publish to `<run>/skills`."""

import json
import shlex
import shutil
import sys
from pathlib import Path

from ....common.json import write_json
from ....logging import get_logger
from ....logging import stage as log_stage
from .agents import event_identity, run_agent, session_identity
from .agents.typing import SessionOptions
from .change_gate import open_items_feedback, run_change_gate
from .change_gate import rejection_feedback as change_rejection_feedback
from .change_warnings import warning_feedback
from .config import EvidenceGateConfig, EvolutionOptions, OfflineEvolutionConfig
from .evidence_gate import rejection_feedback, run_evidence_gate
from .layout import skill_names
from .prompts import lenses, render_prompt
from .typing import EvolutionResult, PreflightContext, RunLayout, RunState, SessionRecord, StageName, TokenUsage

STAGES: tuple[StageName, ...] = ("evidence", "evidence_gate", "change", "change_gate", "publish")
"""Stages in execution order."""


class OfflineEvolution:
    """Improve a skill collection from historical trajectories without rewards or task reruns.

    Each run lives in its own directory. Stages record completion in state.json, so a run
    interrupted by an agent failure continues where it stopped, resuming the failed session.
    """

    def __init__(self, config: OfflineEvolutionConfig, options: EvolutionOptions | None = None) -> None:
        """Bind settings and the agent backend.

        Args:
            config: Validated evolver settings.
            options: Directory-name mutation permissions; None permits all edits and creations.
        """
        self.config = config
        self.options = options if options is not None else EvolutionOptions()

    async def evolve(self, *, skills: Path, trajectories: Path, work_dir: Path) -> EvolutionResult:
        """Snapshot inputs or continue the same run from its committed stage boundary.

        Args:
            skills: Directory of skill directories, or one skill directory.
            trajectories: Trajectory file or directory in any format.
            work_dir: New empty directory or an existing run with matching inputs.

        Returns:
            Outcome and the skills to use next.

        Raises:
            ValueError: The directory is unrecognized, inputs are invalid, or permissions name unknown members.
            RuntimeError: An agent session failed; call `resume` to continue.
        """
        layout = RunLayout(root=work_dir)
        if layout.state_path.exists():
            return await self.resume(work_dir=work_dir)
        if work_dir.exists() and any(work_dir.iterdir()):
            raise ValueError(f"run directory {work_dir} is not empty")
        self.options.allowed_ids(skill_names(skills))
        work_dir.mkdir(parents=True, exist_ok=True)
        layout.snapshot(skills, trajectories)
        state = RunState(config=self.config, options=self.options, inputs_digest=layout.inputs_digest())
        write_json(layout.state_path, state.model_dump(mode="json", exclude={"config": {"agent": {"api_key"}}}))
        return await _Run(layout, state, self.config).execute()

    async def resume(self, *, work_dir: Path) -> EvolutionResult:
        """Continue an existing run from its first incomplete stage.

        Args:
            work_dir: Run directory created by `evolve`.

        Returns:
            Outcome and the skills to use next.

        Raises:
            RuntimeError: The run's inputs were modified after creation.
            RuntimeError: An agent session failed again.
        """
        layout = RunLayout(root=work_dir)
        state = read_state(work_dir)
        if state.options != self.options:
            raise ValueError("Offline evolution permissions changed")
        return await _Run(layout, state, self.config).execute()


def read_state(work_dir: Path) -> RunState:
    """Load the state of a run.

    Args:
        work_dir: Run directory.

    Returns:
        The stored run state, including the settings the run was created with.
    """
    return RunState.model_validate_json(RunLayout(root=work_dir).state_path.read_text(encoding="utf-8"))


class _Run:
    """Stage execution for one run directory."""

    def __init__(self, layout: RunLayout, state: RunState, config: OfflineEvolutionConfig) -> None:
        """Bind the run's paths, persisted state and agent backend."""
        self.layout = layout
        self.state = state
        self.config = state.config
        self.agent_config = config.agent

    async def execute(self) -> EvolutionResult:
        """Run every incomplete stage and return the outcome."""
        self._check_inputs()
        for stage in STAGES:
            if stage in self.state.completed:
                continue
            with log_stage(f"offline_evolution.{stage}"):
                await {
                    "evidence": self._evidence,
                    "evidence_gate": self._evidence_gate,
                    "change": self._change,
                    "change_gate": self._change_gate,
                    "publish": self._publish,
                }[stage]()
            self._check_inputs()
            get_logger(__name__).info(
                "offline_evolution.stage.completed",
                extra={
                    "event_body": {
                        "stage": stage,
                        "accepted_findings": len(self.state.evidence_gate.accepted_findings),
                        "accepted_files": len(self.state.change_gate.accepted_files),
                    }
                },
            )
            self.state.completed.append(stage)
            self._save()
        assert self.state.outcome is not None
        return EvolutionResult(
            run_dir=self.layout.root, outcome=self.state.outcome, skills=self.layout.output_skills, state=self.state
        )

    async def _evidence(self) -> None:
        """Ask the evidence agent for a focused collection of located comparisons and observations."""
        workspace = self.layout.evidence_workspace
        workspace.mkdir(parents=True, exist_ok=True)
        prompt = render_prompt(
            "evidence",
            inputs=self.layout.inputs,
            workspace=workspace,
            lenses=lenses("evidence"),
            check_command=self._check_command("evidence"),
        )
        await self._session(
            self.state.evidence,
            "evidence",
            prompt,
            read=(self.layout.inputs,),
            write=(workspace,),
            deny=(self.layout.host,),
        )

    async def _evidence_gate(self) -> None:
        """Verify the collection; return structural or provenance errors up to `retries` times."""
        record = self.state.evidence_gate
        session = self.state.evidence
        while True:
            if session.pending_feedback is not None:
                await self._feedback_session(session, "evidence", self.layout.evidence_workspace)
            record.attempts += 1
            report = await run_evidence_gate(self.layout, record.attempts, self.config)
            record.accepted = report.status == "accepted"
            record.errors = report.errors
            record.accepted_findings = dict(sorted(report.accepted_findings().items()))
            if not record.accepted and record.attempts <= self.config.retries:
                session.pending_feedback = rejection_feedback(report)
            self._save()
            if session.pending_feedback is None:
                return

    async def _change(self) -> None:
        """Ask the change agent to edit a copy of the skills, unless no accepted finding calls for an edit."""
        if not any(ref.issue for ref in self.state.evidence_gate.accepted_findings.values()):
            self.state.change.skipped = "no accepted finding calls for an edit"
            return
        if (
            not self.state.options.allowed_ids(skill_names(self.layout.input_skills))
            and not self.state.options.allow_create_skills
        ):
            self.state.change.skipped = "all inputs are read-only and creation is forbidden"
            return
        if not self.layout.edited_skills.exists():
            shutil.copytree(self.layout.input_skills, self.layout.edited_skills)
        prompt = render_prompt(
            "change",
            report_md=self.layout.evidence / "report.md",
            report_json=self.layout.evidence / "report.json",
            inputs=self.layout.inputs,
            edited=self.layout.edited_skills,
            workspace=self.layout.change_workspace,
            protected=self._protected_note(),
            lenses=lenses("change"),
            check_command=self._check_command("change"),
        )
        await self._session(
            self.state.change,
            "change",
            prompt,
            read=(self.layout.inputs, self.layout.evidence, self.layout.change_workspace),
            write=self._change_write_scopes(),
            deny=(self.layout.host,),
            readonly=self._protected_dirs(),
        )

    async def _change_gate(self) -> None:
        """Gate changes.json and return what needs an answer to the session: an invalid file, rejected
        changes, unclaimed files, unaddressed findings or unanswered warnings.

        All share the `retries` feedback budget. When it is spent, the last gate result stands: rejected
        changes and unclaimed files stay left out, and unaddressed findings and warnings stay recorded.
        """
        record = self.state.change_gate
        session = self.state.change
        if self.state.change.skipped:
            record.skipped = self.state.change.skipped
            return
        while True:
            if session.pending_feedback is not None:
                await self._feedback_session(
                    session, "change", self.layout.change_workspace, readonly=self._protected_dirs()
                )
            try:
                report = run_change_gate(
                    self.layout,
                    self.state.evidence_gate.accepted_findings,
                    set(self.state.options.allowed_ids(skill_names(self.layout.input_skills))),
                    self.state.options.allow_create_skills,
                )
            except ValueError as error:
                record.changes_error = str(error)
                record.accepted, record.rejected, record.accepted_files, record.warnings = [], [], [], []
                record.unaddressed_findings = []
                feedback = str(error)
            else:
                record.changes_error = None
                record.accepted = [c.id for c in report.changes if c.status == "accepted"]
                record.rejected = [c.id for c in report.changes if c.status == "rejected"]
                record.accepted_files = report.accepted_files
                record.warnings = report.warnings
                record.unaddressed_findings = report.unaddressed_findings
                parts = (
                    change_rejection_feedback(report),
                    open_items_feedback(report),
                    warning_feedback(report.warnings),
                )
                feedback = "\n\n".join(part for part in parts if part)
            if feedback and record.feedback_rounds < self.config.retries:
                record.feedback_rounds += 1
                session.pending_feedback = feedback
            elif record.changes_error:
                record.skipped = "changes.json invalid after retries"
            self._save()
            if session.pending_feedback is None:
                return

    async def _publish(self) -> None:
        """Write the gated candidate, or the unchanged input skills, to `<run>/skills`."""
        updated = bool(self.state.change_gate.accepted_files)
        output = self.layout.output_skills
        if output.exists():  # left by an interrupted publish
            shutil.rmtree(output)
        shutil.copytree(self.layout.candidate_skills if updated else self.layout.input_skills, output)
        self.state.outcome = "updated" if updated else "unchanged"

    async def _feedback_session(
        self, record: SessionRecord, name: str, workspace: Path, *, readonly: tuple[Path, ...] = ()
    ) -> None:
        """Finish saved gate feedback before rechecking the workspace, even after interruption."""
        assert record.pending_feedback is not None
        await self._session(
            record,
            name,
            render_prompt(
                "retry",
                feedback=record.pending_feedback,
                workspace=workspace,
                check_command=self._check_command(name),
            ),
            read=(self.layout.inputs, self.layout.evidence, workspace),
            write=self._change_write_scopes() if name == "change" else (workspace,),
            deny=(self.layout.host,),
            readonly=readonly,
            resume=True,
        )

    async def _session(
        self,
        record: SessionRecord,
        name: str,
        prompt: str,
        *,
        read: tuple[Path, ...],
        write: tuple[Path, ...],
        deny: tuple[Path, ...],
        readonly: tuple[Path, ...] = (),
        resume: bool = False,
    ) -> None:
        """Run one agent session, resuming the stage's previous session after feedback or a failure."""
        self._check_inputs()
        interrupted = record.error
        session_id = self._logged_session(name, record.sessions) or record.session_id
        if interrupted and not resume:
            prompt = (
                f"Your previous session stopped with: {interrupted}\n"
                f"Continue from where you stopped and finish.\n\n{prompt}"
            )
        record.sessions += 1
        self._save()
        self.layout.logs.mkdir(parents=True, exist_ok=True)
        write_json(
            self.layout.preflight_path,
            PreflightContext(
                gate=EvidenceGateConfig.model_validate(self.config.model_dump(exclude={"agent", "retries"})),
                options=self.state.options,
                inputs_digest=self.state.inputs_digest,
                accepted_findings=self.state.evidence_gate.accepted_findings,
            ),
        )
        request = SessionOptions(
            read_dirs=(*read, self.layout.preflight_path),
            write_dirs=write,
            deny_paths=deny,
            readonly_dirs=readonly,
            config_dir=self.layout.host / f".{self.agent_config.class_name}",
            session_id=session_id,
            log_path=self.layout.logs / f"{name}-{record.sessions}.jsonl",
        )
        try:
            result = await run_agent(self.agent_config, user=prompt, workspace=self.layout.root, session=request)
        except Exception as error:
            record.error = str(error)
            record.session_id = self._logged_session(name, record.sessions) or session_id
            self._save()
            raise
        record.error, record.session_id = None, session_identity(result)
        record.pending_feedback = None
        record.tokens += TokenUsage(
            input=result.usage.get("input_tokens") or 0,
            cache_write=result.usage.get("cache_creation_input_tokens") or 0,
            cache_read=result.usage.get("cache_read_input_tokens") or 0,
            output=result.usage.get("output_tokens") or 0,
        )
        self._save()
        self._check_inputs()

    def _check_inputs(self) -> None:
        """Stop before gating, starting another session or publishing altered input snapshots."""
        if self.layout.inputs_digest() != self.state.inputs_digest:
            raise RuntimeError("inputs/ changed since the run was created")

    def _check_command(self, stage: str) -> str:
        """Build the installed preflight command with the service interpreter."""
        return shlex.join(
            [
                sys.executable,
                "-m",
                "mindmemos_skill.algos.evolve.offline_evolution.check",
                stage,
                "--run",
                str(self.layout.root),
            ]
        )

    def _protected_dirs(self) -> tuple[Path, ...]:
        """Deny edits of input members outside the editable subset."""
        names = skill_names(self.layout.input_skills)
        allowed = self.state.options.allowed_ids(names)
        return tuple(self.layout.edited_skills / name for name in sorted(names - allowed))

    def _protected_note(self) -> str:
        """Describe exact mutation permissions for the change session."""
        names = skill_names(self.layout.input_skills)
        readonly = names - self.state.options.allowed_ids(names)
        protected = ", ".join(sorted(readonly)) or "(none)"
        creation = "allowed" if self.state.options.allow_create_skills else "forbidden"
        return (
            f"Read-only skills: {protected}. You may analyze and cite them but must not change or delete them. "
            f"Creating independent new skills is {creation}."
        )

    def _save(self) -> None:
        """Persist state without serializing the model credential into agent-accessible files."""
        write_json(
            self.layout.state_path,
            self.state.model_dump(mode="json", exclude={"config": {"agent": {"api_key"}}}),
        )

    def _logged_session(self, name: str, invocation: int) -> str | None:
        """Recover a streamed identity after process death before the stage state was committed."""
        path = self.layout.logs / f"{name}-{invocation}.jsonl"
        if not invocation or not path.is_file():
            return None
        identity = None
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and (observed := event_identity(event)):
                    identity = observed
        return identity

    def _change_write_scopes(self) -> tuple[Path, ...]:
        """Limit file writes to existing editable members when creation is disabled."""
        if self.state.options.allow_create_skills:
            return (self.layout.change_workspace,)
        changes = self.layout.change_workspace / "changes.json"
        if not changes.exists():
            write_json(changes, {"changes": []})
        names = self.state.options.allowed_ids(skill_names(self.layout.input_skills))
        return (changes, *(self.layout.edited_skills / name for name in sorted(names)))
