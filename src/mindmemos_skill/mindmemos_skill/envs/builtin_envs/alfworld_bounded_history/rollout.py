"""One reset/model/step loop with request-local simulator ownership."""

import re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from ....agents import OpenAIChatAgent
from ....skill import Skill
from ...base import Environment
from ...data import RewardData
from ...typing import RolloutResult, Task
from ..common import skill_text, total_usage
from .config import ALFWorldConfig
from .prompts import NO_HISTORY, WITH_HISTORY
from .reward import evaluate
from .typing import ALFWorldRewardData, ALFWorldTask


def _create_simulator(config: ALFWorldConfig, gamefile: Path):
    """Select one game before upstream initialization scans the dataset."""
    from alfworld.agents.environment import get_environment

    environment_class = get_environment(config.simulator["env"]["type"])

    class SelectedGame(environment_class):
        def collect_game_files(self, verbose=False):
            """Initialize only the explicitly selected game."""
            self.game_files = [str(gamefile)]
            self.num_games = 1

    split = "train"
    if "valid_seen" in gamefile.parts:
        split = "eval_in_distribution"
    elif "valid_unseen" in gamefile.parts:
        split = "eval_out_of_distribution"
    simulator = SelectedGame(deepcopy(config.simulator), train_eval=split).init_env(batch_size=1)
    try:
        simulator.seed(config.seed)
    except BaseException:
        simulator.close()
        raise
    return simulator


@Environment.register("alfworld_bounded_history")
class ALFWorldEnvironment(Environment[ALFWorldConfig]):
    """Run one game with bounded history and no batch scheduling barrier."""

    task_type = ALFWorldTask
    """Public ALFWorld task schema."""

    reward_data_type = ALFWorldRewardData
    """Private ALFWorld scoring schema."""

    config_type = ALFWorldConfig
    """Recorded model, simulator, and history settings."""

    def __init__(self, config: ALFWorldConfig, *, agent: OpenAIChatAgent | None = None) -> None:
        """Initialize the model; defer simulator creation to each rollout.

        Args:
            config: Audited runtime settings.
            agent: Optional model backend with matching settings.
        """
        super().__init__(config)
        if agent is not None and agent.config != config.agent:
            raise ValueError("Injected agent config must match the environment's agent config")
        self.agent = agent if agent is not None else OpenAIChatAgent(config.agent)

    def _run(self, *, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult:
        """Keep simulator state local and close it even if model execution fails."""
        if not isinstance(task, ALFWorldTask) or not isinstance(reward_data, ALFWorldRewardData):
            raise TypeError("ALFWorld requires ALFWorldTask and ALFWorldRewardData")
        for filename, digest in reward_data.game_sha256.items():
            if sha256((task.gamefile.parent / filename).read_bytes()).hexdigest() != digest:
                raise ValueError("ALFWorld game files changed; reload the dataset")
        text = skill_text(skills)
        history, responses, trajectory = [], [], []
        system = "You are an expert agent operating in the ALFRED Embodied Environment."
        simulator = _create_simulator(self.config, task.gamefile)
        done, won, output = False, False, ""
        try:
            observations, infos = simulator.reset()
            observation = observations[0]
            marker = "Your task is to: "
            if marker not in observation:
                raise ValueError("Task description not found in text observation.")
            description = observation.split(marker, 1)[1].strip()
            for step in range(self.config.max_steps):
                admissible = "\n ".join(f"'{action}'" for action in infos["admissible_commands"][0] if action != "help")
                values = {"current_observation": observation, "admissible_actions": admissible}
                if history and self.config.history_length:
                    recent = history[-self.config.history_length :]
                    values.update(
                        task_description=description,
                        step_count=len(history),
                        history_length=len(recent),
                        action_history="\n".join(recent),
                        current_step=len(history) + 1,
                    )
                    prompt = WITH_HISTORY.format(**values)
                else:
                    prompt = NO_HISTORY.format(**values)
                if text.strip():
                    prompt = (
                        "\n\n## Skill Knowledge\n"
                        "Below is a skill document with learned strategies. "
                        "Use these guidelines to inform your decisions:\n\n"
                        f"{text}\n\n{prompt}"
                    )
                (workspace / f"prompt_{step + 1}.md").write_text(prompt, encoding="utf-8")
                response = self.agent.run(system=system, user=prompt)
                if not isinstance(response.output, str):
                    raise TypeError("ALFWorld requires a text agent output")
                responses.append(response)
                output = response.output.strip()
                action_match = re.search(r"<action>(.*?)</action>", output, re.DOTALL)
                if not output:
                    output = "<think>empty model response</think><action>look</action>"
                elif action_match is None:
                    output = "<think>missing action tag</think><action>look</action>"
                action_match = re.search(r"<action>(.*?)</action>", output, re.DOTALL)
                action = action_match.group(1).strip().lower()
                reasoning = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
                history.append(f"[Observation {step + 1}: '{observation}', Action {step + 1}: '{action}']")
                observations, scores, dones, infos = simulator.step([action])
                observation, done = observations[0], bool(dones[0])
                if done:
                    won = bool(infos["won"][0])
                trajectory.append(
                    {
                        "step": step,
                        "action": action_match.group(1).strip(),
                        "reasoning": reasoning.group(1).strip() if reasoning else None,
                        "model_response": output,
                        "env_feedback": observation,
                        "reward": 10.0 * float(infos["won"][0]),
                        "done": done,
                    }
                )
                if done:
                    break
        finally:
            simulator.close()
        return RolloutResult(
            output=output,
            trajectory=trajectory,
            reward=evaluate(won=won, done=done, max_steps=self.config.max_steps).model_copy(
                update={
                    "task_description": description,
                    "task_type": next(
                        (
                            name
                            for name in (
                                "pick_and_place",
                                "pick_two_obj_and_place",
                                "look_at_obj_in_light",
                                "pick_heat_then_place_in_recep",
                                "pick_cool_then_place_in_recep",
                                "pick_clean_then_place_in_recep",
                            )
                            if name in str(task.gamefile)
                        ),
                        "other",
                    ),
                },
            ),
            usage=total_usage(responses),
        )
