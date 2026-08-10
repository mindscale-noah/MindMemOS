"""Single-game ALFWorld runtime preserving Skill-GRPO step semantics."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import re
import traceback
from pathlib import Path
from queue import Empty
from typing import Any


def project_action(model_response: str) -> tuple[str, int]:
    """Apply the original ``alfworld_projection`` rules to one response."""

    original = model_response
    action = model_response.lower()
    valid = 0
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = action.find(start_tag)
    end_idx = action.find(end_tag)
    if re.search(r"[\u4e00-\u9fff]", original):
        valid = 0
    try:
        if start_idx == -1 or end_idx == -1:
            return action[-30:], valid
        action = action[start_idx + len(start_tag) : end_idx].strip().lower()
        valid = 1
    except Exception:
        action = action[-30:]
    if original.find("<think>") == -1 or original.find("</think>") == -1:
        valid = 0
    return action, valid


def _patch_textworld_eval_symbol() -> None:
    from textworld.envs.pddl import textgen

    if getattr(textgen.EvalSymbol.derive, "_mindmemos_skill_py314_patch", False):
        return

    def derive(self, context=None):
        context = context or self.context
        value = eval(self.expression, {}, dict(context["variables"]))
        return [textgen.TerminalSymbol(value)]

    derive._mindmemos_skill_py314_patch = True
    textgen.EvalSymbol.derive = derive


def _load_config() -> dict[str, Any]:
    import yaml

    path = Path(__file__).with_name("config.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _worker_loop(command_queue, result_queue, seed: int, is_train: bool, eval_dataset: str, gamefile: str) -> None:
    try:
        from alfworld.agents.environment import get_environment

        _patch_textworld_eval_symbol()
        config = _load_config()
        env_cls = get_environment(config["env"]["type"])
        original_collect = getattr(env_cls, "collect_game_files", None)
        if original_collect is not None:

            def collect_single_game(self, verbose=False):
                del verbose
                self.game_files = [gamefile]
                self.num_games = 1

            env_cls.collect_game_files = collect_single_game
        try:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    base_env = env_cls(config, train_eval="train" if is_train else eval_dataset)
        finally:
            if original_collect is not None:
                env_cls.collect_game_files = original_collect
        base_env.game_files = [gamefile]
        if hasattr(base_env, "num_games"):
            base_env.num_games = 1
        env = base_env.init_env(batch_size=1)
        env.seed(seed)
        result_queue.put((True, "ready"))
    except BaseException:
        result_queue.put((False, traceback.format_exc()))
        return

    while True:
        command, payload = command_queue.get()
        if command == "close":
            close = getattr(env, "close", None)
            if callable(close):
                close()
            result_queue.put((True, None))
            return
        try:
            if command == "reset":
                result = env.reset()
            elif command == "step":
                result = env.step([payload])
            else:
                raise ValueError(f"Unknown ALFWorld worker command: {command}")
            result_queue.put((True, result))
        except BaseException:
            result_queue.put((False, traceback.format_exc()))


class ALFWorldSimulator:
    """Process-isolated simulator for one fixed ALFWorld gamefile."""

    def __init__(self, *, seed: int, is_train: bool, eval_dataset: str, gamefile: str) -> None:
        start_method = os.environ.get("ALFWORLD_WORKER_START_METHOD") or None
        context = mp.get_context(start_method) if start_method else mp.get_context()
        self._commands = context.Queue(maxsize=1)
        self._results = context.Queue(maxsize=1)
        self._process = context.Process(
            target=_worker_loop,
            args=(self._commands, self._results, seed, is_train, eval_dataset, gamefile),
        )
        self._process.start()
        try:
            ok, payload = self._results.get(timeout=60)
        except Empty as exc:
            self.close(kill=True)
            raise RuntimeError("Timed out starting ALFWorld worker") from exc
        if not ok:
            self.close(kill=True)
            raise RuntimeError(f"Failed to start ALFWorld worker:\n{payload}")
        self.admissible_actions: list[str] = []

    def reset(self) -> tuple[str, dict[str, Any]]:
        self._commands.put(("reset", None))
        observations, infos = self._receive()
        info = _unbatch_info(infos)
        self.admissible_actions = list(info["admissible_commands"])
        return str(observations[0]), info

    def step(self, model_response: str) -> tuple[str, float, bool, dict[str, Any]]:
        action, valid = project_action(model_response)
        self._commands.put(("step", action))
        observations, _scores, dones, infos = self._receive()
        info = _unbatch_info(infos)
        info["is_action_valid"] = valid
        self.admissible_actions = list(info["admissible_commands"])
        reward = 10.0 * float(info["won"])
        return str(observations[0]), reward, bool(dones[0]), info

    def _receive(self, timeout: float | None = None):
        ok, payload = self._results.get(timeout=timeout)
        if not ok:
            raise RuntimeError(f"ALFWorld worker failed:\n{payload}")
        return payload

    def close(self, *, kill: bool = False) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.is_alive() and not kill:
            try:
                self._commands.put(("close", None))
                self._receive(timeout=10)
            except Exception:
                kill = True
        if kill and process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        self._commands.close()
        self._results.close()
        self._process = None


def _unbatch_info(infos: dict[str, Any]) -> dict[str, Any]:
    return {key: value[0] for key, value in infos.items()}


__all__ = ["ALFWorldSimulator", "project_action"]
