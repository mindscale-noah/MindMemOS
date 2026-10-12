"""Own synchronous upstream ALFWorld state in a dedicated subprocess."""

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


def create_simulator(config: dict, gamefile: Path, seed: int):
    """Initialize only the requested game using the supplied upstream settings."""
    from alfworld.agents.environment import get_environment

    environment_class = get_environment(config["env"]["type"])

    class SelectedGame(environment_class):
        def collect_game_files(self, verbose=False):
            """Restrict upstream discovery to this process's selected game."""
            self.game_files = [str(gamefile)]
            self.num_games = 1

    split = (
        "eval_in_distribution"
        if "valid_seen" in gamefile.parts
        else "eval_out_of_distribution"
        if "valid_unseen" in gamefile.parts
        else "train"
    )
    simulator = SelectedGame(config, train_eval=split).init_env(batch_size=1)
    try:
        simulator.seed(seed)
    except BaseException:
        simulator.close()
        raise
    return simulator


def main() -> None:
    """Serve explicit commands while keeping upstream output off the protocol pipe."""
    simulator = None
    try:
        for line in sys.stdin:
            try:
                command = json.loads(line)
                with redirect_stdout(sys.stderr):
                    operation = command["operation"]
                    if operation == "initialize" and simulator is None:
                        simulator = create_simulator(command["config"], Path(command["gamefile"]), command["seed"])
                        result = None
                    elif operation == "reset" and simulator is not None:
                        observations, infos = simulator.reset()
                        result = [
                            list(observations),
                            {"admissible_commands": [list(items) for items in infos["admissible_commands"]]},
                        ]
                    elif operation == "step" and simulator is not None:
                        observations, scores, dones, infos = simulator.step(command["actions"])
                        result = [
                            list(observations),
                            [float(score) for score in scores],
                            [bool(done) for done in dones],
                            {
                                "admissible_commands": [list(items) for items in infos["admissible_commands"]],
                                "won": [bool(won) for won in infos["won"]],
                            },
                        ]
                    else:
                        raise ValueError("Invalid simulator operation")
                encoded = json.dumps({"result": result})
            except Exception as exc:
                encoded = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            print(encoded, flush=True)
    finally:
        if simulator is not None:
            with redirect_stdout(sys.stderr):
                simulator.close()


if __name__ == "__main__":
    main()
