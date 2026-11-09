## 1. Official Dataset

Official source: [ALFWorld](https://github.com/alfworld/alfworld#download-data). Run the project download script
from the repository root with Python 3.12 or 3.13:

```bash
uv run python scripts/download_datasets.py download skill --name alfworld
```

The default destination is `data/mindmemos_skill/alfworld/`. The script uses the sources and pinned versions in
`config/presets/dataset_download_sources.json`.
The download includes the pinned ALFWorld 0.4.2 simulator configuration at
`configs/base_config.yaml`; environment initialization reads this local file.
If the dataset was downloaded before this file was included, use a new
`--data-root` to download the complete payload. The script verifies existing
directories rather than updating them in place.

## 2. Split File Format and Provided Paths

Use the repository-provided Skill-Forest split file:

```text
resources/mindmemos_skill_eval/skill_forest-alfworld/data_splits/splits.json
```

The companion `games.json` in the same directory maps each provided ID to
its official relative `gamefile` path. These IDs are project manifest IDs;
do not assign new IDs by enumerating downloaded games. The mapping preserves
the published Skill-Forest path manifest from
[SkillOpt](https://github.com/microsoft/SkillOpt/tree/main/data/alfworld_path_split).

The split file is a UTF-8 JSON object mapping nonblank split names to ordered
lists of string task IDs. For example:

```json
{"train": ["task-1"], "val": ["task-2"], "test": ["task-3"]}
```

Every ID must exist in the loaded data and must not repeat within or across
splits. Empty splits and partial selection are allowed; unlisted tasks are
excluded. The order within each split is preserved. The example IDs above
illustrate the format; use the provided file with the official dataset.

## 3. Environment Initialization and Rollout

Run the following complete example from the repository root after downloading
the data. Use a Python environment in which `mindmemos_skill` is installed.
The example uses the Skill-Forest paper target model, GPT-5.4-mini, with
medium reasoning and a 16384-token completion budget. Set `api_key` and
`base_url` for your provider; use its equivalent deployment name if needed.
These settings are passed explicitly rather than inherited from generic defaults. This example loads data, applies the
provided split file, runs one no-skill test task, and prints the result.
Each execution uses a new work directory and makes real model calls.

```python
import asyncio
from pathlib import Path
from mindmemos_skill.common.ids import generate_id

import yaml

from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data

root = Path.cwd()

dataset_root = root / "data/mindmemos_skill/alfworld"
data = Data.build("alfworld_bounded_history", {
    "data_path": root / "resources/mindmemos_skill_eval/skill_forest-alfworld/data_splits/games.json",
    "game_root": dataset_root,
})
samples = data.load()
split_file = root / "resources/mindmemos_skill_eval/skill_forest-alfworld/data_splits/splits.json"
splits = data.split(samples, split_file=split_file)

agent = {
    "model": "gpt-5.4-mini",
    "api_key": "your-api-key",
    "base_url": "https://your-model-service.example/v1",
    "max_completion_tokens": 16384,
    "reasoning_effort": "medium",
}

# Read the complete pinned simulator configuration downloaded with the data.
config_path = dataset_root / "configs/base_config.yaml"
config_text = config_path.read_text(encoding="utf-8")
simulator = yaml.safe_load(config_text.replace("$ALFWORLD_DATA", dataset_root.as_posix()))
simulator["general"]["use_cuda"] = False

env = Environment.build("alfworld_bounded_history", {
    "agent": agent,
    "simulator": simulator,
    "seed": 42,
    "max_steps": 50,
    "history_length": 2,
})

async def main():
    """Run one test task and print its rollout result."""
    sample = splits["test"][0]
    work_dir = root / "output/env_examples/alfworld_bounded_history" / generate_id()
    result = await env.rollout(
        task=sample.task,
        reward_data=sample.reward_data,
        work_dir=work_dir,
        resume=False,
    )
    print(result)


asyncio.run(main())
```
