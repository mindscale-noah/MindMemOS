## 1. Official Dataset

Official source: [SpreadsheetBench Verified 400](https://huggingface.co/datasets/KAKA22/SpreadsheetBench). Run the project download script
from the repository root with Python 3.12 or 3.13:

```bash
uv run python scripts/download_datasets.py download skill --name spreadsheetbench
```

The Hugging Face downloads require `huggingface_hub` in the Python environment.
The default destination is `data/mindmemos_skill/spreadsheetbench/`. The script uses the sources and pinned versions in
`config/presets/dataset_download_sources.json`.

## 2. Split File Format and Provided Paths

Use the repository-provided Skill-Forest split file:

```text
resources/mindmemos_skill_eval/skill_forest-spreadsheetbench/data_splits/splits.json
```

TreeSkill also provides a split file for the same Verified 400 dataset:

```text
resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json
```

The examples below select the matching split and environment explicitly.

The split file is a UTF-8 JSON object mapping nonblank split names to ordered
lists of string task IDs. For example:

```json
{"train": ["task-1"], "val": ["task-2"], "test": ["task-3"]}
```

Every ID must exist in the loaded data and must not repeat within or across
splits. Empty splits and partial selection are allowed; unlisted tasks are
excluded. The order within each split is preserved. The example IDs above
illustrate the format; use the provided file with the official dataset.

## 3. Initialize from the Official Dataset

### Native Tool Calling

```python
import asyncio
from pathlib import Path
from uuid import uuid4

from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data

root = Path.cwd()

dataset_root = root / "data/mindmemos_skill/spreadsheetbench/spreadsheetbench_verified_400"
data = Data.build("spreadsheetbench", {
    "data_path": dataset_root / "dataset.json",
    "data_root": dataset_root,
})
samples = data.load()
split_file = root / "resources/mindmemos_skill_eval/skill_forest-spreadsheetbench/data_splits/splits.json"
splits = data.split(samples, split_file=split_file)

agent = {
    "model": "gpt-5.4-mini",
    "api_key": "your-api-key",
    "base_url": "https://your-model-service.example/v1",
    "max_completion_tokens": 16384,
    "reasoning_effort": "medium",
}

env = Environment.build("spreadsheetbench_tool_calling", {
    "agent": agent,
    "max_turns": 30,
    "skill_loading_mode": "tool",
    "stagnation_limit": None,
    "transactional_recalculation": False,
})

async def main():
    """Run one test task and print its rollout result."""
    sample = splits["test"][0]
    work_dir = root / "output/env_examples/spreadsheetbench" / uuid4().hex
    result = await env.rollout(
        task=sample.task,
        reward_data=sample.reward_data,
        work_dir=work_dir,
        resume=False,
    )
    print(result)


asyncio.run(main())
```

### Text Action

```python
import asyncio
from pathlib import Path
from uuid import uuid4

from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data

root = Path.cwd()

config = {
    "agent": {
        "model": "gpt-5.4-mini",
        "api_key": "your-api-key",
        "base_url": "https://your-model-service.example/v1",
        "max_completion_tokens": 16384,
        "reasoning_effort": "medium",
    },
    "transactional_recalculation": True,
    "max_turns": 100,
    "shell_timeout_seconds": 120,
    "stagnation_limit": 5,
}

dataset_root = root / "data/mindmemos_skill/spreadsheetbench/spreadsheetbench_verified_400"
data = Data.build("spreadsheetbench", {
    "data_path": dataset_root / "dataset.json",
    "data_root": dataset_root,
})
samples = data.load()
split_file = root / "resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json"
splits = data.split(samples, split_file=split_file)
env = Environment.build("spreadsheetbench_text_action", config)


async def main():
    """Run one text-action task using the configured LibreOffice runtime."""
    sample = splits["test"][0]
    work_dir = root / "output/env_examples/spreadsheetbench_text_action" / uuid4().hex
    result = await env.rollout(
        task=sample.task,
        reward_data=sample.reward_data,
        work_dir=work_dir,
        resume=False,
    )
    print(result)


asyncio.run(main())
```
