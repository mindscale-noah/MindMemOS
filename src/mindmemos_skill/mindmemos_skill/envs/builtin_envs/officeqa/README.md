## 1. Official Dataset

Official source: [OfficeQA](https://huggingface.co/datasets/databricks/officeqa). Run the project download script
from the repository root with Python 3.12 or 3.13:

```bash
uv run python scripts/download_datasets.py download skill --name officeqa
```

The Hugging Face downloads require `huggingface_hub` in the Python environment.
The command downloads the pinned release to `data/mindmemos_skill/officeqa/` by default.

## 2. Split File Format and Provided Paths

Use the repository-provided Skill-Forest split file:

```text
resources/mindmemos_skill_eval/skill_forest-officeqa/data_splits/splits.json
```

The split file is a UTF-8 JSON object mapping nonblank split names to ordered
lists of string task IDs. For example:

```json
{"train": ["task-1"], "val": ["task-2"], "test": ["task-3"]}
```

Every ID must exist in the loaded data and must not repeat within or across
splits. Empty splits and partial selection are allowed; unlisted tasks are
excluded. The order within each split is preserved. The example IDs above
illustrate the format; use the provided file with the official dataset.

OfficeQA uses the official CSV `uid` values (or `id` for JSON records).

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

from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data

root = Path.cwd()

data = Data.build("officeqa", {
    "data_path": root / "data/mindmemos_skill/officeqa/officeqa_full.csv",
    "docs_root": root / "data/mindmemos_skill/officeqa/treasury_bulletins_parsed",
})
samples = data.load()
split_file = root / "resources/mindmemos_skill_eval/skill_forest-officeqa/data_splits/splits.json"
splits = data.split(samples, split_file=split_file)

agent = {
    "model": "gpt-5.4-mini",
    "api_key": "your-api-key",
    "base_url": "https://your-model-service.example/v1",
    "max_completion_tokens": 16384,
    "reasoning_effort": "medium",
}

env = Environment.build("officeqa", {
    "agent": agent,
    "max_tool_turns": 24,
    "max_tool_chars": 16000,
})

async def main():
    """Run one test task and print its rollout result."""
    sample = splits["test"][0]
    work_dir = root / "output/env_examples/officeqa" / generate_id()
    result = await env.rollout(
        task=sample.task,
        reward_data=sample.reward_data,
        work_dir=work_dir,
        resume=False,
    )
    print(result)


asyncio.run(main())
```
