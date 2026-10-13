## 1. Official Dataset

Evaluation source: [lucadiliello/searchqa](https://huggingface.co/datasets/lucadiliello/searchqa),
pinned to revision `c1a979068ba118d85467179b704031d113d689cc`.

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name searchqa
```

The source Parquet files are stored under `data/mindmemos_skill/searchqa/data/`.
The download receipt records the pinned source and verifies the local files.

## 2. Split File Format and Provided Paths

Use the repository-provided Skill-Forest split file:

```text
resources/mindmemos_skill_eval/skill_forest-searchqa/data_splits/splits.json
```

The split file is a UTF-8 JSON object mapping nonblank split names to ordered
lists of native HF `key` values as task IDs. For example:

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
from uuid import uuid4

from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data

root = Path.cwd()

import json
import pyarrow.parquet as parquet

split_file = root / "resources/mindmemos_skill_eval/skill_forest-searchqa/data_splits/splits.json"
manifest = json.loads(split_file.read_text())
wanted = {identifier for ids in manifest.values() for identifier in ids}
records = []
for path in sorted((root / "data/mindmemos_skill/searchqa/data").glob("*.parquet")):
    for batch in parquet.ParquetFile(path).iter_batches(columns=["key", "question", "context", "answers"]):
        for row in batch.to_pylist():
            if row["key"] in wanted:
                records.append({"id": row["key"], "question": row["question"],
                                "context": row["context"], "answers": row["answers"]})

prepared = root / "output/env_examples/searchqa/items.json"
prepared.parent.mkdir(parents=True, exist_ok=True)
prepared.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
data = Data.build("searchqa", {"data_path": prepared})
samples = data.load()
split_file = root / "resources/mindmemos_skill_eval/skill_forest-searchqa/data_splits/splits.json"
splits = data.split(samples, split_file=split_file)

agent = {
    "model": "gpt-5.4-mini",
    "api_key": "your-api-key",
    "base_url": "https://your-model-service.example/v1",
    "max_completion_tokens": 16384,
    "reasoning_effort": "medium",
}

env = Environment.build("searchqa", {
    "agent": agent,
    "max_turns": 1,
    "max_context_chars": 6000,
})

async def main():
    """Run one test task and print its rollout result."""
    sample = splits["test"][0]
    work_dir = root / "output/env_examples/searchqa" / uuid4().hex
    result = await env.rollout(
        task=sample.task,
        reward_data=sample.reward_data,
        work_dir=work_dir,
        resume=False,
    )
    print(result)


asyncio.run(main())
```
