## 1. Official Dataset

Official source: [SearchQA](https://github.com/nyu-dl/dl4ir-searchQA). Run the project download script
from the repository root with Python 3.12 or 3.13:

```bash
uv run python scripts/download_datasets.py download --benchmark skill --dataset searchqa --source-archive /absolute/path/to/SearchQA.zip
```

The default destination is `data/mindmemos_skill/searchqa/`. The script uses the sources and pinned versions in
`config/presets/dataset_download_sources.json`.

First obtain the official archive from the [upstream Google Drive release](https://drive.google.com/drive/u/2/folders/1kBkQGooNyG0h8waaOJpgdGtOnlb1S649).
Replace the archive argument with its local path; repeat `--source-archive` when
importing multiple archives in the same invocation. The script imports and
extracts archives; it does not convert raw crawler or TSV files. The example
below requires JSON records with `id`, `question`, `context`, and `answers`
(a list of strings), preserving the IDs in our split file. Raw releases lacking
these fields need conversion before this loader can use them; the project does
not currently provide that conversion.

## 2. Split File Format and Provided Paths

Use the repository-provided Skill-Forest split file:

```text
resources/mindmemos_skill_eval/skill_forest-searchqa/data_splits/splits.json
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

# Locate the JSON release with the normalized fields required by this loader.
required = {"id", "question", "context", "answers"}
records = {}
for path in sorted((root / "data/mindmemos_skill/searchqa").rglob("*.json")):
    if path.name == "download_receipt.json":
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload] if "id" in payload else payload.get("data", list(payload.values()))
    if not isinstance(payload, list):
        continue
    for row in payload:
        if isinstance(row, dict) and required <= row.keys():
            identifier = str(row["id"])
            if identifier in records and records[identifier] != row:
                raise ValueError(f"Conflicting SearchQA records for {identifier}")
            records[identifier] = row
if not records:
    raise ValueError("The archive must contain JSON records with id, question, context, and answers")

# Keep generated loader input separate from the verified download payload.
prepared = root / "output/env_examples/searchqa/items.json"
prepared.parent.mkdir(parents=True, exist_ok=True)
prepared.write_text(json.dumps(list(records.values())), encoding="utf-8")
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
