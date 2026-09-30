## 1. Official Dataset

Download [SpreadsheetBench Verified 400](https://huggingface.co/datasets/KAKA22/SpreadsheetBench/blob/main/spreadsheetbench_verified_400.tar.gz).

## 2. Split File Format

Provide a UTF-8 JSON object mapping split names to ordered source task IDs:

```json
{"train": ["task-1"], "val": ["task-2"], "test": ["task-3"]}
```

Use the dataset's `id` values as strings (OfficeQA also accepts `uid`). Every ID
must exist in the loaded dataset. IDs must not repeat across or within splits.
Empty splits and partial dataset selection are allowed.

## 3. Environment Initialization

```python
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data

data = Data.build("spreadsheetbench", {"data_path": "/path/to/spreadsheetbench_verified_400/dataset.json", "data_root": "/path/to/spreadsheetbench_verified_400"})
samples = data.load()
splits = data.split(samples, split_file="/path/to/splits.json")
env = Environment.build("spreadsheetbench", {"agent": {"model": "your-model", "max_completion_tokens": 16384}, "tool_agent": "openai_tool", "max_turns": 30, "skill_loading_mode": "tool", "execution": {"python_command": ["/path/to/venv/bin/python"]}})
```

```python
bash_env = Environment.build("spreadsheetbench", {
    "agent": {"model": "your-model"},
    "interaction_mode": "bash_only",
    "tool_agent": "text_action",
    "skill_loading_mode": "system_prompt",
    "max_turns": 15,
    "stagnation_limit": 5,
    "execution": {"python_command": ["/path/to/venv/bin/python"]},
})
```

For a uv-managed project, use `"python_command": ["uv", "run", "--project", "/path/to/project", "python"]`.
On Windows, an interpreter path can be `"C:/project/.venv/Scripts/python.exe"`.
The default shell is `/bin/sh -c` on macOS/Linux and `cmd.exe /d /s /c` on Windows.
