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
env = Environment.build("spreadsheetbench_tool_calling", {
    "agent": {"model": "your-model", "max_completion_tokens": 16384},
    "max_turns": 30,
    "skill_loading_mode": "tool",
})

text_env = Environment.build("spreadsheetbench_text_action", {
    "agent": {"model": "your-model"},
    "max_turns": 100,
    "stagnation_limit": 5,
    "shell_timeout_seconds": 120,
    "transactional_recalculation": True,
})
```

Environment initialization automatically creates a dedicated Python 3.13
virtual environment and installs its locked requirements through uv. The fixed
paths are `<MindMemOS home>/envs/spreadsheetbench_text_action/bin/python` and
`<MindMemOS home>/envs/spreadsheetbench_tool_calling/bin/python` (Windows uses
`Scripts/python.exe`). `make dev` defaults to the repository's `.mindmemos/` home.
No interpreter setting is needed in the environment or text Action configuration.

Dependencies are packaged under `runtime/requirements/text_action/` (Trace2Skill)
and `runtime/requirements/tool_calling/` (SkillOpt). A completed installation is
reused without reinstalling; a file lock prevents duplicate concurrent setup.
Interrupted installations can retry on the next initialization. Shell `python`
and `python3` use the managed interpreter. Dependencies and workbook read/write
support are checked during initialization. The interpreter path is stored in the
internal `execution.python_command` and read directly when writing
`audit/execution.json`. Package versions are pinned by the requirements files.
LibreOffice/pyuno for formula recalculation remains a separate host dependency.
