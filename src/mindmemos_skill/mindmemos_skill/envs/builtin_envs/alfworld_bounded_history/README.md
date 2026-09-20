## 1. Official Dataset

Download the release from the [official ALFWorld source](https://github.com/alfworld/alfworld#download-data).

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
import yaml
from pathlib import Path

simulator = yaml.safe_load(Path("/path/to/alfworld/base_config.yaml").read_text())
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data

data = Data.build("alfworld_bounded_history", {"data_path": "/path/to/alfworld/items.json", "game_root": "/path/to/alfworld/data"})
samples = data.load()
splits = data.split(samples, split_file="/path/to/splits.json")
env = Environment.build("alfworld_bounded_history", {"agent": {"model": "your-model"}, "simulator": simulator, "history_length": 5})
```
