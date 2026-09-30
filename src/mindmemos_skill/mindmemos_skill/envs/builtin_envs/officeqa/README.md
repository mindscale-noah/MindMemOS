## 1. Official Dataset

Download the release from the [official OfficeQA source](https://github.com/databricks/officeqa).

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

data = Data.build("officeqa", {"data_path": "/path/to/officeqa/questions.csv", "docs_root": "/path/to/officeqa/corpus"})
samples = data.load()
splits = data.split(samples, split_file="/path/to/splits.json")
env = Environment.build("officeqa", {"agent": {"model": "your-model"}})
```
