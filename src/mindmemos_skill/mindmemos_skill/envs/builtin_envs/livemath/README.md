## 1. Official Dataset

Download the monthly `qa_*_final.json` files from the
[official LiveMathematicianBench dataset](https://huggingface.co/datasets/LiveMathematicianBench/LiveMathematicianBench/tree/main/data).

## 2. Split File Format

Provide a UTF-8 JSON object mapping split names to ordered lists of task IDs:

```json
{
  "train": ["202511:1", "202511:2"],
  "val": ["202512:1"],
  "test": ["202601:1"]
}
```

- **Task IDs:** Use `<month>:<no>` from the official records. Every ID must exist in the loaded dataset.
- **Split names:** Include at least one nonblank name. Standard names such as `train`, `val`, and `test`, as well as custom names, are supported.
- **Unique membership:** An ID must not appear more than once, either within a split or across splits.
- **Partial selection:** Empty splits and dataset subsets are allowed. Unlisted tasks are excluded.
- **Ordering:** Tasks retain the order specified within each split.

## 3. Environment Initialization

Replace `data_path` with the downloaded dataset directory or a single
`qa_*_final.json` file. Set `split_file` to a JSON file matching the format above,
and replace `your-model` with your model name.

```python
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data

data = Data.build(
    "livemath",
    {
        "data_path": "/path/to/LiveMathematicianBench/data",
        "choice_seed": 42,
    },
)
samples = data.load()
splits = data.split(samples, split_file="/path/to/splits.json")

env = Environment.build(
    "livemath",
    {"agent": {"model": "your-model"}},
)
```
