## 1. Official Dataset

Download the release from the [official SearchQA source](https://github.com/nyu-dl/SearchQA).

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

data = Data.build("searchqa", {"data_path": "/path/to/searchqa/items.json"})
samples = data.load()
splits = data.split(samples, split_file="/path/to/splits.json")
env = Environment.build("searchqa", {
    "agent": {
        "model": "your-model",
        "api_key": "your-api-key",
        "base_url": "https://gateway.example.com/v1",
        "timeout_s": 180,
        "max_retries": 3,
        "max_completion_tokens": 4096,
        "temperature": None,
        "reasoning_effort": None,
    },
    "max_turns": 1,
    "max_context_chars": 6000,
    "evaluator_version": "searchqa-v1",
})
```

`Environment.build()` receives the inner environment configuration. A training
request uses `env={"class_name": "searchqa", "config": ...}`; the local evaluation
YAML uses the same envelope under `environment`.

| Environment field | Default | Meaning |
| --- | --- | --- |
| `agent` | Required | OpenAI-compatible chat model configuration; supply `model`, `api_key`, and `base_url` for model calls. |
| `max_turns` | `1` | Maximum answering and self-review rounds per task; must be at least 1. Additional rounds do not receive private scoring feedback. |
| `max_context_chars` | `6000` | Public passage character budget; must be at least 1. Truncation prefers `[DOC]` boundaries and uses a character-prefix fallback. |
| `evaluator_version` | `searchqa-v1` | Exact-match, token F1, and feedback protocol identifier recorded for audit and resume checks. |

Connection values come exclusively from configuration. `timeout_s` is the timeout
for each physical request, and `max_retries=3` allows up to four physical requests
per model call. `max_completion_tokens` includes reasoning tokens; `None` omits
the API limit. `temperature=None` and `reasoning_effort=None` omit those request
parameters. Use values supported by the configured model service.

The dataset path belongs to `SearchQADataConfig.data_path`, separate from the
environment. Algorithm models such as Skill-Forest's `optimizer` are configured
independently in the service YAML. The annotated example is
[`skill_forest_searchqa.yaml`](../../../../../../config/mindmemos_eval/skill_evaluation/skill_forest_searchqa.yaml).
