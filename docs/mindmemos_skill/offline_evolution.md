# Offline Evolution 演进算法

`offline_evolution` 基于历史执行轨迹演进一个或多个 Skill，依次完成证据收集、G1 核验、Skill 修改、G2 检查和版本保存。

## 配置与调用

服务配置示例位于 `config/mindmemos_skill/dev.example.yaml`。在 `service.agents` 中配置所选 Agent 的 `api_key`、`base_url` 和 `model`，运行环境需安装对应的 Claude CLI 或 Codex CLI。`offline` pipeline 默认使用 `offline_evolution_claude`；使用 Codex 时，将 `service.evolution_pipelines.offline.config.evolver.config.agent` 设为 `offline_evolution_codex`。

SDK 使用已保存的服务连接，通过 `POST /v1/skills/algos/evolve` 提交演进请求，`pipeline` 填写 `offline`。

输入为已注册的 Skill 精确版本和实际执行轨迹。每次上报绑定执行所用的版本；多个执行逐条上报。完整 JSON 轨迹可放入消息的 `content`：

```python
import json
from pathlib import Path

from mindmemos_sdk import MindMemOSClient
from mindmemos_sdk.skills import EvolveRequest, TaskShowRequest, TrajectoryReportRequest

raw_trajectory = json.loads(Path("trajectory.json").read_text(encoding="utf-8"))
version_ids = ["skv-INPUT_A", "skv-INPUT_B"]
with MindMemOSClient() as client:
    client.skills.report_trajectory(
        TrajectoryReportRequest(
            user_id="example",
            agent_id="example",
            skill_version_id=version_ids[0],
            messages=[{"role": "trajectory", "content": raw_trajectory}],
        )
    )
    task = client.skills.evolve(
        EvolveRequest(
            pipeline="offline",
            skill_version_ids=version_ids,
            auto_accept=False,
        )
    )
    state = client.skills.show_task(TaskShowRequest(task_id=task["task_id"]))
    print(state)
```

通过 `show_task` 查询任务状态，`succeeded` 后从 `result.members` 读取已保存的 Skill 和版本 ID。证据按输入的精确版本读取，默认使用全部可用轨迹并去重，至少需要一条轨迹。

## 修改范围与结果

| 请求字段 | 含义 |
| --- | --- |
| `skill_version_ids` | 输入的精确版本列表，每个 Skill 家族至多一个；单个输入也可使用 `skill_version_id`，两者二选一 |
| `editable_skills` | 允许修改的输入 Skill ID 列表；默认全部可改，`[]` 表示全部只读 |
| `allow_create_skills` | 是否允许新建独立 Skill，默认 `true` |
| `auto_accept` | 默认 `true`，保存后激活已有 Skill 的新版本；`false` 只保存，保留原 active |

全部输入均供分析，`editable_skills` 限制修改范围。已有成员必须保留，未变化的成员沿用原版本。

`auto_accept=False` 时，通过 `show` 读取保存的新版本，使用 `activate` 激活。新建 Skill 的首版本按注册规则成为 active。

算法参数在 `service.evolution_pipelines.offline.config.evolver.config` 中设置，包含 `agent`、`retries`、`script_timeout_s`、`max_invalid_share` 和 `python`。pipeline 的 `traj_collect.trajectory_limit` 控制每个版本的轨迹读取上限，默认不限制。

运行产物保存在 pipeline 的 `work_dir` 下，包含证据报告、修改报告和候选文件。算法重入相同目录时复用兼容检查点；SDK 重复提交 `evolve` 创建新任务和独立运行。
