# WildClawBench + MindMemOS 快速上手

本文档面向外部评测人员：基于最新 MindMemOS 代码和已发布的 Docker Hub 镜像，完整跑完 WildClawBench 60 个任务。



## 0. 设置本地路径

先设置两个路径变量。它们只是给后面的命令复用，避免每条命令都写完整路径。

- `MINDMEMOS_REPO`：MindMemOS 仓库放在本机哪里。
- `WILDCLAWBENCH_DIR`：WildClawBench 仓库放在本机哪里。

如果还没有下载代码，可以先选一个目录：

```bash
mkdir -p ~/code

export MINDMEMOS_REPO=~/code/MindMemOS
export WILDCLAWBENCH_DIR=~/code/WildClawBench
```

如果已经下载过代码，把右边改成已有目录：

```bash
export MINDMEMOS_REPO=/path/to/your/MindMemOS
export WILDCLAWBENCH_DIR=/path/to/your/WildClawBench
```

## 1. 获取最新代码和数据

如果还没有 MindMemOS 代码：

```bash
git clone https://github.com/mindscale-noah/MindMemOS.git "$MINDMEMOS_REPO"
```

如果已经有 MindMemOS 代码，只需要拉到最新：

```bash
cd "$MINDMEMOS_REPO"
git pull
```

如果还没有 WildClawBench 代码：

```bash
git clone https://github.com/InternLM/WildClawBench.git "$WILDCLAWBENCH_DIR"
```

下载 WildClawBench 任务数据：

```bash
cd "$WILDCLAWBENCH_DIR"

hf download internlm/WildClawBench workspace \
  --repo-type dataset \
  --local-dir .
```

校验 60 个任务是否齐全：

```bash
find "$WILDCLAWBENCH_DIR/tasks" -mindepth 2 -maxdepth 2 -name '*.md' | wc -l
# 期望输出：60
```

## 2. 拉取评测镜像

```bash
docker pull clttyou/mindmemos-wildclawbench:1.3-brave-yibu

docker tag \
  clttyou/mindmemos-wildclawbench:1.3-brave-yibu \
  wildclawbench-mindmemos:v1.3-brave-yibu
```

可选校验：

这条命令只用于确认本地镜像是否存在、平台和大小是否正常；不影响评测流程，可以跳过。

```bash
docker image inspect \
  wildclawbench-mindmemos:v1.3-brave-yibu \
  --format 'ID={{.Id}} PLATFORM={{.Os}}/{{.Architecture}} SIZE={{.Size}}'
```

## 3. 准备 MindMemOS 和本地镜像

这一节每次评测前都按顺序执行，不需要判断之前有没有启动过服务，也不需要判断镜像里的代码是否最新。

先生成本次评测专用的 `api_key` 和 `project_id`：

```bash
cd "$MINDMEMOS_REPO"

python3 scripts/wildclawbench/new_key.py \
  --benchmark wildclawbench \
  --memory-algorithm schema \
  --project-override-config config/presets/project_override_wildclawbench_schema.example.yaml \
  --disable-previous
```

该 override 绑定 `config/presets/entity_modeling_wildclawbench.json` 实体模型（task 溯源实体 + environment/method/behavioral 三类经验实体），并把 schema 检索调成快速非 agentic 配置（关闭 entity agent / multi-hop / dual-path / rerank）。

把输出中的 `api_key` 和 `project_id` 填到下面：

```bash
export MINDMEMOS_API_KEY='<上一步输出的 api_key>'
export MINDMEMOS_PROJECT_ID='<上一步输出的 project_id>'
```

然后启动一份新的 MindMemOS API。下面命令会先停掉 8001 端口上的旧 API；如果本来没启动过，也可以直接执行：

```bash
cd "$MINDMEMOS_REPO"

api_pid="$(lsof -tiTCP:8001 -sTCP:LISTEN || true)"
if [ -n "$api_pid" ]; then
  kill $api_pid
fi

make API_HOST=0.0.0.0 API_PORT=8001 dev
```

保持这个终端不要关闭。另开一个终端，确认 API 正常：

```bash
curl http://127.0.0.1:8001/healthz
```

继续在新终端里同步并认证本地评测镜像：

```bash
cd "$MINDMEMOS_REPO"

bash scripts/wildclawbench/sync_image.sh \
  --api-key "$MINDMEMOS_API_KEY"
```

到这里，MindMemOS API 和本地镜像都已经准备好，可以继续配置 WildClawBench。

说明：

- `make dev` 会启动 MindMemOS 依赖服务和 FastAPI，不需要再单独执行 `docker compose -f dockers/docker-compose.memory.yml up -d`。
- `API_PORT=8001` 是本文档后续认证镜像时使用的端口。
- `API_HOST=0.0.0.0` 表示让 API 监听所有网卡。WildClawBench 任务跑在 Docker 容器里，会通过 `http://host.docker.internal:8001` 访问宿主机 API；用 `0.0.0.0` 比只监听 `127.0.0.1` 更稳。
- `sync_image.sh` 会用当前 MindMemOS 代码更新本地镜像，并用本次 `MINDMEMOS_API_KEY` 完成认证。

## 4. 配置 WildClawBench

只需要改 WildClawBench 仓库里的两个文件：

- `.env`：放搜索 key、模型 key、judge 配置。
- `my_api.json`：放模型 endpoint 和模型名。

进入 WildClawBench 仓库：

```bash
cd "$WILDCLAWBENCH_DIR"
```

如果还没有 `.env`，先从示例复制一份：

```bash
cp .env.example .env
```

编辑 `.env`，至少确认下面这些字段存在且值正确：

```bash
DOCKER_IMAGE=wildclawbench-mindmemos:v1.3-brave-yibu
DEFAULT_PARALLEL=1

BRAVE_API_KEY=<搜索服务 API key>

MY_PROXY_API_KEY=<评测方模型服务 API key>
OPENROUTER_API_KEY=<评测方模型服务 API key>
OPENROUTER_BASE_URL=https://<评测方模型服务地址>/v1
JUDGE_MODEL=gpt-5.4
```

编辑 `my_api.json`，例如使用 MiniMax M2.7：

```json
{
  "providers": {
    "custom": {
      "baseUrl": "https://<评测方模型服务地址>/v1",
      "apiKey": "${MY_PROXY_API_KEY}",
      "api": "openai-completions",
      "models": [
          { "id": "MiniMax-M2.7", "name": "MiniMax M2.7" }
      ]
    }
  }
}
```

`--model` 参数必须和 `my_api.json` 里的 provider/model 对上。上面的示例对应：

```bash
--model custom/MiniMax-M2.7
```

## 5. 冒烟测试

全量 60 个任务耗时很长，先跑 1 个较轻的任务，确认模型、judge、MindMemOS 记忆链路能正常启动、执行、评分、写入和 drain。

```bash
cd "$WILDCLAWBENCH_DIR"

DOCKER_IMAGE=wildclawbench-mindmemos:v1.3-brave-yibu \
bash script/run.sh openclaw \
  --task tasks/06_Safety_Alignment/06_Safety_Alignment_task_1_file_overwrite.md \
  --models-config my_api.json \
  --model custom/MiniMax-M2.7

python3 "$MINDMEMOS_REPO/scripts/wildclawbench/wait_drain.py" \
  --project-id "$MINDMEMOS_PROJECT_ID" \
  --timeout 300
```

预期：

- 在当前终端日志里能看到容器启动和模型设置，例如：

  ```text
  Starting container
  Model set: custom/MiniMax-M2.7
  Agent finished successfully
  Grading results written to → .../score.json
  Usage written to .../usage.json
  Container cleaned up
  ```

- `wait_drain.py` 在当前终端最终输出：

  ```text
  no pending add_record, safe to continue
  ```

- 冒烟任务的产物在 WildClawBench 的 `output/openclaw` 目录下。可以用下面命令找到最近一次冒烟结果目录：

  ```bash
  latest_smoke_dir="$(
    find "$WILDCLAWBENCH_DIR/output/openclaw/06_Safety_Alignment/06_Safety_Alignment_task_1_file_overwrite" \
      -mindepth 1 \
      -maxdepth 1 \
      -type d \
      -name 'MiniMax-M2.7_*' \
      -print \
    | sort \
    | tail -n 1
  )"

  echo "$latest_smoke_dir"
  ls "$latest_smoke_dir/score.json" "$latest_smoke_dir/usage.json"
  ```

  这两个文件都存在，就说明本次冒烟至少完成了评分和用量记录。



冒烟测试通过后，不要直接复用这个 `MINDMEMOS_PROJECT_ID` 跑正式全量。冒烟测试已经写入过记忆；正式评测要重新生成一个干净的 key/project：

```bash
cd "$MINDMEMOS_REPO"

python3 scripts/wildclawbench/new_key.py \
  --benchmark wildclawbench \
  --memory-algorithm schema \
  --project-override-config config/presets/project_override_wildclawbench_schema.example.yaml \
  --disable-previous
```

把新的输出重新填到环境变量：

```bash
export MINDMEMOS_API_KEY='<新的 api_key>'
export MINDMEMOS_PROJECT_ID='<新的 project_id>'
```

用新的 key/project 重新准备 MindMemOS API 和本地镜像。

先在 API 终端执行：

```bash
cd "$MINDMEMOS_REPO"

api_pid="$(lsof -tiTCP:8001 -sTCP:LISTEN || true)"
if [ -n "$api_pid" ]; then
  kill $api_pid
fi

make API_HOST=0.0.0.0 API_PORT=8001 dev
```

保持这个终端不要关闭。另开一个终端执行：

```bash
cd "$MINDMEMOS_REPO"

bash scripts/wildclawbench/sync_image.sh \
  --api-key "$MINDMEMOS_API_KEY"
```

## 6. 跑全量 60 个任务

不要使用 WildClawBench 官方的 `--parallel 4` 全量命令。MindMemOS 写入是异步的，全量评测必须使用本仓库的串行 wrapper：每跑完一个任务，等待记忆写入排空后再跑下一个任务。

```bash
cd "$WILDCLAWBENCH_DIR"

export WILDCLAWBENCH_RUN_MARKER="$WILDCLAWBENCH_DIR/.wildclawbench_run_$(date +%Y%m%d_%H%M%S).marker"
touch "$WILDCLAWBENCH_RUN_MARKER"
echo "本次运行 marker: $WILDCLAWBENCH_RUN_MARKER"

WILDCLAWBENCH_DIR="$WILDCLAWBENCH_DIR" \
MINDMEMOS_PROJECT_ID="$MINDMEMOS_PROJECT_ID" \
DOCKER_IMAGE=wildclawbench-mindmemos:v1.3-brave-yibu \
DRAIN_TIMEOUT=300 \
bash "$MINDMEMOS_REPO/scripts/wildclawbench/run_serial.sh" \
  --category all \
  --models-config my_api.json \
  --model custom/MiniMax-M2.7
```

长时间运行建议用后台方式：

```bash
cd "$WILDCLAWBENCH_DIR"

export WILDCLAWBENCH_RUN_MARKER="$WILDCLAWBENCH_DIR/.wildclawbench_run_$(date +%Y%m%d_%H%M%S).marker"
touch "$WILDCLAWBENCH_RUN_MARKER"
echo "本次运行 marker: $WILDCLAWBENCH_RUN_MARKER"

caffeinate -i -s nohup env \
  WILDCLAWBENCH_DIR="$WILDCLAWBENCH_DIR" \
  MINDMEMOS_PROJECT_ID="$MINDMEMOS_PROJECT_ID" \
  DOCKER_IMAGE=wildclawbench-mindmemos:v1.3-brave-yibu \
  DRAIN_TIMEOUT=300 \
  bash "$MINDMEMOS_REPO/scripts/wildclawbench/run_serial.sh" \
    --category all \
    --models-config my_api.json \
    --model custom/MiniMax-M2.7 \
  > ~/wildclawbench_run.log 2>&1 &
```

查看日志：

```bash
tail -f ~/wildclawbench_run.log
```

## 7. 查看结果

结果目录：

```bash
$WILDCLAWBENCH_DIR/output/openclaw
```

说明：

- 每跑一个任务，WildClawBench 都会新建一个带模型名、时间戳和 run id 的结果目录。
- 多次运行不会覆盖历史结果，所以 `output/openclaw` 里通常会有很多旧结果。
- 第 6 步里创建的 `WILDCLAWBENCH_RUN_MARKER` 用来区分“本次运行”和历史运行。
- 如果换了新终端，需要把第 6 步打印出来的 marker 路径重新 export 一次，例如：

  ```bash
  export WILDCLAWBENCH_RUN_MARKER=/path/to/WildClawBench/.wildclawbench_run_20260804_120000.marker
  ```

统计本次运行已经生成了多少个 `score.json`：

```bash
find "$WILDCLAWBENCH_DIR/output/openclaw" \
  -name score.json \
  -newer "$WILDCLAWBENCH_RUN_MARKER" \
  | wc -l
```

这条命令的意思是：在 `output/openclaw` 下查找所有比本次 marker 更新的 `score.json`，再统计数量。全量 60 个任务都跑完时，正常应接近 60；如果少于 60，说明还有任务没跑完，或者有任务在生成分数前异常中断。

列出本次运行的结果文件：

```bash
find "$WILDCLAWBENCH_DIR/output/openclaw" \
  -name score.json \
  -newer "$WILDCLAWBENCH_RUN_MARKER" \
  -print \
  | sort
```

汇总本次运行的分数、token 和耗时：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["WILDCLAWBENCH_DIR"]) / "output" / "openclaw"
marker = Path(os.environ["WILDCLAWBENCH_RUN_MARKER"])
marker_time = marker.stat().st_mtime

rows = []
for score_path in sorted(root.rglob("score.json")):
    if score_path.stat().st_mtime <= marker_time:
        continue

    run_dir = score_path.parent
    usage_path = run_dir / "usage.json"

    score = json.loads(score_path.read_text(encoding="utf-8"))
    usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else {}

    task_id = run_dir.parent.name
    category = run_dir.parent.parent.name
    overall = score.get("overall_score")

    rows.append({
        "category": category,
        "task_id": task_id,
        "overall_score": overall,
        "elapsed_time": usage.get("elapsed_time"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read_tokens": usage.get("cache_read_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": usage.get("cost_usd"),
        "run_dir": str(run_dir),
    })

for row in rows:
    print(
        f"{row['category']}/{row['task_id']} "
        f"score={row['overall_score']} "
        f"elapsed={row['elapsed_time']}s "
        f"tokens={row['total_tokens']} "
        f"cost=${row['cost_usd']} "
        f"dir={row['run_dir']}"
    )

scores = [r["overall_score"] for r in rows if isinstance(r["overall_score"], (int, float))]
total_tokens = sum((r["total_tokens"] or 0) for r in rows)
total_elapsed = sum((r["elapsed_time"] or 0) for r in rows)

print()
print(f"tasks_with_score={len(rows)}")
if scores:
    print(f"avg_score={sum(scores) / len(scores):.4f}")
print(f"total_tokens={total_tokens}")
print(f"total_elapsed_seconds={total_elapsed:.2f}")
PY
```

查看失败或低分任务时，优先看对应结果目录下的：

```bash
score.json
usage.json
chat.jsonl
task_output/
```

## 8. 清理残留容器

正常运行时 `script/run.sh` 会自动清理任务容器。只有中途被打断、电脑重启、或者不确定是否有残留容器时，执行：

```bash
containers="$(
  docker ps -a \
    --filter "ancestor=wildclawbench-mindmemos:v1.3-brave-yibu" \
    -q
)"

if [ -n "$containers" ]; then
  docker rm -f $containers
fi
```

## 9. 关键注意事项

- 每次正式评测都生成一个新的 `MINDMEMOS_API_KEY` / `MINDMEMOS_PROJECT_ID`，避免历史记忆污染。
- 评测前必须运行 `sync_image.sh`，让本地镜像里的 SDK / OpenClaw 插件和当前最新代码一致。
- 冒烟测试通过后，正式全量前要重新生成干净 project，并再次同步认证镜像。
- 全量 60 个任务必须用 `run_serial.sh`，不要并行跑。
- `.env` 里的 `BRAVE_API_KEY`、`MY_PROXY_API_KEY`、`OPENROUTER_API_KEY`、`OPENROUTER_BASE_URL`、`JUDGE_MODEL` 必须和评测方实际服务一致。
- `my_api.json` 里的模型 id 必须和运行命令里的 `--model custom/MiniMax-M2.7` 对上。
- 评测镜像是无凭据发布镜像，外部用户必须先执行第 3 步完成同步和认证。
- 全量运行耗时以小时计，建议预留 20-30GB 磁盘空间。
