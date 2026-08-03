# WildClawBench + MindMemOS 用户评测执行文档

前提：已经按 [WILDCLAWBENCH_SETUP_FROM_ZERO_ZH.md](WILDCLAWBENCH_SETUP_FROM_ZERO_ZH.md) 完成环境搭建——MindMemOS 基础设施和 API 已启动，`wildclawbench-mindmemos:v1.3-brave-yibu` 镜像已构建并验证通过。本文档只讲"怎么按下运行按钮、以及为什么不能直接用官方文档里 `--parallel 4` 的写法"。

## 1. 为什么不能直接抄 WildClawBench 官方的并行命令

WildClawBench README 给的全量评测命令是：

```bash
bash script/run.sh openclaw --category all --parallel 4 --model openrouter/openai/gpt-5.5
```

`--parallel 4` 对绝大多数评测集是安全的，但**对接了 MindMemOS 的这次评测不安全**，原因是两层叠加的异步：

1. **WildClawBench 层面**：60 个任务当前共用同一个 OpenClaw 身份（`appId: wildclawbench`），也就共用同一个 MindMemOS `project_id`。如果 `--parallel > 1`，多个任务的容器会同时调用同一个 project 的 `add`/`search`，任务之间的记忆边界就不再干净——这违背了"每个任务应该只看到之前任务写入的记忆，看不到并发中的其它任务"的假设。
2. **MindMemOS 层面**：OpenClaw 插件配置的是 `addMode: async`。调用 `add` 时 API 立即返回 `status: queued`，真正的写入由宿主机 API 进程里的 Kafka 消费者异步完成（见 [WILDCLAWBENCH_SETUP_FROM_ZERO_ZH.md](WILDCLAWBENCH_SETUP_FROM_ZERO_ZH.md) 第 5 步）。`run_batch.py` 对这个异步过程一无所知：一个任务的容器被清理（`remove_container`）之后，它触发的记忆写入可能还在 `processing`。哪怕 `--parallel 1` 保证了任务容器不重叠，也**不**保证下一个任务开始时，上一个任务的记忆已经写完。

`.env` 里 `DEFAULT_PARALLEL` 目前默认就是 `1`（已经是对的），但只解决了第 1 层问题，没解决第 2 层。第 2 层需要额外的"排空等待"步骤——这正是你问的"Task 之间要去查 add_record 数据库，确认上一个 task 状态是 ok"。

## 2. 排空等待怎么做

`add_record_v1` 是 Qdrant 里记录每次 `add` 请求生命周期的 collection，状态机是：

```
queued → processing → ok | error
```

（源码见 [operation_records.py](/Users/chenliang/code/new/0701/MindMemOS/src/mindmemos/mindmemos/pipelines/memory_db/operation_records.py) 的 `record_add_input` / `mark_add_processing` / `mark_add_completed` / `mark_add_failed`。）

本仓库提供了两个脚本，把"跑一个任务 → 等它写完 → 再跑下一个"这件事自动化：

- [scripts/wildclawbench/wait_drain.py](/Users/chenliang/code/new/0701/MindMemOS/scripts/wildclawbench/wait_drain.py)：直接查询 Qdrant，确认某个 `project_id` 下没有任何 `status` 为 `queued`/`processing` 的记录，才退出（有超时保护，默认 180 秒，超时会非 0 退出，不会假装成功）。
- [scripts/wildclawbench/run_serial.sh](/Users/chenliang/code/new/0701/MindMemOS/scripts/wildclawbench/run_serial.sh)：逐个任务调用 `script/run.sh openclaw --task <file>`，每跑完一个任务就调用上面的等待脚本，确认排空了再继续下一个。

单独测试等待脚本（可用来确认 Qdrant 连通、project_id 填对）：

```bash
python3 /Users/chenliang/code/new/0701/MindMemOS/scripts/wildclawbench/wait_drain.py \
  --project-id proj_wildclawbench_20260706_112221
```

## 3. 冒烟测试（先跑一个任务）

不要一上来就跑全量。先确认链路通：

```bash
cd /Users/chenliang/WildClawBench

DOCKER_IMAGE=wildclawbench-mindmemos:v1.3-brave-yibu \
HTTP_PROXY_INNER=http://host.docker.internal:7890 \
HTTPS_PROXY_INNER=http://host.docker.internal:7890 \
NO_PROXY_INNER=host.docker.internal,127.0.0.1,localhost \
bash script/run.sh openclaw \
  --models-config my_api.json \
  --task tasks/06_Safety_Alignment/06_Safety_Alignment_task_1_file_overwrite.md \
  --model custom/gpt-4.1-mini
```

跑完之后手动确认一下排空脚本能正常工作：

```bash
python3 /Users/chenliang/code/new/0701/MindMemOS/scripts/wildclawbench/wait_drain.py \
  --project-id proj_wildclawbench_20260706_112221
```

应该很快打印 `no pending add_record, safe to continue`。如果一直卡住直到超时，先去检查 MindMemOS API 进程和 Kafka 是否还活着（`docker compose -f dockers/docker-compose.memory.yml ps`），不要跳过这一步直接跑全量。

## 4. 跑单个类别

先按类别小批量跑，比如先跑 6 个任务最少的 `03_Social_Interaction`：

```bash
cd /Users/chenliang/WildClawBench

WILDCLAWBENCH_DIR=/Users/chenliang/WildClawBench \
MINDMEMOS_PROJECT_ID=proj_wildclawbench_20260706_112221 \
bash /Users/chenliang/code/new/0701/MindMemOS/scripts/wildclawbench/run_serial.sh \
  --category 03_Social_Interaction \
  --models-config my_api.json \
  --model custom/gpt-4.1-mini
```

## 5. 跑全量 60 个任务

确认前面的类别小批量跑通、分数看起来合理之后，再跑全部：

```bash
cd /Users/chenliang/WildClawBench

WILDCLAWBENCH_DIR=/Users/chenliang/WildClawBench \
MINDMEMOS_PROJECT_ID=proj_wildclawbench_20260706_112221 \
DRAIN_TIMEOUT=300 \
bash /Users/chenliang/code/new/0701/MindMemOS/scripts/wildclawbench/run_serial.sh \
  --category all \
  --models-config my_api.json \
  --model custom/gpt-4.1-mini
```

说明：

- `--category all` 会遍历 `tasks/` 下全部 6 个类别、60 个任务，逐个跑、逐个排空，全程严格串行，不会出现两个任务容器同时跑、或下一个任务在上一个任务记忆写完之前开始的情况。
- 全程串行意味着比官方 `--parallel 4` 的写法慢很多（60 个任务顺序跑，单任务几分钟到二十多分钟不等，总耗时以小时计），这是为了记忆边界干净所付出的必要代价，不建议为了赶时间调大并行度。
- 如果中途某个任务失败（`run.sh` 非零退出），脚本会跟着 `set -euo pipefail` 中断，方便你先排查这个任务再决定是重跑单个任务还是继续。重跑单个任务用第 3 步的单任务命令即可。
- 如果某次排空等待超时（默认 180 秒，这里示例调到 300 秒），说明 Kafka worker 处理慢或者卡住了，去看 MindMemOS API 进程日志，不要绕过等待直接继续。

## 6. 查看结果

跑完之后（无论是单任务、按类别还是全量），WildClawBench 会自动生成：

- 每个任务：`output/openclaw/<category>/<task_id>/<model_timestamp_runid>/score.json`（各项指标 0.00-1.00）、`usage.json`（token/耗时/花费）、`chat.jsonl`（完整对话轨迹）、`task_output/`（agent 产出的文件）。
- 按类别的 summary（跑批次时终端会打印）。
- 全量跑完后的全局汇总：`output/summary_all.json`。

判断一个低分任务是"环境问题"还是"任务没做对"，参考已有文档 [WILDCLAWBENCH_MINDMEMOS_ZH.md](WILDCLAWBENCH_MINDMEMOS_ZH.md) 第 8 节的判断清单；MindMemOS 集成本身的链路已经验证过，全量跑分数低大概率是模型能力/工具能力问题，不要先怀疑记忆链路。

## 7. 跑完之后清理

```bash
docker ps -a --filter "ancestor=wildclawbench-mindmemos:v1.3-brave-yibu" -q | xargs -r docker rm -f
```
