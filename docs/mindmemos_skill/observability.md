# 算法运行追踪后端

`mindmemos_skill` 的训练始终产生 OpenTelemetry span。算法与 Agent 通过包级 Python `logging` 记录事件，服务级桥接 handler 将日志转成关联当前 span 的 OTel log；没有算法 run 的请求、调度和后台日志也会转发。服务可选配 SQLite/PostgreSQL 审计库、OTLP/HTTP Collector，或同时启用两者。省略配置时仍执行相同的打点，但不持久化或转发。业务 Skill 表不保存调用正文。

分层边界：`logging` 保留普通控制台/文件输出，并用服务级 handler 将包级 Python 日志转发到 OTel；其控制台级别与审计采集级别分别控制。`infra.observability` 创建隔离的 OTel tracer/logger，使用 SDK 回调将记录交给异步审计写入，并用官方 protobuf 编码器及异步 HTTP 批量转发 logs 和 spans。`persistence.observability.AlgorithmAuditPersistence` 在配置数据库时注册审计表，管理 run 状态与查询。业务代码用 `get_logger(__name__)` 记录事件，用运行 session 创建 span；Agent 的 info 日志记录调用状态与 usage，debug 日志记录经过脱敏和截断的输入输出。

```yaml
observability:
  backend:
    class_name: postgres
    config:
      url: postgresql+psycopg://USER:PASSWORD@HOST:5432/observability
  capture_level: info  # 改为 debug 才记录 Agent 输入输出
  otlp:
    endpoint: http://otel-collector:4318
    timeout: 5.0
```

配置位于服务顶层，与 `persistence` 并列。`observability.backend` 和 `observability.otlp` 均可省略其中一项，但至少须配置一项。`endpoint` 是 Collector 的 OTLP/HTTP 基础地址；SDK 分别请求 `/v1/logs` 和 `/v1/traces`。与 Collector 在同一 Docker 网络时可使用示例中的服务名；宿主机进程可用映射端口 `http://localhost:4318`。本地审计库验证可使用 `class_name: sqlite` 和单独的 SQLite URL。PostgreSQL 复用项目现有数据库实现，因此目标实例目前也需要预装 `vector` 扩展。凭据应放在本地配置中，不提交真实 URL。

配置数据库时，`create_service()` 会在启动时创建缺失的 `observability_runs`、`observability_spans`、`observability_logs` 表；它不会修改已有表。历史训练不会被伪造为完整 trace，也不需要重建 Skill 数据。所有 `run_training` 执行都生成算法 span 与运行状态日志；算法 span 继承当前任务或请求的 trace。配置数据库后，恢复执行沿用 `run_id`，并通过 OTel link 关联先前的算法 span。Skill-Forest 的树扩展、候选节点和最终集成各有阶段 span；`OpenAIChatAgent.run()` 产生调用子 span 和含模型、状态、usage 的 info 日志。TreeSkill、Skill-Forest 以及训练后评估通过原生协程传递 trace 上下文；异步任务自动继承创建时的上下文。数据库审计写入由协程执行，并在运行结束时排空；OTLP 使用异步 HTTP 批量导出，服务关闭时等待剩余导出完成，不使用同步导出器线程。运行中的普通包级日志也会进入 OTel，作为 `python.log` 保存原消息。Info Agent 事件不保存输入输出正文；Debug 另发关联日志，并对 Agent 输入输出中的常见凭据键脱敏、限制单个正文长度。公共出口对普通消息、结构化正文和 traceback 统一脱敏与截断；调试输入仍可能含有任务正文，须限制该数据库的读取权限。

读取入口为 `service.algorithm_run(project_id, run_id)`、`service.algorithm_spans(project_id, run_id, limit=..., offset=...)` 和 `service.algorithm_logs(...)`。这些查询只读取配置的审计数据库；仅配置 OTLP 时不可用。Collector 写入的 ClickHouse `otel_logs`、`otel_traces` 由其自身的查询工具读取。查询强制经过项目作用域。当前不暴露 HTTP 审计端点或 UI 页面。

## 统一 Python 日志出口

HTTP 进程启动时显式调用 `configure_logging()`；直接 Python 集成在应用启动时调用该函数。
算法继续使用普通 `logging.getLogger(__name__)`（位于 `mindmemos_skill` 命名空间内），或 `get_logger(__name__)`。
`logger.exception()` 的多行 traceback 会同时进入终端、任务日志和 OTel 的 `exception.stacktrace`。
消息、结构化 `extra={"event_body": ...}` 和 traceback 各限制为 32768 个字符，超出部分添加截断标记。
常见凭据字段、文本中的凭据赋值和 Bearer 凭据会被替换为 `[REDACTED]`；任意无字段标记的敏感文本仍须由调用方识别。

在要监控的同步或异步函数上添加 `@traced()`，无需手动调用 `log_context`。命名及
`name`、`record_args`、`record_result` 参数与 `mindmemos.logging.tracing.traced` 保持一致。
每次调用自动产生子 span 和 INFO 完成日志，包含函数名、状态和毫秒耗时；失败记录 ERROR
日志及 traceback，并原样抛出异常。正常取消记录 INFO 中断日志并继续传播取消信号，不打印业务失败 traceback。

```python
from mindmemos_skill.logging import get_logger, traced

@traced("evaluation", record_args=True, record_result=True)
async def evaluate(run_id: str, project_id: str):
    get_logger(__name__).info("Evaluation started")
    return {"score": 0.9}
```

输入输出默认关闭；开启后统一脱敏和截断。直接命名为 `request_id`、`project_id`、`task_id`、
`attempt`、`run_id`、`service_name` 的字符串或整数参数自动绑定到本次调用的上下文，函数内
普通日志和子协程继承这些字段，调用结束后恢复父上下文。其他关联字段由运行基础设施自动传入。
算法运行内使用该服务的 tracer；独立调用使用全局 OTel provider，即使没有配置导出器也会记录普通日志。
装饰器保留函数名称和签名，支持普通函数、实例方法和协程函数；生成器与异步生成器不支持，
会在装饰时报告错误。`log_context` 保留供基础设施内部管理任务和运行作用域。

任务入口使用 `async with async_task_log_reporter(sink, project_id=..., task_id=..., attempt=...)`，
从执行上下文自动收集普通 logger 消息。算法不传上报器，也不维护日志专用进度对象；普通消息使用 f-string。
消息以可读字符串为核心，按 `事件 | key=value 字段 | 结果或耗时` 组织；
控制台使用 `[UTC 时间] [级别] [logger] | 任务身份 | 消息`，不追加 JSON 正文行。
`payload.message` 是同一可读消息；关联字段和已有可选 `payload.body` 保留供遥测或历史日志消费，不要求算法构造正文。

Kafka worker 自动绑定 `project_id`、`task_id` 和物理 `attempt`，普通 INFO 及以上日志追加到业务任务数据库的
`skill_task_logs_v1` 表。任务表只保存执行、结果和恢复状态，进度由这张日志表读取，不覆盖重试前的记录。
查询使用 `TaskDataService.logs(project_id, task_id, after=0, limit=100)`，按项目隔离。
本地内存任务模式使用相同的日志捕获和查询结构，在内存中保存任务日志，进程关闭后不保留。

任务日志在入队前排除 INFO 级别的正常 `agent.call`，避免逐次模型调用挤占进度分页和写入队列；
WARNING/ERROR 级别的调用诊断仍保留。此筛选只作用于任务日志 handler，OpenTelemetry 仍采集
正常调用的状态、usage 和调用 span；启用 OTLP 导出后，可在 Grafana 查询这些遥测记录。
变更在任务执行进程加载新代码后生效，已写入任务数据库的历史调用日志不会自动删除或隐藏。

任务日志使用每个任务独立的异步 FIFO 写入队列，任务结束或取消时等待已排队日志写完；worker 暴露终态前先排空日志，因此客户端观察终态后可以读取最后的失败诊断。
每个队列最多保存 1000 条待写日志；队列满时该条任务数据库副本会丢弃并向终端报告，终端及 OTel 输出继续执行。
日志入库或 OTel 桥接失败会输出可读且经过脱敏的诊断，不替代业务结果。
进程被强制终止时仍可能丢失尚未落库的日志；当前没有持久化日志补写队列。

`service.task_progress(project_id, task_id, after=0)` 和 `/v1/skills/tasks/progress` 直接返回任务日志，
不再构造专用进度事件。每条记录含 `sequence`、`log_id`、`project_id`、`task_id`、`attempt`、
`occurred_at` 和 `payload`；`payload` 含日志名称、级别、消息、结构化 `body`、trace/span ID 和 traceback。
阶段名称与完成数量由算法日志的 `payload.body` 提供，阶段耗时来自阶段 span 或阶段结束日志。
阶段计时使用公共 `stage()`，阶段名称、状态和 `duration_ms` 同时进入普通日志和子 span。

分页游标为最后读到的 `sequence`，每页最多 100 条。持久化日志序号递增但允许有间隔，
并发任务或恢复 attempt 不要求序号连续。失败诊断保存在对应 attempt 的日志中，任务状态响应不再包含 `progress` 副本。
旧的进度模型、事件表和 reporter 接口已移除，不提供旧本地 Skill 数据库迁移或兼容层。

相关离线验证：

```bash
.venv/bin/pytest -q src/mindmemos_skill/tests/test_logging.py src/mindmemos_skill/tests/test_logging_context.py src/mindmemos_skill/tests/test_task_manager.py src/mindmemos_skill/tests/test_observability.py
```

## Grafana：Skill 运行日志

`dockers/grafana/dashboards/MindMemOS Skill/skill-spans.json` 是自动加载的 **MindMemOS Skill Spans** 看板，位于 Grafana 的 **MindMemOS Skill** 文件夹。它读取 ClickHouse 的 `otel_traces`、`otel_logs` 和 Kafka/Collector 指标表，按 service、project、request、task、attempt、run、trace 和 span 名称筛选，展示 span 数量、错误数、算法执行耗时、最近 spans、关联日志及已上报的 traceback。算法耗时按 `algorithm.train` 或 `algorithm.evolve` span 计算，不含排队时间；错误时间、阶段和 traceback 依赖服务端上报，历史缺失字段显示为空。

要让训练日志进入此页面，在服务配置的 `observability` 中加入 `otlp.endpoint: http://localhost:4318`（宿主机启动服务）或 `http://otel-collector:4318`（同一 Docker 网络）。可同时保留本地审计 `backend`。默认 `config/mindmemos_skill/dev.example.yaml` 已配置 PostgreSQL 审计库和宿主机 OTLP 地址，训练日志与链路经 Collector 写入 ClickHouse。Collector 将日志写入 `otel.otel_logs`；Grafana 的 ClickHouse 数据源及页面由 `dockers/compose.yaml` 自动挂载。首次查看时可在 Grafana 的 MindMemOS Skill 文件夹打开该页面。

页面的项目/运行筛选只是探索工具，不能代替访问控制；授权查看 Grafana 的人可能读取其他项目日志。`capture_level: debug` 时日志正文可能含任务输入输出，需限制 Grafana/ClickHouse 读取权限。OTLP 批量导出失败不会反映在该页面或本地 `audit_status` 中；应另行监控 Collector 导出错误和队列状态。当前 ClickHouse TTL 默认 72 小时，超过保留期的日志不会显示。


发布 `skill_forest` 训练结果时，后端从训练目录的 `result.json` 读取紧凑报告，随版本事务写入 `skill_versions.metadata.algorithm_report`。报告含 `schema_version`、`run_id`、总树数、总节点数、扩展次数、集成策略，以及每棵树的节点数、winner 的 `gain`、验证均分和 `train_top_select_score`（冻结 winner 在完整训练池上的选择分）。选择策略还会记录最终选中的节点 ID。普通版本读取接口会原样返回该 JSON，供后续 UI 展示；未实现报告的训练器不写该键。

配置数据库后，结束的 span 和发出的 log 由异步任务写库，运行结束前等待写入完成，写入异常会把 run 的 `audit_status` 标为 `incomplete`。这个状态只表明本进程检测到的审计库失败；OTLP 批量导出异步执行，其网络失败不会更新 `audit_status`。关闭服务时等待异步导出任务并关闭连接池；瞬时网络错误和可重试 HTTP 状态在配置的超时内重试，最终失败由 OpenTelemetry exporter logger 记录。Collector 端还需独立监控写入失败与积压。当前机制不提供跨进程崩溃后的完整审计保证，也不阻止 Skill 发布。仅配置 OTLP 时，没有本地 run 状态或跨进程查询能力。发布门禁、持久化补写和报告 UI 属于后续接入工作。外部 provider 内部重试目前计入一次逻辑 Agent 调用；需要精确的物理请求次数时，应再在传输层打点。


## 请求、Kafka 与任务链路

公开异步 Service 业务方法默认记录经过脱敏的输入、结果和耗时，覆盖直接 Python 调用。
HTTP 入口创建 `request_id` 和请求 span，Service span 继承请求 span；鉴权、校验和业务错误也记录状态与响应。
HTTP 输入输出最多采样 32768 字节的 JSON；小型 JSON 请求会以相同 ASGI 消息重放，保证鉴权拒绝时也有输入证据。
文件和其他流式响应只记录内容类型、实际发送字节数与结束状态；不读取或缓存完整文件。
超出采样上限的正文只保留大小和截断标记，客户端断开时记录已发送字节数。

提交任务时保存原 Service span 的 W3C `traceparent` / `tracestate` 和 `request_id`。
Kafka publish span 和每个 worker execute span 以这个提交 span 为父级，Kafka headers 携带保存的上下文。
恢复通知和新的 attempt 继续沿用原提交关联；算法 span 继承 worker span，不再创建独立的空 Context 根。
任务数据库仍是执行状态的真相来源；Kafka 只通知，不要求 API 等待 broker 确认。
`task.kafka.enqueued` 表示客户端已接收发送请求，`task.kafka.delivered` 才表示收到 broker 确认。
启动连接流程通过 Kafka Admin 显式检查 topic，日志区分 `topic.created` 与 `topic.ready`，兼容同时创建的竞争。

任务状态返回 `accepted_at`、`started_at`、`finished_at`、`waiting_ms`、`execution_ms` 和 `end_to_end_ms`。
首次接受时间在重启和恢复后保持不变；每次 claim 使用新的 attempt 和开始时间。
`waiting_ms` 表示首次接受至当前 attempt 开始的时间，恢复时也包括此前中断 attempt 所占时间；
`execution_ms` 只衡量当前 attempt，`end_to_end_ms` 包含排队、此前执行和恢复等待。
运行中读取时长随当前时间增长。终态和完成时间在同一次结果更新中提交；写入失败时仍为可恢复的 running，不能显示成功完成。
业务失败有完成时间；正常 worker 关闭和租约丢失记录中断或恢复日志，不伪装成业务失败。
强制杀进程可能丢失未结束 span，但已提交的任务日志和首次接受时间仍保留。

## 算法阶段与异常

```python
from mindmemos_skill.logging import get_logger, stage

with stage("prepare"):
    get_logger(__name__).info("Inputs prepared")

with stage("train"):
    result = await trainer.train(...)
```

公共阶段接口创建子 span，自动记录阶段完成、失败或取消及耗时。
阶段异常事件由 OTel SDK 自动记录，在统一出口对异常信息和 traceback 脱敏及限制大小。
训练 pipeline 已覆盖 prepare、train、evaluate、publish；演进 pipeline 已覆盖 prepare、evolve、publish。
算法中被捕获后继续执行的单项错误通过普通异常日志保存诊断，不改变原有重试或跳过策略。
嵌套阶段、Service 与 worker 边界为同一异常保留一份日志 traceback，外层仍记录各自失败摘要。
并行阶段各自计时，不能把阶段耗时相加视为任务总耗时。

OTLP 导出失败无论是否有算法上下文均在终端记录 traceback，并避免将出口自身失败再次递归送往该出口。
后续发送恢复时输出 `otel.export.recovered` 和累计失败批次数；失败批次当前不提供持久化补写。
Collector 自采集接收/导出指标，即使未启用 Kafka exporter 也保留健康指标采集。
Grafana 新增原始 span 属性、父级和异常事件、输入输出与 traceback 详情，另列数据库任务数量与真实 Kafka lag。
数据库数量每 30 秒取一次聚合样本，Kafka lag 来自 broker；数量面板按 service/project 查看，不由 request/task 筛选推算。
没有最新样本时应查看最后接收时间与 Collector 健康指标，不能把没有数据当成零积压。
