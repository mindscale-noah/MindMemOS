# 算法运行追踪后端

`mindmemos_skill` 的训练始终产生 OpenTelemetry span。算法与 Agent 通过包级 Python `logging` 记录事件，运行上下文内的日志由桥接 handler 转成关联当前 span 的 OTel log。服务可选配一个独立的可观测性数据库，将 OTel 数据持久化；当前支持已有的 `sqlite`、`postgres` 数据库实现。省略配置时仍执行相同的打点，但不创建或写入审计数据库。业务 Skill 表不保存调用正文。

分层边界：`logging` 保留普通控制台/文件输出，并用运行范围内的 handler 将包级 Python 日志转发到 OTel；其控制台级别与审计采集级别分别控制。`infra.observability` 创建隔离的 OTel tracer/logger、接收结束的 span 和发出的 log，并把 OTel 数据转换为可持久化字段；不定义算法、Agent、run 或业务表。`persistence.observability.AlgorithmAuditPersistence` 在配置数据库时注册审计表，把 OTel 输出接到数据库，管理 run 状态与查询。业务代码用 `get_logger(__name__)` 记录事件，用运行 session 创建 span；Agent 的 info 日志记录调用状态与 usage，debug 日志记录经过脱敏和截断的输入输出。

```yaml
observability:
  backend:
    class_name: postgres
    config:
      url: postgresql+psycopg://USER:PASSWORD@HOST:5432/observability
  capture_level: info  # 改为 debug 才记录 Agent 输入输出
```

配置位于服务顶层，与 `persistence` 并列。`ServiceConfig` 只接受一个 `observability.backend`。本地验证可以使用 `class_name: sqlite` 和单独的 SQLite URL。PostgreSQL 复用项目现有数据库实现，因此目标实例目前也需要预装 `vector` 扩展。凭据应放在本地配置或环境变量中，不提交真实 URL。

配置数据库时，`create_service()` 会在启动时创建缺失的 `observability_runs`、`observability_spans`、`observability_logs` 表；它不会修改已有表。历史训练不会被伪造为完整 trace，也不需要重建 Skill 数据。所有 `auto_train_skill` 执行都生成根 trace 与运行状态日志；配置数据库后，恢复执行沿用 `run_id`，新 trace 通过 OTel link 关联旧根 span。Skill-Forest 的树扩展、候选节点和最终集成各有阶段 span；`OpenAIChatAgent.run()` 产生调用子 span 和含模型、状态、usage 的 info 日志。TreeSkill、Skill-Forest 以及训练后评估中的内置线程执行会传递 trace 上下文；自定义线程池可使用 `map_with_context()`。运行中的普通包级日志也会进入 OTel，作为 `python.log` 保存原消息。Info Agent 事件不保存输入输出正文；Debug 另发关联日志，并对 Agent 输入输出中的常见凭据键脱敏、限制单个正文长度。其他普通日志的消息由调用方负责避免敏感内容；调试输入仍可能含有任务正文，须限制该数据库的读取权限。

读取入口为 `service.algorithm_run(project_id, run_id)`、`service.algorithm_spans(project_id, run_id, limit=..., offset=...)` 和 `service.algorithm_logs(...)`。查询强制经过项目作用域。当前不暴露 HTTP 审计端点或 UI 页面。

普通同步或异步函数可直接加 `@log_function`，无需手写起止日志：

```python
from mindmemos_skill.logging import log_function

@log_function
async def score_candidate(candidate, task):
    return await evaluate(candidate, task)
```

装饰器生成 `function.call` 事件，包含函数名、`ok`、`duration_ms`；异常时还包含错误类型和截断到 32 KiB 的 traceback，并原样抛出异常。输入和输出分别作为 `function.input`、`function.output` 的 Debug 事件，按现有规则对常见凭据键脱敏、限制长度；Info 采集等级不入库输入输出。函数在训练运行上下文内调用时，日志自动关联当前 trace/span 并由可选数据库保存。生成器及异步生成器在装饰时会被拒绝，因为仅创建生成器对象无法代表实际执行耗时。

发布 `skill_forest` 训练结果时，后端从已校验的 `forest.json` 和 `ensemble/winners.json` 构造紧凑报告，随版本事务写入 `skill_versions.metadata.algorithm_report`。报告含 `schema_version`、`run_id`、总树数、总节点数、扩展次数、集成策略，以及每棵树的节点数、winner 的 `gain`、验证均分和 `train_top_select_score`（冻结 winner 在完整训练池上的选择分）。选择策略还会记录最终选中的节点 ID。普通版本读取接口会原样返回该 JSON，供后续 UI 展示；未实现报告的训练器不写该键。

配置数据库后，结束的 span 和发出的 log 同步写库，写入异常会把 run 的 `audit_status` 标为 `incomplete`。这个状态只表明本进程检测到的失败；它不提供跨进程崩溃后的完整审计保证，也不阻止 Skill 发布。未配置数据库时，OTel 事件由本地 provider 处理但不持久化，也没有跨进程查询能力。发布门禁、持久化补写、更细粒度的算法阶段 span 和报告 UI 属于后续接入工作。外部 provider 内部重试目前计入一次逻辑 Agent 调用；需要精确的物理请求次数时，应再在传输层打点。
