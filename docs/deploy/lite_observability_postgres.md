# Lite observability 使用 PostgreSQL

`config/mindmemos_lite/dev.example.yaml` 已选择 `observability.exporter: postgres`。
该后端保存 Trace、Span、Span 事件和 LLM 调用统计，使用普通 PostgreSQL 表，不需要 pgvector 扩展。
它不改变应用普通文本日志的输出方式，也不修改 SDK Trace 界面。

## 与记忆库共用 PostgreSQL

示例配置的日志 DSN 默认引用 `database.pgvector.dsn`，日志 schema 默认为 `observability`，记忆 schema 默认为 `mindmemos`。
只需按原方式设置记忆数据库的 `PGVECTOR_DSN`。日志和记忆分别管理连接池；关闭日志存储不会关闭记忆存储。

```yaml
observability:
  enabled: true
  exporter: postgres
  postgres:
    dsn: ${database.pgvector.dsn}
    schema: observability
```

## 使用两个 PostgreSQL 数据库或服务器

设置环境变量 `MINDMEMOS_TRACE_POSTGRES_DSN` 即可覆盖示例配置的日志连接地址，`PGVECTOR_DSN` 继续用于记忆存储。
也可以直接在私有 YAML 中分别配置 `observability.postgres.dsn` 和 `database.pgvector.dsn`。
`MINDMEMOS_TRACE_POSTGRES_SCHEMA` 可单独覆盖日志 schema。

```yaml
observability:
  enabled: true
  exporter: postgres
  postgres:
    dsn: ${oc.env:MINDMEMOS_TRACE_POSTGRES_DSN}
    schema: observability

database:
  pgvector:
    dsn: ${oc.env:PGVECTOR_DSN}
    schema: mindmemos
```

以上均为配置片段，需合并到完整配置。连接串在配置快照中按秘密字段脱敏。
建议同库时也分开 schema；schema 本身不隔离 CPU、磁盘或实例故障。

## 运行行为

- 服务启动时连接日志数据库并初始化四张表、索引及外键。数据库必须事先存在；账号需要建表和读写权限。`create_schema: false` 可用于管理员已建好 schema 的部署。
- 每批写入使用一个事务和 PostgreSQL pipeline；重复的 `(trace_id, span_id)` 更新原 Span，并替换其事件和 LLM 统计。
- 连接池默认最多 2 个连接。连接、取连接和单条 SQL 超时分别由 `connect_timeout_seconds`、`pool_timeout_seconds` 和 `statement_timeout_seconds` 控制，默认均为 5 秒。单条 SQL 超时不是整批总超时。
- `retention_days` 保留现有语义：启动时按 Trace 的结束时间清理，级联删除其 Span、事件和 LLM 统计；`null` 禁用清理。此实现没有后台定时清理。
- 初始化连接失败会使启动失败；运行中导出失败由现有 exporter 记录错误，不把异常抛回业务。队列仍在内存中，没有持久化重试保证。
- SQLite 仍可通过 `exporter: sqlite` 和 `sqlite_path` 使用；配置类保留 SQLite 作为无配置默认值。历史 SQLite 数据不会自动搬迁。

## 验证

使用基于 Lite 包依赖建立的测试环境，设置指向可建删测试 schema 的专用 PostgreSQL 数据库的 `MINDMEMOS_TEST_OBSERVABILITY_POSTGRES_DSN`，运行：

```bash
pytest tests/mindmemos_lite/test_postgres_observability.py \
       tests/mindmemos_lite/test_postgres_observability_integration.py -q
```

集成测试创建唯一临时 schema，并在结束后删除；不需要 pgvector。
