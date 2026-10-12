# 数据库基础设施

`infra` 提供表注册、输入校验和统一的数据操作。业务表定义放在 `persistence`，
通过 `TableRegistry` 集中注册。SQLite 与 PostgreSQL 共用操作和校验代码；
连接、upsert 方言和向量适配位于 `builtin_backends/`。

## 安装与后端选择

在仓库根目录安装独立 Skill 包：

```sh
uv pip install -e ./src/mindmemos_skill
# 使用 PostgreSQL 时：
uv pip install -e './src/mindmemos_skill[postgres]'
```

默认 `SQLiteDatabase()` 使用 `./data/mindmemos.db`，路径相对于启动目录。
目录和连接在首次访问数据库时创建，构造对象和注册表都不会连接或建表。
SQLite 启用 WAL、外键约束，并为每条连接加载 sqlite-vec。
运行环境的 Python sqlite3 必须允许加载扩展；SQLite 需要支持 RETURNING（3.35+）。
内存数据库使用单连接，主要用于测试；同一 SQLite 实例使用协程锁串行化事务。持久化本地数据请选择文件数据库。

远程后端使用 `PostgresDatabase(PostgresConfig(url=...))`。部署时先安装
pgvector 并在目标库执行 `CREATE EXTENSION vector`；运行时只检查扩展，不安装。
SQLAlchemy 连接池按进程管理，容量为 `pool_size + max_overflow`。
读写、查询、初始化和关闭均为异步接口，必须 `await`；事务使用 `async with`。
SQLite 使用 `sqlite+aiosqlite`，PostgreSQL 使用 psycopg 原生异步驱动，不提供同步业务接口或线程适配。
建表使用 SQLAlchemy 官方 `AsyncConnection.run_sync` DDL API，其 I/O 仍由异步驱动执行，不创建工作线程。

已有服务配置通过注册名称切换后端，实例名称和业务依赖引用不变：

```yaml
service:
  home: .mindmemos/skill
  persistence:
    database:
      class_name: sqlite
      config: {}
```

PostgreSQL 使用 `class_name: postgres`，配置 `url`（`postgresql+psycopg://...`）、
`pool_size`、`max_overflow`、`pool_timeout`。不要将真实凭据提交到配置仓库。
切换 URL 不会迁移已有数据；此未上线 Skill 服务不提供旧库迁移逻辑，本地格式冲突按项目约定清理后重新初始化。

## 集中注册业务表

下面的模型和表定义应位于业务的 `persistence` 模块；这里只用示例说明契约。示例中的 `await` 在异步函数内执行。

```python
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, MetaData, String, Table
from mindmemos_skill.infra.database import SQLiteDatabase, TableRegistry, TableSpec, Vector

class ItemInput(BaseModel):
    id: str
    title: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0)
    embedding: list[float]

metadata = MetaData()
items = Table(
    "items", metadata,
    Column("id", String, primary_key=True),
    Column("title", String, nullable=False),
    Column("revision", Integer, nullable=False),
    Column("embedding", Vector(3), nullable=False),
)
tables = TableRegistry([
    TableSpec(name="items", table=items,
              insert_model=ItemInput),
])
db = SQLiteDatabase(tables=tables)
try:
    await db.create_tables()  # Explicit initialization; not schema migration.
    await db.insert("items", [{"id": "a", "title": "Example", "embedding": [1., 0., 0.]}])
    await db.update_many("items", [{"id": "a", "revision": 1}])
    hits = await db.search("items", vector_field="embedding", query_vector=[1., 0., 0.], top_k=5)
finally:
    await db.close()
```

也可以先通过服务配置创建数据库，再由业务组装代码集中调用
`database.tables.register(spec)`。注册表是实例级对象，不是全局单例。
使用同一数据库实例的业务模块共用同一注册表；应在处理请求前完成注册。

表对象描述数据库 schema，Pydantic 模型描述输入约束，两者显式绑定，
不自动从任意模型推导数据库结构。注册时检查字段名称、主键和必需列是否存在；
不声称能静态证明任意自定义验证器与数据库约束等价。
模型字段使用列 key，不支持别名；业务若需别名，应在 persistence 转换。

每张表只绑定一个完整记录模型，不另设更新模型或字段修改白名单。
部分更新复用完整记录模型中的字段类型和 `Field` 约束，只校验和写入显式提供的字段，
不填默认值，也不运行模型上的 `field_validator` 或 `model_validator`。
模型未声明的数据库列仍按列类型校验。哪些字段允许修改、完整记录校验和历史保护由业务代码负责；
基础设施允许更新所有列，包括主键，实际写入仍须满足数据库约束。
输入严格校验类型、拒绝未知字段，并检查非空、向量维度、float32 有限值和非零向量。
模型约束在数据库执行前校验；主键、唯一约束、外键等仍由数据库保证。

## 操作契约

| 方法 | 语义 / 返回值 |
| --- | --- |
| `insert(table, records)` | 插入列表，唯一冲突报错；返回插入条数 |
| `upsert(table, records, conflict_fields=..., update_fields=...)` | 原子新增或更新；冲突键默认为主键；只更新显式列；返回受影响条数 |
| `update(table, values, filters=...)` | 同一组变更应用到匹配行；返回匹配条数 |
| `update_many(table, records)` | 每条记录包含完整主键和各自变更；返回累计匹配条数 |
| `delete(table, filters=...)` | 删除匹配行；返回删除条数 |
| `get(table, key)` | 按主键读取；缺失返回 `None`，复合主键用字典 |
| `select(table, ...)` | 返回记录字典列表；默认上限 100 |
| `count(table, filters=...)` | 返回满足条件的行数 |
| `search(table, ...)` | 返回 `SearchHit(record, distance)` 列表 |
| `transaction()` | 提供绑定同一连接的操作句柄 |
| `create_tables()` | 按外键依赖顺序创建尚不存在的注册表，不迁移既有表 |
| `close()` | 关闭并释放池；此实例不再接受操作 |

`insert` 和 `upsert` 接受完整新增数据。upsert 的冲突字段必须是已声明的主键或
`UniqueConstraint`；暂不支持表达式索引、部分唯一索引作为冲突目标。
每条记录必须包含冲突字段和要更新的字段；`update_fields` 为空时为冲突忽略，
不会覆盖旧值。upsert 输入按完整新增模型校验，字段修改权限由业务代码控制。
数据库生成主键时，可在新增模型中省略该字段，或使用 `id: int | None = None`；
未提供的生成字段不会写入 NULL，显式 NULL 则仍按数据库非空约束拒绝。

`update` 和 `update_many` 不插入不存在的记录。`update` 用 filters 定位，可显式修改主键；
`update_many` 中的主键只用于定位，修改主键时使用 `update`。
未提供字段保持原值；显式 `None` 写入 NULL，但必须符合数据库可空约束。
`update_many` 顺序处理，同一主键出现多次时按输入顺序生效，计数也按操作累计。
批量更新当前在同一事务中逐条执行，以正确处理不同变更字段及精确返回匹配数。

修改和删除需要显式提供 `filters`；`{}` 明确表示全部记录。
查询可传等值字典，或 `Filter` 列表（AND 组合）：

```python
from mindmemos_skill.infra.database import Filter

rows = await db.select(
    "items",
    filters=[Filter(field="revision", op="ge", value=1)],
    fields=["id", "title"], order_by=["-revision"], limit=20, offset=0,
)
```

运算符支持 `eq/ne/gt/ge/lt/le/in/is_null/not_null`，标识符先匹配注册字段，
值通过参数绑定传给数据库。过滤值按列的 Python 类型校验；不套用新增模型的业务范围限制。
空 IN 列表不匹配任何行；NULL 用专门的检查或 `eq/ne`。
普通查询按指定字段排序，并补充主键消除平局；没有指定时按主键排序。

向量与普通字段共用增删改。SQLite 将 float32 向量保存在普通 BLOB 列，
通过 sqlite-vec 距离函数精确检索；PostgreSQL 使用 pgvector 的 vector 列及距离运算符。
`search` 先过滤，再对非 NULL 向量按 cosine distance 排序并取 top-k；平局按主键排序。
距离越小越接近，允许浮点误差。第一版不创建 ANN 索引，不承诺大规模扫描性能。
不要给要求精确检索的表额外配置近似索引，否则 PostgreSQL 可能选择近似执行计划。
原生 SQL 绕过输入校验，不属于这些接口的数据合法性保证范围。

## 事务和错误

```python
async with db.transaction() as tx:
    await tx.insert("items", records)
    await tx.update("items", {"revision": 2}, filters={"id": "a"})
```

独立操作自动提交或回滚。显式事务中的操作共用一个连接；每次操作使用保存点，
即使调用者捕获其中一次批量操作的数据库异常，也不会保留该批次的部分写入。
事务块异常退出时全部回滚；退出后 `tx` 失效。不支持嵌套 `transaction()`。
同一 `tx` 不在并发协程间共享；数据库生命周期属于同一事件循环，服务完成所有操作后调用 `await db.close()`。

未注册表、未知字段、输入不合法等抛出 `ValueError`（包括 Pydantic 验证错误）。
批量输入错误标明表名、从 0 开始的记录序号和字段原因。
数据库约束/连接异常保留 SQLAlchemy 异常，不伪造成功或默认自动重试。

## 验证

```sh
.venv/bin/python -m pytest src/mindmemos_skill/tests/test_infra.py -q
```

测试默认实际运行 SQLite。提供 `MINDMEMOS_TEST_POSTGRES_URL` 后，同一套行为测试
也会运行 PostgreSQL；目标必须是可用于测试的数据库并预装 vector。
测试使用随机表名并清理自身创建的表。没有配置时 PostgreSQL 行为测试明确跳过；
SQL 编译和编解码测试不能替代真实 PostgreSQL 测试。
