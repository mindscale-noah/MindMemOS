# 基础设施

`infra` 按能力分组。根目录只维护公共注册基类和内置加载入口，不放数据库或执行器的具体实现。

```text
infra/
├── base.py                  # Infrastructure 注册基类
├── registry.py              # 内置实现加载清单
├── database/
│   ├── base.py              # Database 抽象接口与公共 CRUD
│   ├── config.py            # SQLite / PostgreSQL 配置
│   ├── typing.py            # 表声明、过滤条件、检索结果
│   ├── tables.py            # 表注册
│   ├── validation.py        # 数据库输入校验
│   ├── vector.py            # 向量类型
│   └── builtin_backends/    # SQLite / PostgreSQL 适配
└── execution/
    ├── config.py            # Python / uv、shell 和超时配置
    ├── constants.py         # 执行相关环境变量名
    └── executor.py          # 跨平台进程启动与清理
```

使用能力模块的公开入口：

```python
from mindmemos_skill.infra import Infrastructure
from mindmemos_skill.infra.database import Database, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.infra.execution import ExecutionConfig, execute
```

`Infrastructure` 保留服务组装用的统一注册入口，SQLite/PostgreSQL 注册名及 YAML 格式不变。`execution` 是无业务依赖的函数能力，不为目录一致性额外包装为注册后端。目录分组不改变公共基础类与服务配置的职责。

业务表与业务存取服务仍归 `persistence`，环境工具和 reward 仍归 `envs`。数据库用法见 [database/README.md](database/README.md)；执行配置示例见 [环境文档](../../../../../docs/mindmemos_skill/benchmark_environments.md)。
