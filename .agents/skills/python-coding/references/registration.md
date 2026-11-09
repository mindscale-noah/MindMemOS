# 统一注册机制

## 适用范围与约定

用于需要按名称选择实现的业务模块。仅在任务涉及注册机制时读取；不为没有注册需求的模块添加注册层。

- **业务基类直接使用业务名称**，例如 `Agent`、`Algorithm`、`Environment`；不强制添加 `Base` 前缀或 `Module` 后缀。用户通过 `Algorithm.register(...)` 清楚感知注册所属的业务类型。
- **注册逻辑只实现一次**：使用共享父类 `Registrable` 提供 `register()` 和 `build()`。业务基类继承它并定义业务接口，不复制注册、查找或构建逻辑。共享注册父类属于行为抽象，使用普通 Python 类；配置数据使用 Pydantic 模型。
- **内置与用户实现统一使用装饰器注册**：在实现类上使用 `@Algorithm.register("name")`。对应业务模块的 `registry.py` 集中维护显式的内置模块加载清单，提供 `ensure_registered()` 导入清单中的全部模块，从而执行装饰器。新增内置实现必须同步更新清单，不依赖自动发现。
- **内置加载由包入口自动完成**：`__init__.py` 先导入业务基类，再调用 `ensure_registered()`。用户正常导入包后即可调用 `Algorithm.build(...)`，无须显式调用注册或加载函数。`ensure_registered()` 保持内部使用，不加入包的 `__all__`。
- **用户扩展通过装饰器注册**：使用 `@Algorithm.register("name")`。用户入口必须显式导入扩展模块，装饰器才会执行；不宣称装饰器能自动发现未导入的模块，不默认扫描目录。
- **所有实现共用同一个注册函数**。示例保留直接调用能力，但内置与用户实现的标准写法均为装饰器。注册对象必须是对应业务基类的具体子类。每个业务基类持有独立注册表，同名实现可以出现在不同业务类型下，不能跨类型误注册。
- **注册名称显式指定**，使用稳定的 `snake_case`。空名称、重复名称和未知名称明确报错，不静默覆盖或回退。示例检查非空及首尾空格，`snake_case` 由命名约定保证。
- **各子类可以使用不同配置**。统一通过 `build(name, options)` 查找实现，用该实现的配置模型校验输入，再调用 `Implementation(config)`。参数约束集中在配置类内部。注册保存实现类，不在注册阶段创建实例或启动外部资源。
- 不引入元类自动注册、目录扫描或通用插件框架；仅在真实需求要求时扩展当前机制。

## 配置声明与示例边界

以下保留已验证的显式双声明示例，不增加类型注解反射：

- `Algorithm[BeamConfig]` 为编辑器及类型检查工具声明 `self.config` 的类型。
- `config_type = BeamConfig` 为运行时解析与校验指定配置模型。
- 两处引用同一个配置类，必须保持一致；示例不额外解析泛型继承信息来检查它们是否一致。

所有实现遵循 `Implementation(config)` 的构造约定。`build()` 执行配置校验；直接调用构造器时，应传入已校验且匹配的配置实例。

注册表在每个子类定义时独立初始化，继承不会自动复制父类注册项。`__init_subclass__` 只初始化存储，不执行自动注册。

`ensure_registered()` 使用普通导入，依赖 Python 模块缓存；重复调用不会重新执行已成功导入模块的装饰器。不要使用 `reload()`，也不需要额外的 `_registered` 标志或在 `build()` 中再次懒加载。该函数不负责重建被人为清空的注册表。

内置实现从 `base.py` 导入业务基类，避免反向导入尚未初始化完的包顶层接口。加载失败必须暴露原始异常，不能吞掉异常后声称注册完成。包导入会加载全部内置实现，因此模块顶层及装饰器只定义和登记类；模型权重加载、服务连接等重资源操作放在实例构造或执行阶段。

## 可运行示例

依赖 Python 3.11+ 和 Pydantic 2。按以下文件名保存代码，在 `registration_demo` 的父目录执行：

```sh
python -m registration_demo
```

示例为了便于阅读，将业务基类与共享父类一起放在 `base.py`，两个配置模型一起放在 `config.py`。实际项目中，共享父类放在公共模块，各实现按功能归入自己的子模块，并遵循各级 `config.py` / `typing.py` 命名约定，不为没有额外数据结构的模块创建空 `typing.py`。

### `registration_demo/base.py`

```python
"""Shared registration and construction behavior."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from inspect import isabstract
from typing import Any, ClassVar, Generic, Self, TypeVar, cast, overload

from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)
ImplementationT = TypeVar("ImplementationT", bound="Registrable[Any]")


class Registrable(Generic[ConfigT]):
    """Provide isolated registries and config-aware construction."""

    config_type: type[ConfigT]
    """Configuration model accepted by this implementation."""

    _registry: ClassVar[dict[str, type["Registrable[Any]"]]]
    """Implementations registered explicitly under this class."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Give each subclass its own initially empty registry."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    def __init__(self, config: ConfigT) -> None:
        """Store the validated implementation-specific configuration."""
        self.config = config

    @overload
    @classmethod
    def register(cls, name: str) -> Callable[[type[ImplementationT]], type[ImplementationT]]: ...

    @overload
    @classmethod
    def register(cls, name: str, implementation: type[ImplementationT]) -> type[ImplementationT]: ...

    @classmethod
    def register(
        cls,
        name: str,
        implementation: type[ImplementationT] | None = None,
    ) -> type[ImplementationT] | Callable[[type[ImplementationT]], type[ImplementationT]]:
        """Register an implementation directly or as a decorator.

        Args:
            name: Unique implementation name within this class's registry.
            implementation: Concrete subclass; None returns a decorator.

        Returns:
            The unchanged implementation class, or its registration decorator.

        Raises:
            ValueError: The name is empty or already registered.
            TypeError: The implementation has an invalid base or config model.
        """
        def _register(candidate: type[ImplementationT]) -> type[ImplementationT]:
            """Validate and store one concrete implementation."""
            if not name or name != name.strip():
                raise ValueError("Registration names must be nonempty and trimmed")
            if name in cls._registry:
                raise ValueError(f"Duplicate {cls.__name__} registration: {name!r}")
            if not isinstance(candidate, type) or not issubclass(candidate, cls) or isabstract(candidate):
                raise TypeError(f"Expected a concrete subclass of {cls.__name__}")
            try:
                config_type = candidate.config_type
            except AttributeError as exc:
                raise TypeError("config_type must be declared on the implementation") from exc
            if not isinstance(config_type, type) or not issubclass(config_type, BaseModel):
                raise TypeError("config_type must be a BaseModel subclass")
            cls._registry[name] = candidate
            return candidate

        return _register if implementation is None else _register(implementation)

    @classmethod
    def build(cls, name: str, options: dict[str, Any]) -> Self:
        """Validate configuration and create the selected implementation.

        Args:
            name: Implementation name registered under this class.
            options: Raw fields for the selected implementation's config model.

        Returns:
            A new instance belonging to this module family.

        Raises:
            ValueError: The name is unknown.
            pydantic.ValidationError: The configuration is invalid.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(f"Unknown {cls.__name__} {name!r}; available: {available}")
        implementation = cls._registry[name]
        config = implementation.config_type.model_validate(options)
        return cast(Self, implementation(config))


class Algorithm(Registrable[ConfigT], ABC):
    """Define the common interface for algorithm implementations."""

    @abstractmethod
    def run(self) -> str:
        """Run the algorithm and return its demonstration result."""
        raise NotImplementedError
```

### `registration_demo/config.py`

```python
"""Configuration models used by the registration demo."""

from pydantic import BaseModel, ConfigDict, Field


class GreedyConfig(BaseModel):
    """Configure the built-in greedy algorithm."""

    model_config = ConfigDict(extra="forbid")
    """Reject fields not declared by this configuration."""

    max_steps: int = Field(default=10, gt=0)
    """Maximum number of search steps."""


class BeamConfig(BaseModel):
    """Configure the user-defined beam search algorithm."""

    model_config = ConfigDict(extra="forbid")
    """Reject fields not declared by this configuration."""

    beam_width: int = Field(default=4, gt=0)
    """Number of candidate branches retained at each search step."""
```

### `registration_demo/builtin.py`

```python
"""Built-in implementations registered through decorators."""

from .base import Algorithm
from .config import GreedyConfig


@Algorithm.register("greedy")
class GreedyAlgorithm(Algorithm[GreedyConfig]):
    """Demonstrate a built-in algorithm with its own config type."""

    config_type = GreedyConfig
    """Configuration model used to validate greedy search options."""

    def run(self) -> str:
        """Return the configured greedy search step limit."""
        return f"greedy: max_steps={self.config.max_steps}"
```

### `registration_demo/registry.py`

```python
"""Load the explicit built-in catalog to execute registration decorators."""

from importlib import import_module


_BUILTIN_MODULES = (".builtin",)
"""Built-in modules whose imports execute their registration decorators."""


def ensure_registered() -> None:
    """Import all built-in implementations to populate the registry."""
    for module_name in _BUILTIN_MODULES:
        import_module(module_name, package=__package__)
```

### `registration_demo/__init__.py`

```python
"""Expose module families after loading the explicit built-in catalog."""

from .base import Algorithm
from .registry import ensure_registered

ensure_registered()

__all__ = ["Algorithm"]
```

### `registration_demo/custom.py`

```python
"""Demonstrate an explicitly imported user extension."""

from . import Algorithm
from .config import BeamConfig


@Algorithm.register("beam")
class BeamAlgorithm(Algorithm[BeamConfig]):
    """Demonstrate a custom algorithm with a different config type."""

    config_type = BeamConfig
    """Configuration model used to validate beam search options."""

    def run(self) -> str:
        """Return the configured beam search width."""
        return f"beam: beam_width={self.config.beam_width}"
```

### `registration_demo/__main__.py`

```python
"""Run both built-in and user-defined implementations."""

from . import Algorithm
from . import custom

print(Algorithm.build("greedy", {"max_steps": 20}).run())
print(Algorithm.build("beam", {"beam_width": 8}).run())
```

预期输出：

```text
greedy: max_steps=20
beam: beam_width=8
```

## 验证重点

验证不同实现的配置解析与构造、重复名称不覆盖原实现、错误父类被拒绝、不同业务注册表隔离、未知名称报错、配置非法值和多余字段被拒绝，以及抽象实现不能注册。内置加载还应在新进程中确认：只导入包即可构建内置实现，无需手动导入对应实现文件或调用 `ensure_registered()`；还要确认重复加载不会重复注册，导入失败不会被静默忽略。
