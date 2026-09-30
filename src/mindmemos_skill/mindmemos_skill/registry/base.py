"""Shared registration, configuration validation, and construction."""

from collections.abc import Callable
from inspect import isabstract
from typing import Any, ClassVar, Generic, Self, TypeVar, cast

from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)
ImplementationT = TypeVar("ImplementationT", bound="Registrable[Any]")


class Registrable(Generic[ConfigT]):
    """Provide an isolated registry for each module family."""

    config_type: type[ConfigT]
    """Model used to validate the concrete implementation's configuration."""

    _registry: ClassVar[dict[str, type["Registrable[Any]"]]] = {}
    """Concrete classes explicitly registered under this family."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize independent storage without registering any classes."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    def __init__(self, config: ConfigT) -> None:
        """Store implementation-specific configuration.

        Args:
            config: Validated configuration for this implementation.
        """
        self.config = config

    @classmethod
    def register(cls, name: str) -> Callable[[type[ImplementationT]], type[ImplementationT]]:
        """Register a concrete subclass with an explicit, case-sensitive name.

        Args:
            name: Unique registration name within this family.

        Returns:
            A decorator that preserves the implementation class.

        Raises:
            ValueError: The name is empty, untrimmed, or already registered.
            TypeError: The class or its configuration model is invalid.
        """
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("Registration names must be nonempty and trimmed")

        def decorator(implementation: type[ImplementationT]) -> type[ImplementationT]:
            """Validate and register one concrete implementation."""
            if name in cls._registry:
                raise ValueError(f"Duplicate {cls.__name__} registration: {name!r}")
            if (
                not isinstance(implementation, type)
                or implementation is cls
                or not issubclass(implementation, cls)
                or isabstract(implementation)
            ):
                raise TypeError(f"Expected a concrete subclass of {cls.__name__}")
            try:
                config_type = implementation.config_type
            except AttributeError as exc:
                raise TypeError("The implementation must declare config_type") from exc
            if not isinstance(config_type, type) or not issubclass(config_type, BaseModel):
                raise TypeError("config_type must be a Pydantic model")
            cls._registry[name] = implementation
            return implementation

        return decorator

    @classmethod
    def name_for(cls, implementation: type) -> str:
        """Return the unambiguous registered name for a concrete class.

        Args:
            implementation: Concrete class to serialize within this family.

        Raises:
            ValueError: The class has no registration or has multiple aliases.
        """
        names = [name for name, candidate in cls._registry.items() if candidate is implementation]
        if len(names) != 1:
            raise ValueError(f"Expected one registered {cls.__name__} name for {implementation.__name__}")
        return names[0]

    @classmethod
    def resolve(cls, name: str) -> type[Self]:
        """Look up an implementation without constructing it.

        Args:
            name: Exact registration name within this family.

        Returns:
            The registered implementation class.

        Raises:
            ValueError: The name has not been registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(f"Unknown {cls.__name__} {name!r}; available: {available}")
        return cast(type[Self], cls._registry[name])

    @classmethod
    def build(cls, name: str, options: dict[str, Any], /, **dependencies: Any) -> Self:
        """Validate configuration and construct the named implementation.

        Args:
            name: Exact registration name within this family.
            options: Raw configuration fields for the selected implementation.
            **dependencies: Existing backend instances passed as constructor keywords.

        Returns:
            A configured implementation of this family.

        Raises:
            ValueError: The name is unknown or configuration is invalid.
            TypeError: Constructor dependency arguments are incompatible.
        """
        implementation = cls.resolve(name)
        config = implementation.config_type.model_validate(options)
        return implementation(config, **dependencies)
