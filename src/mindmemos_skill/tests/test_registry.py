"""Registration isolation, validation, and construction contracts."""

from abc import ABC, abstractmethod

import pytest
from mindmemos_skill.registry import Registrable
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Options(BaseModel):
    """Configuration accepted by test implementations."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown fields."""

    count: int = Field(default=1, gt=0)
    """Positive count used to verify configuration validation."""


def test_family_isolation_and_typed_configuration():
    """The same name resolves independently for different module families."""

    class First(Registrable[Options]):
        pass

    class Second(Registrable[Options]):
        pass

    @First.register("Example")
    class FirstImplementation(First):
        config_type = Options

    @Second.register("Example")
    class SecondImplementation(Second):
        config_type = Options

    first = First.build("Example", {"count": 3})
    assert isinstance(first, FirstImplementation)
    assert first.config.count == 3
    assert isinstance(Second.build("Example", {}), SecondImplementation)
    with pytest.raises(ValueError, match="Duplicate"):
        First.register("Example")(FirstImplementation)
    assert First.resolve("Example") is FirstImplementation
    with pytest.raises(ValueError, match="Unknown"):
        First.build("missing", {})
    for options in ({"count": 0}, {"extra": True}):
        with pytest.raises(ValidationError):
            First.build("Example", options)


@pytest.mark.parametrize("name", ["", " ", " Example", "Example ", None])
def test_invalid_registration_names(name):
    """Reject ambiguous registration names before decoration."""
    with pytest.raises(ValueError):
        Registrable.register(name)


def test_reject_invalid_implementations():
    """Reject unrelated, abstract, and unconfigured implementation classes."""

    class Family(Registrable[Options]):
        pass

    class Abstract(Family, ABC):
        config_type = Options

        @abstractmethod
        async def run(self):
            pass

    class MissingConfig(Family):
        pass

    class InvalidConfig(Family):
        config_type = dict

    for candidate in (Family, object, Abstract, MissingConfig, InvalidConfig, 123):
        with pytest.raises(TypeError):
            Family.register("invalid")(candidate)


def test_each_implementation_can_use_its_own_config():
    """A family does not force all implementations to share config fields."""

    class Family(Registrable):
        pass

    class TextOptions(BaseModel):
        label: str

    @Family.register("numeric")
    class Numeric(Family):
        config_type = Options

    @Family.register("text")
    class Text(Family):
        config_type = TextOptions

    assert Family.build("numeric", {"count": 2}).config.count == 2
    assert Family.build("text", {"label": "hello"}).config.label == "hello"


def test_dependency_names_do_not_collide_with_builder_arguments():
    """Allow constructor dependency keywords called name or options."""

    class Family(Registrable[Options]):
        pass

    @Family.register("Example")
    class Example(Family):
        config_type = Options

        def __init__(self, config, *, name, options):
            super().__init__(config)
            self.name = name
            self.options = options

    dependency = object()
    instance = Family.build("Example", {}, name=dependency, options=dependency)
    assert instance.name is dependency
    assert instance.options is dependency
