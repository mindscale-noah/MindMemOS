from __future__ import annotations

import pytest
from mindmemos_skill.errors import (
    ConfigNotInitializedError,
    EmbeddingDimensionError,
    InvalidConfigError,
    MindMemOSSkillError,
    MindMemosSkillError,
    ModelEndpointNotConfiguredError,
    SkillCapabilityUnavailableError,
    SkillConfigurationError,
    SkillServiceClosedError,
)
from mindmemos_skill.service.errors import (
    SkillCapabilityUnavailableError as ServiceSkillCapabilityUnavailableError,
)
from mindmemos_skill.service.errors import (
    SkillServiceClosedError as ServiceSkillServiceClosedError,
)


def test_all_public_skill_errors_share_the_package_base() -> None:
    error_types = (
        ConfigNotInitializedError,
        InvalidConfigError,
        EmbeddingDimensionError,
        ModelEndpointNotConfiguredError,
        SkillConfigurationError,
        SkillCapabilityUnavailableError,
        SkillServiceClosedError,
    )

    assert MindMemosSkillError is MindMemOSSkillError
    assert all(issubclass(error_type, MindMemOSSkillError) for error_type in error_types)


def test_service_error_imports_remain_compatible() -> None:
    assert ServiceSkillCapabilityUnavailableError is SkillCapabilityUnavailableError
    assert ServiceSkillServiceClosedError is SkillServiceClosedError


def test_structured_errors_keep_operator_context() -> None:
    config_error = InvalidConfigError("llm.model", support="configured")
    dimension_error = EmbeddingDimensionError(expected=1024, actual=2560, model="qwen", task="startup.probe")

    assert config_error.field == "llm.model"
    assert config_error.support == "configured"
    assert "llm.model" in str(config_error)
    assert dimension_error.expected == 1024
    assert dimension_error.actual == 2560
    assert "startup.probe" in str(dimension_error)


def test_compatibility_errors_keep_builtin_exception_types() -> None:
    assert isinstance(SkillConfigurationError("invalid runtime"), ValueError)
    assert isinstance(ModelEndpointNotConfiguredError("chat"), RuntimeError)


@pytest.mark.parametrize(
    "error_type",
    [SkillCapabilityUnavailableError, SkillServiceClosedError],
)
def test_service_errors_are_catchable_by_package_base(error_type: type[Exception]) -> None:
    with pytest.raises(MindMemOSSkillError):
        raise error_type("test")
