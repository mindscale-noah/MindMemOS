from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments import component_selection


def test_custom_dataset_receives_only_explicit_factory_options(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_create(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(component_selection, "create", fake_create)

    dataset = component_selection.build_registered_dataset(
        benchmark="custom-benchmark",
        dataset_ref="user_dataset",
        dataset_options={"source": "fixture", "fold": 2},
        data_root=Path("ignored-for-custom-components"),
        split_dir=Path("also-ignored"),
        seed=7,
        shuffle_choices=False,
    )

    assert dataset is sentinel
    assert captured == {
        "type": component_selection.ComponentType.DATASET,
        "name": "user_dataset",
        "source": "fixture",
        "fold": 2,
    }


def test_custom_env_receives_only_explicit_factory_options() -> None:
    env_ref, options = component_selection.build_registered_env_options(
        benchmark="custom-benchmark",
        env_ref="user_env",
        env_options={"mode": "fixture"},
        max_turns=50,
        env_seed=42,
        shell_timeout=120,
        use_theorem=False,
        use_sketch=False,
    )

    assert env_ref == "user_env"
    assert options == {"mode": "fixture"}


def test_custom_benchmark_requires_explicit_dataset_ref() -> None:
    try:
        component_selection.resolve_dataset_ref(benchmark="custom-benchmark", dataset_ref=None)
    except ValueError as exc:
        assert str(exc) == "benchmark 'custom-benchmark' has no default dataset; set --dataset-ref"
    else:
        raise AssertionError("custom benchmark unexpectedly received a built-in Dataset default")
