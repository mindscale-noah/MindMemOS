from __future__ import annotations

from mindmemos_skill.datasets import (
    ALFWorldPathSplitDataset,
    LiveMathIdSplitDataset,
    SpreadsheetBenchIdSplitDataset,
    TaskDataset,
)
from mindmemos_skill.registry import ComponentType, get_component, list_components


def test_builtin_datasets_use_the_unified_registry() -> None:
    assert list_components(type=ComponentType.DATASET) == {
        "dataset": ["alfworld_path_split", "livemath_id_split", "spreadsheetbench_id_split"]
    }
    assert get_component(type=ComponentType.DATASET, name="alfworld_path_split").factory is ALFWorldPathSplitDataset
    assert get_component(type=ComponentType.DATASET, name="livemath_id_split").factory is LiveMathIdSplitDataset
    assert (
        get_component(type=ComponentType.DATASET, name="spreadsheetbench_id_split").factory
        is SpreadsheetBenchIdSplitDataset
    )


def test_builtin_datasets_live_in_independent_registered_dataset_packages() -> None:
    assert ALFWorldPathSplitDataset.__module__ == (
        "mindmemos_skill.datasets.registered_datasets.alfworld.dataset"
    )
    assert LiveMathIdSplitDataset.__module__ == (
        "mindmemos_skill.datasets.registered_datasets.livemath.dataset"
    )
    assert SpreadsheetBenchIdSplitDataset.__module__ == (
        "mindmemos_skill.datasets.registered_datasets.spreadsheetbench.dataset"
    )
    assert TaskDataset.__module__ == "mindmemos_skill.datasets.base"
