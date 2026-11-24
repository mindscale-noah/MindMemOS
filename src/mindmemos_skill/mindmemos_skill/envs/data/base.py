"""Load dataset samples and create reproducible, disjoint data splits."""

from abc import ABC, abstractmethod
from math import floor, fsum
from pathlib import Path
from random import Random

from pydantic import TypeAdapter

from ...registry.base import ConfigT, Registrable
from .config import SplitConfig
from .typing import DataSample, SplitManifest


class Data(Registrable[ConfigT], ABC):
    """Share loading validation and splitting across environment datasets."""

    def load(self) -> list[DataSample]:
        """Load all samples and validate their structure and unique task IDs.

        Returns:
            Samples in source order, preserving concrete task and reward-data objects.

        Raises:
            ValueError: Samples are invalid or contain duplicate task IDs.
            OSError: The implementation cannot read its data source.
        """
        return list(self._index(self._load()).values())

    @abstractmethod
    def _load(self) -> list[DataSample]:
        """Read the configured source and construct typed task and scoring inputs."""

    def split(
        self,
        samples: list[DataSample],
        *,
        ratios: dict[str, float] | None = None,
        seed: int | None = None,
        split_file: str | Path | None = None,
    ) -> dict[str, list[DataSample]]:
        """Partition samples randomly or select membership from a JSON file.

        Args:
            samples: Loaded samples with unique, stable task IDs.
            ratios: Named nonnegative proportions summing to one in random mode.
            seed: Explicit random seed; does not affect the global random state.
            split_file: JSON mapping split names to ordered IDs; mutually exclusive
                with ratios and seed. Unlisted samples are omitted.

        Returns:
            Disjoint sample lists keyed by split name. Random mode covers all samples.

        Raises:
            ValueError: Options, sample IDs, or manifest membership are invalid.
            OSError: The split file cannot be read.
        """
        options = SplitConfig(ratios=ratios, seed=seed, split_file=split_file)
        index = self._index(samples)
        if options.split_file is not None:
            manifest = SplitManifest.model_validate_json(options.split_file.expanduser().read_text(encoding="utf-8"))
            unknown = {identifier for ids in manifest.root.values() for identifier in ids} - index.keys()
            if unknown:
                raise ValueError(f"Unknown task IDs in split file: {sorted(unknown)}")
            return {name: [index[identifier] for identifier in ids] for name, ids in manifest.root.items()}

        assert options.ratios is not None and options.seed is not None
        ordered = [index[identifier] for identifier in sorted(index)]
        Random(options.seed).shuffle(ordered)
        names = sorted(options.ratios)
        total = fsum(options.ratios.values())
        quotas = {name: len(ordered) * (options.ratios[name] / total) for name in names}
        counts = {name: floor(quotas[name]) for name in names}
        remainder = len(ordered) - sum(counts.values())
        priority = sorted(names, key=lambda name: (-(quotas[name] - counts[name]), name))
        for name in priority[:remainder]:
            counts[name] += 1
        splits: dict[str, list[DataSample]] = {}
        offset = 0
        for name in names:
            splits[name] = ordered[offset : offset + counts[name]]
            offset += counts[name]
        return splits

    def save_splits(self, splits: dict[str, list[DataSample]], path: str | Path) -> None:
        """Save split names and IDs without copying tasks or private evaluation data.

        Args:
            splits: Disjoint named sample lists, optionally covering only a subset.
            path: Destination JSON file; parent directories are created as needed.

        Raises:
            ValueError: Samples or membership are invalid.
            OSError: The destination cannot be written.
        """
        manifest = SplitManifest({name: list(self._index(samples)) for name, samples in splits.items()})
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _index(samples: list[DataSample]) -> dict[str, DataSample]:
        """Validate samples and index them without silently overwriting duplicate IDs."""
        samples = TypeAdapter(list[DataSample]).validate_python(samples)
        index: dict[str, DataSample] = {}
        for sample in samples:
            if sample.task.id in index:
                raise ValueError(f"Duplicate task ID: {sample.task.id!r}")
            index[sample.task.id] = sample
        return index
