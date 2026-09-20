"""Backend registration, construction, and capability validation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any

from .models import BackendConfig, TableSpec
from .vector_store import ScopedVectorStore


class TableRegistry:
    """Immutable-after-bootstrap registry for an arbitrary number of tables."""

    def __init__(self, specs: Iterable[TableSpec] = ()) -> None:
        self._specs: dict[str, TableSpec] = {}
        self._frozen = False
        for spec in specs:
            self.register(spec)

    def register(self, spec: TableSpec) -> None:
        if self._frozen:
            raise RuntimeError("table registry is frozen")
        if spec.name in self._specs:
            raise ValueError(f"table {spec.name!r} is already registered")
        index_owners = {index.name: table.name for table in self._specs.values() for index in table.indexes}
        for index in spec.indexes:
            owner = index_owners.get(index.name)
            if owner is not None:
                raise ValueError(
                    f"index {index.name!r} is already registered for table {owner!r}; "
                    f"table {spec.name!r} cannot reuse it"
                )
        self._specs[spec.name] = spec

    def get(self, name: str) -> TableSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown logical table {name!r}") from exc

    def freeze(self) -> Mapping[str, TableSpec]:
        self._frozen = True
        return MappingProxyType(self._specs)

    @property
    def specs(self) -> tuple[TableSpec, ...]:
        return tuple(self._specs.values())


BackendFactory = Callable[[Mapping[str, Any], TableRegistry], ScopedVectorStore]


class BackendRegistry:
    """Explicit adapter registry; no business layer imports concrete drivers."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("backend name must not be empty")
        if normalized in self._factories:
            raise ValueError(f"backend {normalized!r} is already registered")
        self._factories[normalized] = factory

    def create(self, config: BackendConfig, tables: TableRegistry) -> ScopedVectorStore:
        name = config.provider.strip().lower()
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ValueError(f"unsupported vector backend {config.provider!r}") from exc
        backend = factory(MappingProxyType(dict(config.options)), tables)
        missing = config.required.missing_from(backend.capabilities)
        if missing:
            raise ValueError(f"backend {name!r} is missing required capabilities: {', '.join(missing)}")
        return backend

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
