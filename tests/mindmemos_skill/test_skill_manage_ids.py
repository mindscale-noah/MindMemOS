"""Frozen physical/generation identity formulas (AC-P3-004, P0 test vectors)."""

from __future__ import annotations

from mindmemos_skill.infra.skill_manage import (
    generation_id_from_operation,
    physical_record_id,
)


def test_physical_record_id_matches_p0_frozen_vector() -> None:
    record_id = physical_record_id("gen-001", "node", "skill:example:v1")

    assert record_id == "4ac4c863-f213-511e-8841-112e9bb400cf"


def test_generation_id_matches_p0_frozen_vector() -> None:
    generation_id = generation_id_from_operation("op-12345", 2)

    assert generation_id == "75b33bd9-1ada-5eeb-94ca-a1a9cee7e276"


def test_same_logical_id_differs_across_generations() -> None:
    generation_a = physical_record_id("gen-001", "node", "skill:example:v1")
    generation_b = physical_record_id("gen-002", "node", "skill:example:v1")

    assert generation_a != generation_b


def test_record_kind_is_part_of_identity() -> None:
    node_id = physical_record_id("gen-001", "node", "same")
    mapping_id = physical_record_id("gen-001", "mapping", "same")

    assert node_id != mapping_id
