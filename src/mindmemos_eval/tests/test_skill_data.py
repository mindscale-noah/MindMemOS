"""Verify pinned SearchQA loading without modifying source text or downloads."""

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
from mindmemos_eval.skills.config import ROOT
from mindmemos_eval.skills.data import _searchqa
from mindmemos_skill.common.json import read_json, write_json


@pytest.fixture
def searchqa_source(tmp_path):
    """Create HF records in a different order from the experiment split."""
    source = tmp_path / "download/data"
    source.mkdir(parents=True)
    rows = [
        {"key": key, "question": f"Question {key}", "context": "[DOC] Café  \n", "answers": ["Private answer"]}
        for key in ("second", "unused", "first")
    ]
    parquet.write_table(pa.Table.from_pylist(rows), source / "train.parquet")
    resources = tmp_path / "resources"
    write_json(resources / "data_splits/splits.json", {"train": ["first"], "val": [], "test": ["second"]})
    return source.parent, resources


def test_searchqa_preserves_source_fields_split_order_and_private_answers(tmp_path, searchqa_source):
    source, resources = searchqa_source
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    loader, _ = _searchqa(source, resources, tmp_path / "run")
    samples = loader.load()
    assert [sample.task.id for sample in samples] == ["first", "second"]
    for sample in samples:
        assert sample.task.context == "[DOC] Café  \n"
        assert sample.reward_data.answers == ["Private answer"]
        assert "Private answer" not in sample.task.model_dump_json()
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    splits = loader.split(samples, split_file=resources / "data_splits/splits.json")
    assert splits["train"][0].task.id == "first"
    assert splits["test"][0].task.id == "second"


@pytest.mark.parametrize(
    ("ids", "error"),
    [(["first", "first"], "duplicate task IDs"), (["unknown"], "missing 1 split IDs")],
)
def test_searchqa_rejects_invalid_splits(tmp_path, searchqa_source, ids, error):
    source, resources = searchqa_source
    write_json(resources / "data_splits/splits.json", {"train": ids, "val": [], "test": []})
    with pytest.raises(ValueError, match=error):
        _searchqa(source, resources, tmp_path / "run")


def test_searchqa_rejects_duplicate_source_keys(tmp_path, searchqa_source):
    source, resources = searchqa_source
    original = source / "data/train.parquet"
    (source / "data/duplicate.parquet").write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="Duplicate SearchQA source key"):
        _searchqa(source, resources, tmp_path / "run")


def test_searchqa_rejects_missing_source(tmp_path, searchqa_source):
    source, resources = searchqa_source
    (source / "data/train.parquet").unlink()
    with pytest.raises(ValueError, match="No SearchQA parquet files"):
        _searchqa(source, resources, tmp_path / "run")


def test_committed_searchqa_resources_contain_only_split():
    resources = ROOT / "resources/mindmemos_skill_eval/skill_forest-searchqa/data_splits"
    assert sorted(p.name for p in resources.iterdir()) == ["splits.json"]
    splits = read_json(resources / "splits.json")
    assert {name: len(ids) for name, ids in splits.items()} == {"train": 400, "val": 200, "test": 1400}
    ids = [identifier for partition in splits.values() for identifier in partition]
    assert len(ids) == len(set(ids)) == 2000
