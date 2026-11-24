"""Verify shared export naming, isolation and source fidelity."""

import pytest
from mindmemos_sdk.skills.cli_files import export_source, read_source
from mindmemos_sdk.skills.files import export_skill_files, resolve_skill_export_directory

CONTENT = "---\r\nname: demo\r\ndescription: Example\r\n---\r\nBody\r\n"


def test_export_keeps_parent_files_and_roundtrips_exact_source(tmp_path):
    parent = tmp_path / "exports"
    parent.mkdir()
    (parent / "keep.txt").write_bytes(b"keep")
    sibling = parent / "other-skill"
    sibling.mkdir()
    (sibling / "SKILL.md").write_bytes(b"other")
    target = resolve_skill_export_directory(CONTENT, parent)
    assert target == parent / "demo"
    assert not target.exists()
    result = export_skill_files(CONTENT, {"assets/binary": b"\x00\xff"}, parent)
    assert result == target
    assert read_source(str(result))[1] == CONTENT
    assert (target / "assets/binary").read_bytes() == b"\x00\xff"
    (target / "stale").write_text("remove")
    with pytest.raises(ValueError, match="not empty"):
        export_skill_files(CONTENT, {}, parent)
    export_skill_files(CONTENT, {}, parent, overwrite=True)
    assert sorted(path.name for path in target.iterdir()) == ["SKILL.md"]
    assert (parent / "keep.txt").read_bytes() == b"keep"
    assert (sibling / "SKILL.md").read_bytes() == b"other"


def test_name_is_always_appended_and_nonstandard_versions_use_explicit_name(tmp_path):
    assert export_skill_files(CONTENT, {}, tmp_path / "demo") == tmp_path / "demo/demo"
    result = export_source(
        {
            "skill_version_id": "v1",
            "content": "plain\r\n",
            "resources": {},
            "name": "explicit",
            "standard_format": False,
        },
        str(tmp_path),
        False,
    )
    assert result["directory"] == str(tmp_path / "explicit")
    assert (tmp_path / "explicit/SKILL.md").read_bytes() == b"plain\r\n"


@pytest.mark.parametrize(
    "name", [None, "", " ", ".", "..", "../escape", "/absolute", "a/b", "a\\b", "x\x00y", "C:drive", "CON", "trailing."]
)
def test_invalid_names_fail_before_writing(tmp_path, name):
    with pytest.raises(ValueError, match="single directory name"):
        export_skill_files("plain", {}, tmp_path / "exports", name=name, standard_format=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("link_parent", [False, True])
def test_export_rejects_symlink_child_and_parent(tmp_path, link_parent):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_bytes(b"keep")
    parent = tmp_path / "exports"
    if link_parent:
        parent.symlink_to(outside, target_is_directory=True)
    else:
        parent.mkdir()
        (parent / "demo").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        export_skill_files(CONTENT, {}, parent, overwrite=True)
    assert (outside / "keep").read_bytes() == b"keep"


def test_standard_name_comes_from_content_not_external_alias(tmp_path):
    assert export_skill_files(CONTENT, {}, tmp_path, name="alias") == tmp_path / "demo"
