"""Build checked local tasks and substitute only the external Harbor executable."""

import hashlib
import json
import sys

import pytest
from mindmemos_skill.envs.builtin_envs.skillsbench import SkillsBenchData, SkillsBenchDataConfig
from mindmemos_skill.envs.builtin_envs.skillsbench.constants import SKILLSBENCH_REVISION


@pytest.fixture
def dataset(tmp_path):
    """Create one Harbor-format task with binary assets, private grading and bundled Skills."""
    root = tmp_path / "dataset"
    task = root / "tasks" / "demo"
    files = {
        "instruction.md": "Create /root/result.txt using the public input.",
        "task.toml": 'version = "1.1"\n[agent]\ntimeout_sec = 10\n[verifier]\ntimeout_sec = 10\n',
        "environment/Dockerfile": "FROM python:3.12-slim\nCOPY skills /root/.claude/skills\n",
        "environment/input.bin": b"\x00\xff",
        "environment/skills/original/SKILL.md": "---\nname: original\ndescription: Original workflow\n---\n\nOriginal.\n",
        "tests/test.sh": "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n",
        "tests/reference.txt": "PRIVATE REFERENCE",
        "solution/solve.sh": "#!/bin/sh\necho solved\n",
    }
    for name, content in files.items():
        path = task / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
    excluded = root / "tasks-extra" / "extra"
    excluded.mkdir(parents=True)
    checksums = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    (root / "download_receipt.json").write_text(
        json.dumps({"source": {"revision": SKILLSBENCH_REVISION}, "checksums": checksums})
    )
    return root


@pytest.fixture
def sample(dataset):
    return SkillsBenchData(SkillsBenchDataConfig(benchmark_root=dataset)).load()[0]


@pytest.fixture
def harbor(tmp_path):
    """Keep the actual shared executor while replacing Harbor with a local JSON-producing program."""
    script = tmp_path / "harbor.py"
    script.write_text("""import json, os, pathlib, sys
if sys.argv[1:] == ["--version"]:
    print("0.24.0")
    raise SystemExit(0)
assert sys.argv[1:3] == ["run", "--config"], sys.argv
config = json.loads(pathlib.Path(sys.argv[3]).read_text())
assert config["n_attempts"] == config["n_concurrent_trials"] == 1
assert config["environment"] == {"type":"docker", "delete":True}
task = pathlib.Path(config["tasks"][0]["path"])
assert "PRIVATE REFERENCE" in (task / "tests/reference.txt").read_text()
agent = config["agents"][0]
trial = pathlib.Path(config["jobs_dir"]) / config["job_name"] / "demo__native"
trial.mkdir(parents=True)
result = {"task_name":task.name, "verifier_result":{"rewards":{"reward":1}}, "agent_result":{"n_input_tokens":10,"n_cache_tokens":6,"n_output_tokens":4,"cost_usd":0.01}}
(trial / "result.json").write_text(json.dumps(result))
(trial / "agent").mkdir()
(trial / "agent/trajectory.json").write_text(json.dumps({"steps":[
 {"source":"user","message":(task / "instruction.md").read_text()},
 {"source":"agent","message":"Working","reasoning_content":"HIDDEN REASONING", "tool_calls":[{"tool_call_id":"call-1","function_name":"shell","arguments":{"command":"inspect input"}}],"observation":{"results":[{"source_call_id":"call-1","content":"PUBLIC OBSERVATION"}]}},
 {"source":"agent","message":"Done"}
]}))
print("Native CLI completed")
""")
    return [sys.executable, str(script)]
