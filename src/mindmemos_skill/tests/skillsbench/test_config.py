"""Verify SkillsBench registration without importing Harbor."""

import subprocess
import sys


def test_registering_skillsbench_does_not_import_harbor():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from mindmemos_skill.envs import Environment; assert 'skillsbench' in Environment.registered_types(); assert 'harbor' not in sys.modules",
        ],
        check=True,
    )
