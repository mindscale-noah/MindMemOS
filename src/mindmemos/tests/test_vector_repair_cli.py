from mindmemos.cli.vector_repair import build_parser


def test_vector_repair_cli_parses_bounded_explicit_request() -> None:
    args = build_parser().parse_args(
        [
            "--project-id",
            "project-1",
            "--memory-id",
            "mem-1",
            "--limit",
            "1",
            "--force",
            "--config-path",
            "/tmp/mindmemos.yaml",
        ]
    )

    assert args.project_id == "project-1"
    assert args.memory_id == ["mem-1"]
    assert args.limit == 1
    assert args.force is True
    assert args.config_path == "/tmp/mindmemos.yaml"
