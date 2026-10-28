"""Benchmark matrix config loading: known-key checks and send_images/mode consistency."""

import pytest
from mindmemos_eval.memory.config import load_benchmark_specs

PROFILE = """
algorithm_profiles:
  vanilla:
    memory_algorithm: vanilla
    project_override_config:
      algo_config:
        multimodal:
          mode: {mode}
    search_params:
      top_k: 20
benchmarks:
  {bench}:
    dataset: datasets/{bench}/data
    default_algorithm: vanilla
"""


def _specs(tmp_path, body: str):
    path = tmp_path / "benchmarks.yaml"
    path.write_text(body, encoding="utf-8")
    return load_benchmark_specs(path)


def test_known_search_and_execution_params_load(tmp_path) -> None:
    specs = _specs(
        tmp_path,
        PROFILE.format(mode="enhanced", bench="memeye")
        + """
execution_params:
  memeye:
    send_images: true
    max_questions: 3
    max_answer_images: 8
    image_max_side: 1024
""",
    )
    assert specs["memeye"].execution_params["image_max_side"] == 1024


def test_unknown_profile_search_param_key_rejected(tmp_path) -> None:
    body = PROFILE.format(mode="enhanced", bench="memeye").replace("top_k: 20", "topk: 20")
    with pytest.raises(ValueError, match=r"algorithm_profiles\.vanilla\.search_params: unknown key\(s\) topk"):
        _specs(tmp_path, body)


def test_unknown_benchmark_search_param_key_rejected(tmp_path) -> None:
    body = PROFILE.format(mode="enhanced", bench="memeye") + """
    search_params:
      multimodall: true
"""
    with pytest.raises(ValueError, match=r"benchmarks\.memeye\.search_params: unknown key\(s\) multimodall"):
        _specs(tmp_path, body)


def test_unknown_execution_param_key_rejected(tmp_path) -> None:
    body = PROFILE.format(mode="enhanced", bench="memeye") + """
execution_params:
  memeye:
    send_images: true
    top_k: 20  # search params belong in search_params; no adapter reads this
"""
    with pytest.raises(ValueError, match=r"execution_params\.memeye: unknown key\(s\) top_k"):
        _specs(tmp_path, body)


def test_execution_params_rejected_for_benchmark_without_known_keys(tmp_path) -> None:
    body = PROFILE.format(mode="off", bench="locomo") + """
execution_params:
  locomo:
    add_batch_size: 20  # a personamem key, not a locomo one
"""
    with pytest.raises(ValueError, match="execution_params.locomo: unknown key"):
        _specs(tmp_path, body)


def test_send_images_true_with_mode_off_rejected(tmp_path) -> None:
    body = PROFILE.format(mode='"off"', bench="memeye") + """
execution_params:
  memeye:
    send_images: true
"""
    with pytest.raises(ValueError, match=r"send_images is true but .*mode=off"):
        _specs(tmp_path, body)


def test_unquoted_off_mode_also_rejected(tmp_path) -> None:
    # A bare YAML `off` parses as false, same trap as on the server.
    body = PROFILE.format(mode="off", bench="memeye") + """
execution_params:
  memeye:
    send_images: true
"""
    with pytest.raises(ValueError, match="send_images is true"):
        _specs(tmp_path, body)


def test_send_images_false_with_mode_off_loads(tmp_path) -> None:
    specs = _specs(
        tmp_path,
        PROFILE.format(mode='"off"', bench="memeye")
        + """
execution_params:
  memeye:
    send_images: false
""",
    )
    assert specs["memeye"].execution_params["send_images"] is False


def test_benchmark_override_mode_replaces_profile_mode(tmp_path) -> None:
    body = PROFILE.format(mode="enhanced", bench="memeye") + """
    project_override_config:
      algo_config:
        multimodal:
          mode: "off"
execution_params:
  memeye:
    send_images: true
"""
    with pytest.raises(ValueError, match="mode=off"):
        _specs(tmp_path, body)


def test_benchmark_override_without_algo_config_hides_profile_mode(tmp_path) -> None:
    # Replace semantics: the benchmark override ships alone, so the profile's mode does not
    # reach the server and the check cannot know the effective mode - it must not block.
    body = PROFILE.format(mode='"off"', bench="memeye") + """
    project_override_config:
      llm_config:
        chat_model: other
execution_params:
  memeye:
    send_images: true
"""
    assert "memeye" in _specs(tmp_path, body)


def test_default_send_images_with_mode_off_rejected(tmp_path) -> None:
    # send_images defaults to true in the adapters, so its absence must not dodge the check.
    with pytest.raises(ValueError, match="send_images is true"):
        _specs(tmp_path, PROFILE.format(mode='"off"', bench="memeye"))


def test_non_string_param_key_reports_value_error(tmp_path) -> None:
    # YAML 1.1 parses a bare `on` key as boolean True; the diagnostic must stay a ValueError.
    body = PROFILE.format(mode="enhanced", bench="memeye").replace("top_k: 20", "on: true")
    with pytest.raises(ValueError, match=r"unknown key\(s\) True"):
        _specs(tmp_path, body)
