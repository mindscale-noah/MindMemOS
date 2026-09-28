"""Compare thinking controls on an identical saved tool-call context without executing tools."""

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from mindmemos_eval.swebench.constants import CHILD_API_KEY_ENV, CHILD_BASE_URL_ENV
from mindmemos_eval.swebench.data import write_json
from mindmemos_eval.swebench.typing import ShellArgs
from openai import AsyncOpenAI
from run_experiment import _credentials


async def probe(args) -> None:
    """Submit independent, sequential comparisons and save raw evidence.

    Args:
        args: Parsed CLI settings identifying the source trace, endpoint credentials and output.
    """
    _credentials(args.env_file)
    key = os.environ[CHILD_API_KEY_ENV]
    endpoint = os.environ[CHILD_BASE_URL_ENV]
    if args.synthetic:
        messages = [
            {
                "role": "system",
                "content": "You are a coding agent. Use shell to inspect and fix the supplied toy program. Return a concise answer.",
            },
            {
                "role": "user",
                "content": "Fix apply_discount in this fictional repository. A 20 percent discount on 100 should yield 80, but it currently yields -1900. Also ensure a zero discount preserves the price. Inspect existing tests and propose the next shell command.",
            },
            {
                "role": "assistant",
                "content": "I will inspect the function and tests.",
                "tool_calls": [
                    {
                        "id": "toy_inspect",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command":"cat pricing.py test_pricing.py"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toy_inspect",
                "content": "pricing.py:\ndef apply_discount(price, percent):\n    return price * (1 - percent)\n\ntest_pricing.py:\nfrom pricing import apply_discount\ndef test_discount():\n    assert apply_discount(100, 20) == 80\ndef test_zero():\n    assert apply_discount(100, 0) == 100",
            },
            {
                "role": "assistant",
                "content": "I will check the failures before editing.",
                "tool_calls": [
                    {
                        "id": "toy_test",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command":"pytest -q"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toy_test",
                "content": "FAILED test_pricing.py::test_discount - assert -1900 == 80\n1 failed, 1 passed.\nThe test repository is entirely fictional; no host files have been accessed.",
            },
        ]
        raw = json.dumps(messages).encode()
        source = "synthetic toy dialogue; no real files or prior conversation exported"
    else:
        raw = args.trajectory.read_bytes()
        trace = json.loads(raw)
        if not trace["messages"][-1].get("reasoning_content"):
            raise ValueError("Expected a trace ending in the failing reasoning response")
        messages = [{k: v for k, v in m.items() if k != "agent"} for m in trace["messages"][:-1]]
        source = str(args.trajectory.resolve())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a command inside the task repository",
                "parameters": ShellArgs.model_json_schema(),
            },
        }
    ]
    variants = {
        "thinking_disabled": {"extra_body": {"thinking": {"type": "disabled"}}},
        "enable_thinking_false": {"extra_body": {"enable_thinking": False}},
        "reasoning_effort_none": {"reasoning_effort": "none"},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "input.json",
        {
            "source": source,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "model": args.model,
            "endpoint_host": urlsplit(endpoint).hostname,
            "messages": messages,
            "tools": tools,
            "variants": variants,
            "repeats": args.repeats,
            "max_tokens_sent": False,
            "tools_executed": False,
        },
    )
    results = []
    async with AsyncOpenAI(api_key=key, base_url=endpoint, timeout=180, max_retries=0) as client:
        for repeat in range(1, args.repeats + 1):
            for name, settings in variants.items():
                record = {"variant": name, "repeat": repeat, "timestamp": datetime.now(timezone.utc).isoformat()}
                try:
                    response = await client.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        tools=tools,
                        temperature=0,
                        parallel_tool_calls=False,
                        **settings,
                    )
                    body = response.model_dump(exclude_none=True)
                    serialized = json.dumps(body, ensure_ascii=False)
                    if key in serialized:
                        raise RuntimeError("Credential detected in response; refusing to persist")
                    write_json(args.output / f"{name}-{repeat}.json", body)
                    choice = response.choices[0]
                    message = choice.message.model_dump(exclude_none=True)
                    reasoning = message.get("reasoning_content") or ""
                    content = message.get("content") or ""
                    record.update(
                        status="ok",
                        response_model=response.model,
                        reasoning_chars=len(reasoning),
                        content_chars=len(content),
                        tool_calls=len(message.get("tool_calls") or []),
                        finish_reason=choice.finish_reason,
                        valid_nonreasoning_response=not reasoning and bool(content or message.get("tool_calls")),
                        usage=response.usage.model_dump() if response.usage else None,
                    )
                except Exception as exc:
                    record.update(
                        status="error",
                        error_type=type(exc).__name__,
                        error=str(exc).replace(key, "[REDACTED]").replace(endpoint, "[ENDPOINT]"),
                    )
                results.append(record)
                write_json(args.output / "summary.json", results)
                print(json.dumps(record, ensure_ascii=False), flush=True)


def main() -> None:
    """Run an explicitly requested comparison without restarting the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(probe(args))


if __name__ == "__main__":
    main()
