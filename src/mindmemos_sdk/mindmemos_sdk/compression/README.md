# Working-memory compression

`mindmemos_sdk.compression.compress_memory()` compresses one snapshot of an agent's history through
an OpenAI-compatible Chat Completions endpoint. It returns the complete summary in one blocking call,
without streaming, automatic retries, or stored session state. Inputs are not modified.

Compression runs independently of `MindMemOSClient` and does not require a MindMemOS server.
Its model credentials are supplied explicitly; `mindmemos auth` configures the long-term memory client,
not the compression model connection.

## Install from this checkout

Use Python 3.11–3.13. From the repository root, install the SDK containing this interface:

```bash
python -m pip install ./src/mindmemos_sdk
```

## Compress one snapshot

Set `OPENROUTER_API_KEY` in your environment, then call the SDK:

```python
import os

from mindmemos_sdk.compression import CompressionConfig, CompressionError, compress_memory

config = CompressionConfig(api_key=os.environ["OPENROUTER_API_KEY"])
task = "Fix the application build using the project's existing dependencies."
memory = ["The project uses Python 3.11 and requirements.txt."]
messages = [
    {"role": "assistant", "content": "The build log reports a missing dependency: xyz."},
    {"role": "user", "content": "Check requirements.txt before installing anything."},
]

try:
    result = compress_memory(
        task=task,
        memory=memory,
        messages=messages,
        mode="incremental",
        config=config,
    )
except CompressionError as exc:
    # Keep the original memory and messages. A failed call may still have usage.
    print("Compression failed:", exc.code, "usage:", exc.usage)
else:
    print(result.summary)
    print("Usage:", result.usage)
    print("Latency:", result.latency_seconds)
    # The agent validates the snapshot and decides when to apply this summary.
```

Each invocation makes at most one model request. Call it when the host decides to compress;
checking a local threshold does not require an SDK or model request.

## Inputs and outputs

All arguments are keyword-only. `task`, `memory`, `messages`, `mode`, and `config` are required.

| Argument | Value | Meaning |
| --- | --- | --- |
| `task` | Nonempty `str` | Original task, used to judge which information matters. |
| `memory` | List or tuple of strings | Existing summary blocks in chronological order; use `[]` initially. |
| `messages` | Nonempty list or tuple of JSON-compatible message dictionaries | Exact raw history selected by the host for this call, in chronological order. |
| `mode` | `"incremental"` or `"full"` | Whether the result will be appended to or replace existing memory. |
| `config` | `CompressionConfig` | Model connection and request timeout. |

Both modes receive the task, existing memory, and selected messages. The result contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `summary` | `str` | Complete, nonempty summary text. |
| `usage` | `dict` or `None` | Supported numeric usage and cost fields returned by the provider. Missing usage remains `None`. |
| `latency_seconds` | `float` | Elapsed time for this call. |

The host applies a valid result as follows:

| Mode | Memory update | Raw history update |
| --- | --- | --- |
| `incremental` | Append `summary` as a new block after existing memory. | Remove only the raw messages covered by this snapshot. |
| `full` | Replace existing memory blocks with `[summary]`. | Remove only the raw messages covered by this snapshot. |

Messages arriving while compression runs remain outside that snapshot and must be retained.
The SDK does not return a coverage range or alter the host's message history.

## Model configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `api_key` | Required | Compression provider key. It is not read automatically from the environment. |
| `base_url` | `https://openrouter.ai/api/v1` | OpenAI-compatible model API base URL. |
| `model` | `qwen/qwen3.5-122b-a10b` | Compression model identifier. |
| `reasoning` | `{"effort": "high"}` | Provider reasoning options; use `None` to omit the field. |
| `timeout_seconds` | `240.0` | Positive timeout for the compression call. |
| `headers` | `{}` | Optional additional HTTP headers; authorization comes from `api_key`. |

Override these settings for another provider. Sampling and output-token limits are left to the
provider. Compression thresholds and the main agent's context budget are host settings, not fields
of `CompressionConfig`.

## Failures, timeout, and usage

Invalid arguments raise `ValueError` or `TypeError`. Request and response failures raise
`CompressionError`; inspect `exc.code`:

| Code | Meaning |
| --- | --- |
| `timeout` | The compression deadline expired. |
| `transport_error` | Connection failure or an incomplete HTTP response. |
| `upstream_http_error` | The model endpoint returned an HTTP error. |
| `invalid_result` | Invalid JSON, empty output, refusal, tool-call output, truncated generation, or another rejected response. |
| `audit_write_error` | The optional audit callback failed. |

On timeout, the SDK interrupts the connection and rejects partial output. DNS resolution, connection,
TLS handshake, and response I/O share the deadline. The SDK does not expose active cancellation,
a request handle, or a session registry. Cancelling a host future does not provide a way to stop an
SDK call that is already running.

`CompressionError` also exposes `usage`, `latency_seconds`, and `request_dispatched`. Preserve any
returned usage even when the summary is rejected or never applied. An interrupted connection does
not guarantee that the provider stops processing or billing; missing usage must stay unknown,
not become zero. Do not add cache or reasoning detail fields again to totals that already include them.

Optional `event_sink` and `audit_sink` keyword arguments accept a callable receiving a dictionary:

- `event_sink` receives start and completion metadata, including usage, success, and timeout status.
  Exceptions from this callback are ignored.
- `audit_sink` receives the model request and raw response, excluding connection headers.
  Exceptions from this callback fail the call with `audit_write_error`.

These callbacks run synchronously in the calling thread. Keep them fast and thread-safe; blocking
callback code is not interrupted by the network deadline. Audit records contain task and message
content, so the host decides whether and where to store them. These hooks do not create a client
callback URL or a result-polling API.

## What the host agent must implement

The SDK supplies the compression operation. The agent, or an adapter around its model-call boundary,
owns the following integration behavior:

| Responsibility | Expected behavior |
| --- | --- |
| Enable/disable switch | Provide one host setting, for example `working_memory.enabled`, defaulting to false. When disabled, follow the existing agent path without SDK calls. This name is a host convention, not an SDK argument. |
| State ownership | Keep the task, accumulated memory, and uncompressed messages separately for each session. Preserve system/developer instructions and the original task outside the replaceable history. |
| Message selection | Select an ordered, contiguous prefix of eligible raw history. Preserve recent context and keep tool calls together with all their results. The SDK serializes selected messages as text; the host handles unsupported or multimodal content. |
| Trigger policy | Decide when to compress and which mode to use. Threshold checks are local. No particular fixed or dynamic threshold algorithm is required by the SDK. |
| Request budget | Account for the complete main-model request, including system instructions, task, memory, raw history, tool definitions, and output reserve. A compression trigger threshold alone is not a model context limit. |
| Snapshot identity | Before dispatch, freeze the selected messages and memory. Keep their range, session, and memory revision attached to the local job or future. These bookkeeping fields do not need to be sent to the SDK or exposed to clients. |
| Result application | At a serialized request boundary, verify that the session is active, the job is still current, and its memory revision and covered messages still match. Apply the chosen mode atomically and retain all messages outside the snapshot. Reject summaries that do not meet the host's size or quality checks. |
| Failure recovery | Keep the original state on failure or timeout. If retries are wanted, define their delay and limit in the host. Never delete raw messages merely because a compression request was dispatched. |
| Usage accounting | Track compression usage separately from main-model usage. Include successful but unused summaries and failed calls with returned usage; preserve unknown values when a receipt is missing. |
| Shutdown | Stop scheduling new compression, invalidate jobs that must no longer commit, and allow a bounded wait for outstanding calls. Preserve available receipts and mark unfinished accounting as unknown. |

The blocking SDK can be called inline if the agent is allowed to wait. For asynchronous compression,
run it in a host-owned worker thread or executor and keep the job next to its snapshot. The agent
loop then follows this sequence:

1. Ingest new execution messages and inspect the current compression job without blocking.
2. If the job finished, handle its error or validate and apply its result.
3. If the trigger policy permits, submit a new snapshot. Bound concurrency and avoid duplicate work.
4. Assemble the next main-model request from the preserved prefix, current memory, and remaining raw
   messages. While compression is pending, use the existing state if it still fits the request budget.
5. If the request no longer fits, follow the host's budget policy: wait for compression, use existing
   compaction, or report a context-limit error. The SDK does not pause the agent automatically.

A simple host can allow one compression job at a time. If the host supports replacing an in-flight
job with a newer one, remove the old job's permission to apply results while retaining it for usage
collection and cleanup. Its running SDK call still completes or times out. Coordinate with any
existing host compaction so both mechanisms cannot apply conflicting history replacements.

Integration checks should cover the disabled path, both compression modes, new messages arriving
during compression, stale results, failed or timed-out calls, tool-call pairing, and shutdown with
outstanding work. None of these require the main agent model to choose the compression range.
