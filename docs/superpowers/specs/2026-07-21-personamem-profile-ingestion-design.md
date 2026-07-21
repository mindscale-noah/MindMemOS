# PersonaMem Profile Ingestion Design

## Goal

Keep PersonaMem session-marker behavior while ensuring the benchmark-provided
`Current user persona:` evidence is available to the memory-RAG path exactly
once per scope.

## Chosen approach

Submit the first visible PersonaMem profile as a dedicated synchronous add
request before adding conversation batches. Convert it from `system` to `user`
because MindMemOS intentionally marks system messages as non-extractable. Keep
the original profile text and its session-aware timestamp.

The profile request carries the existing scope metadata plus:

- `source: personamem_persona`
- `content_type: profile`

Ordinary user and assistant messages remain in their existing batches with
their existing metadata. All system messages continue to act as timestamp
session boundaries and are excluded from conversation batches.

## Alternatives considered

1. Prepend the converted profile to the first conversation batch. This saves
   one API call but request-level profile metadata would incorrectly describe
   ordinary conversation messages.
2. Pass the original system message through unchanged. This does not work:
   the add pipeline deliberately marks system messages as non-extractable.
3. Add the profile at every session boundary. PersonaMem repeats the same
   profile per block, so this creates duplicate evidence and unnecessary work.

## Data flow

1. Load the full context to calculate session-aware timestamps.
2. Slice the visible scope using the existing context store.
3. Find the first visible, non-empty system message whose content starts with
   `Current user persona:` (case-insensitive).
4. Submit that content once as a one-message `user` profile request.
5. Exclude all system messages from normal conversation batches and submit the
   remaining messages with their existing roles and timestamps.
6. Count the profile in `total_messages`, `added_messages`, and `add_calls`.

If the profile request fails, the scope remains a build failure. A scope with
no matching profile continues to build only its conversation messages.

## Compatibility and isolation

The change is confined to the PersonaMem evaluation environment. It does not
modify the SDK, FastAPI contract, add pipeline, runtime DTOs, database schema,
or other benchmarks. The existing user/session scope identifiers remain
unchanged.

## Tests

- A repeated profile is submitted exactly once with the profile metadata.
- The profile is converted to an extractable user message and keeps its mapped
  timestamp.
- Non-profile system messages remain session boundaries but are not submitted.
- Conversation messages retain their roles, timestamps, batching, and ordinary
  metadata.
- A context without a profile preserves existing behavior.
