# Internal Console memory reads

These gateway-authenticated endpoints are private to the Console BFF. They require
`memory:read` (or `memory:*`), and the gateway token's project must match
`{project_id}`. Responses use the standard `request_id` / `code` / `message` / `data`
envelope. Never expose the gateway signing credential to a browser.
The BFF also sends `memory:visualize`, but it is not an additional required scope;
`memory:visualize` alone does not authorize these reads.

## Inventory contract

`GET /internal/v1/projects/{project_id}/memories`

| Query | Default | Meaning |
| --- | --- | --- |
| `q` | omitted | Full-text content filter |
| `status` | omitted | Exact lifecycle status filter |
| `algorithm` | omitted | Exact stored `mem_extract_type` filter |
| `mem_type` | omitted | Exact memory-type filter |
| `limit` | 50 | Page size, 1–100 |
| `cursor` | omitted | Opaque `data.next_cursor` from the preceding page |

`data` contains `items` and nullable `next_cursor`. Each item is an allowlisted
projection with these fields:

`memory_id`, `content`, `status`, `mem_type`, `algorithm`, `user_id`, `session_id`,
`created_at`, `updated_at`, `reinforcement_count`, `entity_id`, `entity_type`,
`property_name`, `lifecycle_available`.

Migration from the earlier untyped internal payload:

- Use `memory_id` rather than `id`.
- Use `algorithm` rather than the storage field `mem_extract_type`.
- Use `updated_at` rather than `update_at`.
- Arbitrary payload fields, including `metadata`, are no longer returned by this
  inventory endpoint. Filters are applied before pagination.

This change does **not** change public `POST /v1/memory/list` or
`POST /v1/memory/scroll`, their SDK contracts, or their documented response models.

## Lifecycle and provenance

`GET /internal/v1/projects/{project_id}/memories/{memory_id}/lifecycle`

The response contains `events`, `lineage`, `availability`, and `provenance`.
Provenance has exactly five public fields: `available`, `add_record_id`,
`request_id`, `source_type`, and `submitted_at`. Original Add messages, arbitrary
metadata, provider credentials, and request bodies are never provenance fields.

The data plane retrieves only allowlisted provenance fields. A stored source ID
uses a project-scoped point lookup. Otherwise, a single filtered lookup on the
`memories[].memory_id` Add Record output association replaces the old scan of up
to 1,000 complete Add Records. Startup schema reconciliation adds its keyword
payload index to existing collections; existing Add outputs do not require a
new association backfill. The array-field filter follows the
[Qdrant filtering contract](https://qdrant.tech/documentation/search/filtering/).

Source lookups, including missing associations, are cached for up to 30 seconds
per repository/event loop, with at most 2,048 entries. Add Record writes and
patches invalidate that project's cache. The current memory and its lifecycle
state are still read separately; a provenance cache hit never freezes an old
memory status or reinforcement count.

If the optional provenance read fails, the endpoint still returns the memory's
events and lineage with `provenance.available=false`,
`availability.lifecycle_complete=false`, and `provenance` in `missing_fields`.
A known source ID may still be included, but does not imply the read succeeded.
Authentication and permission failures retain their error responses.

## Insights and caching

`GET /internal/v1/projects/{project_id}/memory-insights`

- `range_days`: 1–90, default 30.
- `scan_limit`: 1–10,000, default 5,000; applies only to the bounded trend scan.
- `refresh`: default false; true starts a new read, bypassing both the cached
  snapshot and any older in-flight loader.

Status totals use exact project-scoped counts. Trends use selected timestamp and
state fields, excluding memory content and unrestricted metadata. If the scan
has more pages, `availability.scanned_all` is false: caching does not turn a
truncated trend into a complete history.

Snapshots are cached for up to 30 seconds, keyed by project, range, and scan
limit. Identical in-flight reads share one loader; cold loaders are limited to
five concurrent requests per cache. Returned values are copied so callers
cannot modify the cached snapshot. Memory upserts, payload updates, patches, and
deletes invalidate that project's snapshots. An older read already in flight
cannot repopulate the cache after invalidation or supersede a newer refresh.
Failed reads are not cached. A failed refresh returns an error to its caller
while preserving any previous snapshot until its original expiration or write
invalidation; it does not extend that snapshot's lifetime.

Caches are local to each event loop/process. Writes in another process become
visible within the TTL, and deployments start with cold caches. `as_of` is the
snapshot generation time, including on cache hits. The platform's
`/api/memory-space/insights?refresh=true` forwards explicit refresh to this
endpoint. These caches reduce repeated reads; an uncached trend request still
performs the bounded database scan.

The platform cache uses the upstream snapshot's remaining freshness rather than
starting another full TTL. Aggregated responses report the oldest available
project snapshot time, so a cache hit is not presented as newly generated data.
