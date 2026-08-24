# Schema Add Version Switch (v1 / v2)

`algo_config.add.schema.version` selects which schema memory-extraction flow
processes `memory add` requests:

| | `v2` (default) | `v1` |
|---|---|---|
| Flow | Rule-based graph fusion: one entity-generation call per episode, deterministic name/type matching for create-vs-update | Develop LLM-heavy flow: entity merge decision, higher-order property generation, property merge/delete, episode search-field augmentation |
| LLM calls per episode | ~5 (objectify, episode entity, episode edges, schema selection, entity generation) | More (adds merge decision, description update, higher-order, search-field calls) |
| Provenance label | `schema_add` (entity `add_algorithm` / memory `mem_extract_version`) | `schema_add_v1` |
| Segmentation prompt | Token-saving boundary prompt (no `reasoning` output field) | Develop boundary prompt (with `reasoning`) |
| Version-specific knobs | `merge.description_rewrite_threshold`, `merge.description_max_chars`, `merge.reference_description_max_chars` | `merge.enable_entity_merge_decision`, `use_property_merge`, `secondary_search_*`, `higher_order.*`, `extraction.use_search_fields`, `extraction.episode_search_fields_augment` |

Invalid values are rejected at config validation (`v1` / `v2` only).

## How to bind a version

- **Deployment default** — set `algo_config.add.schema.version` in the base
  config (`config/mindmemos/dev.yaml` or your deployment YAML). Applies to every
  project without an override.
- **Per project** — set the same field in the project's override config attached
  to its API key. The gateway resolves the key per request and merges
  `tenant_config` + `project_config` on top of the base (project wins), so
  different projects in one deployment can run different versions concurrently.
  Kafka workers rebind the same fragments, so async add jobs see the same
  version as inline ones.

```yaml
algo_config:
  add:
    schema:
      version: v1   # v2 is the default
```

## When a change takes effect

- The schema-add runtime (extractor, planner, chunker, prompts) is resolved per
  drain loop from the request-scoped config — a per-project override change is
  effective on that project's **next add request**, with no restart.
- Base-YAML changes require a **process restart**; the base config is not
  hot-reloaded.
- Records still sitting in the add buffer when the version flips are chunked
  and extracted under the **new** version. Memories already written are never
  rewritten by a version switch.

## Storage compatibility

Both versions write the same collections with the same entity/memory payload
schema, and entity recall is version-agnostic:

- **v2 can read and update v1-written data, and vice versa.** A v2 update to a
  v1-created entity applies the rule-based merge (with bounded description
  growth) to the existing record; nothing needs migration.
- **Mixed histories and back-and-forth switching are safe.**
- Differences are provenance and artifacts, not schema:
  - `mem_extract_version` / `add_algorithm` differ (`schema_add_v1` vs
    `schema_add`) — both are keyword-indexed, so you can filter which version
    produced what, e.g.
    `mindmemos memory get --filter '{"mem_extract_version":"schema_add_v1"}'`.
  - v1-only artifacts (higher-order properties, LLM-augmented episode search
    fields) stay stored and searchable under either version; v2 simply stops
    generating new ones.

## Lifecycle

`v1` is a compatibility mode: it exists to reproduce develop baselines in
benchmarks and to serve as a rollback path while `v2` is validated. It will be
kept at least until `v2` results on LoCoMo / PersonaMem are confirmed; any
deprecation will be announced in the changelog ahead of removal. Switching is
per-project and reversible at any time — storage is compatible in both
directions.

## Regression coverage

- Segmentation prompt parity —
  `test_schema_add_pipeline_segments_with_version_matched_boundary_prompt`
- Boundary prompt versioning — `test_conv_boundary_detection_prompt_is_versioned`
- Metadata label parity — `test_schema_add_writes_version_matched_labels`
