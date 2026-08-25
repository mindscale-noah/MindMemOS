# MindMemOS Configuration Reference

MindMemOS is configured by one YAML file plus a small set of environment
variables. This document describes how configuration is resolved and what
every algorithm-config field means.

## File resolution

- Template: `config/mindmemos/dev.example.yaml` (committed).
- Local overlay: `config/mindmemos/dev.yaml` (gitignored). Copy the template
  here and adjust values for your environment.
- Any path can be passed explicitly via `build_config(config_path=...)`.

YAML files under `config/` are gitignored except `*.example.yaml` and
`config/mindmemos/api_keys.yaml`, so benchmark and local configs never leak
into the repository.

## Environment variables

Connection endpoints and credentials are the only env-configurable fields.
They are read at load time (`.env` at the repository root is honoured):

| Variable | Applies to | Default |
|---|---|---|
| `MINDMEMOS_QDRANT_URL` | `database.qdrant.url` | `http://localhost:6333` |
| `MINDMEMOS_NEO4J_URI` | `database.neo4j.uri` | `bolt://localhost:7687` |

## `algo_config` top-level sections

| Section | Purpose |
|---|---|
| `common` | Shared algorithm settings (e.g. prompt language). |
| `text_processing` | Normalization, tokenization, sparse encoding, and entity rules. |
| `add` | Add-pipeline settings: `schema` (structured add) and `vanilla`. |
| `dreaming` | Background dreaming / consolidation jobs. |
| `search` | Search engines (`default`, `vanilla`, `schema_search`), `agentic`, `rerank`, and `retention`. |
| `skill_evolution` | Skill evolution and rollout knobs. |

## `algo_config.add.schema.chunker` — episode chunking

| Field | Default | Description |
|---|---|---|
| `split_mode` | `llm` | Episode splitting mode: `llm` or `rule`. |
| `min_episode_length` | `1` | Minimum turns per episode. |
| `max_episode_length` | `15` | Maximum turns per episode. |
| `max_buffer_size` | `1000` | Maximum dialogue turns buffered before forced flush. |
| `split_on_user_speaker` | `true` | Prefer splitting when the speaker changes to the user. |
| `max_minutes_from_first` | `30` | Force a split after this many minutes from episode start. |
| `streaming_window_size` | `15` | Turn window used in streaming chunk mode. |

## `algo_config.text_processing` — normalization and sparse encoding

### Normalization

| Field | Default | Description |
|---|---|---|
| `unicode_normal_form` | `NFKC` | Unicode normalization form (`NFC`, `NFKC`, `NFD`, `NFKD`). |
| `strip_zero_width_chars` | `true` | Remove zero-width characters. |
| `normalize_whitespace` | `true` | Collapse whitespace runs. |
| `normalize_lowercase` | `false` | Lowercase all text. |
| `whitespace_regex` | `\\s+` | Regex used for whitespace collapsing. |
| `strip_text` | `true` | Strip leading/trailing whitespace. |
| `content_hash_algorithm` | `md5` | Hash algorithm for content fingerprints. |

### Language detection and tokenization

| Field | Default | Description |
|---|---|---|
| `spacy_en_model` | `en_core_web_sm` | spaCy model for English. |
| `spacy_zh_model` | `zh_core_web_sm` | spaCy model for Chinese. |
| `nlp_max_retries` | `50` | Maximum retries for spaCy pipeline loads. |
| `nlp_retry_base_delay` | `0.1` | Base delay (seconds) for NLP load retries. |
| `explicit_language_confidence` | `1.0` | Confidence when the language is explicitly tagged. |
| `lang_zh_ratio` | `0.35` | CJK ratio above which text is Chinese. |
| `lang_en_latin_ratio` | `0.5` | Latin ratio above which text is English. |
| `lang_mixed_zh_ratio` | `0.35` | CJK ratio for the mixed-language class. |
| `lang_mixed_latin_ratio` | `0.15` | Latin ratio for the mixed-language class. |
| `jieba_cut_all` | `false` | Use jieba full-mode segmentation for Chinese. |

### BM25 / sparse vector encoding

| Field | Default | Description |
|---|---|---|
| `bm25_min_term_len` | `1` | Minimum term length kept for BM25. |
| `bm25_drop_punctuation` | `true` | Drop punctuation tokens. |
| `bm25_lowercase_en` | `true` | Lowercase English terms. |
| `bm25_use_spacy_lemma` | `true` | Use spaCy lemmas when available. |
| `bm25_en_regex_pattern` | `[A-Za-z][A-Za-z0-9_+-]*` | Fallback English term regex. |
| `bm25_use_stem_fallback` | `true` | Stem terms when lemmas are unavailable. |
| `bm25_stemmer_name` | `porter` | NLTK stemmer name. |
| `sparse_hash_dim` | `2000000` | Dimension of the hashed sparse vector. |
| `sparse_hash_algorithm` | `sha1` | Hash algorithm for sparse encoding. |
| `sparse_k1` | `1.5` | BM25 k1 parameter. |
| `sparse_b` | `0.75` | BM25 b parameter. |
| `sparse_fallback_mode` | `log_tf` | Non-BM25 fallback weighting: `tf` or `log_tf`. |
| `sparse_bm25_model_name` | `hash_bm25_v1` | Registered BM25 model name. |
| `sparse_fallback_model_name` | `hash_sparse_tf_v1` | Registered fallback model name. |
| `bm25_idf_smoothing` | `0.5` | IDF smoothing term. |
| `bm25_min_idf_denominator` | `1e-9` | Numerical floor for IDF denominators. |
| `bm25_min_avg_doc_len` | `1.0` | Floor for average document length. |

### Entity rules

| Field | Default | Description |
|---|---|---|
| `entity_fallback_on_empty` | `true` | Fall back to rule-based entities when NLP finds none. |
| `spacy_entity_default_confidence` | `1.0` | Confidence assigned to spaCy entities. |
| `rule_entity_default_confidence` | `0.6` | Confidence assigned to rule-matched entities. |
| `max_entity_count` | `64` | Maximum entities extracted per text. |
| `rule_zh_min_term_len` | `2` | Minimum Chinese term length for rule entities. |
| `entity_rule_find_quoted_text` | `true` | Extract quoted strings as entities. |
| `entity_rule_find_title_case` | `true` | Extract Title Case phrases as entities. |
| `entity_rule_find_acronyms` | `true` | Extract acronyms as entities. |
| `entity_rule_find_file_paths` | `true` | Extract file paths as entities. |
| `entity_rule_find_code_identifiers` | `true` | Extract code identifiers as entities. |
| `entity_rule_find_book_titles` | `true` | extract 《…》 / “…” book-title patterns. |
| `entity_rule_find_english_terms` | `true` | Extract English proper-noun-like terms. |
| `entity_rule_find_long_jieba_terms` | `true` | Extract long jieba segments as entities. |
| `stopwords_zh_path` | *(none)* | Optional Chinese stopword list file. |
| `stopwords_en_path` | *(none)* | Optional English stopword list file. |

## `algo_config.add.vanilla` — vanilla add pipeline budgets

| Field | Default | Description |
|---|---|---|
| `chunk_soft_token_budget` | `26000` | Soft token budget per chunk before compaction. |
| `chunk_hard_token_budget` | `32000` | Hard token budget per chunk. |
| `turn_hard_token_budget` | `16000` | Hard token budget for a single turn. |
| `history_soft_token_budget` | `2000` | Soft budget for carried dialogue history. |
| `history_hard_token_budget` | `4000` | Hard budget for carried dialogue history. |
| `history_min_turn_count` | `1` | Minimum history turns kept. |
| `compaction_soft_token_budget` | `16000` | Soft budget triggering compaction. |
| `compaction_head_tokens` | `4000` | Head tokens preserved verbatim during compaction. |
| `compaction_tail_tokens` | `4000` | Tail tokens preserved verbatim during compaction. |
| `compaction_summary_context_token_budget` | `200000` | Context budget for compaction summarization. |
| `compaction_summary_output_token_budget` | `8000` | Output budget for compaction summarization. |
| `time_gap_threshold_seconds` | `1800` | Session split threshold (seconds). |
| `template_tokens` | `1000` | Tokens reserved for prompt templates. |
| `recall_budget` | `2000` | Token budget for memory recall during add. |
| `output_headroom` | `4000` | Tokens reserved as output headroom. |
| `enable_entities` | `false` | Extract entities during vanilla add. |
| `recall` | *(nested)* | Recall sub-configuration. |
| `safety_gate` | *(nested)* | Safety-gate sub-configuration. |

## `algo_config.search.vanilla` — vanilla hybrid search

| Field | Default | Description |
|---|---|---|
| `recall_size` | `20` | Base recall size for hybrid retrieval. |
| `hybrid_prefetch_factor` | `3` | Multiplier for vector prefetch depth. |
| `hybrid_prefetch_min` | `30` | Minimum prefetch size. |
| `hybrid_prefetch_max` | `300` | Maximum prefetch size. |
| `use_reranker` | `true` | Rerank final vanilla candidates when a reranker is configured. |
| `dedup_enabled` | `true` | Deduplicate near-identical candidates. |
| `dedup_threshold` | `0.6` | Jaccard-style threshold for dedup. |
| `dedup_max_candidates` | `128` | Cap on candidates processed by dedup. |
| `graph_enabled` | `false` | Enable entity-graph path expansion. |
| `shared_entity_graph_enabled` | `false` | Expand memories sharing an entity with seeds. |
| `graph_seed_memory_limit` | `5` | Seed memories used for graph expansion. |
| `graph_related_per_seed` | `3` | Related memories fetched per seed. |
| `shared_entity_graph_limit_per_entity` | `3` | Memories fetched per shared entity. |
| `graph_max_candidates` | `10` | Maximum additional candidates from graph paths. |
| `graph_decay` | `0.5` | Score decay per graph hop. |
| `graph_score` | `0.01` | Score floor for graph-propagated candidates. |

## `algo_config.search.retention` — token-budget retention

Retention is activated per request by passing `token_budget` in the search
request (Python SDK / CLI: the `token_budget` / `--token-budget` option). When
omitted, search behaves exactly as before; when set, results are packed under a
strict token budget after final filtering. Every returned memory keeps its real
memory id, usable with get/update/delete and feedback as usual.

Both request limits bind at the same time, whichever is tighter: the response
contains at most `top_k` memories whose combined estimated tokens stay within
`token_budget`. When rerank is enabled it scores the whole retention candidate
pool (up to `max_candidates`), so budget-aware selection is not pre-narrowed to
`top_k`; lower `max_candidates` to make retention consider fewer (e.g. only
`top_k`) candidates per request. Redundancy is handled at selection time by the
`mixed-v2` MMR packing (`mmr_lambda`), never by merging or rewriting memories.

| Field | Default | Description |
|---|---|---|
| `min_token_budget` | `1` | Minimum token budget accepted for request-gated retention. |
| `max_token_budget` | `128000` | Maximum token budget accepted for request-gated retention. |
| `max_candidates` | `100` | Maximum candidate memories scored per retention pass. |
| `relevance_weight` | `0.50` | Weight of the rerank relevance term in the mixed priority score. |
| `query_overlap_weight` | `0.25` | Weight of the query-term overlap term in the mixed priority score. |
| `recency_weight` | `0.15` | Weight of the recency term in the mixed priority score. |
| `cost_weight` | `0.10` | Weight subtracted for token cost relative to the request budget. |
| `recency_half_life_days` | `30.0` | Half-life in days for the exponential recency decay. |
| `missing_recency_score` | `0.5` | Recency score used when a candidate has no parsable timestamp. |
| `selector_version` | `mixed-v1` | Retention selector implementation: `mixed-v1` or `mixed-v2`. |
| `estimator_version` | `heuristic-v1` | Token estimator implementation (currently heuristic only). |
| `top_m_guarantee` | `5` | Candidates force-kept by relevance before MMR re-ranking (mixed-v2). |
| `mmr_lambda` | `0.70` | Trade-off between priority and redundancy in MMR packing (mixed-v2). |

## Validation

`mindmemos.config.validation.validate_config` enforces choice and range rules
for every documented field; invalid values fail fast at startup with the
offending config path.
