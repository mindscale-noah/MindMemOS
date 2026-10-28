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
| `multimodal` | Images in `vanilla` memory: `mode` (`off` / `basic` / `enhanced`) and the image channel. |

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
strict token budget after final filtering. Retention itself never merges or
rewrites results: every returned item passes through with the id and content
produced by the underlying engine.

Both request limits bind at the same time, whichever is tighter: the response
contains at most `top_k` memories whose combined estimated tokens stay within
`token_budget`. When rerank is enabled it scores the whole retention candidate
pool (up to `max_candidates`), so budget-aware selection is not pre-narrowed to
`top_k`; lower `max_candidates` to make retention consider fewer (e.g. only
`top_k`) candidates per request. Selection never merges or rewrites memories.
Redundancy-aware packing is available by setting `selector_version` to
`mixed-v2` (MMR, `mmr_lambda`); the default `mixed-v1` packs greedily by
priority. Retention applies its own `heuristic-v2` token estimator, which
counts words longer than 8 characters at one token per 8 characters so
spaceless runs (URLs, base64, hashes) scale with length instead of
bypassing the budget at one flat token; the add pipeline's chunking and
compaction estimator is a separate implementation with word-count
semantics.

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
| `estimator_version` | `heuristic-v2` | Token estimator implementation (currently heuristic only). |
| `top_m_guarantee` | `5` | Candidates force-kept by relevance before MMR re-ranking (mixed-v2). |
| `mmr_lambda` | `0.70` | Trade-off between priority and redundancy in MMR packing (mixed-v2). |

## `algo_config.multimodal` — images in vanilla memory

`memory_algorithm: vanilla` handles images when `mode` is not `off`. The mode is a normal config
value, so it can be set per project (`project_override_config`), e.g.
`{"algo_config": {"multimodal": {"mode": "enhanced"}}}`. Quote `"off"` in YAML (a bare `off` is
read as `false`).

| Mode | Add | Search |
|---|---|---|
| `off` (default) | Plain vanilla. Requests with images are rejected (400, `add.images_unsupported`). | Plain vanilla. `query_images` are rejected (400, `search.images_unsupported`); `multimodal` is ignored. |
| `basic` | Text through vanilla; images stored in the blob store and, with `mm_embed_model_router`, embedded into the image collection with the dialogue around them. | Multimodal searches append the images most similar to the query text (`image_top_k`) after the text results. |
| `enhanced` | As `basic`, and a chunk with images is extracted by one VLM call: dialogue facts go to text memory, anchored to their images (`metadata.image_blob_ids`); each image's description and visual facts are stored on its image record, with a text vector (never in text memory). | Multimodal searches add facts recalled through standout image matches, the most relevant visual facts, evidence images and, with `search_strategy: agentic`, a multimodal agent. |

A search is **multimodal** only when the request has `query_images` or sets `multimodal: true`
(`POST /v1/memory/search`, SDK `search(multimodal=True)`). Any other search, and any request
without images, behaves exactly like vanilla in every mode: same results, same model calls, no
image lookups, even for users who stored images. Vanilla text search, related-memory recall during
add, dreaming and feedback never see visual facts.

Enhanced details:

- **VLM extraction** uses the same number of chat calls as vanilla (the chat model must accept
  images); a chunk with more than `max_extraction_images` images adds one visual-only call per
  `max_extraction_images` further images. Each image gets a description including its visible text
  and a list of visual facts.
- **Image scores** mix text similarity (query vs. description + visual facts + dialogue, in the
  image record's `text` vector) with image similarity (multimodal embedding), each normalized over
  its hits, weighted by `image_text_weight`.
- **Visual facts**: the visual facts of the evidence images and of the `visual_fact_images`
  best-scored images are reranked against the query (when the request sets `rerank`); the top
  `visual_fact_limit` are returned as separate results (`memory_type: visual_fact`, with the
  `blob_id` of their image), after the text results.
- **Evidence images**: images that stand out for the query images, the session of the best image
  similarity match, then images anchored to the retrieved facts interleaved with the best-scored
  images. Facts anchored to images that stand out by image similarity are recalled too. Image
  results carry their description (and their visual facts when `visual_fact_limit` is 0).
- **Multimodal agentic retrieval**: a short tool loop (search memories, search images, view images,
  submit) refines the single-pass result; its memory search returns dialogue facts and the visual
  facts of the best-matching images.

| Field | Default | Description |
|---|---|---|
| `mode` | `off` | `off`, `basic` or `enhanced` (above). |
| `image_search` | `true` | Search images by similarity (needs `mm_embed_model_router`). |
| `image_top_k` | `5` | Basic: maximum image results appended after the text results. |
| `image_score_threshold` | `null` | Basic: minimum text-image similarity; `null` keeps the top hits. |
| `image_context_chars` | `500` | Characters of surrounding dialogue stored with each image. |
| `max_images_per_request` | `32` | Add requests with more images are rejected (`add.too_many_images`). |
| `max_occurrences_per_image` | `20` | Appearances remembered per deduplicated image. |
| `evidence_images` | `6` | Enhanced: maximum evidence images per search. |
| `session_expansion_images` | `4` | Enhanced: images from the session of the best image match (in order, the match included); `0` disables. |
| `anchored_fact_limit` | `5` | Enhanced: maximum facts recalled through matching images. |
| `image_hit_margin` | `0.03` | Enhanced: an image hit must exceed the median similarity of the hits by this much to recall facts. |
| `image_text_weight` | `0.5` | Enhanced: weight of text similarity in image scores (the rest is image similarity). |
| `visual_fact_limit` | `10` | Enhanced: visual facts returned as separate results; `0` puts them in the image results instead. |
| `visual_fact_images` | `12` | Enhanced: best-scored images whose visual facts are candidates (evidence images always are). |
| `max_visual_facts_per_image` | `30` | Enhanced: visual facts kept per image (newest first). |
| `extraction_image_detail` | `auto` | Enhanced: image detail for VLM extraction (`auto`, `low`, `high`). |
| `max_extraction_images` | `6` | Enhanced: maximum images in one extraction call; a chunk with more also makes visual-only calls (this many images each) for the rest. |
| `agentic_max_steps` | `4` | Enhanced: tool calls in agentic retrieval before it must submit. |

Deleting memory: archived memories are never recalled through images. Deleting a user's or a
project's images (`ImageMemoryStore.purge_images`) removes their image records and the blobs no
other image record of the project uses (a project purge drops the whole blob namespace); the eval
project reset also deletes the project's blobs.

## `mm_embed_model_router` — multimodal embedding

Same shape as the other routers, and like them requests route through a shared
`litellm.Router`: retries, cooldowns and endpoint failover are litellm's. Each endpoint must
encode text and images into one vector space; images ride the Jina-style extension of the
OpenAI embeddings API (`input` items `{"image": "<data URI or URL>"}`), which litellm passes
through to `openai/`-compatible endpoints verbatim.

| Field | Default | Description |
|---|---|---|
| `model` | — | LiteLLM model identifier with a provider prefix, e.g. `openai/siglip2-base-patch16-384`; litellm strips the prefix on the wire. |
| `api_base` | — | Base URL; litellm appends the embeddings path (`/embeddings`). |
| `api_key` | `null` | Bearer token (optional for local endpoints). |
| `dimensions` | — | Required. Sizes the `image` vector of `database.qdrant.image_collection` and is checked at startup. |
| `image_transport` | `base64` | `base64` data URIs, or `url` to send the original URL when an image came from one. Shared by all endpoints. |
| `timeout` / `num_retries` / `max_batch_size` | `60` / `3` / `16` | Per-request timeout, retries, inputs per request. Retries/failover across endpoints are litellm's; `max_batch_size` is shared. |
| `rpm` / `tpm` | `null` | Optional LiteLLM rate limits per endpoint. |
| `extra_body` | `{}` | Extra request body fields. |
| `routing_strategy` / `allowed_fails` / `cool_down` / `retry_after` | `simple-shuffle` / `null` / `null` / `0` | Router-level LiteLLM routing fields, as on the other routers. |
| `dimensions_supported_models` | `[]` | Model name prefixes (provider prefix stripped) allowed to receive `dimensions` on the wire; litellm otherwise drops it for any openai-compatible model whose name lacks `text-embedding-3` (list `siglip2` to keep it). |

For local development `scripts/mm_embed_server.py` serves SigLIP2 (`siglip2-base-patch16-384`,
768 dimensions) behind this contract on `http://127.0.0.1:8090/v1` — see its docstring for the
run command (`uv run --no-project --with torch --with transformers --with pillow ...`).

With no endpoints, multimodal modes only store images. At startup the server embeds one probe text:
a dimension mismatch (endpoint vs `dimensions` vs an existing image collection) stops startup; an
unreachable endpoint only logs a warning. Changing the model requires a new `image_collection`.

The image collection has two named vectors: `image` (this router) and `text` (the text embedding
router, `database.qdrant.vector_size` dimensions; set in enhanced mode). Startup also stops when an
existing image collection has no `text` vector of that size (created by an older version): use a new
`image_collection` name or drop the collection.

## `blob_store` — raw image storage

| Field | Default | Description |
|---|---|---|
| `backend` | `local` | Storage backend: `local` or `s3` (any S3-compatible endpoint: MinIO, OSS S3 gateway, R2). |
| `local_root` | `data/blobs` | Directory of the local backend; relative to the repository root. |
| `s3_endpoint` | *(required for s3)* | Endpoint URL with scheme, e.g. `http://localhost:9100`; `https://` enables TLS. |
| `s3_region` | `us-east-1` | Bucket region/location; required non-empty with `backend: s3` (an empty region fails validation — it would make presigning do a live location lookup). Baked into every presigned URL; also used when creating the bucket. |
| `s3_bucket` | *(required for s3)* | Bucket name; created lazily on first use if missing. |
| `s3_access_key` | *(required for s3)* | Access key (env: `MINDMEMOS_S3_ACCESS_KEY`). |
| `s3_secret_key` | *(required for s3)* | Secret key (env: `MINDMEMOS_S3_SECRET_KEY`). |
| `s3_public_base_url` | *(none)* | Optional public mirror; when set, `public_url()` returns shareable URLs. |
| `s3_presign_endpoint` | *(none)* | External address baked into the short-lived presigned URLs of `POST /v1/memory/search` hits; unset signs against `s3_endpoint` (env: `MINDMEMOS_S3_PRESIGN_ENDPOINT`). |
| `s3_presign_expiry_seconds` | `3600` | Presigned URL lifetime, `1..604800` seconds (env: `MINDMEMOS_S3_PRESIGN_EXPIRY_SECONDS`). |
| `max_image_bytes` | `20971520` | Larger images are rejected. |
| `allowed_path_roots` | `[]` | Directories `ImageRef.path` may point into; empty disables path inputs. |
| `url_fetch_enabled` | `true` | Download `ImageRef.url` inputs. |
| `url_fetch_timeout` | `20.0` | Download timeout in seconds. |
| `url_allow_private_hosts` | `false` | Allow URLs resolving to private/loopback addresses (SSRF guard; DNS rebinding is not covered). |

Images are deduplicated by SHA-256 per project (blob store) and per project+user (image
collection). Only PNG, JPEG, GIF, WebP and BMP are accepted (checked by content, not extension).
Stored images are readable by the project's keys at `GET /v1/memory/images/{blob_id}?user_id=<owner>`;
the server verifies the declared owner against the image records, so another user's blob id 404s.
On the `s3` backend, search hits instead carry `metadata.image_url` — a presigned GET link (signed
locally, no store I/O) that downloads the bytes directly from MinIO without transiting this API;
the proxy route above remains as the fallback when the backend cannot sign (e.g. `local`).

The `s3` backend needs the `s3` extra (`pip install mindmemos[s3]`; the dev workspace already
includes it). Object keys mirror the local layout (`<namespace-prefix>/<id[:2]>/<id>`); the MIME
type rides on the object's content-type. For local development `make dev-blobs` starts MinIO on
port 9100 (ClickHouse native claims 9000) with credentials taken from the same `MINDMEMOS_S3_*`
environment variables. All `s3_*` fields except `s3_region` can be set via environment
(`MINDMEMOS_S3_ENDPOINT/BUCKET/ACCESS_KEY/SECRET_KEY/PUBLIC_BASE_URL/PRESIGN_ENDPOINT/PRESIGN_EXPIRY_SECONDS`).

## Validation

`mindmemos.config.validation.validate_config` enforces choice and range rules
for every documented field; invalid values fail fast at startup with the
offending config path.
