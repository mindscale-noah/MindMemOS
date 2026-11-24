# Docker and Configuration

This directory defines the database, task queue, and monitoring dependencies for Memory and Skill. For startup and management commands, see the [deployment guide](../docs/deploy/instruction.md#3-select-and-manage-services).

## File Structure

| File or directory | Purpose |
| --- | --- |
| [compose.yaml](compose.yaml) | Single definition of all containers: images, ports, mounts, health checks, and dependencies |
| [otel/collector.yaml](otel/collector.yaml) | Collector configuration template |
| [postgres/init-vector.sql](postgres/init-vector.sql) | Enable pgvector when initializing a new PostgreSQL data volume |
| [grafana/provisioning/](grafana/provisioning/) | ClickHouse and Qdrant data sources and dashboard provisioning |
| [grafana/dashboards/](grafana/dashboards/) | Dashboards for logs, traces, Skill, and Qdrant |

## Containers

| Purpose | Compose service | Image | Function |
| --- | --- | --- | --- |
| Memory storage | `qdrant` | `qdrant/qdrant:v1.18.0` | Vector and sparse index storage |
| Memory storage | `neo4j` | `neo4j:2026.05.0` | Graph storage |
| Skill storage | `skill-postgres` | `pgvector/pgvector:0.8.6-pg16-bookworm` | Application data and vector storage |
| Memory / Skill async tasks | `kafka` | `apache/kafka:3.9.0` | Memory async messages and optional Skill task notifications |
| Memory / Skill async tasks | `kafka-ui` | `provectuslabs/kafka-ui:v0.7.2` | Inspect Kafka topics, messages, and consumer groups |
| Memory / Skill monitoring | `kafka-exporter` | `danielqsj/kafka-exporter:v1.8.0` | Expose Kafka consumer lag and partition metrics |
| Memory / Skill monitoring | `clickhouse` | `clickhouse/clickhouse-server:24.8` | Store logs, traces, and metrics exported by the Collector |
| Memory / Skill monitoring | `otel-collector` | `otel/opentelemetry-collector-contrib:0.127.0` | Receive OTLP logs and traces, scrape Kafka metrics, and export to ClickHouse |
| Memory / Skill monitoring | `grafana` | `grafana/grafana:11.4.0` | Query ClickHouse and display monitoring dashboards; also query Qdrant metadata directly |

Host applications connect through published ports. Containers communicate through service names and internal ports. Changing a host port does not require changing internal container addresses.

| Endpoint | Default host address | Internal container address |
| --- | --- | --- |
| Qdrant HTTP / gRPC | `localhost:6333` / `localhost:6334` | `qdrant:6333` / `qdrant:6334` |
| Neo4j HTTP / Bolt | `localhost:7474` / `localhost:7687` | `neo4j:7474` / `neo4j:7687` |
| Skill PostgreSQL | `localhost:55432` | `skill-postgres:5432` |
| Kafka | `localhost:9092` | `kafka:29092` |
| Kafka UI / exporter | `localhost:8080` / `localhost:9308` | `kafka-ui:8080` / `kafka-exporter:9308` |
| OTLP gRPC / HTTP | `localhost:4317` / `localhost:4318` | `otel-collector:4317` / `otel-collector:4318` |
| ClickHouse HTTP / native | `localhost:8123` / `localhost:9000` | `clickhouse:8123` / `clickhouse:9000` |
| Grafana | `localhost:3000` | `grafana:3000` |

Kafka runs in single-node KRaft mode and does not require ZooKeeper. Services use fixed container names, so different feature selections share the same containers.

## Configuration

The root [.env.example](../.env.example) lists environment variables; store private values in `.env`. Memory and Skill application settings are in `config/mindmemos/dev.yaml` and `config/mindmemos_skill/dev.yaml`, respectively. Skill retains the database type specified in YAML: SQLite does not start PostgreSQL, and external databases do not start corresponding local containers.

| Component | Environment variables and defaults |
| --- | --- |
| Qdrant (Memory) | `MINDMEMOS_QDRANT_HTTP_PORT=6333`<br>`MINDMEMOS_QDRANT_GRPC_PORT=6334`<br>`MINDMEMOS_QDRANT_CPUS=4`<br>`MINDMEMOS_QDRANT_MEMORY_LIMIT=4g`<br>`MINDMEMOS_QDRANT_MEMORY_RESERVATION=2g` |
| Neo4j (Memory) | `MINDMEMOS_NEO4J_HTTP_PORT=7474`<br>`MINDMEMOS_NEO4J_BOLT_PORT=7687`<br>`MINDMEMOS_NEO4J_USERNAME=neo4j`<br>`MINDMEMOS_NEO4J_PASSWORD=mindmemos_dev_password` |
| PostgreSQL (Skill) | `MINDMEMOS_SKILL_POSTGRES_PORT=55432`<br>`MINDMEMOS_SKILL_POSTGRES_DB=mindmemos_skill`<br>`MINDMEMOS_SKILL_POSTGRES_USER=mindmemos_skill`<br>`MINDMEMOS_SKILL_POSTGRES_PASSWORD=mindmemos_dev_password` |
| Kafka (Memory & Skill) | `MINDMEMOS_KAFKA_PORT=9092`<br>`MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS=localhost:9092`<br>`MINDMEMOS_KAFKA_NUM_PARTITIONS=32`<br>`MINDMEMOS_KAFKA_UI_PORT=8080`<br>`MINDMEMOS_KAFKA_EXPORTER_PORT=9308` |
| ClickHouse (Memory & Skill) | `MINDMEMOS_CLICKHOUSE_HTTP_PORT=8123`<br>`MINDMEMOS_CLICKHOUSE_NATIVE_PORT=9000`<br>`MINDMEMOS_CLICKHOUSE_DB=otel`<br>`MINDMEMOS_CLICKHOUSE_USER=mindmemos`<br>`MINDMEMOS_CLICKHOUSE_PASSWORD=mindmemos_dev_password` |
| Collector (Memory & Skill) | `MINDMEMOS_OTEL_HTTP_PORT=4318`<br>`MINDMEMOS_OTEL_GRPC_PORT=4317`<br>`MINDMEMOS_OTEL_TTL=72h` |
| Grafana (Memory & Skill) | `MINDMEMOS_GRAFANA_PORT=3000`<br>`MINDMEMOS_GRAFANA_USER=admin`<br>`MINDMEMOS_GRAFANA_PASSWORD=admin` |

### `.env` Overrides and Port Synchronization

For `make dev`, `make api`, and `make docker`, connection settings follow this precedence: **command-line options / process environment > root `.env` > YAML > application defaults**. Make options control Kafka tasks and OTLP, overriding the corresponding switches in source YAML. Database types and algorithm parameters remain defined by YAML. Only supported environment variables map to application fields; arbitrary variables are not expanded into arbitrary YAML fields.

| `.env` setting | Application settings synchronized at startup |
| --- | --- |
| `MINDMEMOS_QDRANT_HTTP_PORT` / `MINDMEMOS_QDRANT_URL` | Memory Qdrant HTTP address |
| `MINDMEMOS_QDRANT_GRPC_PORT` / `MINDMEMOS_QDRANT_PREFER_GRPC` / `MINDMEMOS_QDRANT_API_KEY` | Corresponding Memory Qdrant connection options |
| `MINDMEMOS_NEO4J_BOLT_PORT` / `MINDMEMOS_NEO4J_URI` | Memory Neo4j address |
| `MINDMEMOS_NEO4J_USERNAME` / `MINDMEMOS_NEO4J_PASSWORD` | Credentials used by Memory and local Neo4j |
| `MINDMEMOS_KAFKA_PORT` / `MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS` | Memory and Skill Kafka addresses and the local broker's advertised address |
| `MINDMEMOS_OTEL_HTTP_PORT` / `MINDMEMOS_TELEMETRY_ENDPOINT` | Memory and Skill OTLP addresses |
| `MINDMEMOS_SKILL_POSTGRES_PORT` / `DB` / `USER` / `PASSWORD` (the last three use the same prefix) | Skill's configured local PostgreSQL URLs, including the PostgreSQL observability backend |

For loopback addresses, published-port variables take precedence. Even if a URL in `.env` still contains an old port, setting `MINDMEMOS_KAFKA_PORT=19092` and running `make dev` makes both the application and the broker's advertised address use `localhost:19092`. The same applies to Qdrant, Neo4j, and OTLP; no YAML port edit is needed. Without environment overrides, the launcher uses YAML connection settings and passes local ports to Compose.

External Kafka, OTLP, Qdrant, and Neo4j endpoints use their respective full-address variables. Local published-port settings do not replace non-loopback hosts or ports. External Skill PostgreSQL connections retain the full URL from YAML and ignore local `POSTGRES_*` overrides. Collector endpoints and scrape targets inside containers, as well as Grafana data source addresses, use the container network and do not change with host ports.

Compose continues to read ports and credentials for containers such as ClickHouse, Grafana, and Kafka UI. Applications connect to the Collector rather than directly to this monitoring ClickHouse instance.

Changing initialization environment variables does not automatically reset users, passwords, or databases in existing database volumes.

## Collector Configuration and Mounts

Applications send OTLP logs and traces to the Collector, which exports them to ClickHouse. The local Kafka exporter provides Prometheus metrics. The current metrics pipeline accepts only Prometheus data, not application OTLP metrics. The ClickHouse exporter creates tables automatically; logs and traces use `otel_logs` and `otel_traces`, with a default retention of `72h`.

The launcher generates Collector configuration from the actual dependencies. It retains Kafka scraping when a local Kafka exporter is selected; otherwise, it removes the corresponding receiver and metrics pipeline. Direct Compose usage mounts the full template.

Generated files are stored at `.local/docker/<Compose project identifier>/<content hash>.yaml` and mounted through `MINDMEMOS_OTEL_CONFIG_FILE`. Files retain environment variable references and do not contain resolved passwords. Different contents use separate files, so existing mounts are not overwritten. Files remain after the launcher exits, allowing containers to keep running and restart. Management commands remove the corresponding files only after the current project's containers have been successfully stopped and removed. This directory is excluded from Git and requires no manual editing.

## Data Volumes

| Named volume | Stored data |
| --- | --- |
| `qdrant_storage` | Memory Qdrant data |
| `neo4j_data`, `neo4j_logs` | Memory Neo4j data and logs |
| `skill_postgres_data` | Skill PostgreSQL data |
| `kafka_data` | Kafka data |
| `clickhouse_data` | Telemetry data |
| `grafana_data` | Grafana state |

The Compose file remains in `dockers/`, preserving the existing default Compose project name and volume keys. Set `COMPOSE_PROJECT_NAME` to change the project name. Actual volume names include the project name as a prefix.
