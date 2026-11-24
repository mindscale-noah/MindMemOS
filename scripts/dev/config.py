"""Resolve development overrides using the applications' existing config loaders."""

import os
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from sqlalchemy.engine import make_url

from . import constants as env
from .typing import DependencyPlan


class LocalProjectConfig(BaseModel):
    """Shared identity and Memory algorithm for local development services."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    """Reject misspelled fields without exposing credentials in validation errors."""

    project_id: str = Field(min_length=1, pattern=r"^\S(?:.*\S)?$")
    """Existing project whose data both services authorize."""

    api_key: SecretStr = Field(min_length=1)
    """Bearer credential accepted by both local services."""

    key_id: str = Field(default="key_local", min_length=1)
    """Memory authentication record ID retained when moving an existing key."""

    memory_algorithm: Literal["vanilla", "schema"] = "vanilla"
    """Memory add and search pipeline family bound to this project."""

    enabled: bool = True
    """Whether the credential can authenticate to the local APIs."""

    scopes: list[Literal["memory:read", "memory:write", "skill:read", "skill:write"]]
    """Explicit permissions granted to this credential by each service."""

    user_override_config: dict | None = None
    """Existing per-user Memory configuration overrides, if configured."""

    project_override_config: dict | None = None
    """Existing per-project Memory configuration overrides, if configured."""


class LocalProjectsConfig(BaseModel):
    """Named local projects and the default used by development commands."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    """Reject unknown settings and keep credential values out of errors."""

    default_project: str = Field(min_length=1)
    """Name of the project selected when no command-line project is supplied."""

    projects: dict[str, LocalProjectConfig] = Field(min_length=1)
    """Project names mapped to their complete identity and permission settings."""

    @model_validator(mode="after")
    def validate_default(self) -> "LocalProjectsConfig":
        """Require a configured default project.

        Returns:
            Validated collection of local projects.
        """
        if self.default_project not in self.projects:
            raise ValueError("default_project must name an entry in projects")
        keys = [project.api_key.get_secret_value() for project in self.projects.values()]
        if len(set(keys)) != len(keys):
            raise ValueError("Each local project must have a unique API key")
        ids = [project.project_id for project in self.projects.values()]
        if len(set(ids)) != len(ids):
            raise ValueError("Each local project must have a unique project_id")
        return self


def is_local(host: str | None) -> bool:
    """Check whether an endpoint belongs to the local development machine.

    Args:
        host: Parsed endpoint hostname without brackets or credentials.

    Returns:
        Whether the hostname is localhost or a loopback IP address.
    """
    if host == "localhost":
        return True
    try:
        return ip_address(host or "").is_loopback
    except ValueError:
        return False


def prepare_environment(path: Path) -> None:
    """Load dotenv below process overrides and derive local endpoint ports.

    Args:
        path: Dotenv file shared by applications and Compose.
    """
    load_dotenv(path, override=False, interpolate=False)
    for address, port, default in (
        (env.QDRANT_URL, env.QDRANT_HTTP_PORT, "http://localhost:6333"),
        (env.NEO4J_URI, env.NEO4J_PORT, "bolt://localhost:7687"),
        (env.OTLP_ENDPOINT, env.OTLP_PORT, "http://localhost:4318"),
        (env.S3_ENDPOINT, env.MINIO_PORT, "http://localhost:9100"),
    ):
        if os.environ.get(port):
            value = os.environ.get(address) or default
            os.environ[address] = _local_url_port(value, int(os.environ[port]))
    if os.environ.get(env.KAFKA_PORT):
        servers = os.environ.get(env.KAFKA_SERVERS) or "localhost:9092"
        if "," not in servers and is_local(urlsplit("//" + servers).hostname):
            os.environ[env.KAFKA_SERVERS] = _local_url_port(
                "//" + servers, int(os.environ[env.KAFKA_PORT])
            ).removeprefix("//")


def _local_url_port(value: str, port: int) -> str:
    """Replace a loopback port without changing the scheme, path or credentials."""
    if not 1 <= port <= 65535:
        raise ValueError("Development ports must be between 1 and 65535")
    parsed = urlsplit(value)
    if not is_local(parsed.hostname):
        return value
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    credentials = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
    return urlunsplit(parsed._replace(netloc=f"{credentials}{host}:{port}"))


def _endpoint(plan: DependencyPlan, value: str, service: str, port_env: str, default_port: int) -> None:
    """Add a connection and bind a loopback endpoint to its Compose service."""
    parsed = urlsplit(value)
    if not parsed.hostname:
        raise ValueError(f"Missing hostname for {service}")
    port = parsed.port or default_port
    if service == "otel-collector" and parsed.port is None:
        port = 443 if parsed.scheme == "https" else 80
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid connection port for {service}")
    plan.connections.add((parsed.hostname, port))
    if is_local(parsed.hostname):
        previous = os.environ.get(port_env)
        if previous and int(previous) != port:
            raise ValueError(f"Conflicting local ports for {service}; configure {port_env}")
        os.environ[port_env] = str(port)
        plan.services.add(service)


def _kafka(plan: DependencyPlan, servers: str) -> None:
    """Resolve broker connections without provisioning an external cluster."""
    brokers = [value.strip() for value in servers.split(",")]
    if len(brokers) > 1 and any(is_local(urlsplit("//" + value).hostname) for value in brokers):
        raise ValueError("Local Compose provides one Kafka broker; use a single local bootstrap address")
    for value in brokers:
        _endpoint(plan, "//" + value, "kafka", env.KAFKA_PORT, 9092)
    if "kafka" in plan.services:
        os.environ[env.KAFKA_SERVERS] = servers
        plan.services.add("kafka-ui")


def _credential(name: str, configured: str | None, default: str) -> str:
    """Match Compose's empty-value fallback while preserving YAML-only settings."""
    value = (os.environ[name] or default) if name in os.environ else (configured or default)
    os.environ[name] = value
    return value


def memory_document(path: Path, plan: DependencyPlan, *, async_tasks: bool = False, otel: bool = False) -> dict:
    """Load Memory configuration and prepare an equivalent runtime YAML document.

    Args:
        path: Original application YAML, never modified by this function.
        plan: Shared dependency plan to extend with enabled Memory dependencies.
        async_tasks: Enable Kafka regardless of the source YAML feature switch.
        otel: Enable OTLP regardless of the source YAML feature switch.

    Returns:
        YAML data with resolved connections and absolute authentication paths.
    """
    from mindmemos.config import build_config

    config = build_config(config_path=path)
    data = yaml.safe_load(path.read_text()) or {}
    qdrant = config.database.qdrant
    neo4j = config.database.neo4j
    _endpoint(plan, qdrant.url, "qdrant", env.QDRANT_HTTP_PORT, 6333)
    if "qdrant" in plan.services:
        os.environ[env.QDRANT_GRPC_PORT] = str(qdrant.grpc_port)
        if qdrant.prefer_grpc:
            plan.connections.add((urlsplit(qdrant.url).hostname, qdrant.grpc_port))
    _endpoint(plan, neo4j.uri, "neo4j", env.NEO4J_PORT, 7687)
    neo4j_user = neo4j.username
    neo4j_password = neo4j.password
    if "neo4j" in plan.services:
        neo4j_user = _credential(env.NEO4J_USER, neo4j.username, "neo4j")
        neo4j_password = _credential(env.NEO4J_PASSWORD, neo4j.password, env.LOCAL_PASSWORD)
    database = data.setdefault("database", {})
    database.setdefault("qdrant", {}).update(
        url=qdrant.url, grpc_port=qdrant.grpc_port, prefer_grpc=qdrant.prefer_grpc, api_key=qdrant.api_key
    )
    database.setdefault("neo4j", {}).update(uri=neo4j.uri, username=neo4j_user, password=neo4j_password)
    data.setdefault("auth", {})["api_key_file"] = config.auth.api_key_file
    data.setdefault("kafka", {})["enabled"] = async_tasks
    data.setdefault("telemetry", {})["enabled"] = otel
    if config.algo_config.multimodal.mode != "off":
        blob = config.blob_store
        port = 443 if urlsplit(blob.s3_endpoint).scheme == "https" else 80
        _endpoint(plan, blob.s3_endpoint, "minio", env.MINIO_PORT, port)
        if "minio" in plan.services:
            os.environ[env.S3_BUCKET] = blob.s3_bucket
            os.environ[env.S3_ACCESS_KEY] = blob.s3_access_key
            os.environ[env.S3_SECRET_KEY] = blob.s3_secret_key
    if async_tasks:
        _kafka(plan, config.kafka.bootstrap_servers)
        data.setdefault("kafka", {})["bootstrap_servers"] = config.kafka.bootstrap_servers
    if otel:
        endpoint = config.telemetry.telemetry_endpoint or "http://localhost:4318"
        _endpoint(plan, endpoint, "otel-collector", env.OTLP_PORT, 4318)
        data["telemetry"]["telemetry_endpoint"] = endpoint
    return data


def _postgres_url(value: str, plan: DependencyPlan) -> str:
    """Overlay PostgreSQL connection settings without changing the selected backend."""
    url = make_url(value)
    if not is_local(url.host):
        plan.connections.add((url.host, url.port or 5432))
        return value
    port = int(os.environ.get(env.POSTGRES_PORT) or url.port or 55432)
    url = url.set(
        port=port,
        username=_credential(env.POSTGRES_USER, url.username, "mindmemos_skill"),
        password=_credential(env.POSTGRES_PASSWORD, url.password, env.LOCAL_PASSWORD),
        database=_credential(env.POSTGRES_DB, url.database, "mindmemos_skill"),
    )
    _endpoint(
        plan,
        f"http://{url.host if ':' not in url.host else '[' + url.host + ']'}:{port}",
        "skill-postgres",
        env.POSTGRES_PORT,
        55432,
    )
    return url.render_as_string(hide_password=False)


def skill_document(
    path: Path, plan: DependencyPlan, host: str, *, async_tasks: bool = False, otel: bool = False
) -> dict:
    """Use Skill's native preflight and preserve all unmodified YAML and secrets.

    Args:
        path: Original Skill YAML, anchoring home and authentication file paths.
        plan: Shared dependency plan to extend with enabled Skill dependencies.
        host: API listen address, checked against the authentication policy.
        async_tasks: Select Kafka submission and consumption; false selects local task execution.
        otel: Enable OTLP while retaining any configured SQL audit backend.

    Returns:
        YAML data with development connection overrides and absolute local paths.
    """
    from mindmemos_skill.config.loader import read_skill_config
    from mindmemos_skill.infra.database.config import PostgresConfig

    config = read_skill_config(path)
    if config.fastapi.auth.mode == "none" and not is_local(host):
        raise ValueError("Unauthenticated Skill HTTP requires a loopback address")
    data = yaml.safe_load(path.read_text())
    service = data["service"]
    service["home"] = str(config.service.home)
    if config.fastapi.auth.api_key_file:
        data["fastapi"]["auth"]["api_key_file"] = config.fastapi.auth.api_key_file
    persistence = config.service.persistence
    if persistence and isinstance(persistence.database.config, PostgresConfig):
        service["persistence"]["database"]["config"]["url"] = _postgres_url(persistence.database.config.url, plan)
    tasks = config.service.task_manager
    if async_tasks:
        if persistence is None:
            raise RuntimeError("Skill asynchronous tasks require service.persistence in the source YAML")
        servers = os.environ.get(env.KAFKA_SERVERS) or (tasks.bootstrap_servers if tasks else None) or "localhost:9092"
        service["task_manager"] = service.get("task_manager") or {}
        service["task_manager"].update(mode="kafka", bootstrap_servers=servers, worker=True)
        _kafka(plan, servers)
    else:
        service["task_manager"] = service.get("task_manager") or {}
        service["task_manager"]["mode"] = "local"
    observability = config.service.observability
    if observability:
        if observability.backend and isinstance(observability.backend.config, PostgresConfig):
            service["observability"]["backend"]["config"]["url"] = _postgres_url(observability.backend.config.url, plan)
    if otel:
        endpoint = os.environ.get(env.OTLP_ENDPOINT) or (
            str(observability.otlp.endpoint) if observability and observability.otlp else "http://localhost:4318"
        )
        service["observability"] = service.get("observability") or {}
        service["observability"]["otlp"] = service["observability"].get("otlp") or {}
        service["observability"]["otlp"]["endpoint"] = endpoint
        _endpoint(plan, endpoint, "otel-collector", env.OTLP_PORT, 4318)
    elif observability:
        if observability.backend:
            service["observability"].pop("otlp", None)
        else:
            service.pop("observability", None)
    return data


def finish_plan(plan: DependencyPlan) -> None:
    """Complete the local telemetry stack without enabling unused Kafka services.

    Args:
        plan: Collected application dependencies to complete in place.
    """
    if "otel-collector" in plan.services:
        plan.services.update({"clickhouse", "grafana"})
        if "kafka" in plan.services:
            plan.services.add("kafka-exporter")
