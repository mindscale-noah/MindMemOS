"""Kafka notifications for durable tasks stored in the existing database."""

import asyncio
import json
from collections.abc import Awaitable
from time import monotonic

from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ...logging import get_logger, log_context
from ...logging.tasks import async_task_log_reporter
from ...logging.tracing import bind_tracer, get_tracer, log_failure
from ...logging.typing import TaskLog
from .base import TaskManager
from .dispatcher import TaskDispatcher
from .typing import TaskHandler, TaskSnapshot, TaskSpec

logger = get_logger(__name__)
ENQUEUE_TIMEOUT_SECONDS = 0.5
"""Maximum API time spent waiting for a client-side buffer slot or metadata."""


class KafkaTaskManager(TaskManager):
    """Use Kafka as a wake-up channel and the borrowed database as task truth."""

    def __init__(
        self,
        *,
        store,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        handler: TaskHandler | None = None,
        tracer: trace.Tracer | None = None,
        max_concurrency: int = 2,
        max_buffered: int = 1000,
        scan_interval_seconds: float = 5,
    ):
        """Configure a dedicated Skill topic, group and optional worker handler.

        Args:
            store: TaskDataService on the existing Skill database.
            bootstrap_servers: Existing Kafka cluster bootstrap address.
            topic: Topic reserved for Skill execution notifications.
            group_id: Consumer group reserved for Skill workers.
            handler: Trusted application dispatcher; None makes this producer-only.
            tracer: Service-owned provider used for publish and execute spans.
            max_concurrency: Maximum simultaneous algorithms per worker process.
            max_buffered: Maximum queued and running notifications admitted locally.
            scan_interval_seconds: Delay between missed-delivery recovery scans.
        """
        if max_concurrency < 1 or scan_interval_seconds <= 0:
            raise ValueError("Worker concurrency and scan interval must be positive")
        self._tracer = tracer or get_tracer()
        self._store = store
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._handler = handler
        self._dispatcher = TaskDispatcher(self._execute, max_concurrency=max_concurrency, max_buffered=max_buffered)
        self._scan_interval = scan_interval_seconds
        self._producer = None
        self._consumer = None
        self._consumer_task: asyncio.Task | None = None
        self._background: set[asyncio.Task] = set()
        self._started = False
        self._closed = False
        self._last_counts_at = float("-inf")
        self._producer_unhealthy = False
        self._consumer_reconnecting = False

    async def start(self) -> None:
        """Start recovery without making API availability depend on Kafka startup."""
        if self._closed:
            raise RuntimeError("Kafka task manager is closed")
        if self._started:
            return
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError as exc:
            raise RuntimeError("Install mindmemos-skill[kafka] to use Kafka tasks") from exc
        self._producer_type = AIOKafkaProducer
        self._consumer_type = AIOKafkaConsumer
        self._started = True
        logger.info(
            "task.kafka.starting",
            extra={
                "event_body": {
                    "topic": self._topic,
                    "group_id": self._group_id,
                    "worker": self._handler is not None,
                }
            },
        )
        self._spawn(self._connect_loop(), name="skill-kafka-connection")
        self._spawn(self._scan(), name="skill-task-recovery")

    async def _connect_loop(self) -> None:
        """Reconnect the shared Kafka clients while DB-backed submissions continue."""
        while True:
            try:
                if self._producer is None:
                    await self._ensure_topic()
                    producer = self._producer_type(
                        bootstrap_servers=self._bootstrap_servers,
                        client_id="mindmemos-skill-tasks",
                        acks="all",
                        enable_idempotence=True,
                    )
                    try:
                        await producer.start()
                    except BaseException:
                        await producer.stop()
                        raise
                    self._producer = producer
                    logger.info("task.kafka.producer.ready", extra={"event_body": {"topic": self._topic}})
                if self._handler is not None and self._consumer is None:
                    consumer = self._consumer_type(
                        self._topic,
                        bootstrap_servers=self._bootstrap_servers,
                        group_id=self._group_id,
                        client_id="mindmemos-skill-worker",
                        enable_auto_commit=False,
                        auto_offset_reset="earliest",
                        max_poll_records=10,
                    )
                    try:
                        await consumer.start()
                    except BaseException:
                        await consumer.stop()
                        raise
                    self._consumer = consumer
                    if self._consumer_reconnecting:
                        logger.info("task.kafka.consumer.recovered", extra={"event_body": {"topic": self._topic}})
                    self._consumer_reconnecting = False
                    logger.info(
                        "task.kafka.consumer.ready",
                        extra={"event_body": {"topic": self._topic, "group_id": self._group_id}},
                    )
                    self._consumer_task = asyncio.create_task(self._consume(), name="skill-task-consumer")
            except asyncio.CancelledError:
                raise
            except Exception:
                self._producer_unhealthy = True
                logger.exception("task.kafka.connect.failed")
            await asyncio.sleep(self._scan_interval)

    async def _ensure_topic(self) -> None:
        """Ensure the configured topic exists, including concurrent startup races."""
        from aiokafka.admin import AIOKafkaAdminClient, NewTopic
        from aiokafka.errors import TopicAlreadyExistsError, for_code

        admin = AIOKafkaAdminClient(bootstrap_servers=self._bootstrap_servers)
        try:
            await admin.start()
            created = False
            if self._topic not in await admin.list_topics():
                try:
                    response = await admin.create_topics(
                        [NewTopic(self._topic, num_partitions=1, replication_factor=1)]
                    )
                except TopicAlreadyExistsError:
                    pass
                else:
                    for topic, code, *_ in response.topic_errors:
                        if code and code != TopicAlreadyExistsError.errno:
                            raise for_code(code)(f"Topic initialization failed: {topic}")
                        created = code == 0
            logger.info(
                "task.kafka.topic.created" if created else "task.kafka.topic.ready",
                extra={"event_body": {"topic": self._topic}},
            )
        finally:
            await admin.close()

    def _spawn(self, operation: Awaitable[None], *, name: str) -> None:
        """Retain a background task and report unexpected loop failures."""
        task = asyncio.create_task(operation, name=name)
        self._background.add(task)

        def finished(done: asyncio.Task) -> None:
            """Release the task and surface unexpected failures."""
            self._background.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logger.error(
                    "task.manager.loop.failed", extra={"event_body": {"loop": name}}, exc_info=done.exception()
                )

        task.add_done_callback(finished)

    async def _notify(self, task_id: str) -> None:
        """Publish using the original submission parent and W3C headers."""
        if self._producer is None:
            raise RuntimeError("Kafka producer is not connected")
        record = await self._store.load(task_id)
        if record is None:
            return
        parent = TraceContextTextMapPropagator().extract(record.trace_context)
        with (
            log_context(request_id=record.request_id, task_id=task_id, project_id=record.project_id),
            self._tracer.start_as_current_span("task.kafka.publish", context=parent) as span,
        ):
            span.set_attribute("task_id", task_id)
            span.set_attribute("project_id", record.project_id)
            span.set_attribute("request_id", record.request_id or "")
            span.set_attribute("messaging.destination.name", self._topic)
            future = await self._producer.send(
                self._topic,
                json.dumps({"task_id": task_id}).encode(),
                key=task_id.encode(),
                headers=[(key, value.encode()) for key, value in record.trace_context.items()],
            )
            logger.info("task.kafka.enqueued", extra={"event_body": {"topic": self._topic}})

            def delivered(done: asyncio.Future) -> None:
                """Observe acknowledgement failures without blocking submissions."""
                if not done.cancelled() and (error := done.exception()) is not None:
                    self._producer_unhealthy = True
                    logger.error(
                        "task.kafka.delivery.failed",
                        extra={"event_body": {"task_id": task_id}},
                        exc_info=(type(error), error, error.__traceback__),
                    )
                elif not done.cancelled():
                    if self._producer_unhealthy:
                        logger.info("task.kafka.producer.recovered", extra={"event_body": {"topic": self._topic}})
                    self._producer_unhealthy = False
                    logger.info("task.kafka.delivered", extra={"event_body": {"topic": self._topic}})

            future.add_done_callback(delivered)

    async def submit(self, spec: TaskSpec) -> TaskSnapshot:
        """Commit the task first, then attempt a non-blocking Kafka notification."""
        if not self._started:
            raise RuntimeError("Kafka task manager is not started")
        record = await self._store.create(spec)
        if self._producer is None:
            return record.snapshot()
        try:
            await asyncio.wait_for(self._notify(spec.task_id), timeout=ENQUEUE_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("task.kafka.enqueue.failed", extra={"event_body": {"task_id": spec.task_id}})
        else:
            try:
                await self._store.defer_notification(spec.task_id)
            except Exception:
                logger.exception(
                    "task.kafka.notification_schedule.failed", extra={"event_body": {"task_id": spec.task_id}}
                )
        return record.snapshot()

    async def get(self, project_id: str, task_id: str) -> TaskSnapshot | None:
        """Read a task from the existing project-owned database."""
        record = await self._store.get(project_id, task_id)
        return record.snapshot() if record is not None else None

    async def logs(self, project_id: str, task_id: str, *, after: int = 0) -> list[TaskLog]:
        """Read correlated task logs committed to the existing database."""
        return await self._store.logs(project_id, task_id, after=after)

    async def _scan(self) -> None:
        """Republish queued tasks and reclaim expired work after process failure."""
        while True:
            try:
                if monotonic() - self._last_counts_at >= 30:
                    with bind_tracer(self._tracer):
                        for project_id, counts in (await self._store.counts()).items():
                            with log_context(project_id=project_id):
                                logger.info(
                                    "task.queue.counts",
                                    extra={
                                        "event_body": {
                                            "queued": counts.get("queued", 0),
                                            "running": counts.get("running", 0),
                                            "succeeded": counts.get("succeeded", 0),
                                            "failed": counts.get("failed", 0),
                                            "topic": self._topic,
                                            "group_id": self._group_id,
                                        }
                                    },
                                )
                    self._last_counts_at = monotonic()
                if self._producer is not None:
                    for record in await self._store.due():
                        try:
                            await asyncio.wait_for(self._notify(record.task_id), timeout=ENQUEUE_TIMEOUT_SECONDS)
                        except Exception:
                            logger.exception(
                                "task.kafka.recovery_notify.failed", extra={"event_body": {"task_id": record.task_id}}
                            )
                        else:
                            await self._store.defer_notification(record.task_id)
                if self._handler is not None:
                    for record in await self._store.expired():
                        if not await self._dispatch(record.task_id):
                            break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("task.kafka.recovery_scan.failed")
            await asyncio.sleep(self._scan_interval)

    async def _consume(self) -> None:
        """Consume durable task notifications without waiting for execution capacity."""
        assert self._consumer is not None
        consumer = self._consumer
        try:
            async for message in consumer:
                try:
                    body = json.loads(message.value)
                    task_id = body["task_id"]
                    if not isinstance(task_id, str) or not task_id:
                        raise ValueError("Kafka task notification requires task_id")
                    await self._dispatch(task_id)
                    # Full queues leave the DB task available for later notification or recovery.
                except Exception:
                    logger.exception("task.kafka.notification.invalid")
                await consumer.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            self._consumer_reconnecting = True
            logger.exception("task.kafka.consumer.failed")
        finally:
            try:
                await consumer.stop()
                logger.info("task.kafka.consumer.stopped", extra={"event_body": {"topic": self._topic}})
            except Exception:
                logger.exception("task.kafka.consumer.stop.failed")
            finally:
                if self._consumer is consumer:
                    self._consumer = None
                self._consumer_task = None

    async def _dispatch(self, task_id: str) -> bool:
        """Admit a notification from either Kafka or recovery without waiting."""
        if self._handler is None:
            return False
        return self._dispatcher.try_submit(task_id, task_id)

    async def _execute(self, task_id: str) -> None:
        """Claim only after obtaining a slot and retain it until execution finishes."""
        claimed = await self._store.claim(task_id)
        if claimed is None:
            return
        record, token = claimed
        cancelled = None
        parent = TraceContextTextMapPropagator().extract(record.trace_context)
        with (
            bind_tracer(self._tracer),
            log_context(
                request_id=record.request_id,
                project_id=record.project_id,
                task_id=task_id,
                attempt=record.attempt,
                service_name="mindmemos-skill",
            ),
            self._tracer.start_as_current_span("task.execute", context=parent) as span,
        ):
            for key, value in {
                "request_id": record.request_id or "",
                "task_id": task_id,
                "project_id": record.project_id,
                "attempt": record.attempt,
            }.items():
                span.set_attribute(key, value)
            try:
                await self._run(record.spec(), token, attempt=record.attempt)
            except asyncio.CancelledError as exc:
                cancelled = exc
                span.set_attribute("status", "recoverable")
        if cancelled is not None:
            raise cancelled

    async def _run(self, spec: TaskSpec, token: str, *, attempt: int) -> None:
        """Run a fenced attempt and drain logs before exposing terminal state."""
        assert self._handler is not None
        current = asyncio.current_task()
        lease_lost = False

        async def heartbeat() -> None:
            """Cancel an attempt that loses its database lease."""
            nonlocal lease_lost
            while True:
                await asyncio.sleep(30)
                try:
                    owned = await self._store.heartbeat(spec.task_id, token)
                except Exception:
                    logger.exception("task.lease.heartbeat.failed", extra={"event_body": {"task_id": spec.task_id}})
                    owned = False
                if not owned:
                    lease_lost = True
                    if current is not None:
                        current.cancel()
                    return

        async with async_task_log_reporter(
            self._store.append_log, project_id=spec.project_id, task_id=spec.task_id, attempt=attempt
        ) as flush_logs:
            record = await self._store.load(spec.task_id)
            logger.info(
                "task.started",
                extra={
                    "event_body": {"attempt": attempt, "waiting_ms": record.snapshot().waiting_ms if record else None}
                },
            )
            pulse = asyncio.create_task(heartbeat())
            try:
                error_type = None
                result = None
                try:
                    result = await self._handler(spec)
                except asyncio.CancelledError:
                    logger.info(
                        "task.lease.lost" if lease_lost else "task.interrupted",
                        extra={"event_body": {"status": "recoverable", "attempt": attempt}},
                    )
                    raise
                except Exception as exc:
                    error_type = type(exc).__name__
                    trace.get_current_span().set_status(trace.Status(trace.StatusCode.ERROR))
                    log_failure(logger, "task.failed", body={"stage": "failed", "error_type": error_type})
                await flush_logs()
                try:
                    committed = await self._store.finish(spec.task_id, token, result=result, error_type=error_type)
                    if committed:
                        terminal = await self._store.load(spec.task_id)
                        snapshot = terminal.snapshot()
                        body = {
                            "status": snapshot.status,
                            "waiting_ms": snapshot.waiting_ms,
                            "execution_ms": snapshot.execution_ms,
                            "end_to_end_ms": snapshot.end_to_end_ms,
                        }
                        for key, value in body.items():
                            trace.get_current_span().set_attribute(key, value)
                        logger.info("task.finished", extra={"event_body": body})
                    else:
                        logger.warning("task.lease.lost", extra={"event_body": {"status": "recoverable"}})
                except Exception:
                    logger.exception(
                        f"task.persistence.failed | task_id={spec.task_id}",
                        extra={"event_body": {"stage": "result_persistence", "status": "failed"}},
                    )
                    raise
            finally:
                pulse.cancel()
                await asyncio.gather(pulse, return_exceptions=True)

    async def close(self) -> None:
        """Stop consumers and leave interrupted attempts for lease recovery."""
        self._closed = True
        self._started = False
        tasks = self._background.copy()
        if self._consumer_task is not None:
            tasks = tasks | {self._consumer_task}
        for task in tasks:
            task.cancel()
        await self._dispatcher.close()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None
        if self._producer is not None:
            try:
                await self._producer.stop()
                logger.info("task.kafka.producer.stopped", extra={"event_body": {"topic": self._topic}})
            except Exception:
                logger.exception("task.kafka.producer.stop.failed")
                raise
            finally:
                self._producer = None
