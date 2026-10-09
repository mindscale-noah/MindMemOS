"""Kafka notifications for durable tasks stored in the existing database."""

import asyncio
import json
from collections.abc import Awaitable

from ...common.asyncio import run_sync
from ...logging import get_logger
from .base import TaskHandler, TaskManager
from .dispatcher import TaskDispatcher
from .progress import EventProgressReporter
from .typing import ProgressEvent, TaskSnapshot, TaskSpec

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
            max_concurrency: Maximum simultaneous algorithms per worker process.
            max_buffered: Maximum queued and running notifications admitted locally.
            scan_interval_seconds: Delay between missed-delivery recovery scans.
        """
        if max_concurrency < 1 or scan_interval_seconds <= 0:
            raise ValueError("Worker concurrency and scan interval must be positive")
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
        self._spawn(self._connect_loop(), name="skill-kafka-connection")
        self._spawn(self._scan(), name="skill-task-recovery")

    async def _connect_loop(self) -> None:
        """Reconnect the shared Kafka clients while DB-backed submissions continue."""
        while True:
            try:
                if self._producer is None:
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
                    self._consumer_task = asyncio.create_task(self._consume(), name="skill-task-consumer")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("task.kafka.connect.failed")
            await asyncio.sleep(self._scan_interval)

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
        """Enqueue without awaiting the broker acknowledgement, as Memory does."""
        if self._producer is None:
            raise RuntimeError("Kafka producer is not connected")
        future = await self._producer.send(self._topic, json.dumps({"task_id": task_id}).encode(), key=task_id.encode())

        def delivered(done: asyncio.Future) -> None:
            """Observe background delivery errors without blocking submitters."""
            if not done.cancelled() and done.exception() is not None:
                logger.error("task.kafka.delivery.failed", extra={"event_body": {"task_id": task_id}})

        future.add_done_callback(delivered)

    async def submit(self, spec: TaskSpec) -> TaskSnapshot:
        """Commit the task first, then attempt a non-blocking Kafka notification."""
        if not self._started:
            raise RuntimeError("Kafka task manager is not started")
        record = await run_sync(self._store.create, spec)
        if self._producer is None:
            return record.snapshot()
        try:
            await asyncio.wait_for(self._notify(spec.task_id), timeout=ENQUEUE_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("task.kafka.enqueue.failed", extra={"event_body": {"task_id": spec.task_id}})
        else:
            try:
                await run_sync(self._store.defer_notification, spec.task_id)
            except Exception:
                logger.exception(
                    "task.kafka.notification_schedule.failed", extra={"event_body": {"task_id": spec.task_id}}
                )
        return record.snapshot()

    async def get(self, project_id: str, task_id: str) -> TaskSnapshot | None:
        """Read a task from the existing project-owned database."""
        record = await run_sync(self._store.get, project_id, task_id)
        return record.snapshot() if record is not None else None

    async def events(self, project_id: str, task_id: str, *, after: int = 0) -> list[ProgressEvent]:
        """Read progress committed to the existing database."""
        return await run_sync(self._store.events, project_id, task_id, after=after)

    async def _scan(self) -> None:
        """Republish queued tasks and reclaim expired work after process failure."""
        while True:
            try:
                if self._producer is not None:
                    for record in await run_sync(self._store.due):
                        try:
                            await asyncio.wait_for(self._notify(record.task_id), timeout=ENQUEUE_TIMEOUT_SECONDS)
                        except Exception:
                            logger.exception(
                                "task.kafka.recovery_notify.failed", extra={"event_body": {"task_id": record.task_id}}
                            )
                        else:
                            await run_sync(self._store.defer_notification, record.task_id)
                if self._handler is not None:
                    for record in await run_sync(self._store.expired):
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
            logger.exception("task.kafka.consumer.failed")
        finally:
            await consumer.stop()
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
        claimed = await run_sync(self._store.claim, task_id)
        if claimed is None:
            return
        record, token = claimed
        await self._run(record.spec(), token)

    async def _run(self, spec: TaskSpec, token: str) -> None:
        """Run a fenced attempt while heartbeats protect its ownership."""
        assert self._handler is not None
        current = asyncio.current_task()

        async def heartbeat() -> None:
            """Cancel an attempt that loses its database lease."""
            while True:
                await asyncio.sleep(30)
                try:
                    owned = await run_sync(self._store.heartbeat, spec.task_id, token)
                except Exception:
                    logger.exception("task.lease.heartbeat.failed", extra={"event_body": {"task_id": spec.task_id}})
                    owned = False
                if not owned:
                    if current is not None:
                        current.cancel()
                    return

        async def record(event: ProgressEvent) -> str:
            """Commit a progress event under the current lease."""
            return await run_sync(self._store.append_progress, spec.task_id, token, event)

        pulse = asyncio.create_task(heartbeat())
        try:
            result = await self._handler(spec, EventProgressReporter(record, task_id=spec.task_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await run_sync(self._store.finish, spec.task_id, token, error_type=type(exc).__name__)
            logger.exception("task.algorithm.failed", extra={"event_body": {"task_id": spec.task_id}})
        else:
            await run_sync(self._store.finish, spec.task_id, token, result=result)
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
            await self._producer.stop()
            self._producer = None
