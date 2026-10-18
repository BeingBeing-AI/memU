import asyncio
import importlib
import json
import logging
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("PG_URL", "sqlite://")
    monkeypatch.setenv("MEMU_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("MEMU_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("CONDENSATION_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("CONDENSATION_MAX_QUEUE_SIZE", "20")
    monkeypatch.setattr(logging, "basicConfig", Mock())
    from ext.app import ext_service

    monkeypatch.setattr(ext_service, "ExtMemoryService", Mock())
    module = importlib.import_module("main")
    monkeypatch.setattr(module, "CONDENSATION_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(module, "CONDENSATION_MAX_QUEUE_SIZE", 2)
    monkeypatch.setattr(module, "_condensation_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(module, "_condensation_admission_lock", asyncio.Lock())
    monkeypatch.setattr(module, "_condensation_tasks", set())
    yield module
    from ext.store.pg_session import shared_engine

    shared_engine.dispose()


def task_registry():
    records = {}

    def get_or_create(*, key, allow_create, **kwargs):
        if key in records:
            return records[key], False
        if not allow_create:
            return None, False
        records[key] = SimpleNamespace(id=len(records) + 1)
        return records[key], True

    return records, Mock(side_effect=get_or_create)


def client_for(api):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")


async def drain(api):
    await asyncio.wait_for(asyncio.gather(*api._condensation_tasks), timeout=5)
    await asyncio.sleep(0)
    assert not api._condensation_tasks


@pytest.mark.parametrize("concurrency", [1, 2])
def test_running_tasks_respect_semaphore(api, monkeypatch, concurrency):
    async def scenario():
        release = asyncio.Event()
        started = asyncio.Event()
        running = 0
        peak = 0
        completed = 0

        async def work(*args):
            nonlocal running, peak, completed
            running += 1
            peak = max(peak, running)
            if running == concurrency:
                started.set()
            try:
                await release.wait()
                completed += 1
            finally:
                running -= 1

        monkeypatch.setattr(api, "_condensation_semaphore", asyncio.Semaphore(concurrency))
        monkeypatch.setattr(api, "_condense_memory", work)
        jobs = [asyncio.create_task(api._run_condensation_task(i, str(i))) for i in range(6)]
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            assert running == concurrency
            assert sum(not job.done() for job in jobs) == 6
        finally:
            release.set()
            await asyncio.gather(*jobs)
        assert peak == concurrency
        assert completed == 6

    asyncio.run(scenario())


@pytest.mark.parametrize("queue_size", [0, 2])
def test_burst_admission_is_bounded_and_full_queue_reuses_task(api, monkeypatch, queue_size):
    async def scenario():
        release = asyncio.Event()
        records, create = task_registry()
        monkeypatch.setattr(api, "CONDENSATION_MAX_QUEUE_SIZE", queue_size)
        monkeypatch.setattr(api, "get_or_create_processing_task", create)

        async def work(*args):
            await release.wait()

        monkeypatch.setattr(api, "_condense_memory", work)
        async with client_for(api) as client:
            try:
                responses = await asyncio.gather(
                    *(client.post("/api/v1/memory/condensation", params={"user_id": str(i)}) for i in range(8))
                )
                accepted = [response for response in responses if response.status_code == 200]
                busy = [response for response in responses if response.status_code == 429]
                assert len(accepted) == 1 + queue_size
                assert len(busy) == 7 - queue_size
                assert all(response.json() == {"status": "BUSY"} for response in busy)
                assert len(api._condensation_tasks) == 1 + queue_size
                assert len(records) == 1 + queue_size

                user_id = next(iter(records)).split(":")[0]
                duplicate = await client.post("/api/v1/memory/condensation", params={"user_id": user_id})
                assert duplicate.json() == {"status": "ACCEPTED", "task_id": 1}

                # A task accepted by a different worker is reused even when this worker is full.
                records[f"remote:{datetime.utcnow().date().isoformat()}"] = SimpleNamespace(id=99)
                remote = await client.post("/api/v1/memory/condensation", params={"user_id": "remote"})
                assert remote.json() == {"status": "ACCEPTED", "task_id": 99}
                assert len(api._condensation_tasks) == 1 + queue_size
            finally:
                release.set()
                await drain(api)
            response = await client.post("/api/v1/memory/condensation", params={"user_id": "next"})
            assert response.status_code == 200
            await drain(api)

    asyncio.run(scenario())


def test_simultaneous_duplicate_requests_schedule_once(api, monkeypatch):
    async def scenario():
        release = asyncio.Event()
        records, create = task_registry()

        async def work(*args):
            await release.wait()

        monkeypatch.setattr(api, "get_or_create_processing_task", create)
        monkeypatch.setattr(api, "_condense_memory", work)
        async with client_for(api) as client:
            try:
                responses = await asyncio.gather(
                    *(client.post("/api/v1/memory/condensation", params={"user_id": "same"}) for _ in range(12))
                )
                assert all(response.json() == {"status": "ACCEPTED", "task_id": 1} for response in responses)
                assert len(records) == 1
                assert len(api._condensation_tasks) == 1
            finally:
                release.set()
                await drain(api)

    asyncio.run(scenario())


@pytest.mark.parametrize("count", [0, 499])
def test_below_threshold_does_not_load_embeddings(api, monkeypatch, count):
    update = Mock()
    load = Mock()
    cluster = Mock()
    monkeypatch.setattr(api, "count_memory_items", Mock(return_value=count))
    monkeypatch.setattr(api, "get_all_memory_items", load)
    monkeypatch.setattr(api, "cluster_memories", cluster)
    monkeypatch.setattr(api, "update_task", update)

    asyncio.run(api._run_condensation_task(1, "small"))

    load.assert_not_called()
    cluster.assert_not_called()
    assert update.call_args.kwargs["status"] == "COMPLETED"
    assert update.call_args.kwargs["content"] == {
        "result": "SKIP",
        "message": "not enough memory items",
        "total_items": count,
    }


def test_threshold_runs_sync_work_in_threads(api, monkeypatch):
    loop_thread = threading.get_ident()
    calls = []
    old_items = [SimpleNamespace(id=str(i)) for i in range(500)]
    new_items = [SimpleNamespace(summary="merged", embedding=None)]

    def threaded(name, result):
        def run(*args, **kwargs):
            assert threading.get_ident() != loop_thread
            calls.append(name)
            return result

        return Mock(side_effect=run)

    load = threaded("load", old_items)
    update = threaded("status", True)
    write = threaded("write", (2, 1))
    monkeypatch.setattr(api, "count_memory_items", threaded("count", 500))
    monkeypatch.setattr(api, "get_all_memory_items", load)
    monkeypatch.setattr(api, "cluster_memories", threaded("cluster", {0: old_items[:2]}))
    monkeypatch.setattr(api, "parse_condensation_result", threaded("parse", new_items))
    monkeypatch.setattr(api, "update_condensation_items", write)
    monkeypatch.setattr(api, "update_task", update)
    monkeypatch.setattr(api, "condensation_memory_items", AsyncMock(return_value=("raw", "merged")))
    monkeypatch.setattr(
        api,
        "memory_service",
        SimpleNamespace(
            llm_client=object(),
            embedding_client=SimpleNamespace(embed=AsyncMock(return_value=[[0.1, 0.2]])),
        ),
    )

    asyncio.run(api._run_condensation_task(1, "large"))

    assert calls == ["status", "count", "load", "cluster", "parse", "write", "status"]
    load.assert_called_once_with("large", include_embedding=True)
    write.assert_called_once_with("large", ["0", "1"], new_items)
    assert new_items[0].embedding == [0.1, 0.2]
    assert update.call_args.kwargs["content"] == {"message": "completed", "delete_count": 2, "insert_count": 1}


def test_failure_releases_capacity_and_marks_failed(api, monkeypatch):
    async def scenario():
        _, create = task_registry()
        update = Mock()
        monkeypatch.setattr(api, "get_or_create_processing_task", create)
        monkeypatch.setattr(api, "count_memory_items", Mock(side_effect=RuntimeError("database unavailable")))
        monkeypatch.setattr(api, "update_task", update)
        async with client_for(api) as client:
            response = await client.post("/api/v1/memory/condensation", params={"user_id": "failure"})
            assert response.status_code == 200
            await drain(api)
            assert update.call_args.kwargs["status"] == "FAILED"
            assert update.call_args.kwargs["increment_failure_count"] is True
            monkeypatch.setattr(api, "count_memory_items", Mock(return_value=0))
            response = await client.post("/api/v1/memory/condensation", params={"user_id": "next"})
            assert response.status_code == 200
            await drain(api)
            assert update.call_args.kwargs["status"] == "COMPLETED"

    asyncio.run(scenario())


def test_cancelled_request_still_schedules_committed_task(api, monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        work = AsyncMock()

        def create(**kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(timeout=5)
            return SimpleNamespace(id=42), True

        monkeypatch.setattr(api, "get_or_create_processing_task", create)
        monkeypatch.setattr(api, "_condense_memory", work)
        request = asyncio.create_task(api.condensation("cancelled"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            request.cancel()
            await asyncio.sleep(0)
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        await drain(api)
        work.assert_awaited_once_with(42, "cancelled")

    asyncio.run(scenario())


def test_task_polling_response_remains_compatible(api, monkeypatch):
    loop_thread = threading.get_ident()
    timestamp = datetime(2026, 10, 18, 2)
    fields = {
        "id": 7,
        "type": "MEMORY_CONDENSATION",
        "name": "memory-condensation-user",
        "key": "user:2026-10-18",
        "status": "PROCESSING",
        "content": json.dumps({"user_id": "user"}),
        "started_at": None,
        "completed_at": None,
        "elapsed_time": None,
        "failure_count": 0,
        "created_at": timestamp,
        "updated_at": timestamp,
    }

    def lookup(task_id):
        assert threading.get_ident() != loop_thread
        return SimpleNamespace(**fields) if task_id == 7 else None

    monkeypatch.setattr(api, "get_task_by_id", lookup)

    async def scenario():
        async with client_for(api) as client:
            response = await client.get("/api/v1/memory/task", params={"task_id": 7})
            expected = fields | {"created_at": timestamp.isoformat(), "updated_at": timestamp.isoformat()}
            assert response.json() == {"status": "SUCCESS", "task": expected}
            response = await client.get("/api/v1/memory/task", params={"task_id": 8})
            assert response.status_code == 404
            assert response.json() == {"detail": "Task not found"}

    asyncio.run(scenario())


def test_count_filters_user_and_deleted_rows_without_selecting_embeddings(api, monkeypatch):
    from ext.store import memory_item_store

    engine = create_engine("sqlite://")
    statements = []
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE memory_items (user_id TEXT, is_deleted BOOLEAN, embedding TEXT)"))
            connection.execute(
                text(
                    "INSERT INTO memory_items VALUES ('a', 0, 'vector'), ('a', 0, 'vector'), "
                    "('a', 1, 'vector'), ('b', 0, 'vector')"
                )
            )

        @event.listens_for(engine, "before_cursor_execute")
        def capture(connection, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        monkeypatch.setattr(memory_item_store, "shared_engine", SimpleNamespace(session=sessionmaker(bind=engine)))
        assert memory_item_store.count_memory_items("a") == 2
        assert memory_item_store.count_memory_items("missing") == 0
        assert all("count(" in statement.lower() and "embedding" not in statement.lower() for statement in statements)
    finally:
        engine.dispose()
