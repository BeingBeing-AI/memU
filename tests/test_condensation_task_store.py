import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def task_store(monkeypatch):
    url = os.getenv("MEMU_TEST_PG_URL")
    if not url:
        pytest.skip("Set MEMU_TEST_PG_URL to a disposable PostgreSQL database for advisory-lock tests")
    monkeypatch.setenv("PG_URL", "sqlite://")
    from ext.store import pg_session
    from ext.store import task_store as store

    schema = f"condensation_test_{uuid.uuid4().hex}"
    admin = create_engine(url)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"}, pool_size=12)
    try:
        store.TaskModel.__table__.create(engine)
        monkeypatch.setattr(store, "shared_engine", SimpleNamespace(session=sessionmaker(bind=engine)))
        yield store
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()
        pg_session.shared_engine.dispose()


def submit(store, key="user:2026-10-18", allow_create=True):
    return store.get_or_create_processing_task(
        task_type="MEMORY_CONDENSATION",
        key=key,
        name="memory-condensation-user",
        content={"user_id": "user"},
        allow_create=allow_create,
    )


def test_concurrent_database_sessions_create_one_processing_task(task_store):
    barrier = threading.Barrier(12, timeout=10)

    def concurrent_submit(_):
        barrier.wait()
        return submit(task_store)

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(concurrent_submit, range(12)))

    assert sum(created for _, created in results) == 1
    assert len({task.id for task, _ in results}) == 1
    assert all(task.status == "PROCESSING" and task.started_at is None for task, _ in results)
    with task_store.shared_engine.session() as session:
        assert session.query(task_store.TaskModel).count() == 1


def test_full_worker_can_reuse_but_cannot_create(task_store):
    assert submit(task_store, allow_create=False) == (None, False)
    task, created = submit(task_store)
    assert created
    reused, created = submit(task_store, allow_create=False)
    assert not created
    assert reused.id == task.id
    assert submit(task_store, key="other:2026-10-18", allow_create=False) == (None, False)


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED"])
def test_terminal_tasks_and_different_dates_do_not_deduplicate(task_store, status):
    first, _ = submit(task_store)
    task_store.update_task(first.id, status=status)
    retry, created = submit(task_store)
    assert created
    assert retry.id != first.id
    next_day, created = submit(task_store, key="user:2026-10-19")
    assert created
    assert next_day.id != retry.id
