# API reference

https://memu.pro/docs#memory-apis

https://memu.pro/docs#response-apis

## 记忆压缩任务

`POST /api/v1/memory/condensation?user_id=...` 接受后台任务，返回 HTTP 200 和
`{"status":"ACCEPTED","task_id":123}`。同一用户在同一 UTC 日期存在 `PROCESSING`
任务时，返回已有 ID；去重覆盖排队中和执行中的任务，并通过 PostgreSQL 事务级
advisory lock 跨 worker 生效。`COMPLETED` 或 `FAILED` 任务不阻止当天重新提交。

| 环境变量 | 缺省值 | 含义 |
| --- | --- | --- |
| `CONDENSATION_MAX_CONCURRENCY` | `1` | 每个进程同时执行的任务数，必须大于 0 |
| `CONDENSATION_MAX_QUEUE_SIZE` | `20` | 每个进程额外等待的任务数，可设为 0 |

运行和等待任务总数达到两项配置之和时，新任务返回 HTTP 429 和 `{"status":"BUSY"}`，
不会创建任务记录或压缩后台任务。重复请求仍返回已有任务 ID。8 个 worker 使用缺省值时，
整个容器最多同时执行 8 个、等待 160 个任务；这不是跨进程共享的总配额。

任务获得执行名额后才查询未删除记忆条数；少于 500 条直接完成，`content` 包含
`result: SKIP` 和 `total_items`，不读取 embedding。同步计数、读写库、结果解析和聚类均在线程中执行。

`GET /api/v1/memory/task?task_id=...` 的响应结构、状态值和字符串形式的 `content` 保持兼容。
排队期间状态仍为 `PROCESSING`，`started_at` 为 `null`；实际开始执行后填写开始时间，
`elapsed_time` 仅计算执行时间。任务不持久化调度，worker 退出后不会自动恢复，
遗留的 `PROCESSING` 记录也不会自动重新执行。

## 本地验证

```bash
uv sync --frozen --no-default-groups --group test --group lint
uv run --no-sync python -m pytest
uv run --no-sync ruff check --no-fix tests/test_condensation.py tests/test_condensation_task_store.py
```

HTTP/mock 和 SQLite 测试覆盖限并发、排队边界、低于阈值跳过 embedding、重复提交、
失败恢复、线程卸载及查询接口兼容性。另有 PostgreSQL 集成测试，使用独立的临时 schema：

```bash
MEMU_TEST_PG_URL=postgresql://localhost/memu_test uv run --no-sync python -m pytest tests/test_condensation_task_store.py
```

该连接必须指向可创建和删除测试 schema 的测试数据库；未设置时跳过 4 项 PostgreSQL 测试。
