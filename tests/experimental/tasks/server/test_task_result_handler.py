"""Tests for TaskResultHandler."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock

import anyio
import pytest

from mcp.server.experimental.task_result_handler import TaskResultHandler
from mcp.shared.exceptions import MCPError
from mcp.shared.experimental.tasks.in_memory_task_store import InMemoryTaskStore
from mcp.shared.experimental.tasks.message_queue import InMemoryTaskMessageQueue, QueuedMessage
from mcp.shared.experimental.tasks.resolver import Resolver
from mcp.shared.message import SessionMessage
from mcp.types import (
    CallToolResult,
    GetTaskPayloadRequest,
    GetTaskPayloadRequestParams,
    GetTaskPayloadResult,
    JSONRPCRequest,
    TaskMetadata,
    TextContent,
)


@pytest.fixture
async def store() -> AsyncIterator[InMemoryTaskStore]:
    """Provide a clean store for each test."""
    s = InMemoryTaskStore()
    yield s
    s.cleanup()


@pytest.fixture
def queue() -> InMemoryTaskMessageQueue:
    """Provide a clean queue for each test."""
    return InMemoryTaskMessageQueue()


@pytest.fixture
def handler(store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue) -> TaskResultHandler:
    """Provide a handler for each test."""
    return TaskResultHandler(store, queue)


@pytest.mark.anyio
async def test_handle_returns_result_for_completed_task(
    store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue, handler: TaskResultHandler
) -> None:
    """Test that handle() returns the stored result for a completed task."""
    task = await store.create_task(TaskMetadata(ttl=60000), task_id="test-task")
    result = CallToolResult(content=[TextContent(type="text", text="Done!")])
    await store.store_result(task.task_id, result)
    await store.update_task(task.task_id, status="completed")

    mock_session = Mock()
    mock_session.send_message = AsyncMock()

    request = GetTaskPayloadRequest(params=GetTaskPayloadRequestParams(task_id=task.task_id))
    response = await handler.handle(request, mock_session, "req-1")

    assert response is not None
    assert response.meta is not None
    assert "io.modelcontextprotocol/related-task" in response.meta


@pytest.mark.anyio
async def test_handle_raises_for_nonexistent_task(
    store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue, handler: TaskResultHandler
) -> None:
    """Test that handle() raises MCPError for nonexistent task."""
    mock_session = Mock()
    request = GetTaskPayloadRequest(params=GetTaskPayloadRequestParams(task_id="nonexistent"))

    with pytest.raises(MCPError) as exc_info:
        await handler.handle(request, mock_session, "req-1")

    assert "not found" in exc_info.value.error.message


@pytest.mark.anyio
async def test_handle_returns_empty_result_when_no_result_stored(
    store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue, handler: TaskResultHandler
) -> None:
    """Test that handle() returns minimal result when task completed without stored result."""
    task = await store.create_task(TaskMetadata(ttl=60000), task_id="test-task")
    await store.update_task(task.task_id, status="completed")

    mock_session = Mock()
    mock_session.send_message = AsyncMock()

    request = GetTaskPayloadRequest(params=GetTaskPayloadRequestParams(task_id=task.task_id))
    response = await handler.handle(request, mock_session, "req-1")

    assert response is not None
    assert response.meta is not None
    assert "io.modelcontextprotocol/related-task" in response.meta


@pytest.mark.anyio
async def test_handle_delivers_queued_messages(
    store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue, handler: TaskResultHandler
) -> None:
    """Test that handle() delivers queued messages before returning."""
    task = await store.create_task(TaskMetadata(ttl=60000), task_id="test-task")

    queued_msg = QueuedMessage(
        type="notification",
        message=JSONRPCRequest(
            jsonrpc="2.0",
            id="notif-1",
            method="test/notification",
            params={},
        ),
    )
    await queue.enqueue(task.task_id, queued_msg)
    await store.update_task(task.task_id, status="completed")

    sent_messages: list[SessionMessage] = []

    async def track_send(msg: SessionMessage) -> None:
        sent_messages.append(msg)

    mock_session = Mock()
    mock_session.send_message = track_send

    request = GetTaskPayloadRequest(params=GetTaskPayloadRequestParams(task_id=task.task_id))
    await handler.handle(request, mock_session, "req-1")

    assert len(sent_messages) == 1


@pytest.mark.anyio
async def test_handle_waits_for_task_completion(
    store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue, handler: TaskResultHandler
) -> None:
    """Test that handle() waits for task to complete before returning."""
    task = await store.create_task(TaskMetadata(ttl=60000), task_id="test-task")

    mock_session = Mock()
    mock_session.send_message = AsyncMock()

    request = GetTaskPayloadRequest(params=GetTaskPayloadRequestParams(task_id=task.task_id))
    result_holder: list[GetTaskPayloadResult | None] = [None]

    async def run_handle() -> None:
        result_holder[0] = await handler.handle(request, mock_session, "req-1")

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_handle)

        # Wait for handler to start waiting (event gets created when wait starts)
        while task.task_id not in store._update_events:
            await anyio.sleep(0)

        await store.store_result(task.task_id, CallToolResult(content=[TextContent(type="text", text="Done")]))
        await store.update_task(task.task_id, status="completed")

    assert result_holder[0] is not None


@pytest.mark.anyio
async def test_deliver_sends_request_and_resolves_resolver(
    store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue, handler: TaskResultHandler
) -> None:
    """Test that _deliver_queued_messages sends requests via session.send_request and sets the resolver result."""
    task = await store.create_task(TaskMetadata(ttl=60000), task_id="test-task")

    resolver: Resolver[dict[str, Any]] = Resolver()
    queued_msg = QueuedMessage(
        type="request",
        message=JSONRPCRequest(
            jsonrpc="2.0",
            id="inner-req-1",
            method="elicitation/create",
            params={"message": "continue?", "requestedSchema": {"type": "object"}},
        ),
        resolver=resolver,
        original_request_id="inner-req-1",
    )
    await queue.enqueue(task.task_id, queued_msg)

    from mcp import types

    mock_session = AsyncMock()
    mock_session.send_request = AsyncMock(return_value=types.ElicitResult(action="accept", content={"ok": True}))

    await handler._deliver_queued_messages(task.task_id, mock_session, "outer-req-1")

    # Assert send_request was called with parsed ElicitRequest!
    mock_session.send_request.assert_called_once()
    args, kwargs = mock_session.send_request.call_args
    request_arg = kwargs.get("request") or args[0]
    assert isinstance(request_arg, types.ElicitRequest)
    assert request_arg.params.message == "continue?"
    assert (kwargs.get("result_type") or args[1]) == types.ElicitResult

    # Assert the resolver got completed with the return value!
    assert resolver.done()
    res = await resolver.wait()
    assert res == {"action": "accept", "content": {"ok": True}}


@pytest.mark.anyio
async def test_deliver_sends_sampling_request_and_resolves_resolver(
    store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue, handler: TaskResultHandler
) -> None:
    """Test that _deliver_queued_messages sends sampling requests and resolves the resolver."""
    task = await store.create_task(TaskMetadata(ttl=60000), task_id="test-task")

    resolver: Resolver[dict[str, Any]] = Resolver()
    queued_msg = QueuedMessage(
        type="request",
        message=JSONRPCRequest(
            jsonrpc="2.0",
            id="req-no-tools",
            method="sampling/createMessage",
            params={"messages": [], "max_tokens": 10},
        ),
        resolver=resolver,
        original_request_id="req-no-tools",
    )
    await queue.enqueue(task.task_id, queued_msg)

    from mcp import types

    mock_session = AsyncMock()
    mock_session.send_request = AsyncMock(
        return_value=types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text="hello"),
            model="mock-model",
        )
    )

    await handler._deliver_queued_messages(task.task_id, mock_session, "outer-req-1")

    # Verify it requested CreateMessageResult
    args, kwargs = mock_session.send_request.call_args
    assert (kwargs.get("result_type") or args[1]) == types.CreateMessageResult
    assert resolver.done()
    res = await resolver.wait()
    assert res["content"]["text"] == "hello"
    assert res["role"] == "assistant"
    assert res["model"] == "mock-model"


@pytest.mark.anyio
async def test_wait_for_task_update_handles_store_exception(
    store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue, handler: TaskResultHandler
) -> None:
    """Test that _wait_for_task_update handles store exception gracefully."""
    task = await store.create_task(TaskMetadata(ttl=60000), task_id="test-task")

    # Make wait_for_update raise an exception
    async def failing_wait(task_id: str) -> None:
        raise RuntimeError("Store error")

    store.wait_for_update = failing_wait  # type: ignore[method-assign]

    # Queue a message to unblock the race via the queue path
    async def enqueue_later() -> None:
        # Wait for queue to start waiting (event gets created when wait starts)
        while task.task_id not in queue._events:
            await anyio.sleep(0)
        await queue.enqueue(
            task.task_id,
            QueuedMessage(
                type="notification",
                message=JSONRPCRequest(
                    jsonrpc="2.0",
                    id="notif-1",
                    method="test/notification",
                    params={},
                ),
            ),
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(enqueue_later)
        # This should complete via the queue path even though store raises
        await handler._wait_for_task_update(task.task_id)


@pytest.mark.anyio
async def test_wait_for_task_update_handles_queue_exception(
    store: InMemoryTaskStore, queue: InMemoryTaskMessageQueue, handler: TaskResultHandler
) -> None:
    """Test that _wait_for_task_update handles queue exception gracefully."""
    task = await store.create_task(TaskMetadata(ttl=60000), task_id="test-task")

    # Make wait_for_message raise an exception
    async def failing_wait(task_id: str) -> None:
        raise RuntimeError("Queue error")

    queue.wait_for_message = failing_wait  # type: ignore[method-assign]

    # Update the store to unblock the race via the store path
    async def update_later() -> None:
        # Wait for store to start waiting (event gets created when wait starts)
        while task.task_id not in store._update_events:
            await anyio.sleep(0)
        await store.update_task(task.task_id, status="completed")

    async with anyio.create_task_group() as tg:
        tg.start_soon(update_later)
        # This should complete via the store path even though queue raises
        await handler._wait_for_task_update(task.task_id)
