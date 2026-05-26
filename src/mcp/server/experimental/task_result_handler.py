"""TaskResultHandler - Integrated handler for tasks/result endpoint.

This implements the dequeue-send-wait pattern from the MCP Tasks spec:
1. Dequeue all pending messages for the task
2. Send them to the client via transport with relatedRequestId routing
3. Wait if task is not in terminal state
4. Return final result when task completes

This is the core of the task message queue pattern.
"""

import logging
from typing import Any

import anyio

from mcp import types
from mcp.server.session import ServerSession
from mcp.shared.exceptions import MCPError
from mcp.shared.experimental.tasks.helpers import RELATED_TASK_METADATA_KEY, is_terminal
from mcp.shared.experimental.tasks.message_queue import QueuedMessage, TaskMessageQueue
from mcp.shared.experimental.tasks.store import TaskStore
from mcp.shared.message import ServerMessageMetadata, SessionMessage
from mcp.types import (
    INVALID_PARAMS,
    GetTaskPayloadRequest,
    GetTaskPayloadResult,
    RelatedTaskMetadata,
    RequestId,
)

logger = logging.getLogger(__name__)


class TaskResultHandler:
    """Handler for tasks/result that implements the message queue pattern.

    This handler:
    1. Dequeues pending messages (elicitations, notifications) for the task
    2. Sends them to the client via the response stream
    3. Waits for responses and resolves them back to callers
    4. Blocks until task reaches terminal state
    5. Returns the final result

    Usage:
        async def handle_task_result(
            ctx: ServerRequestContext, params: GetTaskPayloadRequestParams
        ) -> GetTaskPayloadResult:
            ...

        server.experimental.enable_tasks(
            on_task_result=handle_task_result,
        )
    """

    def __init__(
        self,
        store: TaskStore,
        queue: TaskMessageQueue,
    ):
        self._store = store
        self._queue = queue

    async def handle(
        self,
        request: GetTaskPayloadRequest,
        session: ServerSession,
        request_id: RequestId,
    ) -> GetTaskPayloadResult:
        """Handle a tasks/result request.

        This implements the dequeue-send-wait loop:
        1. Dequeue all pending messages
        2. Send each via transport with relatedRequestId = this request's ID
        3. If task not terminal, wait for status change
        4. Loop until task is terminal
        5. Return final result

        Args:
            request: The GetTaskPayloadRequest
            session: The server session for sending messages
            request_id: The request ID for relatedRequestId routing

        Returns:
            GetTaskPayloadResult with the task's final payload
        """
        task_id = request.params.task_id

        while True:
            task = await self._store.get_task(task_id)
            if task is None:
                raise MCPError(code=INVALID_PARAMS, message=f"Task not found: {task_id}")

            await self._deliver_queued_messages(task_id, session, request_id)

            # Re-query the task status since delivery blocks might have driven it to a terminal state
            task = await self._store.get_task(task_id)
            if task is None:
                raise MCPError(code=INVALID_PARAMS, message=f"Task not found: {task_id}")

            # If task is terminal, return result
            if is_terminal(task.status):
                result = await self._store.get_result(task_id)
                related_task = RelatedTaskMetadata(task_id=task_id)
                related_task_meta: dict[str, Any] = {RELATED_TASK_METADATA_KEY: related_task.model_dump(by_alias=True)}
                if result is not None:
                    result_data = result.model_dump(by_alias=True)
                    existing_meta: dict[str, Any] = result_data.get("_meta") or {}
                    result_data["_meta"] = {**existing_meta, **related_task_meta}
                    return GetTaskPayloadResult.model_validate(result_data)
                return GetTaskPayloadResult.model_validate({"_meta": related_task_meta})

            # Wait for task update (status change or new messages)
            await self._wait_for_task_update(task_id)

    async def _deliver_queued_messages(
        self,
        task_id: str,
        session: ServerSession,
        request_id: RequestId,
    ) -> None:
        """Dequeue and send all pending messages for a task.

        Each message is sent via standard session.send_request inside a task group
        so that responses are automatically matched and routed by the session pipeline.
        """
        metadata = ServerMessageMetadata(related_request_id=request_id)

        async with anyio.create_task_group() as tg:
            while True:
                message = await self._queue.dequeue(task_id)
                if message is None:
                    break

                logger.debug("Delivering queued message for task %s: %s", task_id, message.type)

                if message.type == "request" and message.resolver is not None:
                    tg.start_soon(self._send_and_resolve_request, session, message, metadata)
                elif message.type == "notification":
                    try:
                        # Parse the raw JSON-RPC notification dictionary back
                        # to a standard Pydantic ServerNotification model
                        notification_dict = message.message.model_dump(by_alias=True, mode="json", exclude_none=True)
                        notification = types.server_notification_adapter.validate_python(notification_dict)
                        await session.send_notification(notification, related_request_id=request_id)
                    except Exception:
                        # Fallback for custom or raw/unregistered messages: deliver directly to the write stream
                        session_message = SessionMessage(
                            message=message.message,
                            metadata=ServerMessageMetadata(related_request_id=request_id),
                        )
                        await session.send_message(session_message)

    async def _send_and_resolve_request(
        self,
        session: ServerSession,
        message: QueuedMessage,
        metadata: ServerMessageMetadata,
    ) -> None:
        """Helper to send a single enqueued request and resolve its returned result."""
        try:
            # Parse raw JSON-RPC fields back into standard Pydantic models for session.send_request
            if message.message.method == "elicitation/create":
                params_dict = message.message.params or {}
                if "url" in params_dict:
                    params = types.ElicitRequestURLParams.model_validate(params_dict)
                else:
                    params = types.ElicitRequestFormParams.model_validate(params_dict)
                request = types.ElicitRequest(params=params)
                if params.task is not None:
                    result_type = types.CreateTaskResult
                else:
                    result_type = types.ElicitResult
            elif message.message.method == "sampling/createMessage":
                params_dict = message.message.params or {}
                params = types.CreateMessageRequestParams.model_validate(params_dict)
                request = types.CreateMessageRequest(params=params)
                if params.task is not None:
                    result_type = types.CreateTaskResult
                elif params.tools is not None:
                    result_type = types.CreateMessageResultWithTools
                else:
                    result_type = types.CreateMessageResult
            else:
                raise ValueError(f"Unsupported queued request method: {message.message.method}")

            # Send standard Pydantic request model and block wait for the response model
            res = await session.send_request(
                request=request,
                result_type=result_type,
                metadata=metadata,
            )
            # Safe back-dump to dict for standard raw resolver signature
            assert message.resolver is not None, "Resolver must not be None for queued requests"
            message.resolver.set_result(res.model_dump(by_alias=True, exclude_none=True))
        except Exception as e:
            assert message.resolver is not None, "Resolver must not be None for queued requests"
            logger.exception("Failed to send and resolve enqueued task request")
            message.resolver.set_exception(e)

    async def _wait_for_task_update(self, task_id: str) -> None:
        """Wait for task to be updated (status change or new message).

        Races between store update and queue message - first one wins.
        """
        async with anyio.create_task_group() as tg:

            async def wait_for_store() -> None:
                try:
                    await self._store.wait_for_update(task_id)
                except Exception:
                    pass
                finally:
                    tg.cancel_scope.cancel()

            async def wait_for_queue() -> None:
                try:
                    await self._queue.wait_for_message(task_id)
                except Exception:
                    pass
                finally:
                    tg.cancel_scope.cancel()

            tg.start_soon(wait_for_store)
            tg.start_soon(wait_for_queue)
