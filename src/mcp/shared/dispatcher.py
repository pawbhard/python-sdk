"""Dispatcher abstraction: the wire-protocol layer beneath a session.

A ``Dispatcher`` is responsible for encoding MCP messages for the wire,
correlating request/response pairs, and delivering incoming messages to
session-provided handlers. The session itself deals only in MCP-level
dicts (``{"method": ..., "params": ...}`` for requests/notifications, result
dicts for responses) and never sees the wire encoding.

The default ``JSONRPCDispatcher`` wraps messages in JSON-RPC 2.0 envelopes
and exchanges them over anyio memory streams — this is what every built-in
transport (stdio, Streamable HTTP, WebSocket) feeds into. Custom dispatchers
may use other encodings and request/response models as long as MCP's
request/notification/response semantics are preserved.

Design note — ``ReplyHandle``
------------------------------
``on_request`` delivers a ``ReplyHandle`` alongside each incoming request.
The handle is an **opaque token** whose meaning is determined entirely by the
dispatcher implementation:

* ``JSONRPCDispatcher``: ``ReplyHandle = RequestId`` — it's the numeric JSON-RPC
  request ID, used to build the matching ``JSONRPCResponse`` envelope.
* A gRPC dispatcher: ``ReplyHandle`` would be the ``grpc.ServicerContext`` for that
  call — ``send_response`` writes directly to the call's stream, no ID lookup.
* A per-request HTTP dispatcher: ``ReplyHandle`` would be the open SSE response
  stream for that POST, again no bookkeeping dictionary needed.

The session never inspects the handle; it just stores it in ``RequestResponder``
and passes it back to ``send_response``. Bookkeeping (if any) lives inside the
dispatcher that needs it, not in the protocol boundary.

!!! warning
    The ``Dispatcher`` Protocol is experimental. Custom transports that
    carry JSON-RPC should implement the ``Transport`` Protocol from
    ``mcp.client._transport`` instead — that path is stable.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeAlias

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from mcp.shared.exceptions import MCPError
from mcp.shared.message import MessageMetadata, ServerMessageMetadata, SessionMessage
from mcp.shared.response_router import ResponseRouter
from mcp.types import (
    CONNECTION_CLOSED,
    ErrorData,
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    RequestId,
)

ReplyHandle: TypeAlias = Any
"""Opaque per-transport token for routing a response back to the right peer channel.

The session stores it in ``RequestResponder`` and passes it verbatim to
``send_response``. Dispatchers may use any type — a numeric ID, a connection
object, a stream reference — as long as it uniquely identifies the inbound
request within that transport.
"""

OnRequestFn = Callable[[RequestId, ReplyHandle, dict[str, Any], MessageMetadata], Awaitable[None]]
"""Called when the peer sends us a request.

Receives ``(request_id, reply_handle, {"method", "params"}, metadata)``.

``request_id`` is the MCP-level identifier used for cancellation and progress
tracking. ``reply_handle`` is the opaque routing token — pass it back verbatim
to ``send_response``.
"""

OnNotificationFn = Callable[[dict[str, Any]], Awaitable[None]]
"""Called when the peer sends us a notification. Receives ``{"method", "params"}``."""

OnErrorFn = Callable[[Exception], Awaitable[None]]
"""Called for transport-level errors and orphaned error responses."""


class Dispatcher(Protocol):
    """Wire-protocol layer beneath ``BaseSession``.

    The dispatcher owns request correlation. ``send_request`` generates whatever
    wire-level identifier its protocol needs internally — the session never sees
    it. The session's own ``_request_id`` counter is used only for progress tokens
    and is independent of the dispatcher's internal IDs.

    ``send_response`` takes an opaque ``ReplyHandle`` delivered by ``on_request``.
    For call-based transports (gRPC, per-request HTTP) the handle IS the channel,
    so no bookkeeping dictionary is needed. For stream-based transports (stdio,
    shared SSE) the handle contains the ID needed to look up the right waiter —
    that lookup lives inside the dispatcher, not in the session.

    Implementations must be cancellation-safe: if ``send_request`` is cancelled
    (e.g. by the session's timeout), any internal correlation state for that
    request must be cleaned up.
    """

    def set_handlers(
        self,
        on_request: OnRequestFn,
        on_notification: OnNotificationFn,
        on_error: OnErrorFn,
    ) -> None:
        """Wire incoming-message callbacks. Called once, before ``run``."""
        ...

    async def run(self) -> None:
        """Run the receive loop. Returns when the connection closes.

        Started in the session's task group; cancelled on session exit.
        """
        ...

    async def send_request(
        self,
        request: dict[str, Any],
        metadata: MessageMetadata = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a request and wait for its response.

        ``request`` is ``{"method": str, "params": dict | None}``. The dispatcher
        generates its own wire-level request identifier internally. Returns the raw
        result dict. Raises ``MCPError`` if the peer returns an error response.
        Raises ``TimeoutError`` if no response arrives within ``timeout``.

        The send itself must not be subject to the timeout — only the wait for
        the response — so that requests are reliably delivered even when the
        caller sets an aggressive deadline.
        """
        ...

    async def send_notification(
        self,
        notification: dict[str, Any],
        related_reply_handle: ReplyHandle | None = None,
    ) -> None:
        """Send a fire-and-forget notification. ``notification`` is ``{"method", "params"}``.

        ``related_reply_handle`` is the handle of the inbound request that triggered
        this notification, if any. Transports may use it to route the notification
        to the correct peer channel (e.g. the SSE stream for that HTTP request).
        """
        ...

    async def send_response(
        self,
        handle: ReplyHandle,
        response: dict[str, Any] | ErrorData,
    ) -> None:
        """Send a response to a request we previously received via ``on_request``.

        ``handle`` is the opaque token delivered to ``on_request`` — pass it back
        verbatim. The dispatcher uses it to route the response to the right channel
        without any external bookkeeping.
        """
        ...


class JSONRPCDispatcher:
    """Default dispatcher: JSON-RPC 2.0 over anyio memory streams.

    This is the behaviour ``BaseSession`` had before the dispatcher extraction —
    every built-in transport produces a pair of streams that feed into here.

    ``ReplyHandle`` for this dispatcher is ``RequestId`` (an int). The handle is
    the JSON-RPC ``id`` field from the inbound request, used to build the matching
    ``JSONRPCResponse`` envelope in ``send_response``. The ``_response_streams``
    dict (bookkeeping for outbound requests) lives entirely in this class and is
    invisible to the session.
    """

    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
        write_stream: MemoryObjectSendStream[SessionMessage],
        response_routers: list[ResponseRouter],
    ) -> None:
        self._read_stream = read_stream
        self._write_stream = write_stream
        self._response_routers = response_routers
        self._response_streams: dict[RequestId, MemoryObjectSendStream[JSONRPCResponse | JSONRPCError]] = {}
        self._next_id: int = 0
        self._on_request: OnRequestFn | None = None
        self._on_notification: OnNotificationFn | None = None
        self._on_error: OnErrorFn | None = None

    def set_handlers(
        self,
        on_request: OnRequestFn,
        on_notification: OnNotificationFn,
        on_error: OnErrorFn,
    ) -> None:
        self._on_request = on_request
        self._on_notification = on_notification
        self._on_error = on_error

    async def send_request(
        self,
        request: dict[str, Any],
        metadata: MessageMetadata = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        # Generate wire ID internally — the session never sees this value.
        request_id = self._next_id
        self._next_id += 1

        response_stream, response_stream_reader = anyio.create_memory_object_stream[JSONRPCResponse | JSONRPCError](1)
        self._response_streams[request_id] = response_stream
        try:
            jsonrpc_request = JSONRPCRequest(jsonrpc="2.0", id=request_id, **request)
            await self._write_stream.send(SessionMessage(message=jsonrpc_request, metadata=metadata))
            with anyio.fail_after(timeout):
                response_or_error = await response_stream_reader.receive()
            if isinstance(response_or_error, JSONRPCError):
                raise MCPError.from_jsonrpc_error(response_or_error)
            return response_or_error.result
        finally:
            self._response_streams.pop(request_id, None)
            await response_stream.aclose()
            await response_stream_reader.aclose()

    async def send_notification(
        self,
        notification: dict[str, Any],
        related_reply_handle: ReplyHandle | None = None,
    ) -> None:
        # For JSON-RPC, the reply handle is the RequestId — use it as related_request_id.
        related_request_id: RequestId | None = related_reply_handle
        jsonrpc_notification = JSONRPCNotification(jsonrpc="2.0", **notification)
        session_message = SessionMessage(
            message=jsonrpc_notification,
            metadata=ServerMessageMetadata(related_request_id=related_request_id) if related_request_id else None,
        )
        await self._write_stream.send(session_message)

    async def send_response(
        self,
        handle: ReplyHandle,
        response: dict[str, Any] | ErrorData,
    ) -> None:
        # For JSON-RPC, the handle is the RequestId from the inbound JSONRPCRequest.
        request_id: RequestId = handle
        if isinstance(response, ErrorData):
            message: JSONRPCResponse | JSONRPCError = JSONRPCError(jsonrpc="2.0", id=request_id, error=response)
        else:
            message = JSONRPCResponse(jsonrpc="2.0", id=request_id, result=response)
        await self._write_stream.send(SessionMessage(message=message))

    async def run(self) -> None:
        assert self._on_request is not None
        assert self._on_notification is not None
        assert self._on_error is not None

        async with self._read_stream, self._write_stream:
            try:
                async for message in self._read_stream:
                    if isinstance(message, Exception):
                        await self._on_error(message)
                    elif isinstance(message.message, JSONRPCRequest):
                        # For JSON-RPC, request_id and reply_handle are the same value:
                        # the numeric ID used both for cancellation tracking and for
                        # building the response envelope. Other dispatchers may differ.
                        request_id = message.message.id
                        await self._on_request(
                            request_id,
                            request_id,  # reply_handle
                            message.message.model_dump(by_alias=True, mode="json", exclude_none=True),
                            message.metadata,
                        )
                    elif isinstance(message.message, JSONRPCNotification):
                        await self._on_notification(
                            message.message.model_dump(by_alias=True, mode="json", exclude_none=True)
                        )
                    else:
                        await self._route_response(message)
            except anyio.ClosedResourceError:
                # Expected when the peer disconnects abruptly.
                logging.debug("Read stream closed by client")
            except Exception as e:
                logging.exception(f"Unhandled exception in receive loop: {e}")  # pragma: no cover
            finally:
                # Deliver CONNECTION_CLOSED to every outbound request still awaiting a response.
                for id, stream in self._response_streams.items():
                    error = ErrorData(code=CONNECTION_CLOSED, message="Connection closed")
                    try:
                        await stream.send(JSONRPCError(jsonrpc="2.0", id=id, error=error))
                        await stream.aclose()
                    except Exception:  # pragma: no cover
                        pass
                self._response_streams.clear()
                # Handlers are bound methods of the session; the session holds us. Break
                # the cycle so refcount GC can free both promptly.
                self._on_request = None
                self._on_notification = None
                self._on_error = None

    async def _route_response(self, message: SessionMessage) -> None:
        # Runtime-true (run() only calls us in the response/error branch) but the
        # type checker can't see that, hence the guard.
        if not isinstance(message.message, JSONRPCResponse | JSONRPCError):
            return  # pragma: no cover

        assert self._on_error is not None

        if message.message.id is None:
            error = message.message.error
            logging.warning(f"Received error with null ID: {error.message}")
            await self._on_error(MCPError(error.code, error.message, error.data))
            return

        response_id = self._normalize_request_id(message.message.id)

        # Response routers (experimental task support) get first look.
        if isinstance(message.message, JSONRPCError):
            for router in self._response_routers:
                if router.route_error(response_id, message.message.error):
                    return
        else:
            response_data: dict[str, Any] = message.message.result or {}
            for router in self._response_routers:
                if router.route_response(response_id, response_data):
                    return

        stream = self._response_streams.pop(response_id, None)
        if stream:
            await stream.send(message.message)
        else:
            await self._on_error(RuntimeError(f"Received response with an unknown request ID: {message}"))

    @staticmethod
    def _normalize_request_id(response_id: RequestId) -> RequestId:
        # We send integer IDs; some peers echo them back as strings.
        if isinstance(response_id, str):
            try:
                return int(response_id)
            except ValueError:
                logging.warning(f"Response ID {response_id!r} cannot be normalized to match pending requests")
        return response_id
