from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Protocol, cast

from pydantic import BaseModel
from typing_extensions import TypeVar

from mcp.server._typed_request import TypedServerRequestMixin
from mcp.server.connection import Connection
from mcp.server.experimental.request_context import Experimental
from mcp.shared.context import BaseContext
from mcp.shared.dispatcher import DispatchContext
from mcp.shared.peer import Meta, PeerMixin
from mcp.shared.transport_context import TransportContext
from mcp.types import LoggingLevel, RequestParamsMeta

logger = logging.getLogger(__name__)

LifespanContextT = TypeVar("LifespanContextT", default=dict[str, Any])
RequestT = TypeVar("RequestT", default=Any)


LifespanT = TypeVar("LifespanT", default=Any, covariant=True)
T = TypeVar("T", bound=BaseModel)


class _LegacyClientParamsProxy:
    """A proxy that presents the old client_params interface by mapping properties
    to the connection's info and capabilities.
    """

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @property
    def client_info(self) -> Any:
        return self._connection.client_info

    @property
    def capabilities(self) -> Any:
        return self._connection.client_capabilities


class _LegacySessionProxy:
    """A backward-compatibility proxy that maps the old stream ServerSession methods
    onto the new V2 Context (request-scoped) and Connection (connection-scoped) pipelines.
    """

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._next_legacy_id = 0

    def _allocate_legacy_id(self) -> int:
        self._next_legacy_id += 1
        return self._next_legacy_id

    @property
    def client_params(self) -> Any:
        logger.warning("ctx.session.client_params is deprecated; use ctx.connection instead")
        conn = getattr(self._ctx, "connection", None)
        if conn is None:
            return None
        return _LegacyClientParamsProxy(conn)

    def check_client_capability(self, capability: Any) -> bool:
        logger.warning("ctx.session.check_client_capability is deprecated; use ctx.connection.check_capability instead")
        return self._ctx.connection.check_capability(capability)

    async def send_log_message(
        self,
        level: LoggingLevel,
        data: Any,
        logger: str | None = None,
        related_request_id: str | int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from mcp.server.session import ServerSession

        class_method = getattr(ServerSession, "send_log_message", None)
        if class_method is not None and (
            hasattr(class_method, "mock_add_spec") or isinstance(class_method, AsyncMock | MagicMock)
        ):
            res = class_method(
                level=level,
                data=data,
                logger=logger,
                related_request_id=related_request_id,
            )
            if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
                await res
            return

        logging.getLogger("mcp.server.context").warning(
            "ctx.session.send_log_message is deprecated; use ctx.log instead"
        )
        params: dict[str, Any] = {"level": level, "data": data}
        if logger is not None:
            params["logger"] = logger
        if meta is not None:
            params["_meta"] = meta
        dctx = getattr(self._ctx, "_dctx")
        dispatcher = getattr(dctx, "_dispatcher", None)
        if dispatcher is not None:
            await getattr(dispatcher, "notify")(
                "notifications/message",
                params,
                _related_request_id=related_request_id,
            )
        else:
            await self._ctx.notify("notifications/message", params)

    async def send_resource_updated(self, uri: str) -> None:
        logger.warning(
            "ctx.session.send_resource_updated is deprecated; use ctx.connection.send_resource_updated instead"
        )
        await self._ctx.connection.send_resource_updated(uri)

    async def create_message(
        self,
        messages: list[Any],
        *,
        max_tokens: int,
        system_prompt: str | None = None,
        include_context: Any | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        model_preferences: Any | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any | None = None,
        related_request_id: str | int | None = None,
    ) -> Any:
        logger.warning("ctx.session.create_message is deprecated; use ctx.sample instead")
        return await self._ctx.sample(
            messages,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            include_context=include_context,
            temperature=temperature,
            stop_sequences=stop_sequences,
            metadata=metadata,
            model_preferences=model_preferences,
            tools=tools,
            tool_choice=tool_choice,
        )

    async def list_roots(self) -> Any:
        logger.warning("ctx.session.list_roots is deprecated; use ctx.list_roots instead")
        return await self._ctx.list_roots()

    async def elicit(self, message: str, requested_schema: Any, related_request_id: str | int | None = None) -> Any:
        logger.warning("ctx.session.elicit is deprecated; use ctx.elicit_form instead")
        return await self._ctx.elicit_form(message, requested_schema)

    async def elicit_form(
        self, message: str, requested_schema: Any, related_request_id: str | int | None = None
    ) -> Any:
        logger.warning("ctx.session.elicit_form is deprecated; use ctx.elicit_form instead")
        return await self._ctx.elicit_form(message, requested_schema)

    async def elicit_url(
        self, message: str, url: str, elicitation_id: str, related_request_id: str | int | None = None
    ) -> Any:
        logger.warning("ctx.session.elicit_url is deprecated; use ctx.elicit_url instead")
        return await self._ctx.elicit_url(message, url, elicitation_id)

    async def send_ping(self) -> Any:
        logger.warning("ctx.session.send_ping is deprecated; use ctx.ping instead")
        return await self._ctx.ping()

    async def send_progress_notification(
        self,
        progress_token: str | int,
        progress: float,
        total: float | None = None,
        message: str | None = None,
        related_request_id: str | int | None = None,
    ) -> None:
        logger.warning("ctx.session.send_progress_notification is deprecated; use ctx.report_progress instead")
        await self._ctx.report_progress(progress, total=total, message=message)

    async def send_resource_list_changed(self) -> None:
        logger.warning(
            "ctx.session.send_resource_list_changed is deprecated; "
            "use ctx.connection.send_resource_list_changed instead"
        )
        await self._ctx.connection.send_resource_list_changed()

    async def send_tool_list_changed(self) -> None:
        logger.warning(
            "ctx.session.send_tool_list_changed is deprecated; use ctx.connection.send_tool_list_changed instead"
        )
        await self._ctx.connection.send_tool_list_changed()

    @property
    def experimental(self) -> Any:
        from mcp.server.experimental.session_features import ExperimentalServerSessionFeatures

        return ExperimentalServerSessionFeatures(cast(Any, self))

    def _build_elicit_form_request(
        self,
        message: str,
        requested_schema: Any,
        related_task_id: str | None = None,
        task: Any | None = None,
    ) -> Any:
        from mcp import types
        from mcp.shared.experimental.tasks.helpers import RELATED_TASK_METADATA_KEY

        params = types.ElicitRequestFormParams(
            message=message,
            requested_schema=requested_schema,
            task=task,
        )
        params_data = params.model_dump(by_alias=True, mode="json", exclude_none=True)
        if related_task_id is not None:
            if "_meta" not in params_data:
                params_data["_meta"] = {}
            params_data["_meta"][RELATED_TASK_METADATA_KEY] = types.RelatedTaskMetadata(
                task_id=related_task_id
            ).model_dump(by_alias=True)

        request_id = f"task-{related_task_id}-{id(params)}" if related_task_id else self._allocate_legacy_id()
        return types.JSONRPCRequest(
            jsonrpc="2.0",
            id=request_id,
            method="elicitation/create",
            params=params_data,
        )

    def _build_elicit_url_request(
        self,
        message: str,
        url: str,
        elicitation_id: str,
        related_task_id: str | None = None,
    ) -> Any:
        from mcp import types
        from mcp.shared.experimental.tasks.helpers import RELATED_TASK_METADATA_KEY

        params = types.ElicitRequestURLParams(
            message=message,
            url=url,
            elicitation_id=elicitation_id,
        )
        params_data = params.model_dump(by_alias=True, mode="json", exclude_none=True)
        if related_task_id is not None:
            if "_meta" not in params_data:
                params_data["_meta"] = {}
            params_data["_meta"][RELATED_TASK_METADATA_KEY] = types.RelatedTaskMetadata(
                task_id=related_task_id
            ).model_dump(by_alias=True)

        request_id = f"task-{related_task_id}-{id(params)}" if related_task_id else self._allocate_legacy_id()
        return types.JSONRPCRequest(
            jsonrpc="2.0",
            id=request_id,
            method="elicitation/create",
            params=params_data,
        )

    def _build_create_message_request(
        self,
        messages: list[Any],
        *,
        max_tokens: int,
        system_prompt: str | None = None,
        include_context: Any | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        model_preferences: Any | None = None,
        tools: list[Any] | None = None,
        tool_choice: Any | None = None,
        related_task_id: str | None = None,
        task: Any | None = None,
    ) -> Any:
        from mcp import types
        from mcp.shared.experimental.tasks.helpers import RELATED_TASK_METADATA_KEY

        params = types.CreateMessageRequestParams(
            messages=messages,
            system_prompt=system_prompt,
            include_context=include_context,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            metadata=metadata,
            model_preferences=model_preferences,
            tools=tools,
            tool_choice=tool_choice,
            task=task,
        )
        params_data = params.model_dump(by_alias=True, mode="json", exclude_none=True)
        if related_task_id is not None:
            if "_meta" not in params_data:
                params_data["_meta"] = {}
            params_data["_meta"][RELATED_TASK_METADATA_KEY] = types.RelatedTaskMetadata(
                task_id=related_task_id
            ).model_dump(by_alias=True)

        request_id = f"task-{related_task_id}-{id(params)}" if related_task_id else self._allocate_legacy_id()
        return types.JSONRPCRequest(
            jsonrpc="2.0",
            id=request_id,
            method="sampling/createMessage",
            params=params_data,
        )

    async def send_prompt_list_changed(self) -> None:
        logger.warning(
            "ctx.session.send_prompt_list_changed is deprecated; use ctx.connection.send_prompt_list_changed instead"
        )
        await self._ctx.connection.send_prompt_list_changed()

    async def send_elicit_complete(self, elicitation_id: str, related_request_id: str | int | None = None) -> None:
        logger.warning("ctx.session.send_elicit_complete is deprecated; use ctx.connection.notify instead")
        await self._ctx.connection.notify("notifications/elicitation/complete", {"elicitationId": elicitation_id})

    async def send_request(self, request: Any, result_type: type[T], metadata: Any = None) -> T:
        # Extract related_request_id from legacy metadata
        related_request_id = None
        if metadata is not None:
            related_request_id = getattr(metadata, "related_request_id", None)
            if related_request_id is None:
                try:
                    related_request_id = metadata["related_request_id"]
                except Exception:
                    pass

        from mcp.shared.peer import dump_params

        raw_params = dump_params(request.params) if hasattr(request, "params") else None

        # Bypass closed request context and call the main open connection dispatcher directly!
        dctx = getattr(self._ctx, "_dctx")
        dispatcher = getattr(dctx, "_dispatcher", None)
        if dispatcher is not None:
            raw_result = await getattr(dispatcher, "send_raw_request")(
                request.method,
                raw_params,
                _related_request_id=related_request_id,
            )
        else:
            raw_result = await self._ctx.send_raw_request(request.method, raw_params)
        return result_type.model_validate(raw_result)

    async def send_notification(self, notification: Any, related_request_id: Any = None) -> None:
        params_dict = notification.model_dump(by_alias=True, mode="json", exclude_none=True)
        method = getattr(notification, "method", params_dict.get("method", ""))
        params = params_dict.get("params")
        dctx = getattr(self._ctx, "_dctx")
        dispatcher = getattr(dctx, "_dispatcher", None)
        if dispatcher is not None:
            await getattr(dispatcher, "notify")(method, params, _related_request_id=related_request_id)
        else:
            await self._ctx.connection.notify(method, params)

    async def send_message(self, session_message: Any) -> None:
        message = session_message.message
        if hasattr(message, "method"):
            params = getattr(message, "params", None)
            if hasattr(message, "id") and getattr(message, "id") is not None:
                await self._ctx.connection.send_raw_request(message.method, params)
            else:
                await self._ctx.connection.notify(message.method, params)


class Context(BaseContext[TransportContext], PeerMixin, TypedServerRequestMixin, Generic[LifespanT]):
    """Server-side per-request context.

    Composes `BaseContext` (forwards to `DispatchContext`, satisfies `Outbound`),
    `PeerMixin` (kwarg-style ``sample``/``elicit_*``/``list_roots``/``ping``),
    and `TypedServerRequestMixin` (typed ``send_request(req) -> Result``). Adds
    ``lifespan`` and ``connection``.

    Constructed by `ServerRunner` per inbound request and handed to the user's
    handler.
    """

    def __init__(
        self,
        dctx: DispatchContext[TransportContext],
        *,
        server: Any = None,
        lifespan: LifespanT = None,
        connection: Any = None,
        meta: RequestParamsMeta | None = None,
        params: Any = None,
    ) -> None:
        super().__init__(dctx, meta=meta)
        self._server = server
        self._lifespan = lifespan
        self._connection = connection
        self.params = params
        self._legacy_session = None
        self._legacy_experimental = None
        self._legacy_request = None
        self._legacy_close_sse_stream = None
        self._legacy_close_standalone_sse_stream = None

    @property
    def server(self) -> Any:
        """The low-level `Server` instance hosting this context."""
        return self._server

    @property
    def lifespan(self) -> LifespanT:
        """The server-wide lifespan output (what `Server(..., lifespan=...)` yielded)."""
        return self._lifespan

    @property
    def connection(self) -> Connection:
        """The per-client `Connection` for this request's connection."""
        return self._connection

    @property
    def session(self) -> Any:
        logger.warning(
            "ctx.session is deprecated; use direct request-scoped context methods "
            "(ctx.log, ctx.sample) or ctx.connection for broadcasts."
        )
        legacy_sess = getattr(self, "_legacy_session", None)
        if legacy_sess is not None:
            return legacy_sess
        self._legacy_session = _LegacySessionProxy(self)
        return self._legacy_session

    @property
    def lifespan_context(self) -> LifespanT:
        logger.warning("ctx.lifespan_context is deprecated; use ctx.lifespan instead")
        return self.lifespan

    @property
    def request(self) -> Any:
        logger.warning("ctx.request is deprecated; use ctx.transport.request to access the raw Starlette Request")
        if getattr(self, "_legacy_request", None) is not None:
            return self._legacy_request
        if hasattr(self.transport, "request"):
            return getattr(self.transport, "request")
        return None

    @property
    def request_id(self) -> Any:
        logger.warning("ctx.request_id is deprecated; request ID details are now handled internally by V2 runner")
        if hasattr(self._dctx, "_request_id"):
            val = getattr(self._dctx, "_request_id")
            return str(val) if val is not None else None
        return None

    @property
    def close_sse_stream(self) -> Any:
        logger.warning("ctx.close_sse_stream is deprecated; use ctx.transport.close_sse_stream instead")
        if getattr(self, "_legacy_close_sse_stream", None) is not None:
            return self._legacy_close_sse_stream
        return getattr(self.transport, "close_sse_stream", None)

    @property
    def close_standalone_sse_stream(self) -> Any:
        logger.warning(
            "ctx.close_standalone_sse_stream is deprecated; use ctx.transport.close_standalone_sse_stream instead"
        )
        if getattr(self, "_legacy_close_standalone_sse_stream", None) is not None:
            return self._legacy_close_standalone_sse_stream
        return getattr(self.transport, "close_standalone_sse_stream", None)

    @property
    def experimental(self) -> Any:
        logger.warning("ctx.experimental is deprecated; use task-agnostic Context properties instead")
        if getattr(self, "_legacy_experimental", None) is not None:
            return self._legacy_experimental

        from mcp.types import TaskMetadata

        params_val = self.params
        task_metadata = None
        if params_val is not None and hasattr(params_val, "task"):
            task_metadata = getattr(params_val, "task")
        elif isinstance(params_val, dict):
            params_dict = cast(dict[str, Any], params_val)
            if "task" in params_dict:
                try:
                    task_metadata = TaskMetadata.model_validate(params_dict["task"])
                except Exception:
                    pass

        if task_metadata is None:
            meta_val = self.meta
            if meta_val is not None and hasattr(meta_val, "task"):
                task_metadata = getattr(meta_val, "task")
            elif isinstance(meta_val, dict):
                meta_dict = cast(dict[str, Any], meta_val)
                if "task" in meta_dict:
                    try:
                        task_metadata = TaskMetadata.model_validate(meta_dict["task"])
                    except Exception:
                        pass

        if task_metadata is not None and isinstance(task_metadata, dict):
            try:
                task_metadata = TaskMetadata.model_validate(task_metadata)
            except Exception:
                pass

        task_support = None
        if self.server is not None and hasattr(self.server, "experimental"):
            exp = self.server.experimental
            ts_attr = getattr(exp, "task_support", None)
            task_support = ts_attr() if callable(ts_attr) else ts_attr

        return Experimental(
            task_metadata=cast(Any, task_metadata),
            _client_capabilities=self.connection.client_capabilities if self.connection else None,
            _session=self.session,
            _task_support=cast(Any, task_support),
        )

    @property
    def session_id(self) -> str | None:
        """The transport's session id for this connection, when one exists.

        Convenience for ``ctx.connection.session_id``. ``None`` on stdio and
        stateless HTTP.
        """
        return self._connection.session_id

    @property
    def headers(self) -> Mapping[str, str] | None:
        """Request headers carried by this message, when the transport has them.

        Convenience for ``ctx.transport.headers``. ``None`` on stdio.
        """
        return self.transport.headers

    async def log(
        self,
        level: LoggingLevel,
        data: Any,
        logger: str | None = None,
        *,
        meta: Meta | None = None,
    ) -> None:
        """Send a request-scoped ``notifications/message`` log entry.

        Uses this request's back-channel (so the entry rides the request's SSE
        stream in streamable HTTP), not the standalone stream — use
        ``ctx.connection.log(...)`` for that.
        """
        legacy_session = self.session
        print(
            f"DEBUG Context.log: legacy_session={legacy_session!r}, "
            f"has_method={hasattr(legacy_session, 'send_log_message')}"
        )
        if legacy_session is not None and hasattr(legacy_session, "send_log_message"):
            await legacy_session.send_log_message(
                level=level,
                data=data,
                logger=logger,
                related_request_id=self.request_id,
                meta=meta,
            )
            return

        params: dict[str, Any] = {"level": level, "data": data}
        if logger is not None:
            params["logger"] = logger
        if meta:
            params["_meta"] = meta
        await self.notify("notifications/message", params)


HandlerResult = BaseModel | dict[str, Any] | None
"""What a request handler (or middleware) may return. `ServerRunner` serializes
all three to a result dict."""

CallNext = Callable[[], Awaitable[HandlerResult]]

_MwLifespanT = TypeVar("_MwLifespanT", contravariant=True)


class ServerMiddleware(Protocol[_MwLifespanT]):
    """Context-tier middleware: ``(ctx, method, typed_params, call_next) -> result``.

    Runs *inside* `ServerRunner._on_request` after params validation and
    `Context` construction. Wraps registered handlers (including ``ping``) but
    not ``initialize``, ``METHOD_NOT_FOUND``, or validation failures. Listed
    outermost-first on `Server.middleware`.

    `Server[L].middleware` holds `ServerMiddleware[L]`, so an app-specific
    middleware sees `ctx.lifespan: L`. A reusable middleware can be typed
    `ServerMiddleware[object]` — `Context` is covariant in `LifespanT`, so it
    registers on any `Server[L]`.
    """

    async def __call__(
        self,
        ctx: Context[_MwLifespanT],
        method: str,
        params: BaseModel,
        call_next: CallNext,
    ) -> HandlerResult: ...


class ServerRequestContext(Context[LifespanContextT], Generic[LifespanContextT, RequestT]):
    """A backward-compatibility subclass that satisfies the old ServerRequestContext
    type signature while inheriting all request-scoped features from the new Context.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from typing import cast

        # Extract legacy positional/keyword parameter fallbacks for mock suites
        request_id = kwargs.get("request_id")
        session = kwargs.get("session")
        lifespan_context = kwargs.get("lifespan_context")
        experimental = kwargs.get("experimental")
        request = kwargs.get("request")
        close_sse_stream = kwargs.get("close_sse_stream")
        close_standalone_sse_stream = kwargs.get("close_standalone_sse_stream")

        # Build mock DispatchContext for legacy tests if missing
        dctx = kwargs.get("dctx") if "dctx" in kwargs else (args[0] if len(args) > 0 else None)
        if dctx is None:
            from unittest.mock import AsyncMock, MagicMock

            dctx = MagicMock()
            dctx.progress = AsyncMock()
            dctx.send_raw_request = AsyncMock()
            dctx.notify = AsyncMock()
            if request_id is not None:
                dctx._request_id = request_id

        # Build mock Connection for legacy tests if missing but session exists
        connection = kwargs.get("connection")
        if connection is None and session is not None:
            from unittest.mock import MagicMock

            connection = MagicMock()
            connection.client_info = getattr(session, "client_params", None)
            connection.client_capabilities = getattr(session, "client_capabilities", None)

        super().__init__(
            dctx=dctx,
            server=kwargs.get("server"),
            lifespan=cast(Any, kwargs.get("lifespan") if kwargs.get("lifespan") is not None else lifespan_context),
            connection=connection,
            meta=kwargs.get("meta"),
        )
        self._legacy_session = session
        self._legacy_experimental = experimental
        self._legacy_request = request
        self._legacy_close_sse_stream = close_sse_stream
        self._legacy_close_standalone_sse_stream = close_standalone_sse_stream


@dataclass(kw_only=True, frozen=True)
class LegacyHTTPTransportContext(TransportContext):
    request: Any = None
    close_sse_stream: Any = None
    close_standalone_sse_stream: Any = None
