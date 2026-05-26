from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Protocol, TypeVar, cast

import anyio
import anyio.lowlevel
from anyio.abc import TaskGroup
from pydantic import BaseModel, TypeAdapter

from mcp import types
from mcp.client.experimental import ExperimentalClientFeatures
from mcp.client.experimental.task_handlers import ExperimentalTaskHandlers
from mcp.shared._context import RequestContext
from mcp.shared.context import TransportContext
from mcp.shared.exceptions import MCPError
from mcp.shared.jsonrpc_dispatcher import JSONRPCDispatcher
from mcp.shared.session import ProgressFnT, RequestResponder
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS
from mcp.types._types import RequestParamsMeta

DEFAULT_CLIENT_INFO = types.Implementation(name="mcp", version="0.1.0")

logger = logging.getLogger("client")


class SamplingFnT(Protocol):
    async def __call__(
        self,
        context: RequestContext[ClientSession],
        params: types.CreateMessageRequestParams,
    ) -> types.CreateMessageResult | types.CreateMessageResultWithTools | types.ErrorData: ...  # pragma: no branch


class ElicitationFnT(Protocol):
    async def __call__(
        self,
        context: RequestContext[ClientSession],
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData: ...  # pragma: no branch


class ListRootsFnT(Protocol):
    async def __call__(
        self, context: RequestContext[ClientSession]
    ) -> types.ListRootsResult | types.ErrorData: ...  # pragma: no branch


class LoggingFnT(Protocol):
    async def __call__(self, params: types.LoggingMessageNotificationParams) -> None: ...  # pragma: no branch


class MessageHandlerFnT(Protocol):
    async def __call__(
        self,
        message: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
    ) -> None: ...  # pragma: no branch


async def _default_message_handler(
    message: RequestResponder[types.ServerRequest, types.ClientResult] | types.ServerNotification | Exception,
) -> None:
    await anyio.lowlevel.checkpoint()


async def _default_sampling_callback(
    context: RequestContext[ClientSession],
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult | types.CreateMessageResultWithTools | types.ErrorData:
    return types.ErrorData(
        code=types.INVALID_REQUEST,
        message="Sampling not supported",
    )


async def _default_elicitation_callback(
    context: RequestContext[ClientSession],
    params: types.ElicitRequestParams,
) -> types.ElicitResult | types.ErrorData:
    return types.ErrorData(  # pragma: no cover
        code=types.INVALID_REQUEST,
        message="Elicitation not supported",
    )


async def _default_list_roots_callback(
    context: RequestContext[ClientSession],
) -> types.ListRootsResult | types.ErrorData:
    return types.ErrorData(
        code=types.INVALID_REQUEST,
        message="List roots not supported",
    )


async def _default_logging_callback(
    params: types.LoggingMessageNotificationParams,
) -> None:
    pass


ClientResponse: TypeAdapter[types.ClientResult | types.ErrorData] = TypeAdapter(types.ClientResult | types.ErrorData)


ResultT = TypeVar("ResultT", bound=BaseModel)


class ClientSession:
    def __init__(
        self,
        dispatcher_or_read_stream: Any = None,
        write_stream_or_timeout: Any = None,
        read_timeout_seconds: float | None = None,
        sampling_callback: SamplingFnT | None = None,
        elicitation_callback: ElicitationFnT | None = None,
        list_roots_callback: ListRootsFnT | None = None,
        logging_callback: LoggingFnT | None = None,
        message_handler: MessageHandlerFnT | None = None,
        client_info: types.Implementation | None = None,
        *,
        dispatcher: JSONRPCDispatcher[TransportContext] | None = None,
        read_stream: Any = None,
        write_stream: Any = None,
        sampling_capabilities: types.SamplingCapability | None = None,
        experimental_task_handlers: ExperimentalTaskHandlers | None = None,
    ) -> None:
        # Resolve dispatcher
        disp = dispatcher
        if disp is None:
            if hasattr(dispatcher_or_read_stream, "send_raw_request") or (
                dispatcher_or_read_stream is not None
                and not hasattr(dispatcher_or_read_stream, "read")
                and not hasattr(dispatcher_or_read_stream, "receive")
                and not hasattr(dispatcher_or_read_stream, "send")
            ):
                disp = dispatcher_or_read_stream

        # Resolve streams
        r_stream = read_stream
        if r_stream is None and disp is None:
            r_stream = dispatcher_or_read_stream

        w_stream = write_stream
        if w_stream is None and disp is None:
            w_stream = write_stream_or_timeout

        # Resolve timeout
        timeout = read_timeout_seconds
        if timeout is None and disp is not None:
            timeout = write_stream_or_timeout

        if disp is not None:
            self._dispatcher = disp
            self._read_timeout_seconds = timeout
        else:
            assert r_stream is not None and w_stream is not None, "Streams must be supplied when dispatcher is missing"
            self._dispatcher = JSONRPCDispatcher(r_stream, w_stream)
            self._read_timeout_seconds = timeout

        self._read_stream = r_stream
        self._write_stream = w_stream

        # Register unhandled message callback to surface null-id errors back to the message handler
        def handle_unhandled_message(msg: Any) -> None:
            if isinstance(msg, types.JSONRPCError):
                mcp_err = MCPError(
                    msg.error.code,
                    msg.error.message,
                    getattr(msg.error, "data", None),
                )
                if self._tg is not None:
                    try:
                        self._tg.start_soon(self._message_handler, mcp_err)
                    except Exception:
                        pass

        self._dispatcher._on_unhandled_message = handle_unhandled_message

        self._client_info = client_info or DEFAULT_CLIENT_INFO
        self._sampling_callback = sampling_callback or _default_sampling_callback
        self._sampling_capabilities = sampling_capabilities
        self._elicitation_callback = elicitation_callback or _default_elicitation_callback
        self._list_roots_callback = list_roots_callback or _default_list_roots_callback
        self._logging_callback = logging_callback or _default_logging_callback
        self._message_handler = message_handler or _default_message_handler
        self._tool_output_schemas: dict[str, dict[str, Any] | None] = {}
        self._initialize_result: types.InitializeResult | None = None
        self._experimental_features: ExperimentalClientFeatures | None = None
        self._task_handlers = experimental_task_handlers or ExperimentalTaskHandlers()
        self._active_responders: dict[types.RequestId, Callable[[Any], None]] = {}
        self._exit_stack: AsyncExitStack | None = None
        self._tg: TaskGroup | None = None

    async def __aenter__(self) -> ClientSession:
        from contextlib import AsyncExitStack

        import anyio

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        self._tg = await self._exit_stack.enter_async_context(anyio.create_task_group())

        # Start the background receive loop task immediately and wait for it to be ready!
        async def run_dispatcher(
            *args: Any,
            task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
        ) -> None:
            try:
                await self._dispatcher.run(*args, task_status=task_status)
            except BaseException as e:

                def is_clean_exit_exception(ex: BaseException) -> bool:
                    if isinstance(
                        ex,
                        anyio.ClosedResourceError | anyio.EndOfStream | anyio.get_cancelled_exc_class(),
                    ):
                        return True
                    if hasattr(ex, "exceptions"):
                        return all(is_clean_exit_exception(sub) for sub in ex.exceptions)
                    return False

                if is_clean_exit_exception(e):
                    logger.debug("Dispatcher background receive loop connection closed/cancelled cleanly")
                else:
                    raise

        await self._tg.start(
            run_dispatcher,
            self._on_request,
            self._on_notify,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        if self._tg is not None:
            self._tg.cancel_scope.cancel()
        if self._exit_stack is not None:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

        # Explicitly close custom created streams to trigger EOF on the server side
        if self._read_stream is not None:
            try:
                await self._read_stream.aclose()
            except Exception:
                pass
        if self._write_stream is not None:
            try:
                await self._write_stream.aclose()
            except Exception:
                pass

        self._tg = None
        self._exit_stack = None
        self._read_stream = None
        self._write_stream = None
        return None

    async def _on_request(
        self,
        dctx: Any,
        method: str,
        params: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        raw_req = {
            "jsonrpc": "2.0",
            "id": dctx.request_id,
            "method": method,
            "params": params,
        }
        request = types.server_request_adapter.validate_python(raw_req)

        meta = params.get("_meta") if isinstance(params, dict) else None
        ctx = RequestContext[ClientSession](
            request_id=dctx.request_id,
            meta=meta,
            session=self,
        )

        response_event = anyio.Event()
        response_storage: list[Any] = []

        def on_respond(resp_val: Any) -> None:
            response_storage.append(resp_val)
            response_event.set()

        self._active_responders[dctx.request_id] = on_respond

        try:
            responder = RequestResponder[types.ServerRequest, types.ClientResult](
                request_id=dctx.request_id,
                request_meta=meta,
                request=request,
                session=cast(Any, self),
                on_complete=lambda _: None,
            )

            if self._task_handlers.handles_request(request):
                await self._task_handlers.handle_request(ctx, responder)
            else:
                match request:
                    case types.CreateMessageRequest(params=p):
                        with responder:
                            if p.task is not None:
                                res = await self._task_handlers.augmented_sampling(ctx, p, p.task)
                            else:
                                res = await self._sampling_callback(ctx, p)
                            client_res = ClientResponse.validate_python(res)
                            await responder.respond(client_res)

                    case types.ElicitRequest(params=p):
                        with responder:
                            if p.task is not None:
                                res = await self._task_handlers.augmented_elicitation(ctx, p, p.task)
                            else:
                                res = await self._elicitation_callback(ctx, p)
                            client_res = ClientResponse.validate_python(res)
                            await responder.respond(client_res)

                    case types.ListRootsRequest():
                        with responder:
                            res = await self._list_roots_callback(ctx)
                            client_res = ClientResponse.validate_python(res)
                            await responder.respond(client_res)

                    case types.PingRequest():
                        with responder:
                            await responder.respond(types.EmptyResult())

                    case _:
                        raise MCPError(
                            code=types.METHOD_NOT_FOUND,
                            message=f"Method not found: {method}",
                        )

            await response_event.wait()
            final_response = response_storage[0]
            if isinstance(final_response, types.ErrorData):
                raise MCPError(
                    code=final_response.code,
                    message=final_response.message,
                    data=final_response.data,
                )

            return final_response.model_dump(by_alias=True, exclude_none=True)

        finally:
            self._active_responders.pop(dctx.request_id, None)

    async def _on_notify(
        self,
        dctx: Any,
        method: str,
        params: Mapping[str, Any] | None,
    ) -> None:
        raw_notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            notification = types.server_notification_adapter.validate_python(raw_notification)
        except Exception as e:
            logger.warning(f"Failed to parse incoming notification {method}: {e}")
            return

        match notification:
            case types.LoggingMessageNotification(params=p):
                await self._logging_callback(p)
            case _:
                pass

        await self._message_handler(notification)

    async def _send_response(self, request_id: types.RequestId, response: Any) -> None:
        callback = self._active_responders.get(request_id)
        if callback is not None:
            callback(response)

    async def send_request(
        self,
        request: types.ClientRequest,
        result_type: type[ResultT] = BaseModel,
        *,
        request_read_timeout_seconds: float | None = None,
        progress_callback: ProgressFnT | None = None,
        metadata: Any = None,
    ) -> ResultT:
        raw_params = (
            request.params.model_dump(by_alias=True, mode="json", exclude_none=True)
            if hasattr(request, "params") and request.params is not None
            else None
        )

        opts: dict[str, Any] = {}
        timeout_val = request_read_timeout_seconds or self._read_timeout_seconds
        if timeout_val is not None:
            opts["timeout"] = timeout_val
        if metadata is not None:
            token = getattr(metadata, "resumption_token", None)
            on_token = getattr(metadata, "on_resumption_token_update", None)
            if token is not None:
                opts["resumption_token"] = token
            if on_token is not None:
                opts["on_resumption_token"] = on_token
        if progress_callback is not None:

            async def safe_progress(progress: float, total: float | None, message: str | None) -> None:
                try:
                    await progress_callback(progress, total, message)
                except Exception as e:
                    import mcp.shared.session

                    mcp.shared.session.logging.exception(
                        "Progress callback raised an exception",
                        exc_info=e,
                    )

            opts["on_progress"] = safe_progress

        from opentelemetry.trace import SpanKind

        from mcp.shared._otel import inject_trace_context, otel_span

        target = getattr(request.params, "name", None) if hasattr(request, "params") else None
        span_name = f"MCP send {request.method} {target}" if target else f"MCP send {request.method}"

        params_dict = raw_params if raw_params is not None else {}

        with otel_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={"mcp.method.name": request.method},
        ):
            meta = params_dict.setdefault("_meta", {})
            inject_trace_context(meta)

            res_dict = await self._dispatcher.send_raw_request(
                request.method,
                params_dict,
                opts=cast(Any, opts),
            )
        return result_type.model_validate(res_dict)

    async def send_notification(
        self,
        notification: types.ClientNotification,
    ) -> None:
        raw_params = (
            notification.params.model_dump(by_alias=True, mode="json", exclude_none=True)
            if hasattr(notification, "params") and notification.params is not None
            else None
        )
        await self._dispatcher.notify(notification.method, raw_params)

    @property
    def _in_flight(self) -> dict[Any, Any]:
        return getattr(self._dispatcher, "_in_flight", {})

    @property
    def _task_group(self) -> Any:
        return self._tg

    @property
    def _receive_request_adapter(self) -> TypeAdapter[types.ServerRequest]:
        return types.server_request_adapter

    @property
    def _receive_notification_adapter(self) -> TypeAdapter[types.ServerNotification]:
        return types.server_notification_adapter

    async def initialize(self) -> types.InitializeResult:
        sampling = (
            (self._sampling_capabilities or types.SamplingCapability())
            if self._sampling_callback is not _default_sampling_callback
            else None
        )
        elicitation = (
            types.ElicitationCapability(form=types.FormElicitationCapability(), url=types.UrlElicitationCapability())
            if self._elicitation_callback is not _default_elicitation_callback
            else None
        )
        roots = (
            # TODO: Should this be based on whether we
            # _will_ send notifications, or only whether
            # they're supported?
            types.RootsCapability(list_changed=True)
            if self._list_roots_callback is not _default_list_roots_callback
            else None
        )

        result = await self.send_request(
            types.InitializeRequest(
                params=types.InitializeRequestParams(
                    protocol_version=types.LATEST_PROTOCOL_VERSION,
                    capabilities=types.ClientCapabilities(
                        sampling=sampling,
                        elicitation=elicitation,
                        experimental=None,
                        roots=roots,
                        tasks=self._task_handlers.build_capability(),
                    ),
                    client_info=self._client_info,
                ),
            ),
            types.InitializeResult,
        )

        if result.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise RuntimeError(f"Unsupported protocol version from the server: {result.protocol_version}")

        self._initialize_result = result

        await self.send_notification(types.InitializedNotification())

        return result

    @property
    def initialize_result(self) -> types.InitializeResult | None:
        """The server's InitializeResult. None until initialize() has been called.

        Contains server_info, capabilities, instructions, and the negotiated protocol_version.
        """
        return self._initialize_result

    @property
    def experimental(self) -> ExperimentalClientFeatures:
        """Experimental APIs for tasks and other features.

        !!! warning
            These APIs are experimental and may change without notice.

        Example:
            ```python
            status = await session.experimental.get_task(task_id)
            result = await session.experimental.get_task_result(task_id, CallToolResult)
            ```
        """
        if self._experimental_features is None:
            self._experimental_features = ExperimentalClientFeatures(self)
        return self._experimental_features

    async def send_ping(self, *, meta: RequestParamsMeta | None = None) -> types.EmptyResult:
        """Send a ping request."""
        return await self.send_request(types.PingRequest(params=types.RequestParams(_meta=meta)), types.EmptyResult)

    async def send_progress_notification(
        self,
        progress_token: str | int,
        progress: float,
        total: float | None = None,
        message: str | None = None,
        *,
        meta: RequestParamsMeta | None = None,
    ) -> None:
        """Send a progress notification."""
        await self.send_notification(
            types.ProgressNotification(
                params=types.ProgressNotificationParams(
                    progress_token=progress_token,
                    progress=progress,
                    total=total,
                    message=message,
                    _meta=meta,
                ),
            )
        )

    async def set_logging_level(
        self,
        level: types.LoggingLevel,
        *,
        meta: RequestParamsMeta | None = None,
    ) -> types.EmptyResult:
        """Send a logging/setLevel request."""
        return await self.send_request(
            types.SetLevelRequest(params=types.SetLevelRequestParams(level=level, _meta=meta)),
            types.EmptyResult,
        )

    async def list_resources(self, *, params: types.PaginatedRequestParams | None = None) -> types.ListResourcesResult:
        """Send a resources/list request.

        Args:
            params: Full pagination parameters including cursor and any future fields
        """
        return await self.send_request(types.ListResourcesRequest(params=params), types.ListResourcesResult)

    async def list_resource_templates(
        self, *, params: types.PaginatedRequestParams | None = None
    ) -> types.ListResourceTemplatesResult:
        """Send a resources/templates/list request.

        Args:
            params: Full pagination parameters including cursor and any future fields
        """
        return await self.send_request(
            types.ListResourceTemplatesRequest(params=params),
            types.ListResourceTemplatesResult,
        )

    async def read_resource(self, uri: str, *, meta: RequestParamsMeta | None = None) -> types.ReadResourceResult:
        """Send a resources/read request."""
        return await self.send_request(
            types.ReadResourceRequest(params=types.ReadResourceRequestParams(uri=uri, _meta=meta)),
            types.ReadResourceResult,
        )

    async def subscribe_resource(self, uri: str, *, meta: RequestParamsMeta | None = None) -> types.EmptyResult:
        """Send a resources/subscribe request."""
        return await self.send_request(
            types.SubscribeRequest(params=types.SubscribeRequestParams(uri=uri, _meta=meta)),
            types.EmptyResult,
        )

    async def unsubscribe_resource(self, uri: str, *, meta: RequestParamsMeta | None = None) -> types.EmptyResult:
        """Send a resources/unsubscribe request."""
        return await self.send_request(
            types.UnsubscribeRequest(params=types.UnsubscribeRequestParams(uri=uri, _meta=meta)),
            types.EmptyResult,
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        meta: RequestParamsMeta | None = None,
    ) -> types.CallToolResult:
        """Send a tools/call request with optional progress callback support."""

        result = await self.send_request(
            types.CallToolRequest(
                params=types.CallToolRequestParams(name=name, arguments=arguments, _meta=meta),
            ),
            types.CallToolResult,
            request_read_timeout_seconds=read_timeout_seconds,
            progress_callback=progress_callback,
        )

        if not result.is_error:
            await self._validate_tool_result(name, result)

        return result

    async def _validate_tool_result(self, name: str, result: types.CallToolResult) -> None:
        """Validate the structured content of a tool result against its output schema."""
        if name not in self._tool_output_schemas:
            # refresh output schema cache
            await self.list_tools()

        output_schema = None
        if name in self._tool_output_schemas:
            output_schema = self._tool_output_schemas.get(name)
        else:
            logger.warning(f"Tool {name} not listed by server, cannot validate any structured content")

        if output_schema is not None:
            from jsonschema import SchemaError, ValidationError, validate

            if result.structured_content is None:
                raise RuntimeError(
                    f"Tool {name} has an output schema but did not return structured content"
                )  # pragma: no cover
            try:
                validate(result.structured_content, output_schema)
            except ValidationError as e:
                raise RuntimeError(f"Invalid structured content returned by tool {name}: {e}")
            except SchemaError as e:  # pragma: no cover
                raise RuntimeError(f"Invalid schema for tool {name}: {e}")  # pragma: no cover

    async def list_prompts(self, *, params: types.PaginatedRequestParams | None = None) -> types.ListPromptsResult:
        """Send a prompts/list request.

        Args:
            params: Full pagination parameters including cursor and any future fields
        """
        return await self.send_request(types.ListPromptsRequest(params=params), types.ListPromptsResult)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        *,
        meta: RequestParamsMeta | None = None,
    ) -> types.GetPromptResult:
        """Send a prompts/get request."""
        return await self.send_request(
            types.GetPromptRequest(params=types.GetPromptRequestParams(name=name, arguments=arguments, _meta=meta)),
            types.GetPromptResult,
        )

    async def complete(
        self,
        ref: types.ResourceTemplateReference | types.PromptReference,
        argument: dict[str, str],
        context_arguments: dict[str, str] | None = None,
    ) -> types.CompleteResult:
        """Send a completion/complete request."""
        context = None
        if context_arguments is not None:
            context = types.CompletionContext(arguments=context_arguments)

        return await self.send_request(
            types.CompleteRequest(
                params=types.CompleteRequestParams(
                    ref=ref,
                    argument=types.CompletionArgument(**argument),
                    context=context,
                ),
            ),
            types.CompleteResult,
        )

    async def list_tools(self, *, params: types.PaginatedRequestParams | None = None) -> types.ListToolsResult:
        """Send a tools/list request.

        Args:
            params: Full pagination parameters including cursor and any future fields
        """
        result = await self.send_request(
            types.ListToolsRequest(params=params),
            types.ListToolsResult,
        )

        # Cache tool output schemas for future validation
        # Note: don't clear the cache, as we may be using a cursor
        for tool in result.tools:
            self._tool_output_schemas[tool.name] = tool.output_schema

        return result

    async def send_roots_list_changed(self) -> None:  # pragma: no cover
        """Send a roots/list_changed notification."""
        await self.send_notification(types.RootsListChangedNotification())
