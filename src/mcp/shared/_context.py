"""Request context for MCP handlers."""

from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

from mcp.types import RequestId, RequestParamsMeta

SessionT = TypeVar("SessionT")


@dataclass(kw_only=True)
class RequestContext(Generic[SessionT]):
    """Common context for handling incoming requests.

    For request handlers, request_id is always populated.
    For notification handlers, request_id is None.
    """

    session: SessionT
    request_id: RequestId | None = None
    meta: RequestParamsMeta | None = None
