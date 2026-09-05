"""Synchronous MCP client integration for ChatSession."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Protocol

import anyio
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession
from mcp.client.streamable_http import (
    create_mcp_http_client,
    streamable_http_client,
)

from .models import McpServer


class McpSessionProtocol(Protocol):
    """Minimal synchronous interface used by ChatSession."""

    def list_tools(self) -> Any:
        """List server tools."""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke one server tool."""

    def close(self) -> None:
        """Close session resources."""


McpSessionFactory = Callable[[McpServer], McpSessionProtocol]


class _AsyncMcpClientBridge:
    """Drive one MCP session inside a single portal task.

    anyio task groups (opened inside streamablehttp_client) are task-bound:
    the transport must be opened, used, and closed from the same task. A
    long-lived ``_run`` coroutine therefore owns the whole lifecycle and
    receives commands over a memory stream; the sync methods post a command
    and block on its reply.
    """

    def __init__(self, server) -> None:
        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._commands: Any = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        try:
            # start_task_soon is the cross-thread API: call it from here (the
            # caller thread), never from inside a coroutine on the loop.
            self._portal.start_task_soon(self._run, server)
            self._ready.wait()
            if self._error is not None:
                raise _unwrap_exc(self._error)
        except BaseException:
            self._portal_cm.__exit__(None, None, None)
            raise

    async def _run(self, server) -> None:
        send, recv = anyio.create_memory_object_stream(0)
        self._commands = send
        try:
            async with create_mcp_http_client(
                headers=dict(server.headers)
            ) as http_client:
                async with streamable_http_client(
                    server.url, http_client=http_client
                ) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._ready.set()
                        await self._serve(session, recv)
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    async def _serve(self, session, recv) -> None:
        async for method, args, reply in recv:
            if method == "close":
                await reply.send(None)
                return
            try:
                await reply.send(await getattr(session, method)(*args))
            except BaseException as exc:
                await reply.send(exc)

    def _call(self, method: str, *args: Any) -> Any:
        return self._portal.call(self._acall, method, args)

    async def _acall(self, method: str, args: tuple[Any, ...]) -> Any:
        reply_send, reply_recv = anyio.create_memory_object_stream(1)
        await self._commands.send((method, args, reply_send))
        result = await reply_recv.receive()
        if isinstance(result, BaseException):
            raise result
        return result

    def list_tools(self) -> Any:
        return self._call("list_tools")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._call("call_tool", name, arguments)

    def close(self) -> None:
        try:
            self._call("close")
        finally:
            self._portal_cm.__exit__(None, None, None)


def _unwrap_exc(exc: BaseException) -> BaseException:
    """Collapse a single-child anyio ExceptionGroup to its real cause.

    ``streamablehttp_client`` wraps a lone transport failure in a
    ``BaseExceptionGroup`` whose ``str()`` is the useless "unhandled errors in a
    TaskGroup" line. Unwrap single-child groups so callers see the actual error.
    """
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


def create_mcp_session(server: McpServer) -> McpSessionProtocol:
    """Create one synchronous MCP client session for a server config."""
    return _AsyncMcpClientBridge(server)
