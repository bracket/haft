"""ChatSession implementation for OpenAI-Responses + MCP tool orchestration."""

from __future__ import annotations

import json
import time
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from .mcp import McpSessionFactory, McpSessionProtocol, create_mcp_session
from .models import McpServer, Turn
from .responses import send_responses_request


class ToolNameCollisionError(ValueError):
    """Raised when two MCP servers expose a tool with the same name."""


class MaxIterationsExceededError(RuntimeError):
    """Raised when the model exceeds the configured call loop backstop."""

    def __init__(
        self,
        max_iterations: int,
        *,
        turn: Turn | None = None,
        iterations: int = 0,
    ) -> None:
        super().__init__(f"Exceeded maximum model-call iterations ({max_iterations})")
        self.turn = turn
        self.iterations = iterations


ResponsesClient = Callable[..., dict[str, Any]]


@dataclass(slots=True)
class _ToolBinding:
    session: McpSessionProtocol
    server: McpServer | None = None


@dataclass(slots=True)
class _PendingLocalTool:
    name: str
    handler: Callable[[dict[str, Any]], Any]
    parameters: dict[str, Any]
    description: str


class _LocalToolSession:
    """McpSessionProtocol shim wrapping a single in-process local tool."""

    def __init__(self, pending: _PendingLocalTool) -> None:
        self._pending = pending

    def list_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": self._pending.name,
                    "description": self._pending.description,
                    "inputSchema": self._pending.parameters,
                }
            ]
        }

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._pending.handler(arguments)

    def close(self) -> None:
        return None


class ChatSession:
    """Synchronous chat-with-tools loop for Responses-compatible endpoints."""

    def __init__(
        self,
        model: str,
        base_url: str,
        system_prompt: str = "",
        max_iterations: int | None = 10,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        stream: bool = False,
        request_headers: Mapping[str, str] | None = None,
        _responses_client: ResponsesClient | None = None,
        _mcp_session_factory: McpSessionFactory | None = None,
    ) -> None:
        """Initialize a chat session.

        The optional ``on_event`` callback is invoked inline; it must not raise,
        and any exception it raises propagates to the caller.
        """
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        if max_iterations is not None and max_iterations < 0:
            raise ValueError("max_iterations must be >= 0 or None")
        self.max_iterations = max_iterations
        self.on_event = on_event
        self.stream = stream
        self.request_headers = dict(request_headers or {})
        self.transcript: list[Turn] = []
        self._servers: list[McpServer] = []
        self._sessions: list[McpSessionProtocol] = []
        self._tool_bindings: dict[str, _ToolBinding] = {}
        self._tools_payload: list[dict[str, Any]] = []
        self._pending_local_tools: list[_PendingLocalTool] = []
        self._initialized = False
        self._closed = False
        self._responses_client = _responses_client or send_responses_request
        self._mcp_session_factory = _mcp_session_factory or create_mcp_session

    def _emit(self, event_type: str, **fields: Any) -> None:
        callback = self.on_event
        if callback is None:
            return
        callback({"type": event_type, "ts": time.time(), **fields})

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def add_mcp_server(self, server: McpServer) -> None:
        """Register one MCP server config without opening a connection."""
        if self._initialized:
            raise RuntimeError("Cannot add MCP servers after first send()")
        self._servers.append(server)

    def add_local_tool(
        self,
        name: str,
        handler: Callable[[dict[str, Any]], Any],
        parameters: dict[str, Any],
    ) -> None:
        """Register one in-process local tool without opening a connection."""
        if self._initialized:
            raise RuntimeError("Cannot add local tools after first send()")
        description = textwrap.dedent(handler.__doc__ or "").strip()
        if not description:
            raise ValueError(f"Local tool {name!r} handler must have a non-empty docstring")
        self._pending_local_tools.append(
            _PendingLocalTool(
                name=name,
                handler=handler,
                parameters=parameters,
                description=description,
            )
        )

    def close(self) -> None:
        """Close all open MCP sessions."""
        if self._closed:
            return
        self._closed = True
        for session in reversed(self._sessions):
            session.close()
        self._sessions.clear()
        self._initialized = False

    def send(self, user_prompt: str) -> Turn:
        """Run one logical user turn to completion, including tool round-trips."""
        if self._closed:
            raise RuntimeError("ChatSession is closed")
        self._ensure_initialized()
        turn = Turn(user_prompt=user_prompt, response_text="")
        # OpenRouter's Responses API is stateless: replay the whole input each
        # call, growing it with the assistant's tool calls and our results.
        input_items = self._build_initial_input(user_prompt)

        iterations = 0
        while self.max_iterations is None or iterations < self.max_iterations:
            iterations += 1
            self._emit("iteration", iteration=iterations, max_iterations=self.max_iterations)
            response = self._request_response(input_items)
            output_items = _read_output_items(response)
            turn.output_items.extend(output_items)
            turn.round_trips.append({"request": list(input_items), "response": response})

            function_calls = _read_function_calls(output_items)
            if not function_calls:
                turn.response_text = _extract_response_text(output_items)
                self._emit("finished", iteration=iterations, text=turn.response_text)
                self.transcript.append(turn)
                return turn

            assistant_text = _extract_response_text(output_items)
            if assistant_text:
                self._emit("assistant_text", iteration=iterations, text=assistant_text)
            call_items, output_replies = self._dispatch_tool_calls(function_calls, turn, iterations)
            input_items = input_items + call_items + output_replies

        assert self.max_iterations is not None
        self.transcript.append(turn)
        self._emit("max_iterations", iteration=iterations)
        raise MaxIterationsExceededError(
            self.max_iterations,
            turn=turn,
            iterations=iterations,
        )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._connect_servers()
        for pending in self._pending_local_tools:
            shim = _LocalToolSession(pending)
            self._sessions.append(shim)
            descriptor = _normalize_mcp_tools(shim.list_tools())[0]
            self._register_tool(_ToolBinding(server=None, session=shim), descriptor)
        self._initialized = True

    def _connect_servers(self) -> None:
        for server in self._servers:
            session = self._mcp_session_factory(server)
            self._sessions.append(session)
            self._register_server_tools(server, session)

    def _register_server_tools(self, server: McpServer, session: McpSessionProtocol) -> None:
        for tool in _normalize_mcp_tools(session.list_tools()):
            self._register_tool(_ToolBinding(server=server, session=session), tool)

    def _register_tool(self, binding: _ToolBinding, tool: dict[str, Any]) -> None:
        name = tool["name"]
        if name in self._tool_bindings:
            raise ToolNameCollisionError(f"Duplicate MCP tool name: {name}")
        self._tool_bindings[name] = binding
        self._tools_payload.append(_to_responses_tool(tool))

    def _build_initial_input(self, user_prompt: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if self.system_prompt:
            items.append(_text_message("system", self.system_prompt))
        items.append(_text_message("user", user_prompt))
        return items

    def _request_response(self, input_items: list[dict[str, Any]]) -> dict[str, Any]:
        return self._responses_client(
            base_url=self.base_url,
            model=self.model,
            input_items=input_items,
            tools=self._tools_payload,
            stream=self.stream,
            request_headers=self.request_headers,
        )

    def _dispatch_tool_calls(
        self,
        function_calls: list[dict[str, Any]],
        turn: Turn,
        iteration: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        call_items: list[dict[str, Any]] = []
        output_items: list[dict[str, Any]] = []
        for call in function_calls:
            tool_name = str(call["name"])
            binding = self._tool_bindings.get(tool_name)
            if binding is None:
                raise KeyError(f"Unknown MCP tool requested: {tool_name}")
            arguments = _parse_tool_arguments(call.get("arguments"))
            call_id = _read_call_id(call)
            self._emit(
                "tool_call",
                iteration=iteration,
                name=tool_name,
                call_id=call_id,
                arguments=arguments,
            )
            try:
                result = binding.session.call_tool(tool_name, arguments)
            except Exception as exc:
                self._emit(
                    "tool_error",
                    iteration=iteration,
                    name=tool_name,
                    call_id=call_id,
                    error=str(exc),
                )
                raise
            output_text = _tool_output_text(result)
            self._emit(
                "tool_result",
                iteration=iteration,
                name=tool_name,
                call_id=call_id,
                output=output_text,
            )
            turn.tool_calls.append(_normalize_value(call))
            turn.tool_results.append({"call_id": call_id, "name": tool_name, "result": output_text})
            call_items.append(_normalize_value(call))
            output_items.append(
                {"type": "function_call_output", "call_id": call_id, "output": output_text}
            )
        return call_items, output_items


def _text_message(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}


def _normalize_mcp_tools(raw_tools: Any) -> list[dict[str, Any]]:
    payload = _normalize_value(raw_tools)
    if isinstance(payload, dict) and isinstance(payload.get("tools"), list):
        tools = payload["tools"]
    elif isinstance(payload, list):
        tools = payload
    else:
        raise TypeError("MCP list_tools did not return a tools collection")
    normalized = [tool for tool in tools if isinstance(tool, dict)]
    for tool in normalized:
        if "name" not in tool:
            raise ValueError("MCP tool entry missing name")
    return normalized


def _to_responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": str(tool["name"]),
        "description": str(tool.get("description", "")),
        "parameters": _read_tool_schema(tool),
    }


def _read_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("inputSchema")
    if isinstance(schema, dict):
        return schema
    schema = tool.get("input_schema")
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}


def _read_output_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if isinstance(output, list):
        return [item for item in output if isinstance(item, dict)]
    return []


def _read_function_calls(output_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in output_items if item.get("type") == "function_call"]


def _extract_response_text(output_items: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for item in output_items:
        item_type = item.get("type")
        if item_type == "output_text" and isinstance(item.get("text"), str):
            texts.append(item["text"])
            continue
        if item_type != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"output_text", "text"} and isinstance(block.get("text"), str):
                texts.append(block["text"])
    return "".join(texts)


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        loaded = json.loads(raw_arguments)
        if isinstance(loaded, dict):
            return loaded
    raise ValueError("function_call arguments must be a JSON object")


def _tool_output_text(result: Any) -> str:
    """Flatten an MCP CallToolResult to the plain string the API expects."""
    payload = _normalize_value(result)
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        blocks = payload.get("content")
        if isinstance(blocks, list):
            texts = [b["text"] for b in blocks
                     if isinstance(b, dict) and isinstance(b.get("text"), str)]
            if texts:
                return "".join(texts)
        structured = payload.get("structuredContent")
        if structured is not None:
            return json.dumps(structured)
    return json.dumps(payload)


def _read_call_id(call: dict[str, Any]) -> str:
    call_id = call.get("call_id")
    if isinstance(call_id, str) and call_id:
        return call_id
    fallback_id = call.get("id")
    if isinstance(fallback_id, str) and fallback_id:
        return fallback_id
    raise ValueError("function_call item missing call_id")


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _normalize_value(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    return value
