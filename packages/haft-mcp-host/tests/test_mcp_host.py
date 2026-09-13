"""Unit tests for haft.mcp_host ChatSession and transport helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import pytest

from haft.mcp_host import ChatSession, MaxIterationsExceededError, McpServer
from haft.mcp_host.responses import send_responses_request
from haft.mcp_host.session import ToolNameCollisionError


@dataclass
class _FakeMcpSession:
    tools: list[dict[str, Any]]
    tool_result: Any = field(default_factory=dict)
    tool_error: Exception | None = None
    list_calls: int = 0
    call_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    closed: bool = False

    def list_tools(self) -> dict[str, Any]:
        self.list_calls += 1
        return {"tools": self.tools}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.call_calls.append((name, arguments))
        if self.tool_error is not None:
            raise self.tool_error
        return self.tool_result

    def close(self) -> None:
        self.closed = True


class _FakeResponsesClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.responses[len(self.calls) - 1]


def _assert_event_payloads(events: list[dict[str, Any]]) -> None:
    assert events
    for event in events:
        assert isinstance(event["type"], str)
        assert isinstance(event["ts"], float)


def test_add_mcp_server_is_lazy_until_first_send() -> None:
    session_obj = _FakeMcpSession(
        tools=[{"name": "read_file", "description": "Read a file", "inputSchema": {"type": "object"}}]
    )
    factory_calls = 0

    def factory(server: McpServer) -> _FakeMcpSession:
        nonlocal factory_calls
        factory_calls += 1
        assert server.url == "https://example.test/mcp"
        return session_obj

    responses = _FakeResponsesClient(
        [{"id": "resp-1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}]
    )

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=responses,
        _mcp_session_factory=factory,
    )
    chat.add_mcp_server(McpServer(url="https://example.test/mcp"))

    assert factory_calls == 0
    turn = chat.send("hello")

    assert factory_calls == 1
    assert session_obj.list_calls == 1
    assert turn.response_text == "ok"


def test_send_runs_function_call_loop_and_tracks_turn_artifacts() -> None:
    mcp_session = _FakeMcpSession(
        tools=[{"name": "sum", "description": "Adds", "inputSchema": {"type": "object"}}],
        tool_result={"result": 3},
    )
    responses = _FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "sum",
                        "arguments": '{"a": 1, "b": 2}',
                    }
                ],
            },
            {
                "id": "resp-2",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "final answer"}],
                    }
                ],
            },
        ]
    )

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=responses,
        _mcp_session_factory=lambda _server: mcp_session,
    )
    chat.add_mcp_server(McpServer(url="https://example.test/mcp", headers={"X-Test": "1"}))

    turn = chat.send("compute")

    assert turn is chat.transcript[-1]
    assert turn.response_text == "final answer"
    assert turn.tool_calls[0]["name"] == "sum"
    assert turn.tool_results[0]["name"] == "sum"
    assert mcp_session.call_calls == [("sum", {"a": 1, "b": 2})]
    assert responses.calls[0]["tools"][0]["name"] == "sum"
    assert "previous_response_id" not in responses.calls[1]
    assert responses.calls[1]["input_items"] == [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "compute"}]},
        {"type": "function_call", "call_id": "call-1", "name": "sum",
         "arguments": '{"a": 1, "b": 2}'},
        {"type": "function_call_output", "call_id": "call-1", "output": '{"result": 3}'},
    ]


def test_send_emits_events_for_multi_round_trip() -> None:
    mcp_session = _FakeMcpSession(
        tools=[{"name": "sum", "description": "Adds", "inputSchema": {"type": "object"}}],
        tool_result={"result": 3},
    )
    responses = _FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "working"}]},
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "sum",
                        "arguments": '{"a": 1, "b": 2}',
                    },
                ],
            },
            {
                "id": "resp-2",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "final answer"}],
                    }
                ],
            },
        ]
    )
    events: list[dict[str, Any]] = []

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        on_event=events.append,
        _responses_client=responses,
        _mcp_session_factory=lambda _server: mcp_session,
    )
    chat.add_mcp_server(McpServer(url="https://example.test/mcp"))

    turn = chat.send("compute")

    assert turn.response_text == "final answer"
    assert [event["type"] for event in events] == [
        "iteration",
        "assistant_text",
        "tool_call",
        "tool_result",
        "iteration",
        "finished",
    ]
    assert events[0]["iteration"] == 1
    assert events[0]["max_iterations"] == 10
    assert events[1]["text"] == "working"
    assert events[2]["name"] == "sum"
    assert events[2]["call_id"] == "call-1"
    assert events[2]["arguments"] == {"a": 1, "b": 2}
    assert events[3]["call_id"] == "call-1"
    assert events[3]["output"] == '{"result": 3}'
    assert events[4]["iteration"] == 2
    assert events[5]["text"] == "final answer"
    _assert_event_payloads(events)


def test_send_skips_empty_assistant_text_event() -> None:
    mcp_session = _FakeMcpSession(
        tools=[{"name": "sum", "description": "Adds", "inputSchema": {"type": "object"}}],
        tool_result={"result": 3},
    )
    responses = _FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "sum",
                        "arguments": '{"a": 1, "b": 2}',
                    }
                ],
            },
            {
                "id": "resp-2",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "final answer"}],
                    }
                ],
            },
        ]
    )
    events: list[dict[str, Any]] = []

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        on_event=events.append,
        _responses_client=responses,
        _mcp_session_factory=lambda _server: mcp_session,
    )
    chat.add_mcp_server(McpServer(url="https://example.test/mcp"))

    chat.send("compute")

    assert [event["type"] for event in events] == [
        "iteration",
        "tool_call",
        "tool_result",
        "iteration",
        "finished",
    ]
    _assert_event_payloads(events)


def test_duplicate_tool_names_across_servers_raise_error() -> None:
    first = _FakeMcpSession(tools=[{"name": "search", "inputSchema": {"type": "object"}}])
    second = _FakeMcpSession(tools=[{"name": "search", "inputSchema": {"type": "object"}}])
    responses = _FakeResponsesClient([
        {"id": "resp-1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]}
    ])
    sessions = [first, second]

    def factory(_server: McpServer) -> _FakeMcpSession:
        return sessions.pop(0)

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=responses,
        _mcp_session_factory=factory,
    )
    chat.add_mcp_server(McpServer(url="https://one.example/mcp"))
    chat.add_mcp_server(McpServer(url="https://two.example/mcp"))

    with pytest.raises(ToolNameCollisionError, match="Duplicate MCP tool name"):
        chat.send("hi")


def test_max_iterations_backstop_raises_when_loop_never_finishes() -> None:
    mcp_session = _FakeMcpSession(
        tools=[{"name": "ping", "inputSchema": {"type": "object"}}],
        tool_result={"ok": True},
    )
    responses = _FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "ping",
                        "arguments": "{}",
                    }
                ],
            }
        ]
    )

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        max_iterations=1,
        _responses_client=responses,
        _mcp_session_factory=lambda _server: mcp_session,
    )
    chat.add_mcp_server(McpServer(url="https://example.test/mcp"))

    events: list[dict[str, Any]] = []
    chat.on_event = events.append

    with pytest.raises(MaxIterationsExceededError) as exc_info:
        chat.send("go")

    error = exc_info.value

    assert error.iterations == 1
    assert error.turn is chat.transcript[-1]
    assert "1" in str(error)
    assert error.turn is not None
    assert error.turn.tool_calls[0]["name"] == "ping"
    assert error.turn.tool_results[0] == {"call_id": "call-1", "name": "ping", "result": '{"ok": true}'}
    assert [event["type"] for event in events] == [
        "iteration",
        "tool_call",
        "tool_result",
        "max_iterations",
    ]
    assert events[-1]["iteration"] == 1
    _assert_event_payloads(events)


def test_tool_error_event_is_emitted_before_exception_propagates() -> None:
    expected = RuntimeError("tool failed")
    mcp_session = _FakeMcpSession(
        tools=[{"name": "ping", "description": "Pings", "inputSchema": {"type": "object"}}],
        tool_error=expected,
    )
    responses = _FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "ping",
                        "arguments": "{}",
                    }
                ],
            }
        ]
    )
    events: list[dict[str, Any]] = []

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        on_event=events.append,
        _responses_client=responses,
        _mcp_session_factory=lambda _server: mcp_session,
    )
    chat.add_mcp_server(McpServer(url="https://example.test/mcp"))

    with pytest.raises(RuntimeError, match="tool failed") as exc_info:
        chat.send("go")

    assert exc_info.value is expected
    assert [event["type"] for event in events] == ["iteration", "tool_call", "tool_error"]
    assert events[1]["name"] == "ping"
    assert events[1]["call_id"] == "call-1"
    assert events[2]["error"] == "tool failed"
    _assert_event_payloads(events)


def test_max_iterations_none_allows_multi_step_loop_to_complete() -> None:
    mcp_session = _FakeMcpSession(
        tools=[{"name": "fetch", "inputSchema": {"type": "object"}}],
        tool_result={"payload": "value"},
    )
    responses = _FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "output": [{"type": "function_call", "call_id": "call-1", "name": "fetch", "arguments": "{}"}],
            },
            {
                "id": "resp-2",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
            },
        ]
    )

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        max_iterations=None,
        _responses_client=responses,
        _mcp_session_factory=lambda _server: mcp_session,
    )
    chat.add_mcp_server(McpServer(url="https://example.test/mcp"))

    turn = chat.send("start")

    assert turn.response_text == "done"
    assert len(responses.calls) == 2


def test_context_manager_closes_open_mcp_sessions() -> None:
    mcp_session = _FakeMcpSession(
        tools=[{"name": "noop", "inputSchema": {"type": "object"}}]
    )
    responses = _FakeResponsesClient(
        [{"id": "resp-1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}]
    )

    with ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=responses,
        _mcp_session_factory=lambda _server: mcp_session,
    ) as chat:
        chat.add_mcp_server(McpServer(url="https://example.test/mcp"))
        chat.send("hello")

    assert mcp_session.closed is True


def test_send_responses_request_non_streaming_uses_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"id": "resp-1", "output": []}

    class _FakeClient:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

        def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse()

    monkeypatch.setattr("haft.mcp_host.responses._build_http_client", lambda _headers: _FakeClient())

    result = send_responses_request(
        base_url="https://responses.example.test/",
        model="model-a",
        input_items=[{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        tools=[],
    )

    assert captured["url"] == "https://responses.example.test/responses"
    assert captured["json"]["stream"] is False
    assert result["id"] == "resp-1"


def test_send_responses_request_streaming_reassembles_completed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStreamResponse:
        def __init__(self, lines: list[str]) -> None:
            self._lines = lines

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self) -> list[str]:
            return self._lines

    class _FakeClient:
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

        def stream(self, method: str, url: str, json: dict[str, Any]) -> _FakeStreamResponse:
            assert method == "POST"
            assert url.endswith("/responses")
            assert json["stream"] is True
            return _FakeStreamResponse(
                [
                    "data: {\"type\": \"response.created\", \"response\": {\"id\": \"resp-1\", \"output\": []}}",
                    "data: {\"type\": \"response.output_item.added\", \"item\": {\"type\": \"message\", \"content\": [{\"type\": \"output_text\", \"text\": \"hi\"}]}}",
                    "data: {\"type\": \"response.completed\", \"response\": {\"id\": \"resp-1\", \"output\": [{\"type\": \"message\", \"content\": [{\"type\": \"output_text\", \"text\": \"hi\"}]}]}}",
                    "data: [DONE]",
                ]
            )

    monkeypatch.setattr("haft.mcp_host.responses._build_http_client", lambda _headers: _FakeClient())

    result = send_responses_request(
        base_url="https://responses.example.test",
        model="model-a",
        input_items=[{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        tools=[],
        stream=True,
    )

    assert result["id"] == "resp-1"
    assert result["output"][0]["type"] == "message"


# ---------------------------------------------------------------------------
# Local in-process tool registration
# ---------------------------------------------------------------------------


def test_add_local_tool_is_lazy_until_first_send() -> None:
    def echo_handler(args: dict[str, Any]) -> str:
        """Echo the input back."""
        return str(args)

    responses = _FakeResponsesClient(
        [{"id": "resp-1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]}]
    )

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=responses,
    )
    chat.add_local_tool(
        "echo",
        echo_handler,
        {"type": "object", "properties": {"text": {"type": "string"}}},
    )

    assert responses.calls == []
    assert chat._initialized is False

    turn = chat.send("hello")

    assert turn.response_text == "done"
    assert responses.calls[0]["tools"] == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo the input back.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        }
    ]


def test_local_tool_dispatch_flattens_string_result() -> None:
    received: list[dict[str, Any]] = []

    def echo_handler(args: dict[str, Any]) -> str:
        """Echo the input back."""
        received.append(args)
        return "hello back"

    responses = _FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "echo",
                        "arguments": '{"text": "hi"}',
                    }
                ],
            },
            {
                "id": "resp-2",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "final answer"}],
                    }
                ],
            },
        ]
    )

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=responses,
    )
    chat.add_local_tool("echo", echo_handler, {"type": "object"})

    turn = chat.send("say hi")

    assert received == [{"text": "hi"}]
    assert turn.response_text == "final answer"
    assert turn.tool_calls[0]["name"] == "echo"
    assert turn.tool_results == [{"call_id": "call-1", "name": "echo", "result": "hello back"}]
    assert responses.calls[1]["input_items"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "hello back",
    }


def test_local_tool_dispatch_flattens_mcp_content_dict() -> None:
    def echo_handler(args: dict[str, Any]) -> dict[str, Any]:
        """Echo the input back."""
        return {"content": [{"type": "text", "text": "ok"}]}

    responses = _FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "echo",
                        "arguments": "{}",
                    }
                ],
            },
            {
                "id": "resp-2",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "final answer"}],
                    }
                ],
            },
        ]
    )

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=responses,
    )
    chat.add_local_tool("echo", echo_handler, {"type": "object"})

    turn = chat.send("go")

    assert turn.tool_results == [{"call_id": "call-1", "name": "echo", "result": "ok"}]
    assert responses.calls[1]["input_items"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "ok",
    }


def test_local_tool_name_colliding_with_remote_tool_raises() -> None:
    mcp_session = _FakeMcpSession(
        tools=[{"name": "dup", "description": "Remote dup", "inputSchema": {"type": "object"}}]
    )
    responses = _FakeResponsesClient(
        [{"id": "resp-1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]}]
    )

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=responses,
        _mcp_session_factory=lambda _server: mcp_session,
    )
    chat.add_mcp_server(McpServer(url="https://example.test/mcp"))

    def dup_handler(args: dict[str, Any]) -> str:
        """Handle dup."""
        return "ok"

    chat.add_local_tool("dup", dup_handler, {"type": "object"})

    with pytest.raises(ToolNameCollisionError, match="Duplicate MCP tool name"):
        chat.send("hi")


def test_add_local_tool_requires_non_empty_docstring() -> None:
    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=_FakeResponsesClient([]),
    )

    def no_docstring(args: dict[str, Any]) -> str:
        return "ok"

    with pytest.raises(ValueError, match="non-empty docstring"):
        chat.add_local_tool("tool-a", no_docstring, {"type": "object"})

    def blank_docstring(args: dict[str, Any]) -> str:
        """   """
        return "ok"

    with pytest.raises(ValueError, match="non-empty docstring"):
        chat.add_local_tool("tool-b", blank_docstring, {"type": "object"})


def test_add_local_tool_after_first_send_raises() -> None:
    def echo_handler(args: dict[str, Any]) -> str:
        """Echo the input back."""
        return "ok"

    responses = _FakeResponsesClient(
        [{"id": "resp-1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]}]
    )

    chat = ChatSession(
        model="test-model",
        base_url="https://responses.example.test",
        _responses_client=responses,
    )
    chat.add_local_tool("echo", echo_handler, {"type": "object"})
    chat.send("hello")

    with pytest.raises(RuntimeError, match="after first send"):
        chat.add_local_tool("late", echo_handler, {"type": "object"})
