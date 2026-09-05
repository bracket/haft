"""Synchronous transport helpers for OpenAI-Responses-compatible endpoints."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from importlib import import_module
from typing import Any


class ResponsesTransportError(RuntimeError):
    """Raised when a Responses request cannot be completed."""


def send_responses_request(
    *,
    base_url: str,
    model: str,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    previous_response_id: str | None = None,
    stream: bool = False,
    request_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """POST one Responses request and return the normalized response object."""
    payload: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "tools": tools,
        "stream": stream,
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id

    url = f"{base_url.rstrip('/')}/responses"
    headers = dict(request_headers or {})
    return _stream_request(url, payload, headers) if stream else _json_request(url, payload, headers)


def _json_request(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    client = _build_http_client(headers)
    with client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise ResponsesTransportError("Responses endpoint returned non-object JSON")
    return body


def _stream_request(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    client = _build_http_client(headers)
    with client, client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            event = _parse_sse_event(line)
            if event == "[DONE]":
                break
            if isinstance(event, dict):
                events.append(event)
    return _assemble_streamed_response(events)


def _build_http_client(headers: dict[str, str]) -> Any:
    try:
        httpx = import_module("httpx")
    except ImportError as exc:
        raise RuntimeError("httpx is required for haft-mcp-host Responses transport") from exc
    return httpx.Client(headers=headers, timeout=60.0)


def _parse_sse_event(line: str) -> dict[str, Any] | str | None:
    raw = line.strip()
    if not raw or raw.startswith(":"):
        return None
    if not raw.startswith("data:"):
        return None
    data = raw[5:].strip()
    if not data:
        return None
    if data == "[DONE]":
        return data
    loaded = json.loads(data)
    return loaded if isinstance(loaded, dict) else None


def _assemble_streamed_response(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    output_items: list[dict[str, Any]] = []
    response_body: dict[str, Any] = {"output": output_items}
    for event in events:
        event_type = event.get("type")
        if event_type == "response.completed":
            completed = event.get("response")
            if isinstance(completed, dict):
                return completed
            continue
        if event_type == "response.created":
            created = event.get("response")
            if isinstance(created, dict):
                response_body.update(created)
                current_output = created.get("output")
                if isinstance(current_output, list):
                    output_items[:] = [i for i in current_output if isinstance(i, dict)]
            continue
        if event_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict):
                output_items.append(item)
    return response_body
