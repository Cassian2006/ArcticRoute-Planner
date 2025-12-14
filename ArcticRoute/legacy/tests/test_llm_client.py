from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest
import requests

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.providers.moonshot_k2 import MoonshotK2Client, MoonshotK2Error

pytestmark = pytest.mark.p0


class DummyResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"status {self.status_code}")


def _make_client(monkeypatch: pytest.MonkeyPatch, responses: List[object], sleep_recorder: List[float]) -> MoonshotK2Client:
    session = SimpleNamespace()
    session.calls = []

    def post(url: str, json: dict, headers: dict, timeout: float) -> DummyResponse:
        idx = len(session.calls)
        session.calls.append({"url": url, "payload": json, "timeout": timeout})
        response = responses[idx]
        if isinstance(response, Exception):
            raise response
        return response  # type: ignore[return-value]

    session.post = post  # type: ignore[attr-defined]

    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    client = MoonshotK2Client(
        model="moonshot-k2",
        base_url="https://example.test",
        session=session,  # type: ignore[arg-type]
        sleep_fn=lambda delay: sleep_recorder.append(delay),
        timeout=5,
        max_retries=3,
        backoff_factor=1.0,
    )
    client._session = session  # type: ignore[attr-defined]
    client._session.calls = session.calls  # type: ignore[attr-defined]
    return client


def test_retry_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: List[float] = []
    responses = [
        requests.Timeout(),
        DummyResponse(
            payload={"choices": [{"message": {"content": "all good"}}]},
        ),
    ]
    client = _make_client(monkeypatch, responses, sleeps)

    result = client.complete("hello world", max_tokens=32)

    assert result == "all good"
    assert len(client._session.calls) == 2  # type: ignore[attr-defined]
    assert sleeps == [1.0]


def test_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: List[float] = []
    responses = [requests.Timeout(), requests.Timeout(), requests.Timeout()]
    client = _make_client(monkeypatch, responses, sleeps)

    with pytest.raises(MoonshotK2Error):
        client.complete("hello again")

    assert len(client._session.calls) == 3  # type: ignore[attr-defined]
    assert sleeps == [1.0, 2.0]
