import json

import httpx
import pytest

from nirnay import SarvamClient


def completion(replies, *, prompt_tokens=70, cached=None, completion_tokens=None):
    """A fake Sarvam chat completion response with one choice per reply."""
    details = {"cached_tokens": cached} if cached is not None else None
    return {
        "choices": [
            {"index": i, "message": {"role": "assistant", "content": r}, "finish_reason": "stop"}
            for i, r in enumerate(replies)
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
            if completion_tokens is not None
            else 3 * len(replies),
            "total_tokens": 0,
            "prompt_tokens_details": details,
        },
    }


class FakeSarvam:
    """Replays queued responses and records every request body."""

    def __init__(self):
        self.queue = []
        self.requests = []

    def push(self, status=200, body=None, headers=None):
        self.queue.append((status, body, headers or {}))
        return self

    def reply(self, *replies, **kwargs):
        return self.push(200, completion(list(replies), **kwargs))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        status, body, headers = self.queue.pop(0)
        if isinstance(body, Exception):
            raise body
        return httpx.Response(status, json=body, headers=headers)

    def client(self, **kwargs) -> SarvamClient:
        return SarvamClient(
            api_key="test-key",
            transport=httpx.MockTransport(self.handler),
            sleep=lambda s: None,
            **kwargs,
        )


@pytest.fixture
def fake():
    return FakeSarvam()
