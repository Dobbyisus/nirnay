"""Minimal HTTP client for Sarvam's chat completions API: auth, pacing, retries."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import httpx

BASE_URL = "https://api.sarvam.ai"
V1_CHAT = "/v1/chat/completions"
V2_CHAT = "/v2/chat/completions"

# Worth retrying: rate limits and transient server trouble. Everything else fails fast.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class SarvamError(Exception):
    """An API call failed and was not (or no longer) worth retrying."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class SarvamClient:
    """Sends chat completion requests to Sarvam.

    Retries rate limits (429), 5xx errors and network failures with exponential backoff,
    honouring ``Retry-After`` when the server sends it. ``min_interval_s`` spaces calls out
    to stay under a requests-per-minute limit (e.g. 1.5 s for 40 RPM).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        min_interval_s: float = 0.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not api_key:
            raise SarvamError("No API key: pass api_key or set SARVAM_API_KEY")
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"api-subscription-key": api_key},
            transport=transport,
        )
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self._sleep = sleep
        self._last_call = float("-inf")
        self.last_paced_s = 0.0  # time the last chat() call spent waiting for min_interval_s

    def chat(self, body: dict[str, Any], *, endpoint: str = V1_CHAT) -> dict[str, Any]:
        """POST ``body`` to ``endpoint`` and return the parsed JSON response."""
        self.last_paced_s = 0.0
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                r = self._http.post(endpoint, json=body)
            except httpx.TransportError as e:  # timeouts, connection resets, DNS…
                if attempt < self.max_retries:
                    self._sleep(_backoff(attempt))
                    continue
                raise SarvamError(f"Request failed after {attempt + 1} attempts: {e}") from e

            if r.status_code == 200:
                return r.json()
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                self._sleep(_retry_after(r) or _backoff(attempt))
                continue
            raise SarvamError(
                f"HTTP {r.status_code}: {_error_message(r)}", status=r.status_code, body=r.text
            )
        raise AssertionError("unreachable")

    def _pace(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)
            self.last_paced_s += wait
        self._last_call = time.monotonic()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SarvamClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _backoff(attempt: int) -> float:
    return min(2.0**attempt, 30.0)


def _retry_after(r: httpx.Response) -> float | None:
    try:
        return min(float(r.headers["retry-after"]), 60.0)
    except (KeyError, ValueError):
        return None


def _error_message(r: httpx.Response) -> str:
    try:
        return r.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return r.text[:300]
