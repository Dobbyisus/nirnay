import httpx
import pytest

from nirnay import SarvamClient, SarvamError

from .conftest import completion


def test_sends_key_header_and_returns_json(fake):
    seen = {}

    def handler(request):
        seen["key"] = request.headers["api-subscription-key"]
        seen["path"] = request.url.path
        return httpx.Response(200, json=completion(["B"]))

    client = SarvamClient(api_key="k", transport=httpx.MockTransport(handler))
    assert client.chat({"model": "m"})["choices"][0]["message"]["content"] == "B"
    assert seen == {"key": "k", "path": "/v1/chat/completions"}


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(SarvamError, match="No API key"):
        SarvamClient()


def test_retries_rate_limit_then_succeeds(fake):
    fake.push(429, {"error": {"message": "slow down"}}, {"retry-after": "2"})
    fake.push(503, {"error": {"message": "overloaded"}})
    fake.reply("B")
    assert fake.client().chat({})["choices"][0]["message"]["content"] == "B"
    assert len(fake.requests) == 3


def test_gives_up_after_max_retries(fake):
    for _ in range(3):
        fake.push(429, {"error": {"message": "slow down"}})
    with pytest.raises(SarvamError) as e:
        fake.client(max_retries=2).chat({})
    assert e.value.status == 429 and "slow down" in str(e.value)


def test_client_errors_fail_fast(fake):
    fake.push(403, {"error": {"message": "invalid key"}})
    with pytest.raises(SarvamError) as e:
        fake.client().chat({})
    assert e.value.status == 403 and len(fake.requests) == 1


def test_network_errors_are_retried(fake):
    fake.push(body=httpx.ReadTimeout("timed out"))
    fake.reply("A")
    assert fake.client().chat({})["choices"][0]["message"]["content"] == "A"


def test_network_errors_eventually_raise(fake):
    for _ in range(2):
        fake.push(body=httpx.ConnectError("no route"))
    with pytest.raises(SarvamError, match="after 2 attempts"):
        fake.client(max_retries=1).chat({})


def test_min_interval_paces_calls(fake):
    slept = []
    fake.reply("A").reply("B")
    client = SarvamClient(
        api_key="k",
        transport=httpx.MockTransport(fake.handler),
        min_interval_s=1.5,
        sleep=slept.append,
    )
    client.chat({})
    assert client.last_paced_s == 0
    client.chat({})
    assert len(slept) == 1 and 0 < slept[0] <= 1.5
    assert client.last_paced_s == slept[0]
