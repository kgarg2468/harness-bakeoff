"""Unit tests for our_version's Pi ports: compat flags, reasoning_details merging, retry policy."""

from __future__ import annotations

from email.utils import formatdate
from time import time

import httpx

from bakeoff.our_version.compat import merge_detail, static_body, usage_fields
from bakeoff.our_version.retry import backoff, classify, is_context_overflow, retry_after
from bakeoff.shared.contract import ModelConfig


def model(**kwargs: object) -> ModelConfig:
    return ModelConfig(base_url="http://127.0.0.1:9/v1", **kwargs)  # type: ignore[arg-type]


def test_compat_defaults_by_kind() -> None:
    openrouter = static_body(
        model(model="openai/gpt-5", reasoning={"effort": "high"}), "sys", [], "s1"
    )
    assert openrouter == {
        "model": "openai/gpt-5",
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0.0,
        "reasoning": {"effort": "high"},
        "session_id": "s1",
        "messages": [{"role": "system", "content": "sys"}],
    }
    byok = static_body(
        model(model="deepseek-r1", kind="openai_compat", reasoning={}, temperature=None),
        "sys",
        [],
        "s1",
    )
    assert byok == {
        "model": "deepseek-r1",
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": 4096,
        "reasoning_effort": "medium",
        "messages": [{"role": "system", "content": "sys"}],
    }


def test_compat_overrides() -> None:
    flags = {
        "max_tokens_field": "max_completion_tokens",
        "reasoning_param": "reasoning_effort",
        "developer_role": True,
    }
    body = static_body(
        model(model="anthropic/claude-sonnet-5", reasoning={"effort": "low"}, compat=flags),
        "s",
        [],
        "x",
    )
    assert body["reasoning_effort"] == "low" and "reasoning" not in body
    assert body["max_completion_tokens"] == 4096 and body["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][0]["role"] == "developer"


def test_merge_detail_by_index_and_type() -> None:
    details: list[dict[str, object]] = []
    for fragment in [
        {"type": "reasoning.text", "text": None, "index": 0},
        {"type": "reasoning.text", "text": "a", "index": 0, "signature": None},
        {"type": "reasoning.summary", "summary": "s", "index": 0},
        {"type": "reasoning.text", "text": "b", "index": 0, "signature": "sig", "format": "f"},
        {"type": "reasoning.text", "signature": "later", "format": "g", "index": 0},
        {"type": "reasoning.text", "text": "other", "index": 1},
    ]:
        merge_detail(details, fragment)
    assert details == [
        {"type": "reasoning.text", "text": "ab", "index": 0, "signature": "sig", "format": "f"},
        {"type": "reasoning.summary", "summary": "s", "index": 0},
        {"type": "reasoning.text", "text": "other", "index": 1},
    ]


def test_merge_detail_without_index_keeps_order_and_blocks_apart() -> None:
    details: list[dict[str, object]] = []
    for fragment in [
        {"type": "reasoning.text", "text": "A", "signature": "s1"},
        {"type": "reasoning.encrypted", "data": "E1", "id": "tool_call_1"},
        {"type": "reasoning.encrypted", "data": "E2", "id": "tool_call_2"},
        {"type": "reasoning.text", "text": "B"},
        {"type": "reasoning.text", "text": "C", "signature": "s2"},
    ]:
        merge_detail(details, fragment)
    assert details == [
        {"type": "reasoning.text", "text": "A", "signature": "s1"},
        {"type": "reasoning.encrypted", "data": "E1", "id": "tool_call_1"},
        {"type": "reasoning.encrypted", "data": "E2", "id": "tool_call_2"},
        {"type": "reasoning.text", "text": "BC", "signature": "s2"},
    ]


def test_usage_fields_fallbacks() -> None:
    assert usage_fields({"prompt_tokens": 9, "prompt_cache_hit_tokens": 4}) == {
        "input_tokens": 9,
        "output_tokens": 0,
        "cached_tokens": 4,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "cost_source": "none",
    }
    assert usage_fields({"cost": 0})["cost_source"] == "provider"


def test_classify_by_status_and_message() -> None:
    limited = classify("http", "HTTP 429: slow down", 429, httpx.Headers({"retry-after": "2"}))
    assert limited.retryable and limited.wait_s == 2.0
    assert not classify("http", "HTTP 429: insufficient_quota", 429).retryable
    assert not classify("http", "HTTP 400: bad request", 400).retryable
    assert classify("http", "HTTP 503: busy", 503).retryable
    assert not classify(
        "http", "HTTP 503", 503, httpx.Headers({"x-should-retry": "false"})
    ).retryable
    assert classify("stream", "server_error: upstream died").retryable
    assert not classify("stream", "content policy").retryable
    overflow = classify(
        "http", "HTTP 400: This endpoint's maximum context length is 8192 tokens", 400
    )
    assert overflow.kind == "context_overflow" and not overflow.retryable


def test_status_codes_match_only_as_whole_words() -> None:
    assert classify("stream", "HTTP 502 from upstream").retryable
    assert classify("stream", "error 429").retryable
    assert not classify("stream", "tool call call_5029 has invalid arguments").retryable
    assert not classify("stream", "Invalid request: max_tokens must be <= 128500").retryable


def test_per_minute_quota_is_a_throttle() -> None:
    per_minute = (
        "HTTP 429: Provider returned error\nQuota exceeded for quota metric requests_per_minute"
    )
    assert classify("http", per_minute, 429).retryable
    assert not classify("http", "HTTP 429: Quota exceeded for this month", 429).retryable


def test_retry_after_forms_and_cap() -> None:
    assert retry_after(httpx.Headers({"retry-after-ms": "250", "retry-after": "9"})) == 0.25
    in_30s = retry_after(httpx.Headers({"retry-after": formatdate(time() + 30, usegmt=True)}))
    assert in_30s is not None and 28 <= in_30s <= 30
    assert retry_after(httpx.Headers({"retry-after": "soon"})) is None
    too_long = classify("http", "HTTP 429: wait", 429, httpx.Headers({"retry-after": "3600"}))
    assert not too_long.retryable and too_long.message.startswith(
        "Server requested 3600s retry delay"
    )


def test_overflow_ignores_throttling() -> None:
    assert is_context_overflow("prompt is too long: 213462 tokens > 200000 maximum")
    assert not is_context_overflow("ThrottlingException: Too many tokens, rate limit hit")


def test_backoff_is_jittered_and_capped() -> None:
    assert 0.375 <= backoff(0) <= 0.5
    assert 6.0 <= backoff(10) <= 8.0
