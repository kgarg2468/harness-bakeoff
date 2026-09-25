# Ported from Pi (MIT): packages/ai/src/utils/{retry.ts,provider-retry.ts,overflow.ts} @ 5fd446ca1843682e8da3fec4ceb71c42f56fbace
# Changes: Python; one ProviderError (kind, retryable, wait); Node/websocket-only patterns dropped; retry loop in loop.py.
"""Which provider errors to retry, how long to wait, and context-overflow detection.

Not ported: silent and length-stop overflow detection (needs the model's context window),
Cerebras' body-less 400/413, and Pi's agent-level retry callbacks.
"""

from __future__ import annotations

import math
import random
import re
import time
from email.utils import parsedate_to_datetime

import httpx

MAX_RETRY_DELAY_S = 60.0  # a server asking for a longer wait is not retried


def _any(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


# Quota, budget and billing exhaustion: not transient, even when sent as HTTP 429.
_NON_RETRYABLE = _any(
    "GoUsageLimitError",
    "FreeUsageLimitError",
    "Monthly usage limit reached",
    "available balance",
    "insufficient_quota",
    "out of budget",
    "quota exceeded",
    "billing",
)
# Transient failures, for errors without an HTTP status (mid-stream errors, transport).
_RETRYABLE = _any(
    "overloaded",
    "currently experiencing high demand",
    r"rate.?limit",
    "too many requests",
    *("429", "500", "502", "503", "504", "520", "524"),
    r"service.?unavailable",
    r"server.?error",
    r"internal.?error",
    r"provider.?returned.?error",
    "exceeded request buffer limit while retrying upstream",
    r"network.?error",
    r"connection.?error",
    r"connection.?refused",
    r"connection.?lost",
    "other side closed",
    "fetch failed",
    r"upstream.?connect",
    "reset before headers",
    "socket hang up",
    "socket connection was closed",
    "timed? out",
    "timeout",
    "terminated",
    "ended without",
    "you can retry your request",
    "try your request again",
    "please retry your request",
    "ResourceExhausted",
)
_OVERFLOW = _any(
    r"prompt (?:is )?too long",
    "request_too_large",
    "input is too long for requested model",
    "exceeds the context window",
    r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
    "input token count.*exceeds the maximum",
    r"maximum prompt length is \d+",
    "reduce the length of the messages",
    r"maximum context length is \d+ tokens",
    r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
    r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
    r"exceeds the limit of \d+",
    "exceeds the available context size",
    "greater than the context length",
    "context window exceeds limit",
    "exceeded model token limit",
    r"too large for model with \d+ maximum context length",
    r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
    "model_context_window_exceeded",
    r"prompt too long; exceeded (?:max )?context length",
    "range of input length should be",
    r"context[_ ]length[_ ]exceeded",
    "too many tokens",
    "token limit exceeded",
)
# Throttling messages that would otherwise match an overflow pattern ("Too many tokens, ...").
_NON_OVERFLOW = _any(r"^(Throttling error|Service unavailable):", "rate limit", "too many requests")


class ProviderError(Exception):
    """A failed model request. `kind` is http, stream, network or context_overflow."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
        wait_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind, self.message, self.retryable = kind, message, retryable
        self.status, self.wait_s = status, wait_s


def is_context_overflow(message: str) -> bool:
    """True if the provider says the input does not fit the model's context window."""
    return not _NON_OVERFLOW.search(message) and bool(_OVERFLOW.search(message))


def classify(
    kind: str, message: str, status: int | None = None, headers: httpx.Headers | None = None
) -> ProviderError:
    """Build the error for a failed request, deciding whether (and when) to retry it."""
    if is_context_overflow(message):
        return ProviderError("context_overflow", message, status=status)
    should_retry = headers.get("x-should-retry") if headers is not None else None
    if should_retry in ("true", "false"):
        retryable = should_retry == "true"
    elif _NON_RETRYABLE.search(message):
        retryable = False
    elif status is None:
        retryable = bool(_RETRYABLE.search(message))
    else:
        retryable = status in (408, 409, 429) or status >= 500
    wait = retry_after(headers) if retryable and headers is not None else None
    if wait is not None and wait > MAX_RETRY_DELAY_S:
        message = f"Server requested {math.ceil(wait)}s retry delay (max: {MAX_RETRY_DELAY_S:.0f}s). {message}"
        return ProviderError(kind, message, status=status)
    return ProviderError(kind, message, retryable=retryable, status=status, wait_s=wait)


def retry_after(headers: httpx.Headers) -> float | None:
    """Seconds the server asked us to wait (`retry-after-ms`, or `retry-after` seconds/date)."""
    if (ms := headers.get("retry-after-ms")) is not None:
        try:
            return float(ms) / 1000
        except ValueError:
            pass
    if (value := headers.get("retry-after")) is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def backoff(retry_index: int) -> float:
    """Jittered exponential backoff in seconds: 0.5, 1, 2, 4, 8 (cap), minus up to 25%."""
    return min(0.5 * 2**retry_index, 8.0) * (1 - random.random() * 0.25)
