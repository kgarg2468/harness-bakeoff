"""Build the pydantic-ai model for a `ModelConfig`.

OpenRouter uses `OpenRouterModel` + `OpenRouterProvider`, as the docs recommend, so
`reasoning_details` and the billed `cost` are parsed. A BYOK OpenAI-compatible endpoint uses
`OpenAIChatModel` with `CompatProvider`, which turns our endpoint compat flags into the profile
and settings options pydantic-ai already has.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI, DefaultAsyncHttpxClient, omit
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from bakeoff.shared.contract import ModelConfig

EventHooks = dict[str, list[Callable[[Any], Awaitable[None]]]]


class CompatProvider(OpenAIProvider):
    """An OpenAI-compatible (BYOK) endpoint whose profile follows our compat flags.

    pydantic-ai picks the profile from the model name, and a custom base URL does not change
    that. A non-OpenAI name therefore gets a profile without thinking support and the unified
    `thinking` setting is dropped silently. Endpoint quirks belong in `Provider.model_profile()`
    (the library's own rule), so this provider layers the flags over the name-based profile.
    """

    def __init__(self, *, openai_client: AsyncOpenAI, profile: OpenAIModelProfile) -> None:
        super().__init__(openai_client=openai_client)
        self._compat_profile = profile

    def model_profile(self, model_name: str) -> ModelProfile | None:  # type: ignore[override]
        return merge_profile(super().model_profile(model_name), self._compat_profile)


def build_model(cfg: ModelConfig, event_hooks: EventHooks) -> OpenAIChatModel:
    """The model for `cfg`. Retries are the OpenAI SDK's own (`max_retries`); the hooks only observe."""
    client = AsyncOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        max_retries=cfg.max_retries,
        timeout=cfg.timeout_s,
        # The SDK's default client class (httpx or httpx2, whichever this SDK version uses).
        http_client=DefaultAsyncHttpxClient(event_hooks=event_hooks),
    )
    base: dict[str, Any] = {"max_tokens": cfg.max_tokens}
    if cfg.temperature is not None:
        base["temperature"] = cfg.temperature
    if cfg.kind == "openrouter":
        return OpenRouterModel(
            cfg.model,
            provider=OpenRouterProvider(openai_client=client),
            settings=OpenRouterModelSettings(**base, **_openrouter_settings(cfg)),
        )
    profile, settings = _compat(cfg)
    return OpenAIChatModel(
        cfg.model,
        provider=CompatProvider(openai_client=client, profile=profile),
        settings=OpenAIChatModelSettings(**base, **settings),
    )


def _openrouter_settings(cfg: ModelConfig) -> OpenRouterModelSettings:
    settings = OpenRouterModelSettings()
    if cfg.reasoning:
        settings["openrouter_reasoning"] = cfg.reasoning  # type: ignore[typeddict-item]
    # No library setting exists for these two. `openrouter_cache_messages` would put
    # `cache_control` on whichever message is last, changing that message's shape from one
    # request to the next; OpenRouter's top-level `cache_control` caches the prefix instead.
    extra_body: dict[str, Any] = {}
    if cfg.session_id:
        extra_body["session_id"] = cfg.session_id
    if cfg.model.startswith("anthropic/"):
        extra_body["cache_control"] = {"type": "ephemeral"}
    if extra_body:
        settings["extra_body"] = extra_body
    return settings


def _compat(cfg: ModelConfig) -> tuple[OpenAIModelProfile, OpenAIChatModelSettings]:
    """Map the compat flags (DESIGN.md) onto pydantic-ai; unknown flags are ignored.

    Absent flags keep the library's OpenAI defaults: `max_completion_tokens`, the unified
    `thinking` setting sent as `reasoning_effort`, `stream_options.include_usage`, system role.
    """
    flags = cfg.compat
    reasoning_param = flags.get("reasoning_param", "reasoning_effort")
    profile = OpenAIModelProfile()
    settings = OpenAIChatModelSettings()
    extra_body: dict[str, Any] = {}
    if "max_tokens_field" in flags:
        profile["openai_chat_supports_max_completion_tokens"] = (
            flags["max_tokens_field"] == "max_completion_tokens"
        )
    if flags.get("developer_role"):
        profile["openai_system_prompt_role"] = "developer"
    if cfg.reasoning and reasoning_param == "reasoning_effort":
        profile["supports_thinking"] = True
        settings["thinking"] = cfg.reasoning.get("effort", True)
    elif cfg.reasoning and reasoning_param == "openrouter":
        extra_body["reasoning"] = cfg.reasoning  # OpenAIChatModel has no setting for this shape
    if flags.get("stream_usage") is False:
        # pydantic-ai always sends `stream_options` when streaming; the OpenAI SDK drops any
        # `extra_body` key whose value is its `omit` sentinel.
        extra_body["stream_options"] = omit
    if extra_body:
        settings["extra_body"] = extra_body
    return profile, settings
