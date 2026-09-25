"""Build the pydantic-ai model for an endpoint.

OpenRouter uses `OpenRouterModel` + `OpenRouterProvider`, as the docs recommend, so
`reasoning_details` and the billed `cost` are parsed. A BYOK OpenAI-compatible endpoint uses
`OpenAIChatModel` with a `profile=` built from our endpoint compat flags. OpenAI's Responses API
uses `OpenAIResponsesModel` with `OpenAIResponsesModelSettings`. A model is built once per
endpoint and shared by every thread; the per-thread part (`session_id`) is a run setting.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, get_args

from openai import AsyncOpenAI, DefaultAsyncHttpxClient, omit

# The first OpenAIChatModel (OpenAIResponsesModel) loads the SDK's chat (responses) resources,
# which the SDK imports lazily: that blocks the event loop for about 0.25 s (0.09 s) on openai 2.x
# inside the first turn. Import them here.
from openai.resources.chat import AsyncChat  # noqa: F401
from openai.resources.responses import AsyncResponses  # noqa: F401
from pydantic_ai.models import Model
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings, ThinkingEffort

from bakeoff.shared.contract import ModelConfig

EventHooks = dict[str, list[Callable[[Any], Awaitable[None]]]]


def build_model(cfg: ModelConfig, event_hooks: EventHooks) -> Model:
    """The model for `cfg`'s endpoint (`cfg.session_id` is ignored: see `run_settings`).
    Retries are the OpenAI SDK's own (`max_retries`); the hooks only observe."""
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
    if cfg.kind == "openai_responses":
        # The harness keeps the history (contract rule 2), so OpenAI stores nothing and a
        # reasoning item goes back as its encrypted content. The library asks for that content
        # (`include`) and replays it, with the item ids, for a reasoning profile.
        settings = OpenAIResponsesModelSettings(**base, openai_store=False, **_effort(cfg))
        if summary := (cfg.reasoning or {}).get("summary"):
            settings["openai_reasoning_summary"] = summary  # the API sends none unless asked
        reasons = cfg.compat.get("reasoning_param") != "none"
        return OpenAIResponsesModel(
            cfg.model,
            provider=OpenAIProvider(openai_client=client),
            profile=_REASONING_PROFILE if reasons else None,
            settings=settings,
        )
    # pydantic-ai picks the profile from the model name, and a custom base URL does not change
    # that, so a non-OpenAI name loses thinking support. The documented `profile=` argument is
    # merged over the name-based profile.
    profile, settings = _compat(cfg)
    return OpenAIChatModel(
        cfg.model,
        provider=OpenAIProvider(openai_client=client),
        profile=profile,
        settings=OpenAIChatModelSettings(**base, **settings),
    )


def run_settings(model: Model, cfg: ModelConfig) -> ModelSettings | None:
    """The per-thread settings: OpenRouter's sticky routing `session_id`. Run settings replace the
    model's `extra_body` as a whole (settings merge shallowly), so it carries the model's keys."""
    if cfg.kind != "openrouter" or not cfg.session_id:
        return None
    extra_body = (model.settings or {}).get("extra_body") or {}
    return ModelSettings(extra_body={**extra_body, "session_id": cfg.session_id})


def _openrouter_settings(cfg: ModelConfig) -> OpenRouterModelSettings:
    settings = OpenRouterModelSettings()
    if cfg.reasoning:
        settings["openrouter_reasoning"] = cfg.reasoning  # type: ignore[typeddict-item]
    # No library setting exists for top-level `cache_control`. `openrouter_cache_messages` would
    # put `cache_control` on whichever message is last, changing that message's shape from one
    # request to the next; OpenRouter's top-level `cache_control` caches the prefix instead.
    if cfg.model.startswith("anthropic/"):
        settings["extra_body"] = {"cache_control": {"type": "ephemeral"}}
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
        settings.update(_effort(cfg))
        if "thinking" in settings:
            profile["supports_thinking"] = True
    elif cfg.reasoning and reasoning_param == "openrouter":
        extra_body["reasoning"] = cfg.reasoning  # OpenAIChatModel has no setting for this shape
    if flags.get("stream_usage") is False:
        # pydantic-ai always sends `stream_options` when streaming; the OpenAI SDK drops any
        # `extra_body` key whose value is its `omit` sentinel.
        extra_body["stream_options"] = omit
    if extra_body:
        settings["extra_body"] = extra_body
    return profile, settings


# pydantic-ai picks the profile from the model name, and 2.31.1 predates gpt-6-luna: it would take
# the model for one that does not reason, so it would drop the effort and replay no reasoning (and
# no `phase`). The model reasons even with no `reasoning` config (at its default effort), so a
# Responses model is taken to reason unless compat `reasoning_param` is "none". With these flags
# 2.31.1 asks for, replays and configures reasoning as 2.50.0 does for gpt-6-luna.
_REASONING_PROFILE = OpenAIModelProfile(
    supports_thinking=True,
    openai_supports_reasoning=True,
    openai_supports_encrypted_reasoning_content=True,
    openai_supports_phase=True,
    openai_responses_supports_reasoning_context=True,
)


def _effort(cfg: ModelConfig) -> OpenAIChatModelSettings:
    """The reasoning effort: the unified `thinking` setting where it has the level. The values it
    has no level for ("none" turns reasoning off, which some models need before they accept
    tools; "max") go straight through the documented OpenAI-specific setting."""
    if not cfg.reasoning:
        return OpenAIChatModelSettings()
    effort = cfg.reasoning.get("effort", True)
    if isinstance(effort, bool) or effort in get_args(ThinkingEffort):
        return OpenAIChatModelSettings(thinking=effort)
    return OpenAIChatModelSettings(openai_reasoning_effort=effort)
