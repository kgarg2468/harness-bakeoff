# Checklist: pydantic_version uses pydantic-ai the recommended way

This checklist is for Dylan (or anyone who prefers pydantic-ai) to review. The goal: nobody can
say version A lost because it used the library wrong. Each item links to the docs it follows.
If you'd do something differently, edit this file in a PR, and A will be changed to match.

- [x] **Model**: `OpenRouterModel` + `OpenRouterProvider(openai_client=AsyncOpenAI(base_url=...))` for
      OpenRouter, so `reasoning_details` are parsed. `OpenAIChatModel` for BYOK. ([models/openrouter](https://ai.pydantic.dev/models/openrouter/))
      *Done in `model.py`. Reasoning goes through `openrouter_reasoning`; `session_id` and top-level
      `cache_control` have no library setting, so they go in `extra_body`.*
- [x] **BYOK thinking**: a provider subclass that overrides `model_profile()`, because a custom
      base URL doesn't change profile selection and thinking is otherwise dropped for non-OpenAI
      model names. ([models/openai, "Custom providers for gateways"](https://ai.pydantic.dev/models/openai/))
      *`CompatProvider` layers the compat flags over the name-based profile (`supports_thinking`,
      `openai_chat_supports_max_completion_tokens`, `openai_system_prompt_role`); thinking uses the
      unified `thinking` setting. `stream_usage: false` has no profile flag, so `extra_body` drops
      `stream_options` with the SDK's `omit`.*
- [x] **Loop API**: `Agent.iter()` (or `run_stream_events()`), never `run_stream()`, which stops at the
      first final output. ([agents](https://ai.pydantic.dev/agents/))
      *`Agent.iter()` in its own task, `node.stream()` for request nodes. `tool_call.ready` comes from
      the model stream's `PartEndEvent`, so a resumed call is not announced twice.*
- [x] **Tools**: our shared JSON schemas through `Tool.from_schema` → `FunctionToolset`.
      *With `strict=False`: the default lets the OpenAI schema transformer close every object, which
      would make free-form args such as `validate_pipeline.pipeline` "must be empty".*
- [x] **Approvals**: `ApprovalRequiredToolset(approval_required_func=...)` backed by the shared
      permission rules. `output_type=[str, DeferredToolRequests]`. Resume with `message_history` +
      `DeferredToolResults` (`ToolDenied(message)` for a denial). ([deferred-tools](https://ai.pydantic.dev/deferred-tools/))
      *Crash resume uses the same `DeferredToolResults`, answering open calls with `check()` again;
      the library requires an answer for every open call, so any "ask" pauses before the rest run.*
- [x] **History**: native `ModelMessagesTypeAdapter` JSON, persisted after every model response
      and tool batch (node boundaries in `iter()`), so a crash loses nothing.
      *Tool results are persisted from the next request node, before that request is sent. A
      history that ends with the user's request is passed as-is, with no `user_prompt`.*
- [x] **Cancel**: `CancellationToken`.
      *The token cancels the task that drives the run, so the run gets its own task. Calls a cancel
      leaves open get the same `interrupted` results the library would synthesize, persisted now.*
- [x] **Limits**: `UsageLimits(request_limit=max_steps)`, with `UsageLimitExceeded` mapped to `max_steps`.
      *`max_cost_usd` uses `cost_limit` and maps to `budget`; the response that crosses it is dropped
      from history by the library.*
- [x] **Cost**: OpenRouter's billed cost from `ModelResponse.provider_details["cost"]`. `RunUsage` cost is
      reported only as an estimate.
      *An `after_model_request` hook also sets the billed cost as the response cost, so `RunUsage` and
      `cost_limit` count what was charged. Without it: `estimate` (genai-prices) or `none`.*
- [x] **Retries**: the OpenAI SDK's built-in retries (`max_retries`). The `[retries]` tenacity
      transport is the alternative; reviewer's choice.
      *`retry` events come from httpx event hooks on the SDK's own client class, as each retry
      starts (with the measured wait); the SDK numbers attempts only in its
      `x-stainless-retry-count` header. A stream that fails midway is not retried.*
- [x] **Bad tool arguments**: pydantic-ai's own idiom (`ModelRetry` / retry prompt), with `retries` set
      high enough that one bad call does not end the run. Reviewer's choice of idiom.
      *`retries=max_steps`. `ToolResult` has no failure kind, so every `ok=False` result is a `ModelRetry`.*
- [x] **Quiet**: `PYDANTIC_AI_NO_BANNER=1`, instrumentation off (nothing may write to stdout/stderr
      inside the engine).
      *`pydantic_ai.BANNER_ENABLED = False` (the in-code switch), instrumentation is off by default,
      and the library's "dropped temperature for a reasoning model" warning is filtered.*
- [x] **Versions**: passes on 2.31.1 (fits the engine today) and on the latest release.
      *2.31.1 + openai 2.54.0 and 2.50.0 + openai 3.19.2 (httpx2), same code.*

Reviewed by: _nobody yet_
