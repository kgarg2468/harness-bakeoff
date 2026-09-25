# Checklist: pydantic_version uses pydantic-ai the recommended way

This checklist is for Dylan (or anyone who prefers pydantic-ai) to review. The goal: nobody can
say version A lost because it used the library wrong. Each item links to the docs it follows.
If you'd do something differently, edit this file in a PR, and A will be changed to match.

- [ ] **Model**: `OpenRouterModel` + `OpenRouterProvider(openai_client=AsyncOpenAI(base_url=...))` for
      OpenRouter, so `reasoning_details` are parsed. `OpenAIChatModel` for BYOK. ([models/openrouter](https://ai.pydantic.dev/models/openrouter/))
- [ ] **BYOK thinking**: a provider subclass that overrides `model_profile()`, because a custom
      base URL doesn't change profile selection and thinking is otherwise dropped for non-OpenAI
      model names. ([models/openai, "Custom providers for gateways"](https://ai.pydantic.dev/models/openai/))
- [ ] **Loop API**: `Agent.iter()` (or `run_stream_events()`), never `run_stream()`, which stops at the
      first final output. ([agents](https://ai.pydantic.dev/agents/))
- [ ] **Tools**: our shared JSON schemas through `Tool.from_schema` → `FunctionToolset`.
- [ ] **Approvals**: `ApprovalRequiredToolset(approval_required_func=...)` backed by the shared
      permission rules. `output_type=[str, DeferredToolRequests]`. Resume with `message_history` +
      `DeferredToolResults` (`ToolDenied(message)` for a denial). ([deferred-tools](https://ai.pydantic.dev/deferred-tools/))
- [ ] **History**: native `ModelMessagesTypeAdapter` JSON, persisted after every model response
      and tool batch (node boundaries in `iter()`), so a crash loses nothing.
- [ ] **Cancel**: `CancellationToken`.
- [ ] **Limits**: `UsageLimits(request_limit=max_steps)`, with `UsageLimitExceeded` mapped to `max_steps`.
- [ ] **Cost**: OpenRouter's billed cost from `ModelResponse.provider_details["cost"]`. `RunUsage` cost is
      reported only as an estimate.
- [ ] **Retries**: the OpenAI SDK's built-in retries (`max_retries`). The `[retries]` tenacity
      transport is the alternative; reviewer's choice.
- [ ] **Bad tool arguments**: pydantic-ai's own idiom (`ModelRetry` / retry prompt), with `retries` set
      high enough that one bad call does not end the run. Reviewer's choice of idiom.
- [ ] **Quiet**: `PYDANTIC_AI_NO_BANNER=1`, instrumentation off (nothing may write to stdout/stderr
      inside the engine).
- [ ] **Versions**: passes on 2.31.1 (fits the engine today) and on the latest release.

Reviewed by: _nobody yet_
