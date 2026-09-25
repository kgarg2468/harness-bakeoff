# Checklist: pydantic_version uses pydantic-ai the recommended way

This checklist is for Dylan (or anyone who prefers pydantic-ai) to review. The goal: nobody can
say version A lost because it used the library wrong. Each item links to the docs it follows.
If you'd do something differently, edit this file in a PR, and A will be changed to match.

- [x] **Model**: `OpenRouterModel` + `OpenRouterProvider(openai_client=AsyncOpenAI(base_url=...))` for
      OpenRouter, so `reasoning_details` are parsed. `OpenAIChatModel` for BYOK. ([models/openrouter](https://ai.pydantic.dev/models/openrouter/))
      *Done in `model.py`. Reasoning goes through `openrouter_reasoning`; `session_id` and top-level
      `cache_control` have no library setting, so they go in `extra_body`. The model and the Agent
      are built once (per endpoint, per tool set) and shared by every thread; the per-thread
      `session_id` is a run setting (`model_settings=`), whose `extra_body` repeats the model's own
      keys because settings merge shallowly.*
- [x] **BYOK thinking**: the model's documented `profile=` argument, merged over the provider's
      name-based profile, because a custom base URL doesn't change profile selection and thinking
      is otherwise dropped for non-OpenAI model names. ([models/openai, "Custom providers for gateways"](https://ai.pydantic.dev/models/openai/))
      *`_compat` turns the compat flags into that profile (`supports_thinking`,
      `openai_chat_supports_max_completion_tokens`, `openai_system_prompt_role`); thinking uses the
      unified `thinking` setting. `stream_usage: false` has no profile flag, so `extra_body` drops
      `stream_options` with the SDK's `omit`.*
- [x] **Loop API**: `Agent.iter()` (or `run_stream_events()`), never `run_stream()`, which stops at the
      first final output. ([agents](https://ai.pydantic.dev/agents/))
      *`Agent.iter()` in its own task, `node.stream()` for request nodes. `tool_call.ready` comes from
      the model stream's `PartEndEvent`, so a resumed call is not announced twice.*
- [x] **Tools**: our shared JSON schemas through `Tool.from_schema` → `FunctionToolset`.
      *With `strict=False`: the default lets the OpenAI schema transformer close every object, which
      would make free-form args such as `validate_pipeline.pipeline` "must be empty". A result that
      is not ok maps onto the library's three outcomes by `ToolResult.error`: `invalid_args` →
      `ModelRetry` (retry prompt), `denied` → `ToolDenied` returned from the tool (outcome
      `denied`, text verbatim, as for a user's denial), `failed` → `ToolFailed` (outcome `failed`;
      the library sends it as `{"error": ...}` and uses no retry budget).*
- [x] **Approvals**: `ApprovalRequiredToolset(approval_required_func=...)` backed by the shared
      permission rules. `output_type=[str, DeferredToolRequests]`. Resume with `message_history` +
      `DeferredToolResults` (`ToolDenied(message)` for a denial). ([deferred-tools](https://ai.pydantic.dev/deferred-tools/))
      *Crash resume uses the same `DeferredToolResults`, answering open calls with `check()` again;
      the library requires an answer for every open call, so any "ask" pauses before the rest run.
      A new user message after a pause nobody answered: the open calls get the library's own
      "interrupted" result, saved as items first (the library would send the same result on the
      wire without it ever reaching the log).*
- [x] **History**: native `ModelMessagesTypeAdapter` JSON, persisted after every model response
      and tool batch (node boundaries in `iter()`), so a crash loses nothing.
      *Every item carries the native of exactly what it shows: a response, or one part of a request
      (one tool result). The library merges consecutive requests before it sends them, so the wire
      does not change, and a crash between two result items of one batch keeps the first. Tool
      results are persisted from the next request node, and the request is sent only after the
      runner has handled them. A history that ends with the user's request is passed as-is, with
      no `user_prompt`.*
- [x] **Cancel**: `CancellationToken`.
      *The token cancels the task that drives the run, so the run gets its own task. Calls a cancel
      leaves open get the same `interrupted` results the library would synthesize, persisted now.
      The library replays an interrupted response as it was, so a `ProcessHistory` capability drops
      its unsigned thinking (the signature only arrives at the end of a thinking block, and
      endpoints that check signatures, like Anthropic, reject it); the item's native keeps it.*
- [x] **Limits**: `UsageLimits(request_limit=max_steps)`, with `UsageLimitExceeded` mapped to `max_steps`.
      *`max_cost_usd` uses `cost_limit` and maps to `budget` (checked first, so crossing it on the
      last step is `budget`); the response that crosses it is dropped from history by the library.
      The request limit is only checked before the next request, so `Hooks(before_tool_execute=...)`
      skips (`SkipToolExecution`) the calls of the response that reaches the cap: their results
      could never be sent. Each still gets a result.*
- [x] **Cost**: OpenRouter's billed cost from `ModelResponse.provider_details["cost"]`. `RunUsage` cost is
      reported only as an estimate.
      *An `after_model_request` hook sets the billed cost as the response cost, so `RunUsage` and
      `cost_limit` count what was charged, and queues the step's `usage` event; it sees every
      response, so the one that crosses `cost_limit` (billed, then dropped) is reported too. Without
      a billed cost: `estimate` (genai-prices) when the provider reported usage, else `none`. The
      library drops a billed cost of exactly 0 (`if cost := usage.cost`), which then shows as an
      estimate. Open question: fakeprov's S12b expects `none` for a BYOK `gpt-5.4-mini` endpoint,
      where A reports genai-prices' OpenAI list price, labelled `estimate`.*
- [x] **Retries**: the OpenAI SDK's built-in retries (`max_retries`). The `[retries]` tenacity
      transport is the alternative; reviewer's choice.
      *`retry` events come from httpx event hooks on the SDK's own client class, as each retry
      starts (with the measured wait); the SDK numbers attempts only in its
      `x-stainless-retry-count` header. A stream that fails midway is retried by code A had to add
      (below).*
- [x] **Bad tool arguments**: pydantic-ai's own idiom (`ModelRetry` / retry prompt), with `retries` set
      high enough that one bad call does not end the run. Reviewer's choice of idiom.
      *`retries=max_steps`. Only `invalid_args` results use it. Arguments that are not valid JSON
      never reach ToolHost: the library validates them before it calls the tool and answers with
      its retry prompt (a pydantic error dump), so such a call has no `check()`, `run()` or
      `tool.start`, and the model sees different text than with B.*
- [x] **Quiet**: `PYDANTIC_AI_NO_BANNER=1`, instrumentation off (nothing may write to stdout/stderr
      inside the engine).
      *`pydantic_ai.BANNER_ENABLED = False` (the in-code switch), instrumentation is off by default,
      and two library warnings are filtered: "dropped temperature for a reasoning model" and
      `CostNotFoundWarning` (a cost limit with no known price; the usage events say `none`).*
- [x] **Versions**: passes on 2.31.1 (fits the engine today) and on the latest release.
      *2.31.1 + openai 2.54.0 and 2.50.0 + openai 3.19.2 (httpx2), same code.*

## Code A had to add

The library has no mechanism for these, so A has its own code (counted like everything else):

- **Retrying a stream that fails midway** (`_drive`, `_retry_reason`). The SDK's and tenacity's
  retries act before the body is read, and a request node's stream cannot be restarted. If a
  stream fails with a provider error (429/5xx, a dropped connection, OpenRouter's error chunk),
  the step runs again from the saved history with the SDK's backoff schedule, up to
  `max_retries`; the partial response is dropped, a `retry` event is emitted, and `usage=` carries
  the run's usage so the retry is not a new step. No tool runs before a response is complete, so
  this is safe.
- **Saving the calls of an unanswered pause** (`_run`, 3 lines), and **waiting for the runner
  before each request** (`await out.join()`, 1 line; the same wait orders `tool.start`).
- **Importing the SDK's chat resources at module import** (`model.py`, 1 line): the first
  `OpenAIChatModel` loads them lazily, which blocked the event loop for about 0.3 s inside the
  first turn on openai 2.x.

## Library bugs and behaviors (worked around or recorded)

- **Worked around**: OpenRouter's documented mid-stream error chunk has a string `code`
  (`"server_error"`), but pydantic-ai's `_OpenRouterError` declares `code: int`, so the error
  surfaces as a pydantic `ValidationError` instead of `ModelHTTPError` (2.31.1 and 2.50.0). A
  recognizes it by the model's title, the only private name it refers to.
- **Recorded**: a billed cost of exactly 0 is dropped (above).
- **Recorded**: once any thinking is in a request, `OpenRouterModel` adds `"reasoning": ""` to
  tool-calling assistant messages that have no thinking field, which changes the bytes of messages
  already sent (byte prefix and prompt cache; the semantic prefix holds). `Item.message` does not
  carry it, since it depends on the rest of the request.
- **Recorded**: an interrupted tool call whose arguments were cut off is replayed as
  `{"INVALID_JSON": "..."}`, not as the streamed text.
- **Recorded**: the retry events read the SDK's `x-stainless-retry-count` request header; the
  tenacity transport would give native events but needs the `[retries]` extra.

Reviewed by: _nobody yet_
