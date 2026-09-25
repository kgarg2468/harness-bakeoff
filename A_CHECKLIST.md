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
- [x] **Responses API** (`kind: openai_responses`): `OpenAIResponsesModel` +
      `OpenAIProvider(openai_client=AsyncOpenAI(base_url=...))` with `OpenAIResponsesModelSettings`.
      ([models/openai, "OpenAI Responses API"](https://ai.pydantic.dev/models/openai/))
      *Done in `model.py`. The harness keeps the history, so `openai_store=False`; the effort is the
      unified `thinking` setting (`openai_reasoning_effort` for values it has no level for, as for
      BYOK), and `reasoning.summary` is `openai_reasoning_summary` (the API streams no summary
      unless asked; `live --api responses --reasoning EFFORT` asks for `auto`). What goes back is
      the library's own replay: for a reasoning model it asks for `include:
      ["reasoning.encrypted_content"]` and, with `openai_send_reasoning_ids` at its default (on
      for reasoning models), replays each reasoning item with its id and encrypted content,
      messages with their `phase`, and calls with their ids. 2.31.1 predates `gpt-6-luna`:
      its name-based profile takes it for a model that does not reason, which would drop the
      effort, the encrypted reasoning and the `phase`. The model reasons even with no `reasoning`
      config, so every Responses config passes the documented `profile=` (`supports_thinking`,
      `openai_supports_reasoning`, `openai_supports_encrypted_reasoning_content`,
      `openai_supports_phase`, `openai_responses_supports_reasoning_context`), unless compat
      `reasoning_param` is `"none"` (a model that does not reason). 2.50.0 knows the model and
      sends the same request, `reasoning.context: "all_turns"` included. `Item.message` is the
      chat-shaped view without the reasoning items (it has no field for them); the native
      replays them.*
- [x] **Loop API**: `Agent.iter()` (or `run_stream_events()`), never `run_stream()`, which stops at the
      first final output. ([agents](https://ai.pydantic.dev/agents/))
      *`Agent.iter()` in its own task, `node.stream()` for request nodes and for tool nodes (its
      `FunctionToolResultEvent`s, see History). `tool_call.ready` comes from the model stream's
      `PartEndEvent`, so a resumed call is not announced twice.*
- [x] **Tools**: our shared JSON schemas through `Tool.from_schema` → `FunctionToolset`.
      *With `strict=False`: the default lets the OpenAI schema transformer close every object, which
      would make free-form args such as `validate_pipeline.pipeline` "must be empty". A result that
      is not ok maps onto the library's three outcomes by `ToolResult.error`: `invalid_args` →
      `ModelRetry` (retry prompt), `denied` → `ToolDenied` returned from the tool (outcome
      `denied`, text verbatim, as for a user's denial), `failed` → `ToolFailed` (outcome `failed`;
      the library sends it as `{"error": ...}` and uses no retry budget). A tool that is not
      `read_only` is `sequential=True`: it runs alone, after the calls before it and before those
      after it (B orders its calls the same way).*
- [x] **Approvals**: `ApprovalRequiredToolset(approval_required_func=...)` backed by the shared
      permission rules. `output_type=[str, DeferredToolRequests]`. Resume with `message_history` +
      `DeferredToolResults` (`ToolDenied(message)` for a denial). ([deferred-tools](https://ai.pydantic.dev/deferred-tools/))
      *A crash resume, or an approval that answers only some calls, uses the same
      `DeferredToolResults`. The library needs an answer for every open call, so a call the user
      did not answer is passed as approved and re-checked in `Hooks(before_tool_execute=...)`,
      which the docs name as the hook to defer from: `check()` == "ask" raises `ApprovalRequired`
      again. So the decided calls run now and only the others pause (before, a partial answer was
      lost and the thread could never finish). Open calls that must never run get the library's
      own "interrupted" result, saved as items: their response was cut short by a cancel, or the
      user sent a new message instead of answering. This is decided from the history, so a crash
      right after a cancel cannot run the cancelled call.*
- [x] **History**: native `ModelMessagesTypeAdapter` JSON, persisted after every model response
      and tool batch (node boundaries in `iter()`), so a crash loses nothing.
      *Every item carries the native of exactly what it shows: a response, or one part of a request
      (one tool result). The library merges consecutive requests before it sends them, so the wire
      does not change, and a crash between two result items of one batch keeps the first. A
      sequential tool's result is saved as soon as it finishes (the tool node's
      `FunctionToolResultEvent`, with the results of the calls before it), so a crash never runs a
      finished write again; read-only results are saved from the next request node, in call order,
      and the request is sent only after the runner has handled them. A history that ends with the
      user's request is passed as-is, with no `user_prompt`. A crash resume whose history already
      ends the turn (the final answer, or a response cut short: by a cancel, or by a failure, which
      A saves with finish reason `error` so that the resume ends with an error as the run did) sends
      nothing; a response with neither text nor calls is no answer (the library asks again), unless
      the library raised on it (out of output tokens, or blank and filtered: its saved finish reason
      says so), and then the resume ends with an error as well, and sends nothing. Either way it
      ends with `budget` instead if the turn's cost crossed `max_cost_usd`: the library checks the
      cost as it adds a response, before it reads it. A complete response with nothing to show (only
      a Responses reasoning item, which the library replays, or an empty one) still gets an item,
      with no content: without it the requests on either side would merge on rebuild, and the merge
      puts the retry prompt before the user's message.*
- [x] **Cancel**: `CancellationToken`.
      *The token cancels the task that drives the run, so the run gets its own task. Calls a cancel
      leaves open get the same `interrupted` results the library would synthesize, persisted now
      (or by the crash resume, if a crash came first). The library replays an interrupted
      response as it was, so a `ProcessHistory` capability drops its unsigned thinking (the
      signature only arrives at the end of a thinking block, and endpoints that check signatures,
      like Anthropic, reject it); the item's native keeps it. On the Responses API the signature is
      a reasoning item's encrypted content, which is on its first part only: a reasoning item
      none of whose parts is signed is dropped whole, from any response (below).*
- [x] **Limits**: `UsageLimits(request_limit=max_steps)`, with `UsageLimitExceeded` mapped to `max_steps`.
      *`max_cost_usd` uses `cost_limit` and maps to `budget` (checked first, so crossing it on the
      last step is `budget`). The library drops the response that crosses it from history, though
      it was streamed and billed; A keeps it and closes its calls, so the next turn sees the
      answer. The request limit is only checked before the next request, so
      `Hooks(before_tool_execute=...)` skips (`SkipToolExecution`) the calls of the response that
      reaches the cap: their results could never be sent. Each still gets a result. A resume
      continues the turn: `usage=RunUsage(...)` from the responses since the user's message (the
      same `usage=` the stream retry uses), so the step count and the budget carry over, as in B.*
- [x] **Cost**: OpenRouter's billed cost from `ModelResponse.provider_details["cost"]`. `RunUsage` cost is
      reported only as an estimate.
      *An `after_model_request` hook sets the billed cost as the response cost, so `RunUsage` and
      `cost_limit` count what was charged, and queues the step's `usage` event; it sees every
      response, including the one that crosses `cost_limit`. Without a billed cost: `estimate`
      (genai-prices) on OpenRouter when it reported usage, else `none`. genai-prices knows
      OpenRouter's prices, not what a BYOK endpoint charges, so BYOK is `none`, never guessed
      (fakeprov S12b). A billed cost of exactly 0 is `provider` 0 (library bug, below).*
- [x] **Retries**: the OpenAI SDK's built-in retries (`max_retries`). The `[retries]` tenacity
      transport is the alternative; reviewer's choice.
      *`retry` events come from httpx event hooks on the SDK's own client class, as each retry
      starts (with the measured wait); the SDK numbers attempts only in its
      `x-stainless-retry-count` header. A stream that fails midway (an error event, or a dropped
      connection) is retried by code A had to add (below).*
- [x] **Bad tool arguments**: pydantic-ai's own idiom (`ModelRetry` / retry prompt), with `retries` set
      high enough that one bad call does not end the run. Reviewer's choice of idiom.
      *`retries=max_steps`. Only `invalid_args` results use it. Arguments the library cannot parse
      (not a JSON object) go on like any others through `Hooks(tool_validate_error=...)`, which
      returns empty validated args: the call is checked, ToolHost gets the raw text and answers
      with its own error (rule 5), and the model sees that text in the retry prompt instead of a
      pydantic error dump.*
- [x] **Quiet**: `PYDANTIC_AI_NO_BANNER=1`, instrumentation off (nothing may write to stdout/stderr
      inside the engine).
      *`pydantic_ai.BANNER_ENABLED = False` (the in-code switch), instrumentation is off by default,
      and three library warnings are filtered: "dropped temperature for a reasoning model",
      `CostNotFoundWarning` (a cost limit with no known price; the usage events say `none`) and
      "Handling of this event type is not yet implemented" (a Responses event with no handler,
      such as `error`: below).*
- [x] **Versions**: passes on 2.31.1 (fits the engine today) and on the latest release.
      *2.31.1 + openai 2.54.0 and 2.50.0 + openai 3.19.2 (httpx2; the latest on 2026-09-25), same
      code, S01-S15 and R01-R05. The Responses request carries `reasoning.context: "all_turns"`
      on both: 2.50.0's default for the models it knows support it, and the profile's flag on
      2.31.1. They differ on a response that ran out of `max_output_tokens` (below).*

## Code A had to add

The library has no mechanism for these, so A has its own code (counted like everything else):

- **Retrying a stream that fails midway** (`_drive`, `_retry_reason`). The SDK's and tenacity's
  retries act before the body is read, and a request node's stream cannot be restarted. If a
  stream fails with a provider error (429/5xx, an error event, a dropped connection), the step
  runs again from the saved history with the SDK's backoff schedule, up to `max_retries`; the
  partial response is dropped, a `retry` event is emitted, and `usage=` carries the run's usage
  so the retry is not a new step. No tool runs before a response is complete, so this is safe.
  `_retry_reason` has to recognize errors the library does not wrap (below).
- **Deciding from the history what a resume must do** (`mapping.close_abandoned`, `this_turn`,
  `spent`, `finished`: about 30 lines, and 15 in `_run`): close calls that must never run, end a
  turn whose end is already saved without a request (with the stop the run reported), and carry
  the steps and cost over. The library resumes a history as it is and starts its usage at zero.
- **Keeping the response that crosses the budget** (4 lines in `_run`).
- **Saving a sequential tool's result when it finishes** (`_Turn.tool_done`, the tool-node
  branch in `_run` and a skip in `flush`: about 20 lines); the library adds results to history
  only when the whole batch is done.
- **Waiting for the runner before each request** (`await out.join()`, 1 line; the same wait
  orders `tool.start`).
- **Retrying a Responses stream that failed** (3 lines in `_run`, `_note_events` and 4 lines in
  `_on_response`: about 20 lines, and the warning filter). The library ends the stream as if it
  were done after an `error` event (it has no handler for it) or a `response.failed`, so the
  partial answer would be the final one. A notes the name of each SSE event in the bytes the
  OpenAI SDK parses (an httpx response hook): a stream whose last event is not
  `response.completed` (or `.incomplete`, which the library handles) is raised as a
  `ModelAPIError`, and the retry above takes over. Usage is no sign: a `response.failed` may
  carry it.
- **The chat-shaped view of a Responses response** (2 lines in `mapping._assistant`, and a
  `responses_api` flag from `_Turn` through `to_openai`): `Item.message` leaves the reasoning
  items out; without this they would look like BYOK thinking fields.
- **An item for a response with nothing to show** (2 lines in `to_openai`, 1 in `finished`): a
  reasoning-only or empty response is on the history the library sends from, so it is saved too,
  or the rebuilt history would merge the requests around it (History, above).
- **Dropping a Responses reasoning item that has no encrypted content** (5 lines in
  `mapping.replayable`, and a 2-line `_replayable` that passes the `responses_api` flag).
- **Importing the SDK's chat and responses resources at module import** (`model.py`, 2 lines):
  the first `OpenAIChatModel` (`OpenAIResponsesModel`) loads them lazily, which blocked the event
  loop for about 0.3 s (0.09 s) inside the first turn on openai 2.x. What remains of the first
  turn's setup (about 40 ms, 16 ms for later endpoints) is building the model: the OpenAI client
  and its SSL context.

## Library bugs and behaviors (worked around or recorded)

- **Worked around** (bug): OpenRouter's documented mid-stream error chunk has a string `code`
  (`"server_error"`), but pydantic-ai's `_OpenRouterError` declares `code: int`, so the error
  surfaces as a pydantic `ValidationError` instead of `ModelHTTPError` (2.31.1 and 2.50.0). A
  recognizes it by the model's title, the only private name it refers to. On openai 3.x a
  stream that drops on OpenRouter arrives the same way, with no error body.
- **Worked around** (bug): OpenRouter's billed cost of exactly 0 is dropped (`if cost :=
  usage.cost`), so a free step was an estimate and counted against `max_cost_usd`. A reads it as
  0 when `provider_details` has `is_byok`, which comes with OpenRouter's usage accounting.
- **Worked around**: errors the library does not wrap: on openai 2.x a connection that drops
  mid-stream raises raw `httpx.RemoteProtocolError`, and on `OpenAIChatModel` (BYOK) an
  `{"error": ...}` event mid-stream raises raw `openai.APIError`.
- **Worked around** (gap): the Responses `error` event has no handler (2.31.1 and 2.50.0). The
  library warns with the event (to stderr, unless filtered) and ends the stream as if it were
  done, so its code and message are lost: A's retry reason and error say only "the stream failed
  before response.completed". The OpenAI SDK raises only for an event with a top-level `error`
  object, not for the documented flat `{"type": "error", "code", "message"}` shape.
- **Worked around** (bug): the segment behind a streamed response knows that a Responses stream
  ended without a terminal event (state `incomplete`), but the continuation wrapper that
  `node.stream()` returns reports every finished stream as `complete`, so A reads the stream's
  last event instead (above).
- **Worked around** (gap): on 2.31.1 a `response.failed` and a `response.incomplete` come out alike:
  no finish reason (2.50.0 maps them to `error` and `length`), state `complete`, and their usage, so
  nothing in the `ModelResponse` tells them apart: A reads the event that ended the stream, the same
  on both (above). So when `max_output_tokens` runs out while the model reasons (OpenAI documents a
  reasoning-only `.incomplete` output), 2.31.1 asks again with its retry prompt ("Please return text
  or call a tool."), while 2.50.0 ends the turn with `UnexpectedModelBehavior` (token limit
  exceeded). A crash resume after that response was saved ends the same way on each (History,
  above); its error says only that the turn had already ended with an error, since the library's
  message is not saved. A failed attempt's usage is not reported (no `usage` event), even when
  `response.failed` carries it, as for any stream that fails.
- **Recorded**: the library merges consecutive requests before it sends them, with tool results
  and retry prompts first. A response it sends nothing for (a reasoning-only or empty one) still
  separates two requests in its history, so A saves it (Code A had to add, above).
- **Worked around** (bug): a response that ends (`.incomplete`) before its reasoning item's done
  event keeps a thinking part without its encrypted content, which the library replays with
  `"encrypted_content": null` (on 2.31.1 in the same run, as it asks again); with `store: false`
  the API answers 404 (fakeprov does; whether OpenAI ends a stream that way is not confirmed).
  A's `ProcessHistory` drops such an item.
- **Worked around**: a `cost_limit` drops the response that crosses it from history (above).
- **Worked around**: `DeferredToolResults` needs an answer for every open call, so there is no
  partial approval (above).
- **Recorded**: `Agent.parallel_tool_call_execution_mode("parallel_ordered_events")` would give
  results in call order, but on 2.31.1 it holds every result event until the whole batch is
  done, so A keeps the default mode and orders the early-saved results itself.
- **Recorded**: for BYOK, `cost_limit` counts genai-prices' estimate by model name (the library
  fills it in), while the usage events say `none`.
- **Recorded**: once any thinking is in a request, `OpenRouterModel` adds `"reasoning": ""` to
  tool-calling assistant messages that have no thinking field, which changes the bytes of messages
  already sent (byte prefix and prompt cache; the semantic prefix holds). `Item.message` does not
  carry it, since it depends on the rest of the request.
- **Recorded**: a tool call whose arguments are not a JSON object (cut off, or malformed) is
  replayed as `{"INVALID_JSON": "..."}`, not as the streamed text; `Item.message` shows it so.
- **Recorded**: `OpenAIChatModel` parses BYOK thinking streamed as `<think>` tags into a thinking
  part and sends it back as tags in `content`, joining text pieces with a blank line;
  `mapping._assistant` follows the same rules.
- **Recorded**: the retry events read the SDK's `x-stainless-retry-count` request header; the
  tenacity transport would give native events but needs the `[retries]` extra.

Reviewed by: _nobody yet_
