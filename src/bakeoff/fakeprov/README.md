# fakeprov: scripted OpenRouter-compatible model server

A stdlib `ThreadingHTTPServer` on `127.0.0.1` that answers streaming chat completions from
scenario scripts, so every loop sees exactly the same model behaviour and every request it
sends is recorded. Scenario pass/fail is judged from those recordings and the session log.

```python
from bakeoff.fakeprov.server import FakeProvider

with FakeProvider(wire_dir=Path("out/wire")) as provider:  # scenarios_dir defaults to ./scenarios
    base_url = provider.base_url("S01", "run1", "our")
    # -> http://127.0.0.1:<port>/s/S01/run1/our/v1
```

```bash
uv run python -m bakeoff.fakeprov --check      # validate every scenario file
uv run python -m bakeoff.fakeprov --port 8787  # serve until Ctrl-C
```

## HTTP surface

- `POST <base>/chat/completions`: streaming only (`"stream": true`, else 400). HTTP/1.1
  keep-alive; the SSE body uses `Transfer-Encoding: chunked`, one SSE event per HTTP chunk,
  ending with `data: [DONE]`. Requests without `Content-Length` get 411.
- `GET <base>/models`: `{"object": "list", "data": [{"id": <scenario model>, ...}]}`.
- **Cursors.** Each `(scenario, run, impl)` has its own cursor. Request *n* on a cursor is
  answered by exchange *n*, whatever happens to it (strict rejection, failed expect, 429, ...).
  Past the last exchange: HTTP 500 `{"error": {"message": "script exhausted"}}`.
- **Server-side rejections** (bad body, strict mode, failed expect, exhausted script, broken
  scenario file, failed recording) carry `x-should-retry: false`, which the OpenAI SDK honours.
- A client that disconnects (cancel, SIGKILL) only closes its own connection. A body cut short
  that way is neither answered nor recorded and does not move the cursor. `stop()` ends
  stalled streams and idle keep-alive connections.
- The listen backlog is 128, so every scenario and loop can connect at the same time.

## Wire recording

Every chat request is recorded under `<wire_dir>/<scenario>/<run>/<impl>/`:

- `NNN.json`: the raw request body bytes, verbatim (`NNN` = 001, 002, ... per cursor).
- `NNN.meta.json`: `{"conn_id", "path", "t_us", "status"}`, plus `"error"` when the fake
  server rejected the request itself. `conn_id` numbers TCP connections from 1 (equal ids mean
  the connection was reused); `t_us` is microseconds since `start()`, taken when the body has
  been read.

Headers are never recorded. The first `base_url()` call for a cursor (or its first request,
if the URL was built by hand) empties that directory, so a recording always belongs to one
server run, including a run that sends no request. Two servers must not share a cursor
directory. If a recording cannot be written, the request is answered with 500 `recording
failed: ...` instead. `GET /models` and requests to unknown routes are not recorded.

## Scenario files

`scenarios/<id>.json`, where `id` equals the file name. `script.load_scenario(path)` validates
a file (JSON schema plus cross-checks) and raises `ScenarioError` naming the exact spot, e.g.
`S02.json: $.exchanges[3].respond.stream[1]: expected exactly one of: text, reasoning, ...`.

| Key | Meaning |
|---|---|
| `id`, `title` | `S01` ...; one-line description |
| `system` | the thread's frozen system prompt |
| `model` | `{"kind": "openrouter" \| "openai_compat", "model", "reasoning"?, "compat"?}` → `ModelConfig` |
| `rules` | permission rules, e.g. `{"*": "allow", "write_file": "ask"}` |
| `limits` | `{"max_steps"}` → `Limits` |
| `engine` | `{"delay_ms"}` → `MockEngine(delay_ms=...)` |
| `style` | optional wire style of every exchange: `"openrouter"`, or `"openai"` (BYOK endpoints). Default: `"openai"` for `openai_compat`, else `"openrouter"` |
| `strict` | optional strict modes for every exchange (see below) |
| `driver` | the steps the scenario runner performs, in order |
| `exchanges` | the scripted model responses, in request order |
| `expect` | final assertions, checked after the driver finishes |

### Driver steps

| Step | Meaning |
|---|---|
| `{"user": str, "cancel_after_ms"?: int}` | run a turn with this user message; set `cancel` that many ms after the turn starts |
| `{"approve": {"allow"?: [ids] \| "all", "deny"?: [ids], "reason"?: str}, "new_process": bool}` | answer the pending `permission.asked` calls and resume (`Resume(kind="approval")`), in a fresh process if `new_process` |
| `{"crash_after": "<event type>", "call_id"?: str}` | run the **next** user step in a child process that SIGKILLs itself when its runner publishes the first event of this type; with `call_id`, the first one about that call (see below) |
| `{"resume": "crash"}` | resume the killed turn (`Resume(kind="crash")`) |
| `{"revert": n}` | `Runner.revert()` the thread's n-th turn (1-based, creation order) |
| `{"compact": str}` | `Runner.compact()` with this summary |

**Crash points.** The child kills itself (`os.kill(os.getpid(), signal.SIGKILL)`) inside the
runner's sink, synchronously, so the crash point is exact and never races the child's next
request. The runner publishes an `item` event only after it has saved the item (together with
the events buffered before it), and the loop cannot run again before the sink returns.
`call_id` narrows `tool_call.ready`, `permission.asked`, `tool.start` and `tool.end` to events
whose `data.call_id` matches, and `item` to the tool result whose `message.tool_call_id`
matches. So `{"crash_after": "item", "call_id": "call_S08_1"}` dies once that call's result is
durable and before the next model request. `tool.end` is too early for that: `ToolHost` emits
it inside `run()`, before the loop has the result, so a crash there loses the result and the
crash resume rightly runs the tool again.

### Exchanges

```json
{"note": "why this exchange exists", "style": "openai", "strict": {...},
 "expect": {...}, "respond": {"status": 200, "headers": {...}, "stream": [ops]}}
```

Only `respond` is required. Before answering, the server applies, in order:

1. **Strict modes** (scenario `strict`, overridden key by key by the exchange's `strict`):
   - `reject_params: [keys]`: 400 `Unrecognized request argument supplied: <key>` (OpenAI error
     shape, `param` = key) if the body has any listed top-level key.
   - `reject_unsigned_reasoning: true`: 400 if an assistant message replays a `reasoning.text`
     detail without a non-empty `signature`. Fragments with the same `index` count as one
     detail, so a signature in a metadata-only fragment signs it.
2. **`expect`** (all optional). Any mismatch answers 500 `expect failed (exchange N): ...`:
   `body_has` / `body_lacks` (top-level keys), `model` (equal), `last_role`,
   `last_content_contains` (substring; list content is joined text parts), `messages_len`,
   `tool_result_contains: {call_id: substring}` (a `role: tool` message for that call contains
   it; `""` only checks that the result exists), `body_equals: {"dotted.path": value}` (exact
   value at a path in the body), `messages_at: [{index, role?, contains?}]`
   (checks one message; a negative index counts from the end), and `min_gap_ms` (the request
   must arrive at least this long after the cursor's previous one, e.g. a retry that honours
   `retry-after`).

`respond.status` other than 200 sends a JSON error with `headers` (e.g. `{"retry-after": "1"}`)
and `body`, which defaults to OpenRouter's `{"error": {"code", "message", "metadata"}}` (or
OpenAI's shape in the `openai` style). Status 200 needs a `stream` of ops:

| Op | Sends |
|---|---|
| `{"text": str, "chunks"?: n}` | `delta.content`, split into n near-equal pieces |
| `{"reasoning": str, "chunks"?: n, "field"?: "reasoning" \| "reasoning_content"}` | reasoning text deltas |
| `{"reasoning_details": [detail, ...]}` | one chunk per detail, sent verbatim (metadata-only and unknown fields included). OpenRouter style also mirrors a `reasoning.text` fragment's text in `delta.reasoning` |
| `{"tool_calls": [{"id", "name", "arguments": str \| object}], "pieces"?: n, "interleave"?: bool}` | per call a head delta (`id`, `type`, `function.name`, `arguments: ""`), then the arguments in n pieces; `interleave` sends the calls round-robin. A string is sent verbatim; an object is JSON-encoded. Indexes count from 0 per response |
| `{"comment": str}` | an SSE comment line, e.g. `: OPENROUTER PROCESSING` |
| `{"finish": reason}` | a chunk with `finish_reason` (`stop`, `length`, `tool_calls`, `content_filter`) |
| `{"usage": {"prompt_tokens", "completion_tokens", "cached_tokens"?, "reasoning_tokens"?, "cost"?, "is_byok"?}}` | OpenRouter style: a chunk that repeats the last `finish_reason` and carries `usage` with `total_tokens`, `cost`, `is_byok`, `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens`. OpenAI style: a final `"choices": []` chunk without `cost`, sent only if the request had `stream_options.include_usage` |
| `{"sse_error": {"code"?: "server_error", "message"}}` | OpenRouter's documented mid-stream error chunk after HTTP 200: a top-level `error`, one choice with `delta: {"content": ""}` and `finish_reason: "error"`; the stream ends without `[DONE]`. The default code is the string OpenRouter documents for mid-stream errors (its pre-stream error bodies use numeric codes). S09 sends exactly that shape |
| `{"stall": true}` | stop sending but keep the socket open until the client leaves |

Every op also takes `delay_ms` (a pause before each chunk it sends). `sse_error` and `stall`
must be last. Chunks look like OpenRouter's: `{"id": "gen-<scenario>-<NNN>", "object":
"chat.completion.chunk", "created": 1758758400, "model": <scenario model>, "provider":
"FakeProvider", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "", ...},
"finish_reason", "native_finish_reason"}]}`. The `openai` style drops `provider` and
`native_finish_reason`, uses `chatcmpl-...` ids, and sends `role` only in the first delta.
Responses are deterministic: identical for every run and impl.

### Final `expect`

| Key | Passes when |
|---|---|
| `stops` (required) | the stop of each loop turn the driver runs to its end, in order: `user` (unless crashed), `approve` and `resume` steps. The runner's own `revert`/`compact` turns run no loop and have no stop, so they are not listed. The loader checks the count |
| `files` | `{path: substring}`: the file exists in the working copy and contains it; `{path: null}`: absent |
| `tool_runs` | `{call_id: n}`: `tool.start` events for that call over the whole scenario, all processes included. `ToolHost` emits one per `run()`, so a deny rule or bad arguments count too; no scenario lists such a call |
| `commits` | number of git commits the scenario's turns and reverts created |
| `requests` | number of chat requests recorded for the cursor |
| `text_contains` | the last assistant text of the last turn contains it |
| `cost_usd` | the sum of provider-reported costs over all usage events (compare with a 1e-9 tolerance) |
| `usage` | `{input_tokens, output_tokens, cached_tokens}`: the totals over all usage events |
| `cost_source` | every usage event's `cost_source` equals it: `provider` (billed cost from the provider), `estimate` (a price table) or `none` (no cost available, never guessed) |

Tool call ids are `call_<scenario>_<n>`, unique per scenario; the loader rejects references
to ids that no exchange scripts. Content checks on tool results stay loose (substrings),
because results come from the shared tools and `MockEngine`.

## The scenarios

| ID | Exercises |
|---|---|
| S01 | chat; keep-alive comments; usage chunk repeats `finish_reason` and carries `cost` |
| S02 | pipe build: bad `describe_component` args, `describe_component`, `write_file` (shape of `examples/incorrect/incorrect-data-fed-tool-pipe.pipe`), `validate_pipeline` (lane error), `edit_file` fix, validate ok, answer |
| S03 | three parallel `describe_component` calls, interleaved argument chunks, 300 ms engine |
| S04 | two turns, revert turn 1 (`notes.md` gone, `todo.md` kept), a third turn appends |
| S05 | one batch: `validate_pipeline` (allow) + `write_file` (ask); pause; approve in a new process |
| S06 | deny with a reason; the reason reaches the model as the tool result |
| S07 | cancel during a stall (unsigned reasoning in flight), cancel during a 2 s tool; strict `reject_unsigned_reasoning` |
| S08 | SIGKILL once the `write_file` result is saved (after `tool.end`, before the next request); crash resume must not re-run the tool |
| S09 | 429 with `retry-after: 1`, then OK; next turn: OpenRouter's mid-stream error chunk (`code: "server_error"`), then OK |
| S10a | `reasoning_details` round-trip: split text, metadata-only signature, encrypted detail |
| S10b | as S10a, plus an unknown field on a fragment (informational) |
| S11 | the model calls tools forever; `max_steps: 3` |
| S12 | OpenRouter cost per step and per turn, across two turns |
| S12b | BYOK (`openai_compat`, `openai` style): `stream_options.include_usage: true` must be sent; usage arrives in a trailing `choices: []` chunk without cost, so cost is reported as unavailable |
| S13 | BYOK thinking: `openai_compat`, `qwen3-32b`, `reasoning_effort` on the wire, `reasoning_content` back |
| S14 | BYOK strict endpoint: 400 on `reasoning`, `reasoning_effort` or `stream_options` |
| S15 | compaction: after the summary a request is `[system, summary, new user]`, then append-only |
