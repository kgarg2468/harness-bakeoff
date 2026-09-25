# fakeprov: scripted OpenRouter-compatible model server

A stdlib `ThreadingHTTPServer` on `127.0.0.1` that answers streaming chat completions (and
OpenAI's Responses API, see [Responses API mode](#responses-api-mode)) from scenario scripts, so
every loop sees exactly the same model behaviour and every request it sends is recorded.
Scenario pass/fail is judged from those recordings and the session log.

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
- `POST <base>/responses`: the same, for scenarios scripted for the Responses API. A scenario
  answers only on its own API's endpoint: a request to the other one gets 404 `scenario R01 is
  scripted for POST <base_url>/responses, not /chat/completions` (recorded, and it uses its
  exchange like any rejection).
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

Every model request (either endpoint) is recorded under `<wire_dir>/<scenario>/<run>/<impl>/`:

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
| `model` | `{"kind": "openrouter" \| "openai_compat" \| "openai_responses", "model", "reasoning"?, "temperature"?, "compat"?}` → `ModelConfig` (passed through unchanged; `temperature: null` sends none, absent keeps `ModelConfig`'s default). `openai_responses` scripts the [Responses API](#responses-api-mode) |
| `rules` | permission rules, e.g. `{"*": "allow", "write_file": "ask"}` |
| `limits` | `{"max_steps"}` → `Limits` |
| `engine` | `{"delay_ms"}` → `MockEngine(delay_ms=...)` |
| `style` | optional wire style of every exchange: `"openrouter"`, or `"openai"` (BYOK endpoints). Default: `"openai"` for `openai_compat`, else `"openrouter"`. Not in Responses scenarios (their style is `"responses"`) |
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
   value at a path in the body; an integer segment indexes a list, e.g. `"tools.0.name"`),
   `body_contains: {"dotted.path": value}` (the value at that path is a list that contains
   it), `messages_at: [{index, role?, contains?}]` (checks one message; a negative index
   counts from the end), and `min_gap_ms` (the request must arrive at least this long after
   the cursor's previous one, e.g. a retry that honours `retry-after`). Responses scenarios
   check `input` instead of `messages` (see below).

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
| `tools_overlap` | `[call_id, ...]` (at least 2): the tools of these calls ran at the same time. Each ran once, in one turn, and the last `tool.start` comes before the first `tool.end` (event `t_us`) |
| `cancel_within_ms` | every turn the driver cancels (`cancel_after_ms`) has its `turn.end` at most this many ms after the driver set `cancel`. The driver notes when it set it; `t_us` in the log gives the rest. A turn that ends before its cancel fires passes this key; `stops` judges it |

Tool call ids are `call_<scenario>_<n>`, unique per scenario; the loader rejects references
to ids that no exchange scripts. Content checks on tool results stay loose (substrings),
because results come from the shared tools and `MockEngine`.

## Responses API mode

A scenario whose model kind is `openai_responses` is scripted for OpenAI's Responses API
(API reference: "Streaming events"; `store: false` statelessness, as a harness that keeps its
own history uses it). Requests go to `POST <base>/responses` and are recorded verbatim, like
chat requests. The body must be a JSON object with `stream: true` and `input`: a non-empty list
of item objects, or a string (one user message, as the API reads it); else 400.

**Stream.** Named SSE events, one per HTTP chunk:
`event: <type>\ndata: {"type": <type>, "sequence_number": n, ...}\n\n`, numbered from 0. There
is no `data: [DONE]`: the stream ends after its last event. It always starts with
`response.created` and `response.in_progress`. Their response object (and that of
`response.completed` / `.incomplete` / `.failed`) does not echo the request, so every loop gets the
same bytes: `{"id": "resp_<scenario>_<NNN>", "object": "response", "created_at": 1758758400,
"status", "completed_at", "error", "incomplete_details": null, "instructions": null,
"max_output_tokens": null, "model": <scenario model>, "output", "parallel_tool_calls": true,
"previous_response_id": null, "reasoning": {"effort": <scenario effort>, "summary": null},
"store": false, "temperature": 1, "text": {"format": {"type": "text"}}, "tool_choice": "auto",
"tools": [], "top_p": 1, "truncation": "disabled", "usage", "user": null, "metadata": {}}`.
Delta events (`response.output_text.delta`, `response.function_call_arguments.delta`,
`response.reasoning_summary_text.delta`) carry an `obfuscation` pad, as the API does by
default (deterministic here), unless the request sets `stream_options.include_obfuscation:
false`. `output_index` counts the response's items from 0.

| Op | Sends |
|---|---|
| `{"reasoning_item": {"id", "encrypted_content", "summary"?: [str]}, "chunks"?: n, "done"?: bool}` | `response.output_item.added` with `{"id", "type": "reasoning", "summary": [], "encrypted_content": <its first half>}` (the API documents that the added item's `encrypted_content` may be incomplete: replay only the done item); if the request asks for a summary (`reasoning.summary`, as the API streams none otherwise), per summary part `response.reasoning_summary_part.added` (`summary_index`, `part: {"type": "summary_text", "text": ""}`), the text in n `response.reasoning_summary_text.delta`, `response.reasoning_summary_text.done` (`text`), `response.reasoning_summary_part.done`; then `response.output_item.done` with `{"id", "type": "reasoning", "summary": [{"type": "summary_text", "text"}, ...] (`[]` unless asked for), "encrypted_content"}`. Reasoning ids are unique per scenario |
| `{"text": str, "chunks"?: n, "phase"?: "commentary" \| "final_answer", "done"?: bool}` | an assistant message: `response.output_item.added` (`{"id": "msg_<scenario>_<NNN>_<output_index>", "type": "message", "status": "in_progress", "content": [], "role": "assistant", "phase"?}`), `response.content_part.added` (`part: {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}`), n `response.output_text.delta` (`delta`, `logprobs: []`), `response.output_text.done`, `response.content_part.done`, `response.output_item.done` (status `completed`, the full `content`). The API asks to resend `phase` on assistant messages |
| `{"tool_calls": [{"id", "name", "arguments": str \| object}], "pieces"?: n}` | per call, one after the other: `response.output_item.added` (`{"id": "fc_<id without call_>", "type": "function_call", "status": "in_progress", "arguments": "", "call_id": <id>, "name"}`), the arguments in n `response.function_call_arguments.delta`, `response.function_call_arguments.done` (`name`, `arguments`), `response.output_item.done` (status `completed`, full `arguments`) |
| `{"completed": {"input_tokens", "output_tokens", "cached_tokens"?, "cache_write_tokens"?, "reasoning_tokens"?}}` | `response.completed`: status `completed`, `output` = every done item, `usage: {"input_tokens", "input_tokens_details": {"cached_tokens", "cache_write_tokens"}, "output_tokens", "output_tokens_details": {"reasoning_tokens"}, "total_tokens"}` |
| `{"incomplete": {"input_tokens", "output_tokens", ...as completed, "reason"?: "max_output_tokens" \| "content_filter"}}` | `response.incomplete`, as the API ends a response that ran out of `max_output_tokens` (say, while it reasoned): status `incomplete`, `incomplete_details: {"reason"}` (default `max_output_tokens`), `output` = the items done so far, `usage` as in `completed` |
| `{"error": {"code"?: "server_error", "message"}}` | the `error` event after HTTP 200: `{"type": "error", "sequence_number", "code", "message", "param": null}`; the stream ends |
| `{"failed": {"code"?: "server_error", "message"}}` | `response.failed`: status `failed`, `error: {"code", "message"}`, `output` = the items done so far |
| `{"stall": true}` | as in chat: nothing more, the socket stays open until the client leaves |

Every stream ends with exactly one of `completed`, `incomplete`, `error`, `failed` or `stall`,
as its last op. `"done": false` (on `text` or `reasoning_item`) cuts the stream before that
item is done: its added and delta events go out, none of its done events (a reasoning item is
cut inside its last summary part), and the next op must be `incomplete`, `error`, `failed` or
`stall`. `delay_ms` works as in chat. `status` other than 200 sends OpenAI's error shape (or
the script's `body`), e.g. a 429 with `retry-after`.

**Strict modes** (`strict`, as in chat): `reject_params: [keys]` answers 400 `Unsupported
parameter: '<key>'.` (`param` = key, `code: "unsupported_parameter"`).
`reject_unencrypted_reasoning: true` answers as the API does with `store: false`, where a
reasoning item exists only as its encrypted content: an input reasoning item without
`encrypted_content` gets 404 `Item with id '<id>' not found. Items are not persisted when
`store` is set to false. Try again with `store` set to true, or remove this item from your
input.`; one whose `encrypted_content` is not the one its done event sent (the added item's
incomplete one, any for an item that was cut short, or any for an item that no earlier response
of the cursor sent) gets 400 `The encrypted content for item <id> could not be verified.`
(`code: "invalid_encrypted_content"`).

**Exchange `expect`.** `body_has`, `body_lacks`, `model`, `body_equals`, `body_contains` and
`min_gap_ms` read the body as in chat. The others read `input` without the system prompt
(`instructions`, or the system/developer messages `input` starts with), so they hold wherever a
loop puts it. An item without a `type` but with a `role` is a `message`.

| Key | Passes when |
|---|---|
| `system_contains` | the system prompt (`instructions` plus those leading messages) contains it |
| `input_len` | the number of input items after the system prompt |
| `last_type` | the last item's type: `message`, `function_call`, `function_call_output` or `reasoning` |
| `last_role` | the last item's `role` (`user`, `assistant`) |
| `last_content_contains` | the last item's text contains it: a message's text (a string or text parts), a `function_call_output`'s `output`, a `function_call`'s `arguments`, a reasoning summary |
| `input_at` | `[{index, type?, role?, phase?, contains?}]`: checks one item; a negative index counts from the end |
| `tool_result_contains` | `{call_id: substring}`: a `function_call_output` with that `call_id` contains it (`""`: it exists) |
| `reasoning_replayed` | `[reasoning id]`: `input` has that reasoning item exactly once, equal to its done item (same keys and values; `encrypted_content` byte for byte; with a summary if this request asks for one, since a thread's requests all do or all don't). The loader checks that an earlier exchange sends it |

A sample (R02's first answer, shortened: one summary part, arguments in 2 pieces; the
`response.created` and `response.in_progress` data are elided):

```
event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_R02_001",...,"status":"in_progress",...,"output":[],...,"usage":null,...}}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1,"response":{...}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"rs_R02_1","type":"reasoning","summary":[],"encrypted_content":"gAAAAABpWe"}}

event: response.reasoning_summary_part.added
data: {"type":"response.reasoning_summary_part.added","sequence_number":3,"item_id":"rs_R02_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","sequence_number":4,"item_id":"rs_R02_1","output_index":0,"summary_index":0,"delta":"**Plan**\n\nLook it up.","obfuscation":"sTDlM7na"}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","sequence_number":5,"item_id":"rs_R02_1","output_index":0,"summary_index":0,"text":"**Plan**\n\nLook it up."}

event: response.reasoning_summary_part.done
data: {"type":"response.reasoning_summary_part.done","sequence_number":6,"item_id":"rs_R02_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":"**Plan**\n\nLook it up."}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":{"id":"rs_R02_1","type":"reasoning","summary":[{"type":"summary_text","text":"**Plan**\n\nLook it up."}],"encrypted_content":"gAAAAABpWeVd...R02_1"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":8,"output_index":1,"item":{"id":"fc_R02_1","type":"function_call","status":"in_progress","arguments":"","call_id":"call_R02_1","name":"describe_component"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":9,"item_id":"fc_R02_1","output_index":1,"delta":"{\"name\": \"","obfuscation":"wKHeiD52LQrfPdd"}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":10,"item_id":"fc_R02_1","output_index":1,"delta":"tool_pipe\"}","obfuscation":"tBkpy0cUX4Ki"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","sequence_number":11,"item_id":"fc_R02_1","output_index":1,"name":"describe_component","arguments":"{\"name\": \"tool_pipe\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":12,"output_index":1,"item":{"id":"fc_R02_1","type":"function_call","status":"completed","arguments":"{\"name\": \"tool_pipe\"}","call_id":"call_R02_1","name":"describe_component"}}

event: response.completed
data: {"type":"response.completed","sequence_number":13,"response":{"id":"resp_R02_001",...,"status":"completed","completed_at":1758758401,...,"output":[<the two done items>],...,"usage":{"input_tokens":2100,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":180,"output_tokens_details":{"reasoning_tokens":150},"total_tokens":2280},...}}
```

The next request replays the done reasoning item verbatim, the function call and its output:
`input: [{"role": "user", ...}, {"id": "rs_R02_1", "type": "reasoning", "summary": [...],
"encrypted_content": "..."}, {"type": "function_call", "call_id": "call_R02_1", "name":
"describe_component", "arguments": "{\"name\": \"tool_pipe\"}", ...}, {"type":
"function_call_output", "call_id": "call_R02_1", "output": "..."}]`.

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
| R01 | Responses API, `gpt-6-luna` at effort `xhigh`: the request shape (`store: false`, `include` has `reasoning.encrypted_content`, Responses-style tools, no `temperature`/`max_tokens`/`reasoning_effort`); reasoning with a summary, then the answer |
| R02 | reasoning (two summary parts), one `describe_component` call, its output, then the answer; the reasoning item is replayed exactly as its done event sent it |
| R03 | reasoning, a `commentary` message and three calls in one response (`validate_pipeline` and `describe_component` allowed, `write_file` asks); approve in a new process; the resumed request replays everything, `phase` included |
| R04 | 429 with `retry-after: 1`, then OK; next turn: a complete reasoning item, cut text and an `error` event mid-stream, then the same request again, keeping nothing of the failed attempt |
| R05 | cancel while a reasoning item streams (cut before its done event); strict `reject_unencrypted_reasoning`: the next turn must not replay the cut item |

Every R scenario uses model kind `openai_responses`, `gpt-6-luna`, `reasoning: {"effort":
"xhigh"}`, `temperature: null`, and strict `reject_params` (`temperature`, `max_tokens`,
`max_completion_tokens`, `reasoning_effort`) plus `reject_unencrypted_reasoning`.
