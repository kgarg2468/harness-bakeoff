# Design

This repo builds **one agent harness twice** and compares the two builds:

- **`our_version`**: our own lean, latency-optimized Python loop. It talks to OpenAI-compatible
  chat completions (OpenRouter and BYOK endpoints) with raw `httpx`. Where it saves work, it
  ports pieces from MIT/Apache-2.0 agents (Pi, OpenCode).
- **`pydantic_version`**: the same harness on [pydantic-ai](https://ai.pydantic.dev), used the
  way its docs recommend.

Everything except the loop is **shared**: the contract, session log, runner, tools, permission
rules, git working copy, fake model server, scenarios, invariants, metrics and report. So the
line counts compare only the part the decision is about.

The harness targets RocketRide's "Rocket Agent v2": a Python module that will run inside the
RocketRide engine process (one asyncio loop, Python 3.12, direct engine calls). Nothing here
touches production. Everything runs locally, and tests never leave `127.0.0.1`.

## Layout

```
src/bakeoff/
  shared/
    contract.py        the seam: Loop protocol, Item/Event/ToolCall types, rules (read this first)
    sessionlog.py      SQLite log: threads, turns, items, events (items/events append-only)
    runner.py          drives one turn: TurnInput -> loop events -> persist -> publish -> git commit
    workcopy.py        git working copy per thread; one commit per completed turn; revert = new commit
    toolhost.py        tool registry, JSON-schema validation, permission rules, tool.start/end events
    permissions.py     allow/ask/deny rules with wildcard patterns (ported from OpenCode)
    skills.py          RocketRide skills: frontmatter, the system-prompt section, load_skill paths
    tools/             file tools + RocketRide engine tools + load_skill
    engine/            Engine protocol, MockEngine (real node definitions), RealEngine (optional)
    invariants.py      I1-I7 checks over wire recordings and the session log
    scenario.py        runs a scripted scenario (driver steps) against the fake server
  our_version/         loop B   (counted)
  pydantic_version/    loop A   (counted)
  hybrid_version/      loop A'  (our loop + pydantic_ai.direct model layer; counted; phase 3)
  fakeprov/            scripted OpenAI/OpenRouter-compatible SSE server (chat + Responses API) + scenarios
  metrics/             loc, deps, bench -> out/metrics.json
  report/              builds out/report.html (side-by-side replay, wire diff, scorecard)
  loops.py             registry of the loops (lazy imports; each loop's documented failures)
  live.py              `bakeoff live` / `chat` against a real model endpoint
  cli.py               `bakeoff` command
  data/                RocketRide node catalog, example pipes, agent skills (MIT, see notices)
tests/                 pytest; the scenario matrix runs every scenario against every loop
out/                   generated: log.sqlite, wc/, wire/, metrics.json, report.html (gitignored)
```

## The seam

Read `src/bakeoff/shared/contract.py`. In short, a `Loop` gets a `TurnInput` (frozen system
prompt, full history, optional resume decisions, limits, model config), a `ToolHost`, and a
cancel `asyncio.Event`. It yields `Event`s. The shared runner:

1. creates the turn row and appends the user's message as an `Item` (so `history` already ends
   with the user item when the loop starts; on resume there is no new user item);
2. emits `turn.start`, then consumes the loop's events, stamping each with
   `{v, thread, turn, impl, seq, t_us, type, data}` (`seq` has no gaps per thread; `t_us` is
   microseconds since turn start);
3. persists `item` and `tool.start` events immediately (one SQLite transaction each; a
   `tool.start` before its tool runs, so a crash cannot hide a run) and other events in
   batches (flush on item, on `tool.start`, on `turn.end`, and every 64 events), then publishes
   each event to the sink (CLI printer, ndjson mirror, tests);
4. on `turn.end` with stop `end_turn`, `max_steps`, `budget`, `cancelled` or `error`, commits the
   working copy (`git add -A && git commit --allow-empty`), stores the sha on the turn row
   together with the `commit` event (one transaction), then publishes `commit` (always the final
   event of a completed turn, right after the loop's `turn.end`). A `paused` turn is not committed until the resumed turn finishes.

`ToolHost` is built by the runner with an `emit` callback, so `tool.start` and `tool.end` are
timed identically for every loop.

### Items and history

`Item.message` is an OpenAI chat-completions message, exactly as it goes on the wire:

- user: `{"role": "user", "content": "..."}` (created by the runner)
- assistant: `{"role": "assistant", "content": "..." | None, "tool_calls": [...], "reasoning_details": [...]}`.
  `tool_calls[].function.arguments` is the raw streamed string. `reasoning_details` is kept
  verbatim (never rebuilt).
- tool result: `{"role": "tool", "tool_call_id": "...", "content": "..."}`. One item per call.

With `ModelConfig.kind == "openai_responses"` a loop speaks OpenAI's Responses API
(`<base_url>/responses`: `input` items, named SSE events). Its items keep this chat-shaped
`Item.message` all the same (the log, the UI and the invariants read it), and the loop keeps the
exact Responses items it must replay in `Item.native`: with `store: false` the harness owns the
history, so reasoning items go back verbatim with their `encrypted_content`.

`Item.native` is loop-private. `pydantic_version` stores pydantic-ai's native `ModelMessage` JSON
there (each item gets the native of exactly what it shows) and rebuilds its history from it.
Items the runner created (user messages, revert notes, compaction summaries) have `native=None`, and every
loop must handle them. A compaction item (`Item.compaction=True`, a user message starting with
`[harness] Conversation summary:`) replaces everything before it (contract rule 8).

### Resume

- **Approval**: the user answers `permission.asked` calls. The runner starts a new turn with
  `Resume(kind="approval", decisions={call_id: "allow"|"deny"}, reason=...)`. History ends with
  the assistant message whose calls are pending (plus results for calls that already ran).
  The loop runs the allowed ones, feeds `ToolResult(ok=False, content="Denied by user: <reason>")`
  for denied ones, and continues. Until then the runner refuses a new user message, a
  compaction and a revert: each would come between the pending calls and their results.
- **Crash**: the worker process died mid-turn. The runner restarts the turn with
  `Resume(kind="crash")`. The loop continues from history. Calls that have no result are
  re-checked: `ask` pauses again, `allow` runs. A tool whose result item was already persisted
  must not run again.

Approval and crash resume are the same code path: rebuild from history, then continue.

## Tools

Tool names, descriptions and argument schemas for the engine tools are taken from the engine's
built-in MCP tool registry (`packages/ai/src/ai/modules/mcp/tools/introspection.py` on
rocketride-server `develop`). `specs()` always returns tools in this order:

| Tool | Args | read_only | Backing |
|---|---|---|---|
| `list_components` | none | yes | `Engine.get_services()` |
| `describe_component` | `name` | yes | `Engine.get_service(name)` |
| `validate_pipeline` | `pipeline` (object) **or** `path` (file in the working copy) | yes | `Engine.validate(pipeline)` |
| `list_files` | `path="."` | yes | working copy |
| `read_file` | `path` | yes | working copy |
| `write_file` | `path`, `content` | no | working copy |
| `edit_file` | `path`, `old_string`, `new_string`, `replace_all=false` | no | working copy; forgiving matcher ported from OpenCode |
| `load_skill` | `name`, `file=None` | yes | RocketRide skills bundle (`src/bakeoff/data/skills/`), see Skills below |

`MockEngine` is built from the real RocketRide node definitions (`src/bakeoff/data/catalog.json`,
generated by `scripts/sync_rocketride_data.py` from `nodes/src/nodes/*/services*.json`). It applies
the engine's structural rules (`pipeline_config.cpp`) plus the lane check. It is labelled stricter
than the engine's `validate` on lanes: the engine only reports lane errors when a pipeline is
opened with `use()`. `RealEngine` (optional) wraps `RocketRideClient(uri=..., auth="local", env={})`
against a local OSS-mode engine. Never let it default to a production URI.

Permission rules (OpenCode style), per thread:

```json
{"*": "allow", "write_file": {"*.pipe": "allow", "*": "ask"}, "edit_file": "ask"}
```

A tool maps to a decision, or to `{glob-on-path: decision}` with the first match winning. Paths
that escape the working copy are always denied.

## Skills

RocketRide ships five pipeline skills in rocketride-server `docs/agents/skills/`: building (the
orchestrator), designing, configuring, running and debugging pipelines. Each is a `SKILL.md` with
`name` and `description` frontmatter plus reference files (gate protocol, patterns,
anti-patterns, error table, node index, example pipes). `scripts/sync_rocketride_data.py` copies
their text files (`.md`, `.json`, `.pipe`, not the `tools/*.py` helpers) to
`src/bakeoff/data/skills/` and records the commit in `SOURCE.json`.

Progressive disclosure: `skills_prompt()` (`shared/skills.py`) is the system-prompt section. It
lists only `name: description` per skill and tells the model to call `load_skill` before acting on
a matching task; it is deterministic, so appending it keeps the thread's system prompt frozen.
`load_skill` returns the `SKILL.md`, or a file it mentions, after one harness paragraph: the
ToolHost's real tool names, and that instructions to run scripts or tools not in that list do not
apply here. `file` is read as the skill writes it: relative to the skill (`GATE_PROTOCOL.md`,
`../MCP_TOOL_CONTRACT.md`), or to the skills root, or as a bare file name that names exactly one
file of the bundle (the skills write `PIPELINE_ANTIPATTERNS.md`, which lives in the configuring
skill). Only files of the bundle (`.md`, `.json`, `.pipe`, no dotfiles) are ever returned. An
unknown skill or file is `invalid_args` listing the valid names, plus a "did you mean" path when a
wrong directory names a known file; the model's value is echoed at most 100 characters long.
`file` may be `null` (= `SKILL.md`). The tool description lists the skill names too, so a thread
whose system prompt lacks `skills_prompt()` can still find them.

## Fake model server (`fakeprov`)

It's a stdlib `ThreadingHTTPServer` that speaks OpenAI/OpenRouter-style streaming chat
completions. `base_url = http://127.0.0.1:<port>/s/<scenario>/<run>/<impl>/v1`. Each
`(scenario, run, impl)` gets its own cursor over the scenario's exchanges, so every loop sees the
identical script. The server:

- records every request body verbatim to `out/wire/<scenario>/<run>/<impl>/NNN.json` (never
  headers);
- checks each exchange's `expect` (e.g. `{"body_has": ["reasoning"], "last_role": "tool"}`) and
  answers HTTP 500 with an explanation on mismatch, so the scenario fails;
- keeps running when a client process is killed, so crash scenarios work.

Response primitives (a script's `respond` is a status/headers plus a list of these):

`text` (split into `chunks`, optional `delay_ms`), `reasoning_details` (fragments sent verbatim,
including metadata-only fragments), `tool_calls` (arguments split into N pieces, optionally
interleaved across indexes), `comment` (`: OPENROUTER PROCESSING`), `usage` (OpenRouter style:
repeats `finish_reason`, carries `cost`, `prompt_tokens_details.cached_tokens`,
`completion_tokens_details.reasoning_tokens`), `finish`, `sse_error` (an error chunk after HTTP
200, with `finish_reason: "error"`), `stall` (stop sending but keep the socket open),
`status: 429` plus a `retry-after` header and an OpenRouter error body.

Strict modes emulate real providers: `reject_params: [...]` returns 400 if the body contains any
listed top-level key. `reject_unsigned_reasoning` returns 400 if any assistant message
replays a `reasoning.text` detail without a signature, as Anthropic does.

A scenario with model kind `openai_responses` speaks OpenAI's Responses API instead, on
`<base_url>/responses` only (the other endpoint gets 404): named SSE events
(`response.created`, `response.output_item.added`/`.done`, `response.output_text.delta`,
`response.function_call_arguments.delta`, `response.reasoning_summary_text.delta`,
`response.completed` with `usage`, the `error` event, ...), no `[DONE]`. Its primitives are
reasoning items (`encrypted_content`, summary deltas if the request asks for them; the added
item's `encrypted_content` is incomplete, as the API documents), messages (with `phase`),
function calls, `completed` (usage), `incomplete` (usage; out of `max_output_tokens`), `error`
and `failed` mid-stream, `stall`, and items cut short before their done events. Its strict mode
`reject_unencrypted_reasoning` answers as the API does with `store: false`: a replayed reasoning
item without its `encrypted_content` is 404, one whose `encrypted_content` is not what an earlier
response's done event sent is 400. An input item of the wrong shape (say, a `function_call_output`
without `output`) is 400 in the API's error shape. Its `expect` checks read `input` (e.g.
`input_len`, `last_type`, `tool_result_contains` by `call_id`, `reasoning_replayed`: the item
exactly as sent). Details: `fakeprov/README.md`.

## Scenarios

Each scenario is a JSON file: a `driver` (user turns, approvals, cancel-after-ms, crash-after-event,
resume) plus the fake server `exchanges`. Pass or fail is judged **only** from the wire
recordings and the session log, never from what a loop says about itself.

| ID | What it exercises | Pass |
|---|---|---|
| S01 | chat; keep-alive comments; usage chunk repeats finish_reason and carries cost | exact text; usage counted once; cost = provider cost |
| S02 | pipe build: `describe_component` -> `write_file` -> `validate_pipeline` (lane error) -> `edit_file` fix -> validate ok -> answer; one call with bad args | results match MockEngine; bad args -> ok=False result, loop continues; one commit |
| S03 | 3 parallel `describe_component` calls, interleaved argument chunks, 300 ms tools | all run; tool time spans overlap; results in call order |
| S04 | 2-turn thread plus revert of turn 1 | revert = new commit + new turn; next request only appends |
| S05 | approval across processes: one batch with `validate_pipeline` (allow) + `write_file` (ask); the process exits; `bakeoff approve` resumes in a new process | write runs once; validate not re-run; history prefix holds across resume; file in the commit |
| S06 | deny with a reason | tool not run; the denial reaches the model as the tool result; file absent |
| S07 | cancel mid-stream (stall) and during a slow tool | stops within 200 ms; next turn succeeds against `reject_unsigned_reasoning`; no orphan tool calls |
| S08 | crash: SIGKILL the worker right after the tool-result item for the scripted call is persisted (`{"crash_after": "item", "call_id": ...}`), before the next request; resume | tool not re-run; resumed request continues the same history |
| S09 | 429 with `retry-after: 1`, then OK; variant: `sse_error` mid-stream | wire: exactly 2 attempts, waited per Retry-After. Observability (a visible `retry` event) is scored separately |
| S10a | `reasoning_details` round-trip with known fields, split across chunks, incl. metadata-only fragments | semantic equality of {type,text,signature,data,format,index}; byte equality reported as info |
| S10b | same plus a synthetic unknown field | informational footnote only |
| S11 | step cap: the model calls tools forever | stop=max_steps after exactly `max_steps` requests; no orphan calls; the last step's calls do not run (their results could never be sent) |
| S12 | cost metering on OpenRouter: provider `cost` per step, across two turns | per-step and per-turn totals exact |
| S12b | cost metering on a BYOK OpenAI-compatible endpoint: usage only when `stream_options.include_usage` is sent, no `cost` | usage counted; cost reported as unavailable, never guessed |
| S13 | BYOK thinking: `openai_compat` endpoint, non-OpenAI model name, reasoning requested | the reasoning parameter reaches the wire |
| S14 | BYOK strict endpoint: 400 if body has `reasoning`, `reasoning_effort` or `stream_options` (configured via compat flags) | the turn finishes |
| S15 | compaction hand-off: runner appends a summary item; loop sends [system, summary, new user] | prefix resets only at the compaction boundary |
| R01 | Responses API (`gpt-6-luna`, effort `xhigh`, summary `auto`): text only | `store: false`, `include` has `reasoning.encrypted_content`, `reasoning.summary` sent, Responses tools; exact text; usage from `response.completed` |
| R02 | reasoning + one function call + its `function_call_output`, then the answer | the reasoning item replayed exactly as sent; tool ran once |
| R03 | commentary + 3 function calls in one response; `write_file` asks; approve in a new process | as S05; the resumed request replays reasoning, commentary (`phase`) and all calls |
| R04 | 429 with `retry-after: 1`, then OK; next turn an `error` event mid-stream, then OK | waited per Retry-After; nothing of the failed attempt is replayed |
| R05 | cancel while reasoning streams | stops within 200 ms; next turn passes `reject_unencrypted_reasoning` |

The driver (`shared/scenario.py`) runs the steps with the real runner, session log, working copy
and ToolHost on MockEngine. `approve` with `new_process` and the user turn after `crash_after` run
as separate OS processes (`bakeoff approve`, `bakeoff turn`); the crash is a SIGKILL the child
sends itself from its runner's sink, so for `item`, `tool.start` and `turn.end` (stored before
they are published) it comes right after the event is durable. Every key of the final `expect`
and the invariants I1, I2, I3, I5 and I7 decide pass or fail. The timing criteria above are
final-expect keys too: `tools_overlap` (S03: the tools' `tool.start`..`tool.end` spans share a
moment) and `cancel_within_ms` (S07: `turn.end` at most 200 ms after the driver set `cancel`).
For I5 the driver captures this process's stdout/stderr from the loop's creation until it is
closed, so background tasks and threads count too, and reads the child processes' pipes (a
worker prints only its one-line JSON summary). Tasks a loop leaves running after `aclose()` are
cancelled before judging and fail the scenario, so nothing they do lands in the next scenario.
If one is still running 2 s after its cancel, `bakeoff scenario` runs nothing more in that
process (it would share the event loop with it) and exits 1; `summary.json` covers the runs so
far, and its `stopped` names the run and the tasks.
A loop's documented failures are listed in `bakeoff/loops.py` with the exact checks they fail:
the matrix runs them as strict xfails, so a fix shows up as well as a regression, and a cell
that fails any other way is a plain failure.

## Invariants (checked on every scenario)

- **I1 append-only**: each request's `messages` (Responses API: `instructions`, then the `input`
  items) are a prefix of the next request's (semantic equality; byte equality reported
  separately). Resets only at a compaction item.
- **I2** every tool call gets exactly one result; no call id runs twice (`tool.start` count);
  every run ends before its turn's `turn.end` (no orphan tools); and every result comes from a
  run, unless the user denied the call or its turn stopped early (cancelled, max_steps, budget,
  error).
- **I3** `seq` has no gaps; `item` events == item rows.
- **I5** the loop writes nothing to stdout/stderr.
- **I6** no connection leaves 127.0.0.1 (socket guard in tests and in every `bakeoff` command;
  `live` and `chat` also allow the model endpoint's host).
- **I7** one commit per completed turn; `turns.commit_sha == git rev-parse`.

## `our_version`: lean and fast

The harness should add close to nothing on top of the model's own latency. Required:

1. **Connection reuse**: one pooled `httpx.AsyncClient` per base URL per loop instance. No TLS
   handshake per step. (HTTP/2 is optional; `h2` ships in the engine.)
2. **Serialize once**: each history message is serialized once and cached by item id. The
   static prefix (model, system, tools) is serialized once per thread. The request body is
   built by joining cached fragments. This is fast, and it makes the prefix byte-identical,
   which keeps provider prompt caches warm.
3. **Thin hot path**: `aiter_lines`, skip `:` comments, one `json.loads` per chunk, no pydantic on
   the hot path, dispatch on the keys present, no logging per chunk.
4. **Eager tools**: when a streamed call is complete (the next index starts, or the stream
   ends) and its tool is `read_only` with `check() == "allow"`, start it immediately while the
   stream continues.
5. **Parallel tools**: run all allowed calls of a step concurrently; append results in call order.
6. **Prompt caching and routing** (OpenRouter): send `session_id` (sticky provider) and, for
   `anthropic/*` models, top-level `cache_control: {"type": "ephemeral"}`.
7. **Cheap cancel**: a watcher task closes the response and cancels running tools when `cancel`
   is set. There is no per-chunk polling.
8. **Retries**: honour `Retry-After` (capped), jittered exponential backoff, retry mid-stream
   errors only before any tool side effect, surface each retry as a `retry` event.
9. **Endpoint compat flags** (ported from Pi's `openai-completions.ts`): at least
   `max_tokens_field` (`max_tokens` | `max_completion_tokens`), `reasoning_param`
   (`openrouter` | `reasoning_effort` | `none`), `stream_usage` (send
   `stream_options.include_usage`), `developer_role`. Defaults come from `ModelConfig.kind`.
10. **No new dependencies**: `httpx` and the standard library only.

Keep it small. Every line must earn its place: no speculative abstractions and no plugin
systems. `metrics/bench.py` measures harness overhead per turn (model time excluded),
per-chunk overhead p50/p95, and events per second. CI fails on large regressions.

## `pydantic_version`: used the recommended way

See `A_CHECKLIST.md`. It must pass on both pydantic-ai 2.31.1 (the newest release that fits the
engine's dependency set) and the latest release (CI matrix).

## Porting and attribution

Porting (including translating TypeScript to Python) is allowed from **MIT or Apache-2.0**
projects only. Every ported file starts with:

```python
# Ported from <project> (<license>): <repo path> @ <commit sha>
# Changes: <one line>
```

Add an entry to `THIRD_PARTY_NOTICES.md`, with the upstream copyright line and license text. If an
upstream file credits its own sources (e.g. OpenCode's edit tool credits Cline and gemini-cli,
both Apache-2.0), credit those too. Never use leaked or unlicensed code.

## Safety

- Tests and scripted runs never leave 127.0.0.1.
- The only live network use is `bakeoff live` / `bakeoff chat`, which read `OPENAI_API_KEY`
  from the environment or from an env file given by `--env-file` or `BAKEOFF_ENV_FILE`, hold it
  in memory, never print or record it, and only talk to the endpoint's host (`api.openai.com`
  by default).
- Wire recordings contain request bodies only, never headers. Live recordings are not committed.
- No code path may default to a production RocketRide URI.
