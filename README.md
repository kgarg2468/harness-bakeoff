# harness-bakeoff

**The same agent harness built twice, compared side by side.**

RocketRide's next agent ("Rocket Agent v2") needs a Python agent loop that runs inside the
engine. There are two ways to build it:

| | Folder | What it is |
|---|---|---|
| **A** | [`src/bakeoff/pydantic_version/`](src/bakeoff/pydantic_version) | the loop on [pydantic-ai](https://ai.pydantic.dev), used the way its docs recommend |
| **B** | [`src/bakeoff/our_version/`](src/bakeoff/our_version) | our own lean, latency-optimized loop on raw `httpx`, with pieces ported from Pi and OpenCode |

Everything else (tools, session log, git working copy, permission rules, a fake OpenRouter
server, test scenarios, metrics and the report) lives in [`src/bakeoff/shared/`](src/bakeoff/shared)
and is identical for both. So the comparison counts only the loop.

- **How it's built:** [`DESIGN.md`](DESIGN.md)
- **How we keep it fair:** [`FAIRNESS.md`](FAIRNESS.md) and [`A_CHECKLIST.md`](A_CHECKLIST.md)
- **What we expected before writing it:** [`PREDICTIONS.md`](PREDICTIONS.md)

## Quick start

```bash
uv sync --all-extras
uv run pytest                              # every scenario against every loop, fully offline
uv run bakeoff scenario --all --impl our   # the scenario matrix, with a one-line reason per failure
./scripts/engine_fit.sh                    # which loop's dependencies fit into the RocketRide engine
```

`bakeoff report` and `bakeoff bench` arrive as those pieces land; see `DESIGN.md`.

## CLI

Scenarios run offline against the fake model server (`fakeprov`), with the real runner, session
log, git working copy and tools. Loops that are not built yet are skipped; a loop that exists
but fails to import is an error (exit 1), so it never drops out of a comparison unnoticed.

```bash
bakeoff scenario S01 S05 --impl our,pydantic   # or --all; --out out (default), --run-id ID
```

A run id is never reused: `out/runs/<run_id>` must not exist yet (the default is a new
timestamped id).

It prints a matrix and exits 1 on any `FAIL` or `XPASS`. `xfail` is a failure that the loop's
registry entry (`bakeoff/loops.py`) documents, with exactly the checks it documents; a documented
cell that fails any other way is a `FAIL`. `XPASS` means a documented failure is fixed, so its
entry must go. A loop that leaves a task running even after it was cancelled stops the command
after that run (exit 1; `stopped` in summary.json says which). Each run writes:

```
out/runs/<run_id>/summary.json                   matrix + one-line reasons, git sha, loop versions
out/runs/<run_id>/<scenario>/<impl>/result.json  every expect key and invariant, with details
                                    log.sqlite   the session log
                                    wire/        every request body the fake server received
                                    events.ndjson, wc/ (the git working copy)
out/runs/latest -> <run_id>
```

Live runs talk to a real OpenAI-compatible endpoint (OpenAI by default). The key is
`OPENAI_API_KEY` from the environment, or from an env file (`--env-file` or `BAKEOFF_ENV_FILE`)
that the code reads; it is never printed or stored. It goes only over https, and only to
`api.openai.com` or a host named with `--key-host HOST` on the command line (never to a host that
only a session log names). The network guard allows only loopback and the endpoint's host.

```bash
bakeoff live --impl our,pydantic --model gpt-6-luna --reasoning none --max-steps 8 \
    --prompt "Build a chat pipeline, save it as chat.pipe, and validate it."
bakeoff chat --impl our            # REPL: approvals prompted, /revert N, /compact TEXT, /exit
# Lands with the loop PRs (see below): no loop speaks the Responses API yet
bakeoff live --impl our,pydantic --api responses --reasoning xhigh --prompt "..."
```

`--api responses` uses OpenAI's Responses API (endpoint kind `openai_responses`, same
`--base-url`, `https://api.openai.com/v1` by default) instead of chat completions: `gpt-6-luna`
takes function tools on chat completions only with reasoning effort `none`, so tools plus
reasoning need it. There `--reasoning EFFORT` also asks for a reasoning summary (`"summary":
"auto"`, unless the effort is `none`): the API streams none otherwise. `--kind` picks the chat
completions kind (`openai_compat` or `openrouter`). Only the shared groundwork is in so far
(this option, fakeprov's Responses mode, scenarios R01-R05); loop support follows in the next
PRs. Until a loop speaks the API (its R01-R05 cells pass instead of `xfail`), `--api responses`
cannot complete a run with it.

In `chat`, Ctrl-C during a turn cancels the turn, and at a prompt it ends the chat. `chat` exits 1
if any turn stopped short (error, max_steps, budget, cancelled) or an invariant failed, else 0.
A run id is never reused: `out/live/<run_id>/<impl>` must not exist yet.

`live` streams each loop's run (text inline, tool calls and results on one line each), then
prints tokens, time to first token, total time and steps side by side, and writes
`out/live/<run_id>/<impl>/` (same files as a scenario run; `result.json` adds `model`,
`base_url`, `prompt`, `final_text`, `latency`, and what the session log does not keep:
`max_steps` and `attended`, true when a person answers approvals). A run that fails, is
interrupted or cannot even start (say, git fails) still writes its `result.json`, with the error.
`--interactive` asks before each write (`attended`); `--base-url` points it at another endpoint
(`{impl}` in the URL is replaced by the loop name).

The commands below act on one thread of a session log from a separate process. The scenario
driver uses them for the cross-process steps (approve in a new process, crash and resume):

```bash
bakeoff turn    THREAD --db out/.../log.sqlite --user TEXT [--crash-after TYPE --call-id ID]
bakeoff approve THREAD --db ... [--allow ID ...] [--deny ID ... --reason TEXT]
bakeoff deny    THREAD --db ... [--call ID ...] --reason TEXT
bakeoff resume  THREAD --db ...    # crash resume
bakeoff cancel  THREAD --db ...    # stops the turn a worker is running
bakeoff fakeprov --port 8787       # serve the fake model server
```

## Status

Work in progress. Each piece lands through a reviewed pull request.

## License

MIT. Ported code is credited in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
