<h1 align="center">harness-bakeoff</h1>

<p align="center">
  <strong>One agent harness, built twice.</strong>
</p>

<p align="center">
  RocketRide's next agent, Rocket Agent v2, needs a Python agent loop that runs inside the engine. This repo builds the same harness twice: once on <a href="https://ai.pydantic.dev">pydantic-ai</a>, used the way its docs recommend, and once as our own lean loop on raw httpx. Everything except the loop is shared, and both are judged by the same scenarios and checks, so the comparison counts only the loop.
</p>

<p align="center">
  <a href="#results-at-a-glance"><strong>Results</strong></a> ·
  <a href="DESIGN.md"><strong>Design</strong></a> ·
  <a href="FAIRNESS.md"><strong>Fairness rules</strong></a> ·
  <a href="A_CHECKLIST.md"><strong>A checklist</strong></a> ·
  <a href="docs/learn/bakeoff-101.html"><strong>Bakeoff 101 (course)</strong></a>
</p>

<p align="center">
  <a href="https://github.com/kgarg2468/harness-bakeoff/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/kgarg2468/harness-bakeoff/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="MIT" src="https://img.shields.io/badge/License-MIT-0F7C86?style=flat-square">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-2D2A26?style=flat-square">
  <img alt="Tests: loopback only" src="https://img.shields.io/badge/Tests-loopback_only-6A3FD1?style=flat-square">
  <img alt="Verdict: none, people decide" src="https://img.shields.io/badge/Verdict-none,_people_decide-2D2A26?style=flat-square">
</p>

<p align="center">
  <img src="docs/assets/two-loops.svg" alt="Two loop folders sit on top and are the only code that is counted: A, pydantic_version, the loop on pydantic-ai with Agent.iter and deferred tools, used the way its docs recommend; and B, our_version, our own low-latency loop on raw httpx with some pieces ported from Pi. Both implement one seam, shared/contract.py: a loop gets the history and the tools, and only yields events. Everything under the seam is shared, identical for both and not counted: the runner, session log, git working copy, RocketRide tools, permission rules and skills that run a turn, and the fake model server, 22 scenarios, invariants I1, I2, I3, I5 and I7, metrics and report page that judge it." width="880">
</p>

## Results at a glance

| | **A** · pydantic-ai | **B** · our loop |
| --- | --- | --- |
| Lines of its own code | 778 | 869 (468 ported from Pi) |
| Packages it installs | 35 · 35.4 MB | 12 · 3.9 MB |
| Cold import | 1.19 s | 71.4 ms |
| Harness overhead per turn (p50) | 369.8 ms (184.1 µs per chunk) | 12.8 ms (6.4 µs per chunk) |
| Scenarios S01–S15 and R01–R05 | 22 / 22 pass | 22 / 22 pass |
| Fits in the engine's Python | up to 2.31.1; 2.32+ needs openai 3 | yes, at the engine's own versions |
| Live `gpt-6-luna` runs that built and validated `chat.pipe` | 5 / 5 | 5 / 5 |
| Live, chat completions, reasoning `none`: time and input tokens per step (median of 3; within 10%) | 1.60 s · 21,433 tokens | 1.49 s · 20,475 tokens |
| Live, Responses API, reasoning `xhigh`: time and input tokens per step (median of 2) | 7.19 s · 25,818 tokens | 6.31 s · 24,827 tokens |
| Lines added for OpenAI's Responses API ([#14](https://github.com/kgarg2468/harness-bakeoff/pull/14), [#13](https://github.com/kgarg2468/harness-bakeoff/pull/13)) | +97 (+46 in the first cut) | +156, 136 of them ported from Pi (+140 in the first cut) |

`gpt-6-luna` takes tools together with reasoning only on `/v1/responses`, so both loops had to learn that API. pydantic-ai already speaks it, so A's first cut mostly wired settings: +46 lines against B's +140. Fixes found in review and by the scenarios (how a stream ends or fails, crash resume, reasoning summaries, a call cut off at the output limit) brought that to +97 against +156. That's a real advantage for A. There are only 2 Responses live runs per loop, and most of each step there is the model reasoning, so read their time per step as a hint rather than a measure of the loop. Which side each number favours, and by how much, is under [Where each wins](#where-each-wins).

The full comparison is one page, `out/report.html`: a scorecard, the scenario matrix, a side-by-side replay of every scenario, a diff of what each loop sent to the model, the live runs, and where each loop wins. Build it with `bakeoff report` (see [Quick start](#quick-start)) and open it in a browser; it has no external assets. A copy built from the runs behind this README is committed as [`docs/report.html`](docs/report.html).

New to the repo? [`docs/learn/bakeoff-101.html`](docs/learn/bakeoff-101.html) is an interactive course (45 short screens) that explains the whole thing from the machine up: which processes run, what goes over the network, then the contract, the shared harness and both loops with real code excerpts, and finally every number in this README and the report. GitHub shows HTML files as source, so clone the repo (or download the whole `docs/` folder, keeping `docs/learn/` inside it, since the course links to `../report.html`) and open them in a browser; neither loads anything from the internet.

<details>
<summary><strong>Where these numbers come from</strong></summary>

- **Lines of code:** `uv run python -m bakeoff.metrics.loc` on `main`. Code lines only (no comments, docstrings or blanks); ported files are marked at the top and counted separately.
- **Packages, import time and overhead:** `out/metrics.json` from `python -m bakeoff.metrics.collect --deps --bench full`, at commit `b6acae3` (`main` with both Responses API PRs merged). Each loop's dependency set is installed alone in a fresh virtual environment. The benchmark turn is a tool step of 2,000 streamed chunks plus a one-chunk answer, against the local fake server, 100 measured turns per loop; overhead is the loop's turn time minus a bare `httpx` client reading the same stream. AMD Ryzen 9 6900HS, Python 3.12.3.
- **Scenarios:** `uv run bakeoff scenario --all --impl our,pydantic`. All 22 pass for both loops, R01–R05 (the Responses API) included.
- **Engine fit:** `./scripts/engine_fit.sh` asks uv to resolve each loop's dependencies together with the engine's pins.
- **Live runs:** `bakeoff live` on 2026-09-25, `gpt-6-luna` on api.openai.com, a 20-step cap, unattended, and the prompt *"Build a RocketRide pipeline that answers questions from a chat using an LLM, save it as chat.pipe, and validate it."* Three runs per loop on chat completions with reasoning `none` (7 or 8 steps each), and two per loop on the Responses API (`--api responses`) with reasoning `xhigh` (10 to 13 steps). Medians per step, because the model chooses how many steps to take. Every run ended with `end_turn`, its one `validate_pipeline` call returned 0 errors and 0 warnings (on MockEngine, which carries the real RocketRide node catalog), and the invariants checked on live runs (I2, I3, I5, I7) held. Each run's key fields, validation result included, are kept in [`docs/results/live-2026-09-25.json`](docs/results/live-2026-09-25.json) (chat completions) and [`docs/results/live-responses-2026-09-25.json`](docs/results/live-responses-2026-09-25.json) (Responses API), written by [`scripts/snapshot_live.py`](scripts/snapshot_live.py), so these figures can be checked without paying for new runs.
- **Responses API lines:** `bakeoff.metrics.loc` before and after each loop's PR. B: 713 → 869 ([#13](https://github.com/kgarg2468/harness-bakeoff/pull/13); 332 → 468 ported from Pi). A: 681 → 778 ([#14](https://github.com/kgarg2468/harness-bakeoff/pull/14)). The first-cut figures are the line tables in the PRs' descriptions (B 853, A 727).

</details>

## How it works

<p align="center">
  <img src="docs/assets/one-turn.svg" alt="How one turn runs. Your message arrives. The loop builds a request from the full history and streams the model's answer: text and tool calls. The shared runner saves each new item to the session log and emits an event. With no tool calls, the turn ends with one git commit. With tool calls, the shared permission rules check each one: allow, ask or deny. Allowed calls run, their results are saved, and the loop takes the next step. A call that asks pauses the turn until a person approves or denies it, later and from any process, and then the turn resumes. A denied call never runs; the model gets the reason as its result. If the worker crashes, bakeoff resume rebuilds the turn from the saved log, and a saved tool result never runs twice. Only Build request and Stream the model are the loop's own code; everything else is shared." width="880">
</p>

A turn starts with your message. The shared runner saves it and hands the loop the whole history. Boxes with a coloured border are the loop's own code, A or B; the grey ones are shared.

- **Each step is one streamed model request.** The loop builds it from the full history and streams the answer: text, reasoning and tool calls. B builds each request from cached bytes and starts read-only tools while the model is still streaming; A does it all through pydantic-ai's `Agent.iter()`.
- **The loop only yields events.** It never writes the log, git or the screen itself. The shared runner saves each item to a SQLite session log, publishes each event, and ends a completed turn with one git commit in the thread's working copy.
- **Tools go through one shared ToolHost.** It checks arguments against each tool's JSON schema, applies the permission rules (allow, ask or deny, per tool and per file pattern) and times every run. The tools are RocketRide's: list, describe and validate pipeline components against the real node catalog, read and edit files, and load the RocketRide pipeline skills.
- **Ask means pause.** The turn ends as `paused`. You approve or deny later, even from another process (`bakeoff approve`), and the turn picks up where it stopped. A denied call never runs; the model gets the reason as its result.
- **A crash is just another resume.** If the worker dies mid-turn, `bakeoff resume` rebuilds the turn from the saved history. A tool whose result was saved never runs twice.

The seam is [`src/bakeoff/shared/contract.py`](src/bakeoff/shared/contract.py): one `Loop` protocol and eight rules every loop follows. [`DESIGN.md`](DESIGN.md) has the rest.

## How we keep it fair

<p align="center">
  <img src="docs/assets/fairness.svg" alt="The same test for both loops. A, pydantic-ai, and B, our own loop, each talk to the same fake model server, fakeprov on 127.0.0.1, which plays 22 scripted scenarios, with a fresh copy of each script per loop. Every request a loop sends, the session log and the git working copy are recorded. Both loops are judged the same way from those recordings only: each scenario's checks and the invariants I1, I2, I3, I5 and I7 give a pass or fail matrix (I6, no traffic beyond this machine, is enforced by a network guard instead). The rules from FAIRNESS.md: predictions were written before either loop; A is used the way the pydantic-ai docs recommend, per A_CHECKLIST.md, which its reviewer can change; both loops have the same feature floor (retries, cost, cancel, approvals, crash resume); and there is no automatic winner: the report shows evidence and people decide." width="880">
</p>

- **One seam.** Only `our_version/` and `pydantic_version/` are counted. Everything else is shared and identical for both.
- **Same scenarios, same fake model.** `fakeprov` is a scripted OpenAI- and OpenRouter-style streaming server on 127.0.0.1. Each loop gets its own copy of the same script, and every request body is recorded.
- **Judged from the outside.** Pass or fail comes only from the recorded requests and the session log, never from what a loop reports about itself. Every scenario also checks the invariants: **I1** history is append-only, **I2** every tool call gets exactly one result, **I3** the event sequence has no gaps, **I5** the loop prints nothing, **I7** one git commit per completed turn. A socket guard enforces **I6**: no connection leaves 127.0.0.1.
- **Predictions first.** [`PREDICTIONS.md`](PREDICTIONS.md) was committed before either loop was written.
- **A is used the recommended way.** [`A_CHECKLIST.md`](A_CHECKLIST.md) ties each choice to the pydantic-ai docs, and its reviewer can change A to match. It also lists the code A had to add and the library behaviours it works around.
- **Same feature floor.** Both loops have retries with backoff, provider-reported cost, feedback on bad tool arguments, cancel, a step cap, approvals that survive a restart, and crash resume. Each is covered by a test, so neither loop looks small by skipping work.
- **No automatic winner.** The report shows evidence for each side, including scenarios that favour A. People decide.

## Quick start

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and git. These run offline once `uv sync` has fetched the packages; `engine_fit.sh` asks PyPI.

```bash
uv sync --all-extras                                # both loops and the dev tools
uv run pytest                                       # every scenario against every loop
uv run bakeoff scenario --all --impl our,pydantic   # the scenario matrix, one line per failure
./scripts/engine_fit.sh                             # which loop's dependencies fit into the engine
```

To build the report page, collect the metrics first (`--deps` installs each loop's dependencies from PyPI into a fresh environment; about 2 minutes in all), then open `out/report.html` in a browser:

```bash
uv run python -m bakeoff.metrics.collect --deps --bench full   # out/metrics.json
uv run bakeoff report                                          # out/report.html
```

`uv run python -m bakeoff.metrics.bench --quick` runs just the overhead benchmark, the way CI does.

### Live runs

`bakeoff live` sends one prompt to a real model through each loop and prints them side by side. It costs money, so it's never part of the tests. The OpenAI key is read at runtime from an env file, or from `OPENAI_API_KEY`, and never printed or stored. For example, with [tokenstash](https://github.com/kgarg2468/tokenstash) putting the key in `.env.local`:

```bash
tokenstash need OPENAI_API_KEY
uv run bakeoff live --impl our,pydantic --model gpt-6-luna --reasoning none --max-steps 20 \
    --env-file .env.local \
    --prompt "Build a RocketRide pipeline that answers questions from a chat using an LLM, save it as chat.pipe, and validate it."
```

The key only goes over https, and only to `api.openai.com` or a host you name with `--key-host`. Each run writes `out/live/<run_id>/<loop>/`, which `bakeoff report` picks up.

## What's in the repo

<details>
<summary><strong>The layout</strong></summary>

```
src/bakeoff/
  pydantic_version/    A: the loop on pydantic-ai (counted)
  our_version/         B: our loop on raw httpx (counted; files ported from Pi say so at the top)
  shared/              everything else, identical for both
    contract.py        the seam: the Loop protocol, Item and Event types, the rules (read this first)
    runner.py          drives one turn: loop events -> session log -> publish -> git commit
    sessionlog.py      SQLite log of threads, turns, items and events
    workcopy.py        one git working copy per thread; one commit per completed turn
    toolhost.py        tool registry, argument checks, permission checks, tool timing
    permissions.py     allow / ask / deny rules with wildcards (ported from OpenCode)
    tools/             file tools, RocketRide engine tools, load_skill
    engine/            MockEngine over the real RocketRide node catalog; RealEngine (optional)
    skills.py          the RocketRide pipeline skills in the system prompt
    invariants.py      I1, I2, I3, I7 over the wire recordings and the session log (I5 and I6: driver and netguard)
    scenario.py        runs one scripted scenario against the fake server
    netguard.py        the loopback-only network guard (I6)
  fakeprov/            the scripted fake model server, and scenarios/*.json
  metrics/             loc, deps and bench -> out/metrics.json
  report/              builds out/report.html
  loops.py             the loop registry, with each loop's documented failures
  live.py              bakeoff live and chat, against a real endpoint
  cli.py               the bakeoff command
  data/                RocketRide node catalog, example pipes and skills (MIT, see notices)
scripts/engine_fit.sh  does each loop's dependency set fit into the engine?
tests/                 pytest; the scenario matrix runs every scenario against every loop
docs/assets/           the diagrams in this README
out/                   generated runs, metrics and report (gitignored)
```

</details>

## Scenarios

Each scenario is a JSON script: the user's turns, approvals, cancels and crashes, plus what the fake model answers. Every loop plays the same script.

<details>
<summary><strong>All 22 scenarios</strong></summary>

| ID | What happens | Passes when |
| --- | --- | --- |
| S01 | A plain chat; the stream has keep-alive comments and a usage chunk that carries the cost | exact text; usage counted once; cost is the provider's |
| S02 | Build a pipeline: describe a component, write it, validate (a lane error), fix it with an edit, validate again; one call has bad arguments | results match the engine; the bad call gets an error result and the loop carries on; one commit |
| S03 | Three parallel `describe_component` calls with interleaved argument chunks and 300 ms tools | all run, overlapping in time; results in call order |
| S04 | Two turns, then revert turn 1 | the revert is a new commit and a new turn; the next request only appends |
| S05 | One batch: validate (allowed) and write (asks); the process exits and a new process approves | the write runs once; validate doesn't run again; the file is in the commit |
| S06 | Deny with a reason | the tool doesn't run; the model sees the denial; no file |
| S07 | Cancel mid-stream, then during a slow tool | stops within 200 ms; the next turn passes an endpoint that rejects unsigned reasoning |
| S08 | The worker is killed right after a tool result is saved; resume | the tool doesn't run again; the resumed request continues the same history |
| S09 | 429 with `retry-after: 1`, then OK; variant: an error mid-stream | exactly 2 attempts, waiting as told |
| S10a | `reasoning_details` split across chunks, with metadata-only fragments | sent back unchanged in meaning |
| S10b | The same, plus an unknown field | informational only |
| S11 | The model calls tools forever | stops at `max_steps` after exactly that many requests; no orphan calls |
| S12 | OpenRouter cost per step, across two turns | per-step and per-turn totals exact |
| S12b | A BYOK OpenAI-compatible endpoint: usage only when asked for, no cost | usage counted; cost reported as unknown, never guessed |
| S13 | A BYOK endpoint, a non-OpenAI model name, reasoning asked for | the reasoning parameter reaches the wire |
| S14 | A strict BYOK endpoint that answers 400 to `reasoning`, `reasoning_effort` or `stream_options` | the turn finishes |
| S15 | Compaction: the runner adds a summary item | the next request is system prompt, summary, new message |
| R01 | Responses API, `gpt-6-luna` at effort `xhigh`: text only | a stateless request (`store: false`, encrypted reasoning); exact text; usage |
| R02 | Reasoning and one function call, then the answer | the reasoning item is replayed exactly as sent; the tool runs once |
| R03 | Commentary and three calls in one response; the write asks; approve in a new process | as S05, and the resumed request replays reasoning, commentary and all calls |
| R04 | 429, then OK; next turn an error event mid-stream, then OK | waited per `Retry-After`; nothing of the failed attempt is replayed |
| R05 | Cancel while reasoning streams | stops within 200 ms; the next turn passes the encrypted-reasoning check |

Both loops pass R01–R05: B since [#13](https://github.com/kgarg2468/harness-bakeoff/pull/13), A since [#14](https://github.com/kgarg2468/harness-bakeoff/pull/14). A loop's expected failures (none on `main` now) are listed in [`loops.py`](src/bakeoff/loops.py) with the exact checks they fail, so a fix shows up as clearly as a regression.

</details>

## Where each wins

Only what the measurements show, plus a few design properties labelled as such. A timing counts as a win only when it's at least 10% better.

**B, our loop:**

- **Installs less:** 12 packages (3.9 MB) against 35 (35.4 MB).
- **Starts faster:** a cold import takes 71.4 ms against 1.19 s.
- **Adds less on top of the model:** 12.8 ms against 369.8 ms per benchmark turn (p50; p95 18.1 ms against 379.8 ms), or 6.4 against 184.1 µs per streamed chunk. At its peak the process holds 3.3 MB more than before the loop was imported, against 67.4 MB more (import, warm-up and benchmark turns).
- **Starts tools early:** 20 read-only tool runs started while the model was still streaming, against 0, across the 22 scenarios.
- **Reuses connections:** 29 connections against 56 for the same 59 requests, in the 22 scenarios where both loops sent the same number of requests.
- **Less time per step on the Responses API, over 2 runs:** median 6.31 s against 7.19 s with reasoning `xhigh` (10 and 12 steps against 12 and 13). Only 2 runs each, and the time is mostly the model's, so this is a hint, not a settled result.
- **Fits the engine as it moves:** it needs only packages the engine already ships. A can't take pydantic-ai 2.32 or later until the engine's crewai node accepts openai 3.
- **Nothing to work around (by design):** [`A_CHECKLIST.md`](A_CHECKLIST.md) lists the library behaviours A had to work around, such as OpenRouter's string error code and retrying a stream that fails midway.

**A, pydantic-ai:**

- **Less of its own code:** 778 lines against 869.
- **New provider APIs are mostly settings:** Responses API support took +97 lines against +156 for B, 136 of those ported from Pi ([#14](https://github.com/kgarg2468/harness-bakeoff/pull/14) and [#13](https://github.com/kgarg2468/harness-bakeoff/pull/13)). The first cuts were +46 against +140.
- **Features come with the library (by design):** retries, usage limits, approvals (deferred tools), cancellation and the message history format are pydantic-ai's, so fixes and new features arrive with upgrades.
- **Many providers behind one interface (by design):** OpenAI, Anthropic, Gemini and more, should the engine ever need more than OpenAI-compatible endpoints.

**Too close to call:**

- **Scenarios:** 22 / 22 each, R01–R05 (the Responses API) included.
- **Live runs:** all 10 runs wrote `chat.pipe` and validated it with 0 errors. On chat completions, time per step (median 1.49 s against 1.60 s) and input tokens per step (20,475 against 21,433) are within 10%. On the Responses API, input tokens per step are too (24,827 against 25,818).

## More

<details>
<summary><strong>Command overview</strong></summary>

| Command | What it does |
| --- | --- |
| `bakeoff scenario` | Run scenarios (`S01 S05`, or `--all`) against loops (`--impl our,pydantic`) and print the matrix. Exit 1 on any `FAIL` or `XPASS` |
| `bakeoff report` | Build `out/report.html` from `out/runs/latest`, `out/live` and `out/metrics.json` |
| `bakeoff live` | One prompt against a real model, per loop, side by side |
| `bakeoff chat` | A REPL on one loop: approvals prompted, `/revert N`, `/compact TEXT`, `/exit` |
| `bakeoff turn` · `approve` · `deny` · `resume` · `cancel` | Act on one thread of a session log from a separate process; the scenario driver uses these for approval in a new process and for crash resume |
| `bakeoff fakeprov` | Serve the fake model server on 127.0.0.1 (`--port`), or check every scenario file (`--check`) |
| `python -m bakeoff.metrics.collect` | Lines, dependencies (`--deps`) and overhead (`--bench full`) into `out/metrics.json` |
| `python -m bakeoff.metrics.bench` | The harness overhead benchmark alone |
| `python -m bakeoff.metrics.loc` | Lines of code per loop, ported lines counted separately |

**Reading the matrix:** `xfail` is a failure that the loop's entry in [`loops.py`](src/bakeoff/loops.py) documents, with exactly the checks it documents; a documented cell that fails any other way is a plain `FAIL`. `XPASS` means a documented failure is fixed, so its entry must go. A loop that isn't built yet is skipped; one that exists but fails to import is an error, so it never drops out of a comparison unnoticed. A loop that leaves a task running after it was cancelled stops the command after that run (exit 1; `stopped` in `summary.json` names it).

</details>

<details>
<summary><strong>Live runs and chat, in detail</strong></summary>

- **The key:** `--env-file` first, then `OPENAI_API_KEY` in the environment, then the file named by `BAKEOFF_ENV_FILE`. It's held in memory, never printed or recorded, and goes only over https to `api.openai.com` or a host named with `--key-host` on the command line (never to a host that only a session log names). The network guard allows loopback and the endpoint's host, nothing else. Pointed at a loopback address, `live` and `chat` send a dummy key.
- **Options both take:** `--model`, `--reasoning EFFORT`, `--max-steps`, `--max-tokens`, `--base-url` (`{impl}` becomes the loop name), `--kind` (`openai_compat`, the default, or `openrouter`) and `--api responses`.
- **`--api responses`** uses OpenAI's Responses API instead of chat completions: `gpt-6-luna` takes function tools on chat completions only with reasoning `none`, so tools plus reasoning need it. There `--reasoning` also asks for a reasoning summary. Both loops speak it ([#13](https://github.com/kgarg2468/harness-bakeoff/pull/13), [#14](https://github.com/kgarg2468/harness-bakeoff/pull/14)).
- **`live`** streams each loop's run (text inline, each tool call and result on one line), then prints tokens, time to first token, total time and steps side by side. `--interactive` asks before each write. Without it the run is unattended: the system prompt tells the model that nobody can answer the skills' approval gates, so it goes on, while checks such as validation still have to pass. A run that fails, is interrupted or can't start still writes its `result.json`, with the error.
- **`chat`** is a REPL on one loop in which writes ask for approval. Ctrl-C during a turn cancels the turn, and at a prompt it ends the chat. It exits 1 if any turn stopped short (error, `max_steps`, budget, cancelled) or an invariant failed.

</details>

<details>
<summary><strong>What a run writes</strong></summary>

```
out/runs/<run_id>/summary.json                   the matrix, one-line reasons, git sha, loop versions
out/runs/<run_id>/<scenario>/<loop>/result.json  every check and invariant, with details
                                    log.sqlite   the session log
                                    wire/        every request body the fake server received
                                    events.ndjson, wc/ (the git working copy)
out/runs/latest -> <run_id>
out/live/<run_id>/<loop>/result.json             a live run: model, prompt, final answer, stops, steps,
                                                 tokens, latency, tool runs, invariants (I2, I3, I5, I7)
                        log.sqlite, events.ndjson, wc/   as above; no wire/ (live request bodies are not recorded)
out/live/latest -> <run_id>
```

A run id is never reused. Wire recordings (scenario runs only) hold request bodies only, never headers. `out/` is not committed; the live-run figures above are kept in [`docs/results/`](docs/results/) (see [Where these numbers come from](#results-at-a-glance)), and `uv run python scripts/snapshot_live.py OUT.json RUN_ID...` writes the same snapshot for your own runs. It copies the prompt and the final answer as written, with key-shaped strings redacted, so read it before you share it.

</details>

<details>
<summary><strong>Status</strong></summary>

Both loops are complete: all 22 scenarios, chat completions and the Responses API, live runs on `gpt-6-luna`. Each piece landed through a reviewed pull request. A third loop, `hybrid` (our loop on pydantic-ai's model layer), is registered in `loops.py` but not built yet.

</details>

## License

MIT. Ported code (Pi, OpenCode, Cline, gemini-cli) and RocketRide data are credited in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
