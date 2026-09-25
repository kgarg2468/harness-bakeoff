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
uv run pytest            # every scenario against every loop, fully offline
./scripts/engine_fit.sh  # which loop's dependencies fit into the RocketRide engine
```

More commands (`bakeoff scenario`, `bakeoff chat --live`, `bakeoff report`, `bakeoff bench`)
arrive as the pieces land; see `DESIGN.md`.

## Status

Work in progress. Each piece lands through a reviewed pull request.

## License

MIT. Ported code is credited in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
