# Predictions (written before either loop exists)

Krish's hunch, recorded 2026-09-25 before any loop code was written:

- **B (our loop) is small.** Rocket Agent is not a general coding agent. It builds pipelines and
  apps through a handful of tools, so the loop should come in well under Pi's ~2k-line agent core.
  Guess: B ≤ 1,200 code lines including the ported provider quirks.
- **B adds no dependencies** to the engine (only `httpx`, which the engine already ships).
- **Harness overhead is negligible for B**: almost all turn latency is the model's. The loop itself
  adds well under a millisecond per step.
- **A is shorter in our own code** but depends on far more library code. It also cannot use the
  newest pydantic-ai inside the engine without resolving the `openai` v2/v3 conflict.

Context: Dylan estimated the *whole* harness (sessions, storage, compaction, tools, UI protocol,
tests) at 25–30k lines. That is a different number from the loop + provider layer measured here.
