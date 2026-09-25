# Fairness rules

1. **One seam.** Only `src/bakeoff/our_version/` and `src/bakeoff/pydantic_version/` (and
   `hybrid_version/`) are counted. Everything else is shared and identical for both.
2. **Same spec, same tests.** Both loops implement `shared/contract.py` and run the same scenario
   scripts against the same fake server. Pass/fail is judged only from wire recordings and the
   session log.
3. **A is used the recommended way.** See `A_CHECKLIST.md`. Its reviewer can change it.
4. **Same feature floor.** Both have retries with backoff, provider-reported cost, bad-argument
   feedback, cancel, a step cap, approvals that survive a restart, and crash resume. Each is
   covered by a test, so B can't look small by skipping work.
5. **Scenarios that favour A are included** (cancel repair, built-in features), and the report
   has a "where A wins" section.
6. **Predictions first.** `PREDICTIONS.md` was committed before either loop was written.
7. **No automatic winner.** The report shows evidence; people decide.
8. **Ported code is labelled.** Lines in B that came from Pi or OpenCode are marked and counted
   separately in the report.
