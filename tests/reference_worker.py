"""`bakeoff` for a scenario's child processes, with the reference Responses client
(`responses_reference.py`) registered as the loop "reference". A cross-process step (`approve`
with `new_process`, a crash turn) then loads it through the registry by the thread's impl,
exactly as it loads a real loop. Use: `run_scenario(..., worker=[<path of this file>])`.
"""

from bakeoff import cli, loops

# Run as a script, so this file's directory (tests/) is on sys.path and the target imports.
loops.REGISTRY["reference"] = loops.LoopEntry(
    "reference", "responses_reference:ReferenceLoop", ("httpx",)
)

if __name__ == "__main__":
    raise SystemExit(cli.main())
