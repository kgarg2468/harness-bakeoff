"""R01-R05 can pass: a minimal correct Responses client (`responses_reference.py`) passes each
one through the real scenario driver, with every final `expect` key and invariant. R03 approves
in a new process, as scripted: the child loads the client through `reference_worker.py`."""

import json
from pathlib import Path

import pytest
from responses_reference import ReferenceLoop

from bakeoff.fakeprov.server import FakeProvider
from bakeoff.shared.scenario import run_scenario

WORKER = [str(Path(__file__).with_name("reference_worker.py"))]


@pytest.mark.parametrize("sid", ["R01", "R02", "R03", "R04", "R05"])
async def test_a_correct_responses_client_passes(sid: str, tmp_path: Path) -> None:
    with FakeProvider(wire_dir=tmp_path / "wire") as provider:
        result = await run_scenario(
            sid, "reference", out=tmp_path / "out", run_id="ref", provider=provider,
            loop_factory=ReferenceLoop, worker=WORKER,
        )  # fmt: skip
    failed = [
        f"{key}: {check['detail']}"
        for group in ("expect", "invariants")
        for key, check in result[group].items()
        if not check["ok"]
    ]
    assert result["passed"], (result["error"], failed)
    i1 = result["invariants"]["I1"]["info"]
    assert i1["apis"] == ["responses"] and i1["byte_prefix"]
    # R03's approval resumed the paused turn in a child process, from the session log alone.
    children = [(p["command"], p["exit"], p["summary"]["stop"]) for p in result["processes"]]
    assert children == ([("approve", 0, "end_turn")] if sid == "R03" else [])


async def test_a_client_that_drops_encrypted_reasoning_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strict mode answers as the API does with store false, so the turn errors."""
    import responses_reference

    keep = responses_reference._input

    def without_encrypted_content(item):  # type: ignore[no-untyped-def]
        return [
            {k: v for k, v in entry.items() if k != "encrypted_content"} for entry in keep(item)
        ]

    monkeypatch.setattr(responses_reference, "_input", without_encrypted_content)
    with FakeProvider(wire_dir=tmp_path / "wire") as provider:
        result = await run_scenario(
            "R02", "reference", out=tmp_path / "out", run_id="ref", provider=provider,
            loop_factory=ReferenceLoop,
        )  # fmt: skip
    assert result["stops"] == ["error"]
    wire = tmp_path / "out" / "runs" / "ref" / "R02" / "reference" / "wire"
    meta = json.loads((wire / "002.meta.json").read_text())
    assert meta["status"] == 404
    assert meta["error"].startswith("Item with id 'rs_R02_1' not found.")
