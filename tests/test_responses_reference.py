"""R01-R05 can pass: a minimal correct Responses client (`responses_reference.py`) passes each
one through the real scenario driver, with every final `expect` key and invariant."""

import json
from pathlib import Path

import pytest
from responses_reference import ReferenceLoop

from bakeoff.fakeprov.script import SCENARIOS_DIR
from bakeoff.fakeprov.server import FakeProvider
from bakeoff.shared.scenario import run_scenario


@pytest.mark.parametrize("sid", ["R01", "R02", "R03", "R04", "R05"])
async def test_a_correct_responses_client_passes(sid: str, tmp_path: Path) -> None:
    data = json.loads((SCENARIOS_DIR / f"{sid}.json").read_text())
    for step in data["driver"]:
        if step.get("new_process"):  # a child process loads its loop by name, not this one
            step["new_process"] = False
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / f"{sid}.json").write_text(json.dumps(data))
    with FakeProvider(scenarios, tmp_path / "wire") as provider:
        result = await run_scenario(
            sid, "our", out=tmp_path / "out", run_id="ref", provider=provider,
            loop_factory=ReferenceLoop,
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
            "R02", "our", out=tmp_path / "out", run_id="ref", provider=provider,
            loop_factory=ReferenceLoop,
        )  # fmt: skip
    assert result["stops"] == ["error"]
    wire = tmp_path / "out" / "runs" / "ref" / "R02" / "our" / "wire"
    meta = json.loads((wire / "002.meta.json").read_text())
    assert meta["status"] == 404
    assert meta["error"].startswith("Item with id 'rs_R02_1' not found.")
