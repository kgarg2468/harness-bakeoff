"""The scenario matrix: every scenario file against every loop that imports here.

A loop's documented failures (`loops.REGISTRY[...].known_failures`) are strict xfails, so the
matrix records what each loop really does: a fix turns the xfail into a failure (XPASS) until the
entry is removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bakeoff import loops
from bakeoff.fakeprov.server import FakeProvider
from bakeoff.shared.scenario import reason, run_scenario, scenario_ids

LOOPS = list(loops.available())


def cases() -> list[object]:
    params = []
    for sid in scenario_ids():
        for impl in LOOPS:
            why = loops.known_failure(impl, sid)
            marks = [pytest.mark.xfail(strict=True, reason=why)] if why else []
            params.append(pytest.param(sid, impl, id=f"{sid}-{impl}", marks=marks))
    return params


@pytest.fixture(scope="session")
def provider(tmp_path_factory: pytest.TempPathFactory):
    # One server for the whole matrix: every (scenario, run, impl) has its own cursor.
    with FakeProvider(wire_dir=tmp_path_factory.mktemp("wire")) as served:
        yield served


@pytest.fixture(scope="session")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("out")


@pytest.mark.parametrize(("sid", "impl"), cases())
async def test_scenario(sid: str, impl: str, provider: FakeProvider, out: Path) -> None:
    result = await run_scenario(sid, impl, out=out, run_id="matrix", provider=provider)
    failed = [
        f"  {key}: {check['detail']}"
        for group in ("expect", "invariants")
        for key, check in result[group].items()
        if not check["ok"]
    ]
    error = [f"  error: {result['error']}"] if result["error"] else []
    assert result["passed"], "\n".join([f"{sid}/{impl}: {reason(result)}", *error, *failed])


def test_our_loop_is_in_the_matrix() -> None:
    assert "our" in LOOPS


def test_no_registered_loop_is_broken() -> None:
    # A loop that is not built yet is skipped; one that exists but fails to import is a bug.
    broken = {name: exc.reason for name, exc in loops.unavailable().items() if not exc.missing}
    assert not broken


def test_known_failures_name_real_scenarios() -> None:
    known = set(scenario_ids())
    for entry in loops.REGISTRY.values():
        assert set(entry.known_failures) <= known, entry.name
