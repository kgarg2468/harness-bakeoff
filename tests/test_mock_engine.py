import copy
import json
import re
import sys
import time

import pytest

from bakeoff.shared.engine.base import envelope
from bakeoff.shared.engine.mock import DEFAULT_CATALOG, MockEngine
from bakeoff.shared.engine.real import RealEngine

EXAMPLES = DEFAULT_CATALOG.parent / "examples"
INCORRECT = EXAMPLES / "incorrect"
# Engine rules MockEngine does not model; their examples validate clean here.
OWNERSHIP_SKIP = (
    "MockEngine does not model the engine's sub-pipeline ownership rules "
    "(stack.cpp computeLifecycleRegions, checked when a pipeline is opened)"
)
UNSUPPORTED = {
    "incorrect-second-start-feeds-subpipe.pipe",
    "incorrect-second-start-feeds-subpipe-node.pipe",
    "incorrect-subpipe-merges-into-main.pipe",
    "incorrect-shared-subpipe-node.pipe",
}

BASE = {
    "source": "chat_1",
    "components": [
        {"id": "chat_1", "provider": "chat", "config": {}},
        {
            "id": "llm_1",
            "provider": "llm_openai",
            "config": {},
            "input": [{"lane": "questions", "from": "chat_1"}],
        },
        {
            "id": "out_1",
            "provider": "response_answers",
            "config": {},
            "input": [{"lane": "answers", "from": "llm_1"}],
        },
    ],
}


@pytest.fixture
def engine():
    return MockEngine()


def _readme_errors() -> dict[str, str]:
    """File -> verbatim engine error, from the README table (rows with a quoted message)."""
    errors = {}
    for line in (INCORRECT / "README.md").read_text().splitlines():
        row = re.match(r"\| `([^`]+\.pipe)` \|.*\| `([^`]+)` \|$", line)
        if row:
            errors[row[1]] = row[2]
    return errors


async def test_catalog_has_real_nodes(engine):
    services = await engine.get_services()
    assert len(services) > 150
    assert {"chat", "llm_openai", "response_answers", "tool_pipe"} <= services.keys()
    llm = await engine.get_service("llm_openai")
    assert set(llm) == {"title", "classType", "lanes", "description", "fields"}
    assert llm["lanes"] == {"questions": ["answers"]}
    assert await engine.get_service("no_such_node") is None


def test_catalog_source_and_size():
    catalog = json.loads(DEFAULT_CATALOG.read_text())
    assert catalog["source"]["repo"] == "rocketride-org/rocketride-server"
    assert re.fullmatch(r"[0-9a-f]{40}", catalog["source"]["commit"])
    assert DEFAULT_CATALOG.stat().st_size < 1_500_000


async def test_delay_ms_sleeps_before_answering():
    engine = MockEngine(delay_ms=50)
    start = time.perf_counter()
    await engine.get_service("chat")
    assert time.perf_counter() - start >= 0.05


async def test_catalog_is_read_on_first_use(tmp_path):
    engine = MockEngine(catalog_path=tmp_path / "missing.json")  # a ToolHost can still be built
    with pytest.raises(FileNotFoundError):
        await engine.get_services()


async def test_custom_catalog_path(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"providers": {"only": {"lanes": {}}}}))
    assert list(await MockEngine(catalog_path=path).get_services()) == ["only"]


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.pipe")), ids=lambda p: p.name)
async def test_valid_examples_validate_ok(engine, path):
    result = await engine.validate(json.loads(path.read_text()))
    assert result == {"ok": True, "errors": [], "warnings": []}


@pytest.mark.parametrize("path", sorted(INCORRECT.glob("*.pipe")), ids=lambda p: p.name)
async def test_incorrect_examples_report_the_readme_error(engine, path):
    if path.name in UNSUPPORTED:
        pytest.skip(OWNERSHIP_SKIP)
    expected = _readme_errors()[path.name]
    result = await engine.validate(json.loads(path.read_text()))
    assert result["ok"] is False
    assert any(expected in error for error in result["errors"]), result["errors"]


async def test_base_pipeline_is_valid(engine):
    assert (await engine.validate(BASE))["ok"]
    assert (await engine.validate({"pipeline": BASE}))["ok"]


def _llm(p):
    return p["components"][1]


# Messages are the engine's own (engLib/test/store/pipeline/pipeline_config.cpp).
STRUCTURE_CASES = [
    ("version not a number", lambda p: p.update(version="one"), "'pipeline.version' must be a number"),
    ("version not integral", lambda p: p.update(version=1.5), "'pipeline.version' must be a number"),
    ("version beyond int32", lambda p: p.update(version=2**31), "'pipeline.version' must be a number"),
    ("version too low", lambda p: p.update(version=0), "'pipeline.version' is unsupported"),
    ("version too high", lambda p: p.update(version=3), "'pipeline.version' is unsupported"),
    ("components missing", lambda p: p.pop("components"), "'pipeline.components' must be an array"),
    ("components invalid", lambda p: p.update(components=42), "'pipeline.components' must be an array"),
    ("component invalid", lambda p: p["components"].__setitem__(1, 42), "Component must be an object"),
    ("id missing", lambda p: _llm(p).pop("id"), "Component 'id' must be a non-empty string"),
    ("id empty", lambda p: _llm(p).update(id=""), "Component 'id' must be a non-empty string"),
    ("provider invalid", lambda p: _llm(p).update(provider=42), "Component llm_1 'provider' must be a non-empty string"),
    ("config missing", lambda p: _llm(p).pop("config"), "Component llm_1 missing 'config' object"),
    ("profile invalid", lambda p: _llm(p).update(config={"profile": 42}), "Component llm_1 config 'profile' must be a non-empty string"),
    ("profile section invalid", lambda p: _llm(p).update(config={"profile": "p", "p": 1}), "Component llm_1 config missing profile object 'p'"),
    ("parameters invalid", lambda p: _llm(p).update(config={"parameters": 1}), "Component llm_1 config 'parameters' must be an object"),
    (
        "secure parameters invalid",
        lambda p: _llm(p).update(config={"parameters": {"secureParameters": 1}}),
        "Component llm_1 config 'parameters': 'secureParameters' must be an object",
    ),
    (
        "secure token missing",
        lambda p: _llm(p).update(config={"profile": "p", "p": {"secureParameters": {"secure": []}}}),
        "Component llm_1 config profile 'p': 'secureParameters' must have a 'token' string",
    ),
    ("duplicate id", lambda p: p["components"].append({"id": "chat_1", "provider": "chat", "config": {}}), "Duplicate component chat_1"),
    ("unknown source", lambda p: p.update(source="unknown_source"), "'pipeline.source' references unknown component id: unknown_source"),
    ("source a number", lambda p: p.update(source=5), "'pipeline.source' must be a non-empty string"),
    ("input invalid", lambda p: _llm(p).update(input=42), "Component llm_1 input must be an array"),
    ("input entry invalid", lambda p: _llm(p).update(input=[42]), "Component llm_1 input entries must be objects"),
    ("lane missing", lambda p: _llm(p)["input"][0].pop("lane"), "Component llm_1 input 'lane' must be a non-empty string"),
    ("lane unknown", lambda p: _llm(p)["input"][0].update(lane="bogus"), "Component llm_1 input has unknown lane bogus"),
    ("from missing", lambda p: _llm(p)["input"][0].pop("from"), "Component llm_1 input 'from' must be a non-empty string"),
    ("from unknown", lambda p: _llm(p)["input"][0].update({"from": "nope"}), "Component llm_1 input references unknown component id: nope"),
    ("control invalid", lambda p: _llm(p).update(control={}), "Component llm_1 control must be an array"),
    ("control classType missing", lambda p: _llm(p).update(control=[{"from": "chat_1"}]), "Component llm_1 control 'classType' must be a non-empty string"),
    ("control from unknown", lambda p: _llm(p).update(control=[{"classType": "llm", "from": "x"}]), "Component llm_1 control references unknown component id: x"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("mutate", "message"), [c[1:] for c in STRUCTURE_CASES], ids=[c[0] for c in STRUCTURE_CASES]
)
async def test_structural_rules(engine, mutate, message):
    pipeline = copy.deepcopy(BASE)
    mutate(pipeline)
    result = await engine.validate(pipeline)
    assert result["ok"] is False
    assert result["errors"][0] == message


async def test_only_the_first_structural_error_is_reported(engine):
    pipeline = copy.deepcopy(BASE)
    pipeline["components"][1]["input"] = 42
    pipeline["components"][2]["input"] = 42
    assert (await engine.validate(pipeline))["errors"] == ["Component llm_1 input must be an array"]


async def test_non_object_pipeline(engine):
    assert (await engine.validate({"pipeline": 42}))["errors"] == [
        "'pipeline' is missing or invalid"
    ]


async def test_source_is_optional(engine):
    pipeline = copy.deepcopy(BASE)
    del pipeline["source"]
    assert (await engine.validate(pipeline))["ok"]


# JsonCpp's isInt() takes an integral real; a source that does not convert to text is no source.
@pytest.mark.parametrize(
    "change", [{"version": 1.0}, {"version": 2.0}, {"source": None}, {"source": True}]
)
async def test_what_the_engine_lets_through(engine, change):
    assert (await engine.validate({**BASE, **change}))["ok"]


async def test_unknown_provider(engine):
    pipeline = copy.deepcopy(BASE)
    pipeline["components"][1]["provider"] = "llm_nope"
    result = await engine.validate(pipeline)
    # Edges touching the unknown component get no lane errors.
    assert result["errors"] == ["Component llm_1: unknown provider 'llm_nope'"]


async def test_unknown_provider_is_reported_with_a_structural_error(engine):
    pipeline = copy.deepcopy(BASE)
    pipeline["components"][2]["provider"] = "nope"
    pipeline["source"] = "missing"
    assert (await engine.validate(pipeline))["errors"] == [
        "'pipeline.source' references unknown component id: missing",
        "Component out_1: unknown provider 'nope'",
    ]


async def test_lane_not_accepted_by_target(engine):
    pipeline = copy.deepcopy(BASE)
    pipeline["components"][1]["input"][0]["lane"] = "text"
    assert (await engine.validate(pipeline))["errors"] == [
        "Component llm_1 input lane text not found in service definition "
        "(llm_openai accepts: questions)"
    ]


async def test_lane_not_produced_by_source(engine):
    pipeline = copy.deepcopy(BASE)
    pipeline["components"][2]["input"][0]["from"] = "chat_1"
    assert (await engine.validate(pipeline))["errors"] == [
        "Component out_1 input lane answers is not produced by component chat_1 "
        "(chat produces: questions)"
    ]


def test_envelope_adds_version_and_unwraps():
    assert envelope({"components": []}) == {"pipeline": {"components": [], "version": 1}}
    assert envelope({"pipeline": {"version": 2}}) == {"pipeline": {"version": 2}}


def test_real_engine_requires_uri():
    with pytest.raises(ValueError, match="explicit engine uri"):
        RealEngine("")


def test_real_engine_import_error_is_clear(monkeypatch):
    monkeypatch.setitem(sys.modules, "rocketride", None)
    with pytest.raises(ImportError, match="rocketride"):
        RealEngine("ws://127.0.0.1:5565")
