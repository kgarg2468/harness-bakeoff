import json

import pytest

from bakeoff.shared.engine.mock import DEFAULT_CATALOG, MockEngine
from bakeoff.shared.toolhost import MAX_OUTPUT
from bakeoff.shared.tools import PathError, ToolContext, ToolError
from bakeoff.shared.tools.engine_tools import (
    ENGINE_TOOLS,
    describe_component,
    list_components,
    validate_pipeline,
)

VALID = {
    "components": [
        {"id": "chat_1", "provider": "chat", "config": {}},
        {
            "id": "out_1",
            "provider": "response_questions",
            "config": {},
            "input": [{"lane": "questions", "from": "chat_1"}],
        },
    ]
}
LANE_ERROR = {
    "components": [
        VALID["components"][0],
        {**VALID["components"][1], "input": [{"lane": "text", "from": "chat_1"}]},
    ]
}


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workdir=tmp_path.resolve(), engine=MockEngine())


def test_specs_follow_the_engine_mcp_registry():
    specs = {tool.spec.name: tool.spec for tool in ENGINE_TOOLS}
    assert list(specs) == ["list_components", "describe_component", "validate_pipeline"]
    assert all(spec.read_only for spec in specs.values())
    assert specs["describe_component"].parameters["required"] == ["name"]
    assert specs["describe_component"].description == (
        "Describe a single RocketRide component: full metadata, lanes, and config schema."
    )
    validate = specs["validate_pipeline"].parameters
    assert set(validate["properties"]) == {"pipeline", "path"}
    # Providers reject top-level combinators in tool schemas.
    assert not {"oneOf", "anyOf", "allOf", "required"} & set(validate)


async def test_list_components_fits_the_output_cap(ctx):
    out = await list_components({}, ctx)
    assert len(out) < MAX_OUTPUT
    data = json.loads(out)
    catalog = json.loads(DEFAULT_CATALOG.read_text())["providers"]
    assert data["ok"] is True
    assert [c["name"] for c in data["components"]] == list(catalog)
    chat = next(c for c in data["components"] if c["name"] == "chat")
    assert chat["category"] == catalog["chat"]["classType"]
    assert catalog["chat"]["description"].startswith(chat["summary"])


async def test_describe_component(ctx):
    data = json.loads(await describe_component({"name": "llm_openai"}, ctx))
    assert data["ok"] is True and data["name"] == "llm_openai"
    assert data["lanes"] == {"questions": ["answers"]}
    assert data["fields"]


async def test_describe_unknown_component(ctx):
    with pytest.raises(ToolError, match="Unknown component: nope"):
        await describe_component({"name": "nope"}, ctx)


async def test_validate_inline(ctx):
    assert json.loads(await validate_pipeline({"pipeline": VALID}, ctx)) == {
        "ok": True,
        "errors": [],
        "warnings": [],
    }
    result = json.loads(await validate_pipeline({"pipeline": LANE_ERROR}, ctx))
    assert result["ok"] is False
    assert "input lane text" in result["errors"][0]


async def test_validate_path_matches_inline(ctx):
    (ctx.workdir / "p.pipe").write_text(json.dumps(LANE_ERROR))
    by_path = await validate_pipeline({"path": "p.pipe"}, ctx)
    assert by_path == await validate_pipeline({"pipeline": LANE_ERROR}, ctx)


def test_validate_needs_exactly_one_source():
    check = ENGINE_TOOLS[2].check_args
    assert check is not None
    for bad in ({}, {"pipeline": VALID, "path": "p.pipe"}):
        assert check(bad) == "pass exactly one of 'pipeline' or 'path'"
    assert check({"pipeline": VALID}) is None
    assert check({"path": "p.pipe"}) is None


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "No pipeline file at p.pipe"), ("{nope", "not valid JSON"), ("[1]", "JSON object")],
)
async def test_validate_path_errors(ctx, content, message):
    if content is not None:
        (ctx.workdir / "p.pipe").write_text(content)
    with pytest.raises(ToolError, match=message):
        await validate_pipeline({"path": "p.pipe"}, ctx)


async def test_validate_path_cannot_escape(ctx):
    with pytest.raises(PathError):
        await validate_pipeline({"path": "../p.pipe"}, ctx)
