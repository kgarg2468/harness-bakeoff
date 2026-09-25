"""RocketRide skills: the bundled data, frontmatter parsing, the prompt section, `load_skill`."""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

from bakeoff.shared import skills
from bakeoff.shared.contract import ToolCall
from bakeoff.shared.skills import SKILLS_DIR, load_skills, parse_frontmatter, skills_prompt
from bakeoff.shared.toolhost import MAX_OUTPUT, build_toolhost

NAMES = [
    "rocketride-building-pipelines",
    "rocketride-configuring-pipelines",
    "rocketride-debugging-pipelines",
    "rocketride-designing-pipelines",
    "rocketride-running-pipelines",
]
FIRST_SEVEN = [
    "list_components",
    "describe_component",
    "validate_pipeline",
    "list_files",
    "read_file",
    "write_file",
    "edit_file",
]
BUILDING = "rocketride-building-pipelines"
CONFIGURING = "rocketride-configuring-pipelines"
DESIGNING = "rocketride-designing-pipelines"


def call(args, call_id="c1"):
    return ToolCall(id=call_id, name="load_skill", arguments=json.dumps(args))


@pytest.fixture
def host(tmp_path):
    return build_toolhost(tmp_path, {"*": "allow"}, lambda event: None)


# --- frontmatter ------------------------------------------------------------------------------


def test_frontmatter_basic():
    text = "---\nname: x\ndescription: Use when: things happen\n---\n# Body\n"
    assert parse_frontmatter(text) == (
        {"name": "x", "description": "Use when: things happen"},
        "# Body\n",
    )


def test_frontmatter_quotes_folding_comments_and_crlf():
    text = (
        "﻿---\r\n"
        "# a comment\r\n"
        "name: 'it''s'\r\n"
        'title: "a: b"\r\n'
        "description: >\r\n"
        "  first line\r\n"
        "  second line\r\n"
        "plain: one\r\n"
        "   two\r\n"
        "---\r\n"
        "body"
    )
    fields, body = parse_frontmatter(text)
    assert fields == {
        "name": "it's",
        "title": "a: b",
        "description": "first line second line",
        "plain": "one two",
    }
    assert body == "body"


@pytest.mark.parametrize("text", ["# no frontmatter\n", "---\nname: x\n", ""])
def test_no_or_unclosed_frontmatter(text):
    assert parse_frontmatter(text) == ({}, text)


def test_frontmatter_rejects_non_key_lines():
    with pytest.raises(ValueError, match="key: value"):
        parse_frontmatter("---\njust text\n---\n")


# --- the bundled data -------------------------------------------------------------------------


def test_bundle_has_the_five_skills():
    bundle = load_skills()
    assert list(bundle.skills) == NAMES
    for skill in bundle.skills.values():
        meta, _ = parse_frontmatter((SKILLS_DIR / skill.name / "SKILL.md").read_text())
        assert skill.description == meta["description"] and skill.description.startswith("Use")
        assert skill.files[0] == "SKILL.md"
        assert skill.files[-2:] == ("../MCP_TOOL_CONTRACT.md", "../README.md")
    assert "LAYER1_NODE_INDEX.json" in bundle.skills["rocketride-designing-pipelines"].files


def test_bundle_is_text_only_and_records_its_source():
    files = [p for p in SKILLS_DIR.rglob("*") if p.is_file()]
    assert files and all(p.suffix in {".md", ".json", ".pipe"} for p in files)
    source = json.loads((SKILLS_DIR / "SOURCE.json").read_text())
    assert source["repo"] == "rocketride-org/rocketride-server"
    assert source["path"] == "docs/agents/skills"
    assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    assert "SOURCE.json" not in load_skills().paths  # provenance, not skill content


def test_skill_name_must_match_its_folder(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "SKILL.md").write_text("---\nname: b\ndescription: d\n---\n")
    with pytest.raises(ValueError, match="name: a"):
        load_skills(tmp_path)


def test_folders_without_skill_md_are_not_skills(tmp_path):
    (tmp_path / "a" / "ref").mkdir(parents=True)
    (tmp_path / "a" / "SKILL.md").write_text("---\nname: a\ndescription: Use for A.\n---\n")
    (tmp_path / "a" / "ref" / "x.md").write_text("x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "y.json").write_text("{}")
    (tmp_path / "SHARED.md").write_text("s")
    bundle = load_skills(tmp_path)
    assert list(bundle.skills) == ["a"]
    assert bundle.skills["a"].files == ("SKILL.md", "ref/x.md", "../SHARED.md")


def test_stray_files_are_not_skill_files(tmp_path):
    (tmp_path / "a" / "tools").mkdir(parents=True)
    (tmp_path / "a" / "SKILL.md").write_text("---\nname: a\ndescription: Use for A.\n---\n")
    (tmp_path / "a" / "tools" / "run.py").write_text("print()")
    (tmp_path / "a" / ".SKILL.md.swp").write_bytes(b"\xff")
    (tmp_path / "a" / "SKILL.md~").write_text("backup")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "x.md").write_text("x")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\xff")
    bundle = load_skills(tmp_path)
    assert bundle.paths == {"a/SKILL.md"}
    assert bundle.skills["a"].files == ("SKILL.md",)


# --- the prompt -------------------------------------------------------------------------------


def test_prompt_lists_names_and_descriptions_only():
    prompt = skills_prompt()
    assert prompt.startswith("## Skills\n")
    assert "call load_skill" in prompt
    entries = [line for line in prompt.splitlines() if line.startswith("- ")]
    assert entries == [f"- {s.name}: {s.description}" for s in load_skills().skills.values()]
    # Progressive disclosure: no skill bodies or reference files in the prompt.
    assert "IRON LAW" not in prompt and "GATE_PROTOCOL" not in prompt
    assert len(prompt) < 2_000
    assert skills_prompt() == prompt  # frozen system prompt: deterministic


def test_no_skills_no_prompt_section(tmp_path):
    assert skills_prompt(tmp_path) == ""


# --- load_skill through the real ToolHost -----------------------------------------------------


def test_tool_order_is_unchanged_and_load_skill_is_last(host):
    specs = host.specs()
    assert [s.name for s in specs[:7]] == FIRST_SEVEN
    assert specs[7].name == "load_skill" and specs[7].read_only
    assert len(specs) == 8


def test_tool_description_names_the_skills(host):
    """Without `skills_prompt()` in the system prompt the model still sees the names."""
    description = host.specs()[7].description
    assert "system prompt" not in description
    assert f"Skills: {', '.join(NAMES)}." in description


async def test_load_skill_returns_note_then_skill_md(host):
    result = await host.run(call({"name": BUILDING}))
    assert result.ok and result.error is None
    note, _, rest = result.content.partition("\n\n--- ")
    assert note.startswith("[harness] ") and "\n" not in note
    for spec in host.specs():  # the real tool names, all of them
        assert spec.name in note
    assert "does not apply here" in note
    assert "Files of rocketride-building-pipelines: SKILL.md, GATE_PROTOCOL.md" in note
    skill_md = (SKILLS_DIR / BUILDING / "SKILL.md").read_text()
    assert rest == f"{BUILDING}/SKILL.md ---\n{skill_md}"
    assert host.run_counts == {"c1": 1}


async def test_null_file_is_skill_md(host):
    result = await host.run(call({"name": BUILDING, "file": None}))
    assert result.ok
    assert result.content == (await host.run(call({"name": BUILDING}, "c2"))).content


@pytest.mark.parametrize(
    ("name", "file", "path"),
    [
        (BUILDING, "GATE_PROTOCOL.md", f"{BUILDING}/GATE_PROTOCOL.md"),
        (BUILDING, "SKILL.md", f"{BUILDING}/SKILL.md"),
        (BUILDING, "./pipeline-patterns.md", f"{BUILDING}/pipeline-patterns.md"),
        # Shared and cross-skill links, as the skill texts write them.
        (BUILDING, "../MCP_TOOL_CONTRACT.md", "MCP_TOOL_CONTRACT.md"),
        (
            "rocketride-running-pipelines",
            "../rocketride-building-pipelines/GATE_PROTOCOL.md",
            f"{BUILDING}/GATE_PROTOCOL.md",
        ),
        # Relative to the skills root, as the building skill names the node index.
        (
            BUILDING,
            "rocketride-designing-pipelines/LAYER1_NODE_INDEX.json",
            "rocketride-designing-pipelines/LAYER1_NODE_INDEX.json",
        ),
        (
            "rocketride-designing-pipelines",
            "examples/simple-chat-rag.pipe",
            "rocketride-designing-pipelines/examples/simple-chat-rag.pipe",
        ),
        # Bare file names, as the skills write them for files in another skill or a subfolder.
        (BUILDING, "PIPELINE_ANTIPATTERNS.md", f"{CONFIGURING}/PIPELINE_ANTIPATTERNS.md"),
        (BUILDING, "PIPELINE_RULES_SUMMARY.md", f"{DESIGNING}/PIPELINE_RULES_SUMMARY.md"),
        (BUILDING, "LAYER1_NODE_INDEX.json", f"{DESIGNING}/LAYER1_NODE_INDEX.json"),
        (
            "rocketride-debugging-pipelines",
            "PIPELINE_ANTIPATTERNS.md",
            f"{CONFIGURING}/PIPELINE_ANTIPATTERNS.md",
        ),
        (DESIGNING, "FAILURE_SCENARIOS.md", f"{DESIGNING}/examples/FAILURE_SCENARIOS.md"),
        (DESIGNING, "agentic-chat.pipe", f"{DESIGNING}/examples/agentic-chat.pipe"),
        # Bare names that exist in several places keep their direct meaning.
        (DESIGNING, "README.md", "README.md"),
        (DESIGNING, "SKILL.md", f"{DESIGNING}/SKILL.md"),
    ],
)
async def test_load_skill_reference_files(host, name, file, path):
    result = await host.run(call({"name": name, "file": file}))
    assert result.ok
    assert result.content.endswith(f"\n\n--- {path} ---\n{(SKILLS_DIR / path).read_text()}")
    assert f"Files of {name}: " in result.content


def test_every_bundle_file_a_skill_mentions_loads_by_that_name():
    """Guards the next sync: a new duplicate file name would break the skills' bare references."""
    bundle = load_skills()
    names = {path.rpartition("/")[2] for path in bundle.paths}
    checked = 0
    for path in sorted(bundle.paths):
        skill = path.partition("/")[0]
        if skill not in bundle.skills:
            continue  # the shared files at the top belong to no skill
        text = (SKILLS_DIR / path).read_text()
        for mentioned in sorted(
            n for n in names if re.search(rf"(?<![\w./-]){re.escape(n)}", text)
        ):
            skills.resolve(skill, mentioned)
            checked += 1
    assert checked > 10  # 17 at the synced commit: the scan itself still finds references


async def test_largest_file_is_not_truncated(host):
    largest = max((p for p in SKILLS_DIR.rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    skill, _, file = largest.relative_to(SKILLS_DIR).as_posix().partition("/")
    result = await host.run(call({"name": skill, "file": file}))
    assert result.ok and len(result.content) <= MAX_OUTPUT
    assert result.content.endswith(largest.read_text())


async def test_unknown_skill_lists_the_valid_names(host):
    bad = call({"name": "rocketride-flying-pipelines"})
    assert host.check(bad) == "allow"  # run() rejects it without executing; the model retries
    result = await host.run(bad)
    assert (result.ok, result.error) == (False, "invalid_args")
    assert result.content == (
        "Invalid arguments for load_skill: Unknown skill: 'rocketride-flying-pipelines'. "
        f"Skills: {', '.join(NAMES)}"
    )
    assert host.run_counts == {}


@pytest.mark.parametrize(
    "file",
    [
        "NOPE.md",
        "../../catalog.json",  # data/ next to the bundle
        "../../../shared/skills.py",
        "../../../../../../../../etc/passwd",
        "/etc/passwd",
        str(SKILLS_DIR / BUILDING / "SKILL.md"),  # absolute, even inside the bundle
        "tools/fetch-doc.py",  # upstream scripts are not copied
        "../SOURCE.json",
        "..",
        ".",
        "",
        "GATE_PROTOCOL.md\x00",
    ],
)
async def test_unknown_or_escaping_files_are_invalid_args(host, file):
    result = await host.run(call({"name": BUILDING, "file": file}))
    assert (result.ok, result.error) == (False, "invalid_args")
    assert result.content.startswith("Invalid arguments for load_skill: Unknown file ")
    assert result.content.endswith(
        f" in skill {BUILDING!r}. Files: SKILL.md, GATE_PROTOCOL.md, ROCKETRIDE_DOC_MAP.md, "
        "pipeline-patterns.md, ../MCP_TOOL_CONTRACT.md, ../README.md"
    )
    assert host.run_counts == {}


async def test_unknown_file_message_echoes_the_value(host):
    result = await host.run(call({"name": BUILDING, "file": "NOPE.md"}))
    assert result.content == (
        f"Invalid arguments for load_skill: Unknown file 'NOPE.md' in skill {BUILDING!r}. "
        "Files: SKILL.md, GATE_PROTOCOL.md, ROCKETRIDE_DOC_MAP.md, pipeline-patterns.md, "
        "../MCP_TOOL_CONTRACT.md, ../README.md"
    )


@pytest.mark.parametrize(
    ("name", "file", "hint"),
    [
        # A wrong directory for a known file name: point at where it is, as the skill writes it.
        (
            BUILDING,
            f"../{DESIGNING}/PIPELINE_ANTIPATTERNS.md",
            f"../{CONFIGURING}/PIPELINE_ANTIPATTERNS.md",
        ),
        (DESIGNING, "./FAILURE_SCENARIOS.md", "examples/FAILURE_SCENARIOS.md"),
        (CONFIGURING, "/etc/PIPELINE_ANTIPATTERNS.md", "PIPELINE_ANTIPATTERNS.md"),
    ],
)
async def test_wrong_directory_gets_a_hint_not_the_file(host, name, file, hint):
    result = await host.run(call({"name": name, "file": file}))
    assert (result.ok, result.error) == (False, "invalid_args")
    assert (
        f"Unknown file {file!r} in skill {name!r}. Did you mean {hint}? Files: " in result.content
    )
    assert host.run_counts == {}


@pytest.mark.parametrize("field", ["name", "file"])
async def test_huge_values_are_not_echoed_back(host, field):
    args = {"name": BUILDING, field: "x" * 100_000}
    result = await host.run(call(args))
    assert (result.ok, result.error) == (False, "invalid_args")
    assert f"{'x' * 100}'... (100000 chars)" in result.content
    assert len(result.content) < 500


def test_read_refuses_paths_resolve_would_not_return():
    with pytest.raises(skills.SkillError):
        skills.read("../catalog.json")


async def test_schema_errors_come_before_skill_lookup(host):
    result = await host.run(call({"file": "SKILL.md"}))
    assert (result.ok, result.error) == (False, "invalid_args")
    assert "'name' is a required property" in result.content


# --- the sync script --------------------------------------------------------------------------


def _sync_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sync_rocketride_data.py"
    spec = importlib.util.spec_from_file_location("sync_rocketride_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_copies_text_files_and_records_the_commit(tmp_path):
    repo = tmp_path / "repo"
    files = {
        "docs/agents/skills/README.md": "top",
        "docs/agents/skills/s/SKILL.md": "---\nname: s\ndescription: Use for S — ok.\n---\n",
        "docs/agents/skills/s/ex/a.pipe": "{}",
        "docs/agents/skills/s/index.json": "[]",
        "docs/agents/skills/s/tools/run.py": "print()",
        "docs/other.md": "not a skill",
    }
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")

    config = ("user.name=t", "user.email=t@t", "commit.gpgsign=false")

    def git(*args):
        flags = [part for c in config for part in ("-c", c)]
        cmd = ["git", "-C", str(repo), *flags, *args]
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "skills")
    commit = git("rev-parse", "HEAD").strip()
    out = tmp_path / "data"
    (out / "skills" / "stale").mkdir(parents=True)
    (out / "skills" / "stale" / "old.md").write_text("removed upstream")

    assert _sync_module().copy_skills(repo, commit, out) == 4
    copied = sorted(p.relative_to(out / "skills").as_posix() for p in out.rglob("*.*"))
    assert copied == ["README.md", "SOURCE.json", "s/SKILL.md", "s/ex/a.pipe", "s/index.json"]
    assert (out / "skills" / "s" / "SKILL.md").read_bytes() == files[
        "docs/agents/skills/s/SKILL.md"
    ].encode()
    source = json.loads((out / "skills" / "SOURCE.json").read_text())
    assert source == {
        "repo": "rocketride-org/rocketride-server",
        "commit": commit,
        "path": "docs/agents/skills",
    }
    assert list(load_skills(out / "skills").skills) == ["s"]
