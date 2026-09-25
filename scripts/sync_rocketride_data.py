# Ported from RocketRide (MIT): nodes/test/framework/discovery.py @ a1fa4f15b5c61c51e450ffb6d24057b7edcf13d3
# Changes: only the service*.json reader (comments, trailing commas); it parses text read via git.
"""Regenerate `src/bakeoff/data/` from rocketride-server's `origin/develop`.

The repository is read with `git show` / `git ls-tree` only (never checked out or modified):

    uv run python scripts/sync_rocketride_data.py [REPO]

Writes `catalog.json` (one compact entry per node provider), copies a few valid example
pipelines plus all of `examples/incorrect/` into `src/bakeoff/data/examples/`, and copies the
agent skills' text files from `docs/agents/skills/` into `src/bakeoff/data/skills/` (with
`SOURCE.json` naming the commit).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from bakeoff.shared.skills import SKILL_SUFFIXES

REF = "origin/develop"
UPSTREAM = "rocketride-org/rocketride-server"
OUT = Path(__file__).resolve().parents[1] / "src" / "bakeoff" / "data"
SERVICE_FILE = re.compile(r"^nodes/src/nodes/[^/]+/services[^/]*\.json$")
# Valid examples that pass MockEngine's rules (structure, known providers, lanes).
VALID_EXAMPLES = (
    "agent-workflow.pipe",
    "document-processor.pipe",
    "llm-benchmark.pipe",
    "rag-pipeline.pipe",
    "tool-pipe-diamond.pipe",
)
# Field keys an agent needs to write a config; UI-only keys (ui, hidden, format, ...) are dropped.
FIELD_KEYS = (
    "type",
    "title",
    "description",
    "default",
    "enum",
    "minimum",
    "maximum",
    "optional",
    "secure",
    "items",
    "object",
    "properties",
    "conditional",
)
PROFILES_PLACEHOLDER = "*>preconfig.profiles.*.title"
SKILLS = "docs/agents/skills"
SKILLS_SOURCE = "SOURCE.json"


def _remove_json_comments(content: str) -> str:
    """Remove JavaScript-style comments from JSON content."""
    # Process line by line to avoid matching // inside strings
    lines = content.split("\n")
    result_lines = []

    in_multiline_comment = False

    for line in lines:
        # Handle multi-line comments
        if in_multiline_comment:
            if "*/" in line:
                line = line[line.index("*/") + 2 :]
                in_multiline_comment = False
            else:
                result_lines.append("")
                continue

        if "/*" in line:
            # Check if it's not inside a string (simple heuristic: before any quote)
            comment_pos = line.find("/*")
            quote_pos = line.find('"')
            if quote_pos == -1 or comment_pos < quote_pos:
                if "*/" in line[comment_pos:]:
                    # Single line /* */ comment
                    end_pos = line.index("*/", comment_pos) + 2
                    line = line[:comment_pos] + line[end_pos:]
                else:
                    line = line[:comment_pos]
                    in_multiline_comment = True

        # Remove single-line comments, but only if // is not inside a string
        # and not preceded by : (which would be in a URL like "http://")
        if "//" in line:
            in_string = False
            i = 0
            while i < len(line) - 1:
                if line[i] == '"' and (i == 0 or line[i - 1] != "\\"):
                    in_string = not in_string
                elif line[i : i + 2] == "//" and not in_string:
                    if i == 0 or line[i - 1] != ":":
                        line = line[:i]
                        break
                i += 1

        result_lines.append(line)

    return "\n".join(result_lines)


def _remove_trailing_commas(content: str) -> str:
    """Remove trailing commas before } or ]."""
    return re.sub(r",(\s*[}\]])", r"\1", content)


def _parse_service_json(content: str, name: str) -> dict[str, Any] | None:
    """Parse a service*.json text, handling comments and trailing commas."""
    try:
        content = _remove_json_comments(content)
        content = _remove_trailing_commas(content)
        # strict=False allows control characters (tabs, etc.) in strings
        return json.loads(content, strict=False)
    except Exception as e:
        print(f"Warning: Failed to parse {name}: {e}", file=sys.stderr)
        return None


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _text(value: Any) -> str:
    """Service descriptions are strings or lists of line fragments."""
    if isinstance(value, list):
        return " ".join(str(part).strip() for part in value if str(part).strip())
    return str(value)


def _enum(values: list[Any], profiles: list[str]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value == PROFILES_PLACEHOLDER:
            out.extend(profiles)
        elif isinstance(value, list) and value:
            out.append(value[0])  # [value, label]
        else:
            out.append(value)
    return out


def _fields(service: dict[str, Any]) -> dict[str, Any]:
    profiles = list(((service.get("preconfig") or {}).get("profiles") or {}).keys())
    out: dict[str, Any] = {}
    for name, spec in (service.get("fields") or {}).items():
        if not isinstance(spec, dict):
            continue
        entry = {key: spec[key] for key in FIELD_KEYS if key in spec}
        if "description" in entry:
            entry["description"] = _text(entry["description"])
        if isinstance(entry.get("enum"), list):
            entry["enum"] = _enum(entry["enum"], profiles)
        out[name] = entry
    return out


def build_catalog(repo: Path, commit: str) -> dict[str, Any]:
    """Parse every node service definition into `{"source", "providers"}`."""
    providers: dict[str, Any] = {}
    for path in _git(repo, "ls-tree", "-r", "--name-only", commit, "nodes/src/nodes").split():
        if not SERVICE_FILE.match(path):
            continue
        service = _parse_service_json(_git(repo, "show", f"{commit}:{path}"), path)
        # Files without a protocol (core/services.common*.json) are shared field fragments.
        if not service or not service.get("protocol"):
            continue
        name = service["protocol"].replace("://", "")
        if name in providers:
            raise SystemExit(f"duplicate provider {name!r} in {path}")
        providers[name] = {
            "title": service.get("title", name),
            "classType": service.get("classType", []),
            "lanes": service.get("lanes", {}),
            "description": _text(service.get("description", "")),
            "fields": _fields(service),
        }
    return {"source": {"repo": UPSTREAM, "commit": commit}, "providers": providers}


def copy_examples(repo: Path, commit: str, out: Path) -> None:
    """Copy the chosen valid examples and every file in examples/incorrect/."""
    incorrect = _git(repo, "ls-tree", "--name-only", commit, "examples/incorrect/").split()
    for src in [f"examples/{name}" for name in VALID_EXAMPLES] + incorrect:
        dest = out / "examples" / src.removeprefix("examples/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_git(repo, "show", f"{commit}:{src}"), encoding="utf-8")


def copy_skills(repo: Path, commit: str, out: Path) -> int:
    """Replace `out/skills/` with the text files of `docs/agents/skills/`; return how many."""
    dest_root = out / "skills"
    # Build the new bundle next to the old one and swap only when it is complete, so a failed
    # git read or write never leaves the bundle missing or half-copied.
    staging = out / ".skills.staging"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        # -z: without it git quotes non-ASCII paths ("r\303\251sum\303\251.md"), which the suffix
        # filter would silently drop. SKILL_SUFFIXES leaves out the skills' tools/*.py helpers.
        paths = _git(repo, "ls-tree", "-r", "-z", "--name-only", commit, f"{SKILLS}/").split("\0")
        copied = [path for path in paths if path.endswith(SKILL_SUFFIXES)]
        for src in copied:
            dest = staging / src.removeprefix(f"{SKILLS}/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Bytes, not text: the copy stays byte-identical to the upstream blob.
            blob = subprocess.run(
                ["git", "-C", str(repo), "show", f"{commit}:{src}"], check=True, capture_output=True
            ).stdout
            dest.write_bytes(blob)
        source = {"repo": UPSTREAM, "commit": commit, "path": SKILLS}
        (staging / SKILLS_SOURCE).write_text(json.dumps(source, indent=1) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    # Generated: files removed upstream must go too, so the old bundle is replaced whole.
    retired = out / ".skills.old"
    shutil.rmtree(retired, ignore_errors=True)
    if dest_root.exists():
        dest_root.rename(retired)
    staging.rename(dest_root)
    shutil.rmtree(retired, ignore_errors=True)
    return len(copied)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "repo", nargs="?", type=Path, default=Path("/home/kg/work/rr/OSS/rocketride-server")
    )
    args = parser.parse_args(argv)
    commit = _git(args.repo, "rev-parse", REF).strip()
    catalog = build_catalog(args.repo, commit)
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(catalog, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    (OUT / "catalog.json").write_text(text, encoding="utf-8")
    copy_examples(args.repo, commit, OUT)
    skill_files = copy_skills(args.repo, commit, OUT)
    print(f"{len(catalog['providers'])} providers from {commit[:12]}, {len(text)} bytes")
    print(f"{skill_files} skill files from {SKILLS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
