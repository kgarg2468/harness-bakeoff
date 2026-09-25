"""Lines of code per loop package, and for `shared/` as the common baseline.

Every physical line gets exactly one kind, in this order of precedence:

- code: it holds a token that is neither a comment nor part of a docstring. Continuation lines
  of a multi-line string or of a bracketed expression are code too.
- docstring: it is part of a module, class or function docstring (the first statement of the
  body, a plain string expression).
- comment: it holds a comment.
- blank: anything else.

`code` is the scorecard number: non-blank, non-comment, non-docstring lines, the same counter
the reviews use. `statements` counts `ast.stmt` nodes, docstrings excluded, so formatting
choices cannot move it.

A file whose leading comment block has `# Ported from <project> (...)` (DESIGN.md, "Porting and
attribution") is counted as ported from that project. Its lines are reported per project, apart
from the original ones (FAIRNESS.md rule 8).
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import tokenize
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from bakeoff.metrics import LOOP_PACKAGES, REPO_ROOT

# Tokens that mark structure, not content: they never make a line count as code.
_LAYOUT = frozenset(
    {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
)
_PORTED = re.compile(r"#\s*Ported from\s+(?P<project>[^(:]+?)\s*(?:[(:]|$)")
_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(slots=True)
class Counts:
    """Line and statement counts of one file, or the sum over several."""

    files: int = 0
    code: int = 0
    comment: int = 0
    docstring: int = 0
    blank: int = 0
    statements: int = 0

    def add(self, other: Counts) -> None:
        """Add `other` into this total."""
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    @property
    def lines(self) -> int:
        """All physical lines."""
        return self.code + self.comment + self.docstring + self.blank


def _docstrings(tree: ast.Module) -> list[ast.Expr]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, _DOCUMENTED) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.append(first)
    return found


def count_source(source: str) -> Counts:
    """Classify every physical line of one Python source (see the module docstring)."""
    lines = io.StringIO(source).readlines()
    tree = ast.parse(source)
    docstrings = _docstrings(tree)
    # ast columns are UTF-8 byte offsets; tokenize columns are characters.
    spans = [((d.lineno, d.col_offset), (d.end_lineno, d.end_col_offset)) for d in docstrings]
    code: set[int] = set()
    doc: set[int] = set()
    comment: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in _LAYOUT:
            continue
        (row, col), (end_row, end_col) = tok.start, tok.end
        # A token that ends at column 0 (an f-string part ending in a newline) stops before
        # that line.
        rows = range(row, max(row, end_row if end_col else end_row - 1) + 1)
        if tok.type == tokenize.COMMENT:
            comment.update(rows)
            continue
        start = (row, len(lines[row - 1][:col].encode()))
        in_doc = any(first <= start < last for first, last in spans)  # type: ignore[operator]
        (doc if in_doc else code).update(rows)
    doc -= code
    comment -= code | doc
    statements = sum(isinstance(node, ast.stmt) for node in ast.walk(tree)) - len(docstrings)
    return Counts(
        files=1,
        code=len(code),
        comment=len(comment),
        docstring=len(doc),
        blank=len(lines) - len(code) - len(doc) - len(comment),
        statements=statements,
    )


def ported_from(source: str) -> str | None:
    """The project named by a `# Ported from <project>` line in the leading comment block."""
    for line in io.StringIO(source):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            return None  # the leading comment block has ended
        if match := _PORTED.match(stripped):
            return match["project"]
    return None


def read_source(path: Path) -> str:
    """A source file as text, honouring its encoding cookie or BOM like the interpreter."""
    raw = path.read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    return raw.decode(encoding)


def count_package(package: Path, root: Path) -> dict[str, Any]:
    """Per-file counts and totals of every `.py` file under `package`, split ported/original."""
    total, original = Counts(), Counts()
    ported: dict[str, Counts] = {}
    files = []
    for path in sorted(p for p in package.rglob("*.py") if "__pycache__" not in p.parts):
        source = read_source(path)
        counts, project = count_source(source), ported_from(source)
        total.add(counts)
        (original if project is None else ported.setdefault(project, Counts())).add(counts)
        path_ = path.relative_to(root).as_posix()
        files.append({"path": path_, **asdict(counts), "ported_from": project})
    return {
        "path": package.relative_to(root).as_posix(),
        "total": asdict(total),
        "original": asdict(original),
        "ported": {name: asdict(ported[name]) for name in sorted(ported)},
        "files": files,
    }


def measure(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Counts for each loop package present under `src/bakeoff`, and for `shared/`."""
    src = root / "src" / "bakeoff"
    return {
        "loops": {
            name: count_package(src / name, root) for name in LOOP_PACKAGES if (src / name).is_dir()
        },
        "shared": count_package(src / "shared", root),
    }


_COLUMNS = ("files", "code", "comment", "docstring", "blank", "statements")


def table(report: dict[str, Any], *, per_file: bool = True) -> str:
    """The report as a fixed-width text table."""

    def row(label: str, counts: dict[str, Any]) -> str:
        return f"{label:<52}" + "".join(f"{counts[c]:>11}" for c in _COLUMNS)

    out = [f"{'':<52}" + "".join(f"{c:>11}" for c in _COLUMNS)]
    packages = [*report["loops"].items(), ("shared (baseline)", report["shared"])]
    for name, package in packages:
        out.append(row(name, package["total"]))
        for project, counts in package["ported"].items():
            out.append(row(f"  ported from {project}", counts))
        if package["ported"]:
            out.append(row("  original", package["original"]))
        if per_file:
            for file in package["files"]:
                out.append(row(f"    {file['path'].removeprefix('src/bakeoff/')}", file))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Print the lines-of-code table (or JSON) for the repository."""
    parser = argparse.ArgumentParser(prog="python -m bakeoff.metrics.loc", description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--summary", action="store_true", help="omit the per-file rows")
    args = parser.parse_args(argv)
    report = measure(args.root.resolve())
    print(json.dumps(report, indent=2) if args.json else table(report, per_file=not args.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
