"""The line counter: code / comment / docstring / blank per line, statements, ported headers."""

from __future__ import annotations

import io
import json
import textwrap
from dataclasses import asdict
from pathlib import Path

from bakeoff.metrics import REPO_ROOT, loc
from bakeoff.metrics.loc import Counts, bakeoff_imports, count_package, count_source, ported_from


def counts(source: str) -> dict[str, int]:
    return asdict(count_source(textwrap.dedent(source).lstrip("\n")))


def test_docstrings_comments_blanks() -> None:
    source = '''
        """Module docstring.

        The blank line above is part of the docstring.
        """
        # a comment line

        import os  # a trailing comment makes no comment line


        class A:
            """Class docstring."""

            def f(self) -> None:
                """Function docstring."""
                return None

            async def g(self) -> None:
                """Async function docstring."""
    '''
    assert counts(source) == {
        "files": 1,
        "code": 5,  # import, class, def f, return, async def g
        "comment": 1,
        "docstring": 7,
        "blank": 5,
        "statements": 5,  # import, class, def, return, async def (docstrings excluded)
    }


def test_strings_that_are_not_docstrings_are_code() -> None:
    source = '''
        x = 1
        """Not first in the module, so not a docstring."""
        TEXT = """a multi-line string

        with a blank line inside"""
        def f():
            y = 2
            """Not first in the function either."""
    '''
    assert counts(source) == {
        "files": 1,
        "code": 8,
        "comment": 0,
        "docstring": 0,
        "blank": 0,
        "statements": 6,
    }


def test_line_with_code_and_docstring_is_code() -> None:
    source = '''
        def f(): """Docstring on the def line."""; return 1
        def g(): """Starts on the def line,
            ends here."""
        def h():
            (
                """Parenthesised docstring."""
            )
    '''
    result = counts(source)
    # f: code. g: code, then a docstring line. h: code, then three docstring lines.
    assert (result["code"], result["docstring"]) == (3, 4)
    assert result["statements"] == 4  # three defs and f's return


def test_non_ascii_before_a_docstring_on_the_same_line() -> None:
    # ast columns are UTF-8 bytes and tokenize columns are characters; they differ here.
    source = '''
        def café(): """First line,
            second line."""
    '''
    assert (counts(source)["code"], counts(source)["docstring"]) == (1, 1)


def test_multiline_fstring_and_continuations() -> None:
    source = """
        name = "x"
        text = f'''hello
        {name}

        bye'''
        total = (1 +
                 2)
        items = [
            1,

            2,
        ]
    """
    result = counts(source)
    assert result["code"] == 11  # every line but the blank one inside the list
    assert result["blank"] == 1


def test_every_line_gets_exactly_one_kind() -> None:
    for path in sorted((REPO_ROOT / "src" / "bakeoff").rglob("*.py")):
        source = loc.read_source(path)
        result = count_source(source)
        assert result.lines == len(io.StringIO(source).readlines()), path


def test_ported_header() -> None:
    assert ported_from("# Ported from Pi (MIT): a/b.ts @ abc\n# Changes: x\nimport os\n") == "Pi"
    assert ported_from("#!/usr/bin/env python\n\n# Ported from OpenCode (MIT): x\n") == "OpenCode"
    assert ported_from("# Ported from Some Project: x\n") == "Some Project"
    # Only the leading comment block counts: not a later comment, not a docstring.
    assert ported_from("import os\n# Ported from Pi (MIT): x\n") is None
    assert ported_from('"""Ported from Pi (MIT)."""\n') is None
    assert ported_from("") is None


def test_package_split_and_deterministic_json(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "bakeoff" / "demo"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__pycache__").mkdir()
    (pkg / "__init__.py").write_text('"""Demo."""\n')
    (pkg / "ported.py").write_text("# Ported from Pi (MIT): x @ y\n# Changes: z\nA = 1\nB = 2\n")
    (pkg / "sub" / "own.py").write_text("C = 3\n\n# note\n")
    (pkg / "__pycache__" / "junk.py").write_text("D = 4\n")
    report = count_package(pkg, tmp_path)
    assert report["path"] == "src/bakeoff/demo"
    assert [f["path"] for f in report["files"]] == [
        "src/bakeoff/demo/__init__.py",
        "src/bakeoff/demo/ported.py",
        "src/bakeoff/demo/sub/own.py",
    ]
    assert report["total"] == asdict(Counts(3, 3, 3, 1, 1, 3))
    assert report["ported"] == {"Pi": asdict(Counts(1, 2, 2, 0, 0, 2))}
    assert report["original"] == asdict(Counts(2, 1, 1, 1, 1, 1))
    assert report["files"][1]["ported_from"] == "Pi"
    assert report["imports_outside"] == [] and report["other_files"] == []
    assert json.dumps(report) == json.dumps(count_package(pkg, tmp_path))


def test_bakeoff_imports() -> None:
    source = textwrap.dedent("""
        import os
        import bakeoff.metrics.bench as b
        from bakeoff import our_version, shared
        from bakeoff.shared.contract import Loop
        from . import sibling
        from .sub.mod import x
        from .. import fakeprov
        from ..our_version.loop import OurLoop
        import httpx
    """)
    assert bakeoff_imports(source, "bakeoff.hybrid_version") == {
        "bakeoff.metrics.bench",
        "bakeoff.our_version",
        "bakeoff.shared",
        "bakeoff.shared.contract",
        "bakeoff.hybrid_version",
        "bakeoff.hybrid_version.sub.mod",
        "bakeoff.fakeprov",
        "bakeoff.our_version.loop",
    }


def test_code_that_lives_elsewhere_is_listed(tmp_path: Path) -> None:
    """A loop cannot look small by importing another loop's code or keeping logic in data."""
    pkg = tmp_path / "src" / "bakeoff" / "hybrid_version"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from bakeoff.our_version.loop import OurLoop as HybridLoop\n"
        "from bakeoff.shared.contract import Loop\n"
        "from .sub import helper\n"
    )
    (pkg / "sub" / "helper.py").write_text("from ...fakeprov import server\nfrom .. import x\n")
    (pkg / "sub" / "prompts.toml").write_text("a = 1\nb = 2\n")
    report = count_package(pkg, tmp_path)
    assert report["imports_outside"] == ["bakeoff.fakeprov", "bakeoff.our_version.loop"]
    assert report["other_files"] == [
        {"path": "src/bakeoff/hybrid_version/sub/prompts.toml", "lines": 2}
    ]
    loops = {"our_version": report, "hybrid_version": report}
    table = loc.table({"loops": loops, "shared": report}, per_file=False)
    assert "  imports bakeoff.our_version.loop (counted in our_version)" in table
    assert "  imports bakeoff.fakeprov (not counted)" in table
    assert (
        "  not Python, not counted: src/bakeoff/hybrid_version/sub/prompts.toml (2 lines)" in table
    )


def test_measure_real_repo() -> None:
    report = loc.measure()
    assert "our_version" in report["loops"]
    assert report["shared"]["path"] == "src/bakeoff/shared"
    ours = report["loops"]["our_version"]
    assert ours["total"]["code"] == ours["original"]["code"] + sum(
        c["code"] for c in ours["ported"].values()
    )
    assert ours["imports_outside"] == []
    # shared/scenario.py, the scenario driver, uses the fake provider and the loop registry;
    # nothing in shared/ imports a loop's code.
    outside = {name.split(".")[1] for name in report["shared"]["imports_outside"]}
    assert outside <= {"fakeprov", "loops"}, report["shared"]["imports_outside"]
    table = loc.table(report)
    assert "our_version" in table and "shared (baseline)" in table and "ported from" in table
