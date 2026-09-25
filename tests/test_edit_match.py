import pytest

from bakeoff.shared.tools.edit_match import (
    REPLACERS,
    EditError,
    MultipleMatches,
    NoMatch,
    block_anchor_replacer,
    context_aware_replacer,
    escape_normalized_replacer,
    indentation_flexible_replacer,
    levenshtein,
    line_trimmed_replacer,
    multi_occurrence_replacer,
    replace,
    simple_replacer,
    trimmed_boundary_replacer,
    whitespace_normalized_replacer,
)


def test_chain_order_matches_opencode():
    assert [r.__name__ for r in REPLACERS] == [
        "simple_replacer",
        "line_trimmed_replacer",
        "block_anchor_replacer",
        "whitespace_normalized_replacer",
        "indentation_flexible_replacer",
        "escape_normalized_replacer",
        "trimmed_boundary_replacer",
        "context_aware_replacer",
        "multi_occurrence_replacer",
    ]


@pytest.mark.parametrize(
    ("replacer", "content", "find", "expected"),
    [
        (simple_replacer, "abc", "b", ["b"]),
        (line_trimmed_replacer, "x\n  a  \n\tb\ny", "a\nb\n", ["  a  \n\tb"]),
        (line_trimmed_replacer, "a\nb", "a\nc", []),
        # Anchors match, the middle line is similar enough.
        (
            block_anchor_replacer,
            "def f():\n    x = compute(1)\n    return x\n",
            "def f():\n    x = compute(2)\n    return x",
            ["def f():\n    x = compute(1)\n    return x"],
        ),
        # Anchors match but the middle line is too different.
        (
            block_anchor_replacer,
            "def f():\n    completely different\n    return x",
            "def f():\n    x = 1\n    return x",
            [],
        ),
        # Several candidates: the most similar block wins.
        (
            block_anchor_replacer,
            "start\nalpha beta\nend\nstart\nalpha betx\nend",
            "start\nalpha betx\nend",
            ["start\nalpha betx\nend"],
        ),
        (block_anchor_replacer, "a\nb", "a\nb", []),  # needs 3+ lines
        (whitespace_normalized_replacer, "x  =   1\ny", "x = 1", ["x  =   1"]),
        (whitespace_normalized_replacer, "call(a,   b) now", "a, b", ["a,   b"]),
        (
            whitespace_normalized_replacer,
            "if  x:\n  y =  1\nz",
            "if x:\ny = 1",
            ["if  x:\n  y =  1"],
        ),
        (
            indentation_flexible_replacer,
            "class A:\n    def f(self):\n        pass\n",
            "def f(self):\n    pass",
            ["    def f(self):\n        pass"],
        ),
        # Relative indentation must still agree.
        (indentation_flexible_replacer, "  a\n  b", "a\n  b", []),
        (escape_normalized_replacer, 'print("hi")\n', 'print(\\"hi\\")', ['print("hi")'] * 2),
        (escape_normalized_replacer, "a\nb", "a\\nb", ["a\nb", "a\nb"]),
        (escape_normalized_replacer, 'say \\"x\\"', 'say "x"', ['say \\"x\\"']),
        (trimmed_boundary_replacer, "foo bar\nbaz", "  foo bar  ", ["foo bar", "foo bar"]),
        (trimmed_boundary_replacer, "foo", "foo", []),  # already trimmed
        # Anchors match and at least half of the middle lines match exactly.
        (
            context_aware_replacer,
            "begin\nsame\nchanged a lot\nend",
            "begin\nsame\nnot the same\nend",
            ["begin\nsame\nchanged a lot\nend"],
        ),
        (
            context_aware_replacer,
            "begin\nno\nmatch\nend",
            "begin\nzero\noverlap\nend",
            [],
        ),
        (multi_occurrence_replacer, "a-a-a", "a", ["a", "a", "a"]),
    ],
)
def test_replacer(replacer, content, find, expected):
    assert list(replacer(content, find)) == expected


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("same", "same") == 0


def test_exact_unique_match():
    assert replace("a = 1\nb = 2\n", "b = 2", "b = 3") == "a = 1\nb = 3\n"


def test_tolerant_match_replaces_the_span_found_in_the_file():
    # Indentation differs, so line_trimmed_replacer finds the span; new_string is used as is.
    content = "def f():\n    if x:\n        return 1\n"
    new = "    if y:\n        return 2"
    assert replace(content, "if x:\n    return 1", new) == "def f():\n" + new + "\n"


def test_ambiguous_match_raises():
    with pytest.raises(MultipleMatches, match="multiple matches"):
        replace("x\nx\n", "x", "y")


def test_replace_all():
    assert replace("x\nx\n", "x", "y", replace_all=True) == "y\ny\n"


def test_not_found_raises():
    with pytest.raises(NoMatch, match="Could not find old_string"):
        replace("abc", "zzz", "y")


@pytest.mark.parametrize(("old", "new"), [("a", "a"), ("", "b")])
def test_noop_and_empty_old_raise(old, new):
    with pytest.raises(EditError):
        replace("abc", old, new)


def test_disproportionate_match_is_refused():
    # One line of literal "\n" escapes unescapes to five lines of content.
    content = "a\nb\nc\nd\ne"
    with pytest.raises(EditError, match="much larger"):
        replace(content, "a\\nb\\nc\\nd\\ne", "x")


def test_empty_candidate_span_is_never_replaced():
    # Upstream would replace "" (between every character) here; the port skips empty spans.
    with pytest.raises(NoMatch):
        replace("a\n\nb", "\t", "X", replace_all=True)
