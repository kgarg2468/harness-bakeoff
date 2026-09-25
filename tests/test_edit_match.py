import random
import time

import pytest

from bakeoff.shared.tools import edit_match
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


def _table_levenshtein(a: str, b: str) -> int:
    """OpenCode's dynamic-programming version, as the reference."""
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[len(b)]


def test_levenshtein_matches_the_table_version():
    rng = random.Random(7)
    for _ in range(500):
        alphabet = rng.choice(["ab", "abcdefgh", "aé日 "])
        a, b = ("".join(rng.choices(alphabet, k=rng.randrange(20))) for _ in range(2))
        assert levenshtein(a, b) == _table_levenshtein(a, b), (a, b)


def test_long_middle_lines_stay_fast():
    # The table version took seconds here, blocking the event loop.
    rng = random.Random(3)
    line = "".join(rng.choices("abcdefgh", k=10_000))
    other = "".join(rng.choices("abcdefgh", k=10_000))
    start = time.perf_counter()
    with pytest.raises(NoMatch):
        replace(f"{{\n{line}\n}}\n", f"{{\n{other}\n}}", "x")
    assert time.perf_counter() - start < 2


def test_single_candidate_stops_once_similar_enough(monkeypatch):
    calls = []

    def counting(a, b):
        calls.append((a, b))
        return levenshtein(a, b)

    monkeypatch.setattr(edit_match, "levenshtein", counting)
    content = "start\n1\n2\n3\n4\nend"
    # Each identical middle line adds 1/4; the third reaches 0.65.
    assert list(block_anchor_replacer(content, content)) == [content]
    assert len(calls) == 3


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


def test_block_anchor_near_tie_is_ambiguous():
    """Two anchored blocks about equally close to old_string: no guess, and replace() refuses."""
    content = "start\nalpha one two\nend\nstart\nalpha one tw0\nend\n"
    assert list(block_anchor_replacer(content, "start\nalpha one twx\nend")) == [
        edit_match.AMBIGUOUS
    ]
    with pytest.raises(MultipleMatches):
        replace(content, "start\nalpha one twx\nend", "x")


def test_block_anchor_clear_winner_is_used():
    content = "start\nalpha one two three\nend\nstart\nzzz qqq\nend\n"
    assert list(block_anchor_replacer(content, "start\nalpha one two thrEE\nend")) == [
        "start\nalpha one two three\nend"
    ]


def test_block_anchor_tie_resolved_by_a_later_replacer():
    """Greptile's case: block-anchor similarity can't separate two blocks, but only one has an
    exact middle line, so the context-aware replacer still edits it."""
    content = "start\nfoo bar\nbaz quX\nend\nstart\nfoo baR\nbaz quX\nend\n"
    old = "start\nfoo bar\nbaz qux\nend"
    assert list(block_anchor_replacer(content, old)) == [edit_match.AMBIGUOUS]
    assert replace(content, old, "NEW") == "NEW\nstart\nfoo baR\nbaz quX\nend\n"


def test_empty_spans_are_not_mistaken_for_ambiguity():
    empty = "".join([])  # a real (empty) span, as a replacer could produce
    assert empty is not edit_match.AMBIGUOUS
    assert isinstance(edit_match.AMBIGUOUS, str)
