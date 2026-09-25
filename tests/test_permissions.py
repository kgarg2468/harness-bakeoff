import pytest

from bakeoff.shared.permissions import evaluate, match, validate_rules

DESIGN_RULES = {"*": "allow", "write_file": {"*.pipe": "allow", "*": "ask"}, "edit_file": "ask"}


@pytest.mark.parametrize(
    ("rules", "tool", "path", "expected"),
    [
        # The DESIGN.md example.
        (DESIGN_RULES, "read_file", "notes.txt", "allow"),
        (DESIGN_RULES, "write_file", "a.pipe", "allow"),
        (DESIGN_RULES, "write_file", "dir/sub/a.pipe", "allow"),  # "*" crosses "/"
        (DESIGN_RULES, "write_file", "a.txt", "ask"),
        (DESIGN_RULES, "edit_file", "a.pipe", "ask"),
        (DESIGN_RULES, "list_components", None, "allow"),
        # Nothing matches -> ask.
        ({}, "read_file", "a", "ask"),
        ({"write_file": {"*.pipe": "allow"}}, "write_file", "a.txt", "ask"),
        # The tool key beats "*"; an unmatched tool mapping falls through to "*".
        ({"*": "deny", "read_file": "allow"}, "read_file", "a", "allow"),
        ({"*": "allow", "read_file": "deny"}, "read_file", "a", "deny"),
        ({"*": "deny", "write_file": {"*.pipe": "allow"}}, "write_file", "a.txt", "deny"),
        # First match wins inside a mapping.
        ({"write_file": {"*": "deny", "*.pipe": "allow"}}, "write_file", "a.pipe", "deny"),
        ({"write_file": {"*.pipe": "allow", "*": "deny"}}, "write_file", "a.pipe", "allow"),
        # A call without a path matches its globs against "".
        ({"validate_pipeline": {"*.pipe": "allow", "*": "ask"}}, "validate_pipeline", None, "ask"),
        # "*" may be a mapping too.
        ({"*": {"docs/*": "allow", "*": "deny"}}, "read_file", "docs/a.md", "allow"),
        ({"*": {"docs/*": "allow", "*": "deny"}}, "read_file", "src/a.py", "deny"),
        # "?" is one character.
        ({"write_file": {"?.pipe": "allow"}}, "write_file", "a.pipe", "allow"),
        ({"write_file": {"?.pipe": "allow"}}, "write_file", "ab.pipe", "ask"),
    ],
)
def test_evaluate(rules, tool, path, expected):
    assert evaluate(rules, tool, path) == expected


@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        ("a.pipe", "*.pipe", True),
        ("a.pipex", "*.pipe", False),
        ("apipe", "*.pipe", False),  # "." is literal
        ("a+b(1).pipe", "a+b(1).pipe", True),  # regex characters are literal
        ("aab(1).pipe", "a+b(1).pipe", False),
        ("dir\\a.pipe", "dir/*.pipe", True),  # backslashes are normalized
        ("line1\nline2", "line1*", True),  # "*" spans newlines
        ("ls", "ls *", True),  # a trailing " *" is optional
        ("ls -la", "ls *", True),
        ("lsx", "ls *", False),
        ("anything", "*", True),
        ("", "*", True),
    ],
)
def test_match(text, pattern, expected):
    assert match(text, pattern) is expected


@pytest.mark.parametrize(
    "rules",
    [{"*": "yes"}, {"write_file": {"*.pipe": "alow"}}, {"read_file": ["allow"]}],
)
def test_validate_rules_rejects_bad_decisions(rules):
    with pytest.raises(ValueError, match="Invalid permission rule"):
        validate_rules(rules)


def test_validate_rules_accepts_design_example():
    validate_rules(DESIGN_RULES)
