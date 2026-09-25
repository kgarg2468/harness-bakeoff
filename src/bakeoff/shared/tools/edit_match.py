# Ported from OpenCode (MIT): packages/opencode/src/tool/edit.ts @ 16c56fe5ecc3305028d1f0a9cff5806e51c9d480
# Changes: the replacer chain and replace() only, in Python; empty candidate spans are skipped.
# OpenCode credits these approaches to Cline (Apache-2.0):
#   evals/diff-edits/diff-apply/diff-06-23-25.ts and diff-06-26-25.ts
# and gemini-cli (Apache-2.0): packages/core/src/utils/editCorrector.ts
"""Forgiving search-and-replace for `edit_file`.

`replace` tries a chain of replacers, from an exact match to whitespace-, indentation-,
escape- and anchor-tolerant matches. Each replacer yields candidate spans of `content`; the
first span that occurs exactly once is replaced (or every occurrence, with `replace_all`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

Replacer = Callable[[str, str], Iterator[str]]

# Similarity thresholds for block anchor fallback matching
SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.65
MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.65


class EditError(ValueError):
    """An edit that cannot be applied."""


class NoMatch(EditError):
    """`old_string` was not found."""


class MultipleMatches(EditError):
    """`old_string` matches more than one place."""


def levenshtein(a: str, b: str) -> int:
    """Edit distance between `a` and `b`."""
    if a == "" or b == "":
        return max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[len(b)]


def _span(lines: list[str], start: int, end: int) -> str:
    """The text of `lines[start..end]` (inclusive) as it appears in the content."""
    return "\n".join(lines[start : end + 1])


def simple_replacer(content: str, find: str) -> Iterator[str]:
    """The exact text."""
    yield find


def line_trimmed_replacer(content: str, find: str) -> Iterator[str]:
    """Lines that match after trimming each line."""
    original_lines = content.split("\n")
    search_lines = find.split("\n")
    if search_lines[-1] == "":
        search_lines.pop()
    for i in range(len(original_lines) - len(search_lines) + 1):
        if all(
            original_lines[i + j].strip() == search_lines[j].strip()
            for j in range(len(search_lines))
        ):
            yield _span(original_lines, i, i + len(search_lines) - 1)


def _middle_similarity(original: list[str], search: list[str], start: int, end: int) -> float:
    """Mean similarity of the lines between the anchors (empty pairs count as 0).

    OpenCode stops early for a single candidate once the threshold is reached; the sum only
    grows, so computing it in full gives the same decision.
    """
    actual_size = end - start + 1
    lines_to_check = min(len(search) - 2, actual_size - 2)
    if lines_to_check <= 0:
        return 1.0  # no middle lines: the anchors decide
    similarity = 0.0
    for j in range(1, min(len(search), actual_size) - 1):
        original_line = original[start + j].strip()
        search_line = search[j].strip()
        max_len = max(len(original_line), len(search_line))
        if max_len == 0:
            continue
        similarity += 1 - levenshtein(original_line, search_line) / max_len
    return similarity / lines_to_check


def block_anchor_replacer(content: str, find: str) -> Iterator[str]:
    """Blocks of 3+ lines whose first and last lines match, with similar middle lines."""
    original_lines = content.split("\n")
    search_lines = find.split("\n")
    if len(search_lines) < 3:
        return
    if search_lines[-1] == "":
        search_lines.pop()

    first = search_lines[0].strip()
    last = search_lines[-1].strip()
    size = len(search_lines)
    max_line_delta = max(1, size // 4)

    candidates: list[tuple[int, int]] = []
    for i, line in enumerate(original_lines):
        if line.strip() != first:
            continue
        for j in range(i + 2, len(original_lines)):
            if original_lines[j].strip() == last:
                if abs((j - i + 1) - size) <= max_line_delta:
                    candidates.append((i, j))
                break  # only the first occurrence of the last line

    if len(candidates) == 1:
        start, end = candidates[0]
        similarity = _middle_similarity(original_lines, search_lines, start, end)
        if similarity >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
            yield _span(original_lines, start, end)
        return

    best: tuple[int, int] | None = None
    max_similarity = -1.0
    for start, end in candidates:
        similarity = _middle_similarity(original_lines, search_lines, start, end)
        if similarity > max_similarity:
            max_similarity, best = similarity, (start, end)
    if best and max_similarity >= MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD:
        yield _span(original_lines, *best)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def whitespace_normalized_replacer(content: str, find: str) -> Iterator[str]:
    """Text that matches once runs of whitespace are collapsed."""
    normalized_find = _normalize_whitespace(find)
    lines = content.split("\n")
    for line in lines:
        if _normalize_whitespace(line) == normalized_find:
            yield line
        elif normalized_find in _normalize_whitespace(line):
            # Find the actual substring in the original line that matches
            words = find.strip().split()
            if words:
                found = re.search(r"\s+".join(re.escape(word) for word in words), line)
                if found:
                    yield found.group(0)

    find_lines = find.split("\n")
    if len(find_lines) > 1:
        for i in range(len(lines) - len(find_lines) + 1):
            block = "\n".join(lines[i : i + len(find_lines)])
            if _normalize_whitespace(block) == normalized_find:
                yield block


def _remove_indentation(text: str) -> str:
    lines = text.split("\n")
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return text
    indent = min(len(line) - len(line.lstrip()) for line in non_empty)
    return "\n".join(line if not line.strip() else line[indent:] for line in lines)


def indentation_flexible_replacer(content: str, find: str) -> Iterator[str]:
    """Blocks that match once their common indentation is removed."""
    normalized_find = _remove_indentation(find)
    content_lines = content.split("\n")
    size = len(find.split("\n"))
    for i in range(len(content_lines) - size + 1):
        block = "\n".join(content_lines[i : i + size])
        if _remove_indentation(block) == normalized_find:
            yield block


_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "'": "'",
    '"': '"',
    "`": "`",
    "\\": "\\",
    "\n": "\n",
    "$": "$",
}


def _unescape(text: str) -> str:
    return re.sub(r"\\([ntr'\"`\\\n$])", lambda m: _ESCAPES[m.group(1)], text)


def escape_normalized_replacer(content: str, find: str) -> Iterator[str]:
    """Text that matches once escape sequences (`\\n`, `\\"`, ...) are unescaped."""
    unescaped_find = _unescape(find)
    if unescaped_find in content:
        yield unescaped_find
    lines = content.split("\n")
    size = len(unescaped_find.split("\n"))
    for i in range(len(lines) - size + 1):
        block = "\n".join(lines[i : i + size])
        if _unescape(block) == unescaped_find:
            yield block


def multi_occurrence_replacer(content: str, find: str) -> Iterator[str]:
    """Every exact occurrence, so `replace_all` can take them all."""
    start = 0
    while (index := content.find(find, start)) != -1:
        yield find
        start = index + len(find)


def trimmed_boundary_replacer(content: str, find: str) -> Iterator[str]:
    """The text without its leading and trailing whitespace."""
    trimmed = find.strip()
    if trimmed == find:
        return
    if trimmed in content:
        yield trimmed
    lines = content.split("\n")
    size = len(find.split("\n"))
    for i in range(len(lines) - size + 1):
        block = "\n".join(lines[i : i + size])
        if block.strip() == trimmed:
            yield block


def context_aware_replacer(content: str, find: str) -> Iterator[str]:
    """Blocks of 3+ lines with matching anchors where at least half the middle lines match."""
    find_lines = find.split("\n")
    if len(find_lines) < 3:
        return
    if find_lines[-1] == "":
        find_lines.pop()
    content_lines = content.split("\n")
    first = find_lines[0].strip()
    last = find_lines[-1].strip()

    for i, line in enumerate(content_lines):
        if line.strip() != first:
            continue
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() != last:
                continue
            block_lines = content_lines[i : j + 1]
            if len(block_lines) == len(find_lines):
                matching = total = 0
                for k in range(1, len(block_lines) - 1):
                    block_line, find_line = block_lines[k].strip(), find_lines[k].strip()
                    if block_line or find_line:
                        total += 1
                        matching += block_line == find_line
                if total == 0 or matching / total >= 0.5:
                    yield "\n".join(block_lines)
            break  # only the first occurrence of the last line


REPLACERS: tuple[Replacer, ...] = (
    simple_replacer,
    line_trimmed_replacer,
    block_anchor_replacer,
    whitespace_normalized_replacer,
    indentation_flexible_replacer,
    escape_normalized_replacer,
    trimmed_boundary_replacer,
    context_aware_replacer,
    multi_occurrence_replacer,
)


def replace(content: str, old: str, new: str, replace_all: bool = False) -> str:
    """Replace `old` with `new` in `content`, tolerating small differences in `old`.

    Raises `NoMatch` if nothing matches, `MultipleMatches` if every match is ambiguous, and
    `EditError` for a no-op or empty `old`.
    """
    if old == new:
        raise EditError("No changes to apply: old_string and new_string are identical.")
    if old == "":
        raise EditError(
            "old_string cannot be empty when editing an existing file. Provide the exact text "
            "to replace, or use write_file for an intentional full-file replacement."
        )

    not_found = True
    for replacer in REPLACERS:
        for search in replacer(content, old):
            index = content.find(search)
            # An empty span would "match" between every character.
            if not search or index == -1:
                continue
            not_found = False
            if _is_disproportionate(search, old):
                raise EditError(
                    "Refusing replacement because the matched span is much larger than "
                    "old_string. Re-read the file and provide the full exact old_string for the "
                    "intended replacement."
                )
            if replace_all:
                return content.replace(search, new)
            if index != content.rfind(search):
                continue
            return content[:index] + new + content[index + len(search) :]

    if not_found:
        raise NoMatch(
            "Could not find old_string in the file. It must match exactly, including "
            "whitespace, indentation, and line endings."
        )
    raise MultipleMatches(
        "Found multiple matches for old_string. Provide more surrounding context to make the "
        "match unique."
    )


def _is_disproportionate(search: str, old: str) -> bool:
    old_lines = len(old.split("\n"))
    search_lines = len(search.split("\n"))
    if search_lines >= max(old_lines + 3, old_lines * 2):
        return True
    if old_lines == 1:
        return False
    return len(search.strip()) > max(len(old.strip()) + 500, len(old.strip()) * 4)
