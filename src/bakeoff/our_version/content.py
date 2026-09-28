"""A user message's content parts as OpenAI's Responses API takes them.

The runner keeps a user message as it came: text, or chat content parts (`text`, `image_url`).
Chat completions take those as they are. The Responses API names them `input_text` and
`input_image`, with the image's URL flat and a `detail` (`auto` unless the part says).
"""

from __future__ import annotations

from typing import Any

from bakeoff.shared.contract import UserContent


def input_content(content: UserContent) -> UserContent:
    """A message's content as Responses `input` content; text stays text."""
    if isinstance(content, str):
        return content
    return [_input_part(part) for part in content]


def _input_part(part: dict[str, Any]) -> dict[str, Any]:
    if part["type"] == "text":
        return {"type": "input_text", "text": part["text"]}
    image = part["image_url"]
    return {"type": "input_image", "image_url": image["url"], "detail": image.get("detail", "auto")}
