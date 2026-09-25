"""scripts/snapshot_live.py, run as a script on live runs written as `bakeoff live` writes them."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bakeoff.shared.contract import Item, ModelConfig
from bakeoff.shared.sessionlog import SessionLog

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "snapshot_live.py"
KEY = "sk-test-never-in-a-snapshot"
REPORT = {"ok": False, "errors": ["no source"], "warnings": []}


def write_run(
    live: Path, run_id: str, impl: str, *, thread: str | None = None, prompt: str = "Build it.",
    final_text: str = "done",
) -> None:  # fmt: skip
    """One loop's live run: result.json, and a session log with the thread's model config (key
    removed, as the runner saves it) and a turn that calls `validate_pipeline` twice."""
    folder = live / run_id / impl
    folder.mkdir(parents=True)
    config = ModelConfig(
        base_url="https://api.openai.com/v1", model="gpt-test", api_key=KEY,
        kind="openai_responses", reasoning={"effort": "xhigh", "summary": "auto"},
    )  # fmt: skip
    model = asdict(config)
    del model["api_key"]
    log = SessionLog(folder / "log.sqlite")
    log.create_thread(f"live-{impl}", impl=impl, system="sys", meta={"rules": {}, "model": model})
    turn = log.start_turn(f"live-{impl}", "user")["id"]
    calls = [("c1", "validate_pipeline"), ("c2", "read_file"), ("c3", "validate_pipeline")]
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Build it."}]
    for call_id, name in calls:
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}},
        ]})  # fmt: skip
        content = json.dumps(REPORT) if call_id == "c1" else "not json" if call_id == "c3" else "x"
        messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    for i, message in enumerate(messages):
        log.append_item(f"live-{impl}", Item(id=f"i{i}", turn_id=turn, message=message))
    log.close()
    (folder / "result.json").write_text(json.dumps({
        "v": 1, "run_id": run_id, "impl": impl, "model": "gpt-test",
        "base_url": "https://api.openai.com/v1", "base_url_template": "https://api.openai.com/v1",
        "max_steps": 20, "attended": False, "prompt": prompt, "final_text": final_text,
        "stops": ["end_turn"], "steps": 4, "requests": 4,
        "usage": {"input_tokens": 400, "cached_tokens": 100, "output_tokens": 20},
        "latency": {"ttft_ms": 812.5, "total_ms": 3000.0}, "duration_ms": 3010.0,
        "tool_runs": {"c1": 1, "c2": 1, "c3": 1}, "files": ["chat.pipe"], "passed": True,
        "error": None, "thread": thread or f"live-{impl}",
        "invariants": {"I2": {"ok": True, "detail": "3 calls"}, "I5": {"ok": False, "detail": "x"}},
    }))  # fmt: skip


def snapshot(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "snap.json"), *args,
         "--live-dir", str(tmp_path / "live"), "--what", "Test runs."],
        capture_output=True, text=True, check=False,
    )  # fmt: skip


def test_snapshot_rows_carry_result_fields_and_what_only_the_log_keeps(tmp_path: Path) -> None:
    for impl in ("pydantic", "our"):
        write_run(tmp_path / "live", "R1", impl)
    done = snapshot(tmp_path, "R1")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().startswith("2 rows")
    text = (tmp_path / "snap.json").read_text()
    assert KEY not in text
    snap = json.loads(text)
    assert snap["what"].startswith("Test runs. Copied field by field")
    assert [(r["run_id"], r["loop"]) for r in snap["runs"]] == [("R1", "our"), ("R1", "pydantic")]
    row = snap["runs"][0]
    assert row["kind"] == "openai_responses"
    assert row["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    # Only validate_pipeline results, in order; one that isn't JSON is kept as its text.
    assert row["validations"] == [REPORT, "not json"]
    assert row["first_token_ms"] == 812.5 and row["duration_ms"] == 3010.0
    assert (row["input_tokens"], row["cached_tokens"], row["output_tokens"]) == (400, 100, 20)
    assert row["tool_runs"] == 3 and row["steps"] == 4 and row["files"] == ["chat.pipe"]
    assert row["invariants"] == {"I2": True, "I5": False}


def test_a_log_with_one_thread_is_read_whatever_thread_the_result_names(tmp_path: Path) -> None:
    write_run(tmp_path / "live", "R1", "our", thread="renamed")
    assert snapshot(tmp_path, "R1").returncode == 0
    (row,) = json.loads((tmp_path / "snap.json").read_text())["runs"]
    assert row["kind"] == "openai_responses" and len(row["validations"]) == 2


def test_a_run_without_loop_results_fails_and_writes_nothing(tmp_path: Path) -> None:
    (tmp_path / "live" / "R2").mkdir(parents=True)
    done = snapshot(tmp_path, "R2")
    assert done.returncode == 1 and "no loop results" in done.stderr
    assert not (tmp_path / "snap.json").exists()


def test_a_run_id_with_no_folder_fails_with_a_message_not_a_traceback(tmp_path: Path) -> None:
    write_run(tmp_path / "live", "R1", "our")
    done = snapshot(tmp_path, "R1", "NOPE")
    assert done.returncode == 1 and "no loop results in" in done.stderr and "NOPE" in done.stderr
    assert "Traceback" not in done.stderr
    assert not (tmp_path / "snap.json").exists()


def test_key_shaped_strings_a_prompt_or_answer_carries_are_redacted(tmp_path: Path) -> None:
    pasted = "sk-proj-" + "A1b2C3d4" * 4
    token = "ghp_" + "x" * 36
    write_run(tmp_path / "live", "R1", "our", prompt=f"Use key {pasted} to build it.",
              final_text=f"Saved {token}; see chat.pipe.")  # fmt: skip
    done = snapshot(tmp_path, "R1")
    assert done.returncode == 0, done.stderr
    assert "redacted 2 key-shaped string(s)" in done.stderr
    text = (tmp_path / "snap.json").read_text()
    assert pasted not in text and token not in text
    (row,) = json.loads(text)["runs"]
    assert row["prompt"] == "Use key [redacted] to build it."
    assert row["final_text"] == "Saved [redacted]; see chat.pipe."
    assert row["validations"][0] == REPORT  # nothing else changes


def test_a_snapshot_with_nothing_key_shaped_says_nothing_about_redaction(tmp_path: Path) -> None:
    write_run(tmp_path / "live", "R1", "our", prompt="Keep the sk-short name.")
    done = snapshot(tmp_path, "R1")
    assert done.returncode == 0 and "redacted" not in done.stderr
    (row,) = json.loads((tmp_path / "snap.json").read_text())["runs"]
    assert row["prompt"] == "Keep the sk-short name."
