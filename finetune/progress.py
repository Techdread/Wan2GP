"""The line protocol between the trainer process and the API server.

The trainer prints ordinary log lines plus one JSON event per line prefixed
with `@@ft `. The server keeps every line in the job's train.log and folds the
events into the job record: overall progress, step counters and a live
`result` dict (phase, loss, speed, files) that clients poll.
"""

from __future__ import annotations

import json
from typing import Any

PREFIX = "@@ft "

# Share of the job's progress bar each phase covers. Training dominates.
PHASES = {
    "starting": (0.0, 0.01),
    "download": (0.01, 0.04),
    "data": (0.04, 0.05),
    "load": (0.05, 0.06),
    "check": (0.06, 0.07),
    "train": (0.07, 0.90),
    "merge": (0.90, 0.94),
    "gguf": (0.94, 0.99),
    "lmstudio": (0.99, 1.0),
    "done": (1.0, 1.0),
}


def event_line(event: str, **fields: Any) -> str:
    return PREFIX + json.dumps({"event": event, **fields}, default=str)


def parse_line(line: str) -> dict | None:
    if not line.startswith(PREFIX):
        return None
    try:
        value = json.loads(line[len(PREFIX):])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and isinstance(value.get("event"), str) else None


def new_state() -> dict:
    return {"phase": "starting", "progress": 0.0, "step": 0, "total_steps": 0, "result": {"phase": "starting"}}


def apply_event(state: dict, ev: dict) -> dict:
    """Fold one event into the state; returns the same dict."""
    kind = ev.get("event")
    result = state["result"]
    if kind == "phase":
        phase = ev.get("phase")
        if phase in PHASES:
            state["phase"] = phase
            result["phase"] = phase
            state["progress"] = max(state["progress"], PHASES[phase][0])
        if ev.get("message"):
            result["message"] = ev["message"]
    elif kind == "data":
        result["data"] = {k: v for k, v in ev.items() if k != "event"}
    elif kind == "plan":
        state["total_steps"] = int(ev.get("total_steps") or 0)
        result["plan"] = {k: v for k, v in ev.items() if k != "event"}
    elif kind == "check":
        result["check"] = {k: v for k, v in ev.items() if k != "event"}
    elif kind == "step":
        step, total = int(ev.get("step") or 0), int(ev.get("total_steps") or state["total_steps"] or 0)
        state["step"], state["total_steps"] = step, total
        lo, hi = PHASES["train"]
        if total:
            state["progress"] = max(state["progress"], lo + (hi - lo) * min(1.0, step / total))
        result["train"] = {k: v for k, v in ev.items() if k != "event"}
        history = result.setdefault("loss_history", [])
        if ev.get("loss") is not None:
            history.append([step, ev["loss"]])
            if len(history) > 2000:  # keep the record small: thin the older half
                del history[: len(history) // 2 : 2]
    elif kind == "eval":
        result.setdefault("eval_history", []).append([int(ev.get("step") or 0), ev.get("eval_loss")])
    elif kind == "checkpoint":
        result.setdefault("checkpoints", []).append({k: v for k, v in ev.items() if k != "event"})
    elif kind == "file":
        result.setdefault("files", []).append({k: v for k, v in ev.items() if k != "event"})
    elif kind == "lmstudio":
        result["lmstudio"] = {k: v for k, v in ev.items() if k != "event"}
    elif kind == "error":
        result["error"] = ev.get("message")
    elif kind == "done":
        state["phase"] = "done"
        result["phase"] = "done"
        state["progress"] = 1.0
    return state
