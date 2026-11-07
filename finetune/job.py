"""Running a fine-tune job for the agent API: the trainer is a separate Python
process (like Whisper), so a crash or out-of-memory error ends the job, not
the server. This module knows nothing about HTTP; agent_api_server.py calls
`run_job` from its worker thread with a few callbacks.

Stopping is cooperative: the server writes STOP into the job folder and the
trainer saves a checkpoint and exits. Only if it has not exited `grace_s`
seconds later is the process terminated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from finetune import progress
from finetune.train import EXIT_STOPPED

WANGP_ROOT = Path(__file__).resolve().parent.parent


def job_dir_for(outputs_root: Path, job_id: str) -> Path:
    return Path(outputs_root) / "finetune" / job_id


def resume_dir_error(outputs_root: Path, job_id: str) -> str | None:
    """Why an earlier job can't be resumed from, or None if it can."""
    d = job_dir_for(outputs_root, job_id)
    if not (d / "spec.json").is_file():
        return f"no fine-tune job {job_id} on this machine"
    if not (d / "resume.pt").is_file():
        return f"job {job_id} saved no checkpoint to resume from"
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_job(*, job_id: str, spec: dict, outputs_root: Path,
            update: Callable[..., Any], get_status: Callable[[], str | None],
            release_vram: Callable[[], None] | None = None,
            log_event: Callable[..., None] | None = None,
            command: list | None = None, grace_s: float = 300.0,
            persist_every_s: float = 1.0) -> str:
    """Run the trainer to the end. Returns the final status it recorded."""
    log_event = log_event or (lambda *a, **k: None)
    started = time.time()
    job_dir = job_dir_for(outputs_root, job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    spec = dict(spec)
    if spec.get("resume_from"):
        spec["resume_dir"] = str(job_dir_for(outputs_root, spec["resume_from"]))
    (job_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    state = progress.new_state()
    state["result"]["job_dir"] = str(job_dir)
    update(status="running", started_at=_now_iso(), queue_position=None, progress=0.0, result=dict(state["result"]))

    if release_vram is not None:
        try:
            release_vram()  # the image/video model this server may be holding
        except Exception as exc:  # noqa: BLE001
            log_event("finetune_release_failed", level="warn", job_id=job_id, error=str(exc))

    cmd = command or [sys.executable, "-u", "-m", "finetune.train", "--job-dir", str(job_dir)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    log_path = job_dir / "train.log"
    proc = subprocess.Popen(cmd, cwd=str(WANGP_ROOT), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)

    stop_asked = threading.Event()
    finished = threading.Event()

    def watch_cancel() -> None:
        asked_at = None
        while not finished.wait(1.0):
            if asked_at is None and get_status() == "cancelling":
                (job_dir / "STOP").write_text(_now_iso(), encoding="utf-8")
                stop_asked.set()
                asked_at = time.time()
                log_event("finetune_stop_requested", job_id=job_id)
            if asked_at is not None and time.time() - asked_at > grace_s and proc.poll() is None:
                log_event("finetune_terminated", level="warn", job_id=job_id, after_s=grace_s)
                proc.terminate()
                return

    watcher = threading.Thread(target=watch_cancel, daemon=True, name=f"finetune-cancel-{job_id}")
    watcher.start()

    last_persist = 0.0
    immediate = {"phase", "error", "done", "checkpoint", "file", "lmstudio", "data", "plan", "check", "eval"}
    try:
        with open(log_path, "a", encoding="utf-8") as logf:
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                ev = progress.parse_line(line.rstrip("\n"))
                if ev is None:
                    continue
                progress.apply_event(state, ev)
                now = time.time()
                if ev["event"] in immediate or now - last_persist >= persist_every_s:
                    update(progress=round(state["progress"], 4), step=state["step"],
                           total_steps=state["total_steps"], result=dict(state["result"]))
                    last_persist = now
        rc = proc.wait()
    finally:
        finished.set()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    result = dict(state["result"])
    files = [f["path"] for f in result.get("files", []) if isinstance(f, dict) and f.get("path")]
    files += [str(p) for p in (log_path, job_dir / "metrics.jsonl") if p.is_file()]
    duration = round(time.time() - started, 2)
    common = dict(completed_at=_now_iso(), duration_seconds=duration, files=files, result=result,
                  step=state["step"], total_steps=state["total_steps"])
    if rc == 0 and state["phase"] == "done":
        update(status="completed", progress=1.0, **common)
        return "completed"
    if rc == EXIT_STOPPED or stop_asked.is_set():
        update(status="cancelled", **common)
        return "cancelled"
    error = result.get("error") or f"trainer exited with code {rc}; see {log_path}"
    update(status="failed", error=error, **common)
    return "failed"
