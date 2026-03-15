#!/usr/bin/env python3
"""
Plan test for agent_api.py
===========================
Validates the WanGP Agent API wrapper without running actual generation.
Tests construction, method signatures, settings assembly, and API compatibility.

Run:  python test_agent_api.py
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

WANGP_ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------ #
#  Mock objects matching the real shared.api shapes
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class FakeGenerationError:
    message: str
    task_index: int = None
    task_id: object = None
    stage: str = None

@dataclass(frozen=True)
class FakeGenerationResult:
    success: bool
    generated_files: list
    errors: list
    total_tasks: int
    successful_tasks: int
    failed_tasks: int

@dataclass(frozen=True)
class FakeProgressUpdate:
    phase: str
    status: str
    progress: int
    current_step: int = None
    total_steps: int = None

@dataclass(frozen=True)
class FakeSessionEvent:
    kind: str
    data: object = None
    timestamp: float = 0.0


class FakeStream:
    def __init__(self, events):
        self._events = events
        self._closed = False

    def iter(self, timeout=None):
        for e in self._events:
            yield e
        self._closed = True

    @property
    def closed(self):
        return self._closed


class FakeJob:
    def __init__(self, result_obj, events=None):
        self._result = result_obj
        self.events = FakeStream(events or [])
        self.done = True
        self.cancel_requested = False

    def result(self, timeout=None):
        return self._result

    def join(self, timeout=None):
        return self._result

    def cancel(self):
        self.cancel_requested = True


class FakeSession:
    """Records all submit calls for inspection."""
    def __init__(self):
        self.submitted_tasks = []
        self.submitted_manifests = []
        self.closed = False

    def _make_result(self, n=1):
        return FakeGenerationResult(
            success=True,
            generated_files=[f"/tmp/output_{i}.png" for i in range(n)],
            errors=[],
            total_tasks=n,
            successful_tasks=n,
            failed_tasks=0,
        )

    def submit_task(self, settings):
        self.submitted_tasks.append(settings)
        progress_events = [
            FakeSessionEvent("progress", FakeProgressUpdate(
                phase="inference", status="Step 1/8", progress=12,
                current_step=1, total_steps=8)),
            FakeSessionEvent("progress", FakeProgressUpdate(
                phase="inference", status="Step 8/8", progress=100,
                current_step=8, total_steps=8)),
        ]
        return FakeJob(self._make_result(), events=progress_events)

    def submit_manifest(self, settings_list):
        self.submitted_manifests.append(settings_list)
        return FakeJob(self._make_result(len(settings_list)))

    def close(self):
        self.closed = True


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def test_construction():
    """Test that WanGPAgent can be constructed with various args."""
    print("\n--- Construction ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent()
    check("default construction", agent._root == WANGP_ROOT)
    check("default profile", agent._profile == 4)
    check("default attention", agent._attention == "sage2")
    check("session initially None", agent._session is None)

    agent2 = WanGPAgent(profile=3, attention="flash", output_dir="/tmp/out", verbose=False)
    check("custom profile", agent2._profile == 3)
    check("custom attention", agent2._attention == "flash")
    check("custom output_dir", agent2._output_dir == Path("/tmp/out"))
    check("verbose=False", agent2._verbose is False)


def test_image_settings():
    """Test that generate_image assembles correct settings."""
    print("\n--- Image Settings ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    session = FakeSession()
    agent._session = session

    result = agent.generate_image(
        prompt="A test image",
        model="z_image",
        resolution="1024x1024",
        steps=8,
        seed=42,
        guidance_scale=0,
        negative_prompt="ugly",
        loras=["test.safetensors"],
        loras_multipliers="1.0",
        output_filename="test_out",
    )

    check("task submitted", len(session.submitted_tasks) == 1)
    s = session.submitted_tasks[0]
    check("model_type", s["model_type"] == "z_image")
    check("prompt", s["prompt"] == "A test image")
    check("resolution", s["resolution"] == "1024x1024")
    check("steps", s["num_inference_steps"] == 8)
    check("seed", s["seed"] == 42)
    check("guidance_scale", s["guidance_scale"] == 0)
    check("image_mode=1", s["image_mode"] == 1)
    check("negative_prompt", s["negative_prompt"] == "ugly")
    check("loras", s["activated_loras"] == ["test.safetensors"])
    check("loras_multipliers", s["loras_multipliers"] == "1.0")
    check("output_filename", s["output_filename"] == "test_out")
    check("result success", result["success"] is True)
    check("result files", len(result["files"]) == 1)
    check("result errors empty", result["errors"] == [])
    check("duration present", "duration_seconds" in result)


def test_video_settings():
    """Test that generate_video assembles correct settings."""
    print("\n--- Video Settings ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    session = FakeSession()
    agent._session = session

    result = agent.generate_video(
        prompt="A test video",
        model="wan21_t2v_14B",
        resolution="832x480",
        steps=30,
        frames=81,
        seed=123,
        guidance_scale=5.0,
        flow_shift=3.0,
        negative_prompt="blurry",
        fps=16,
        start_image="/tmp/start.png",
        end_image="/tmp/end.png",
    )

    check("task submitted", len(session.submitted_tasks) == 1)
    s = session.submitted_tasks[0]
    check("model_type", s["model_type"] == "wan21_t2v_14B")
    check("prompt", s["prompt"] == "A test video")
    check("resolution", s["resolution"] == "832x480")
    check("steps", s["num_inference_steps"] == 30)
    check("video_length", s["video_length"] == 81)
    check("seed", s["seed"] == 123)
    check("guidance_scale", s["guidance_scale"] == 5.0)
    check("flow_shift", s["flow_shift"] == 3.0)
    check("image_mode=0", s["image_mode"] == 0)
    check("force_fps as string", s["force_fps"] == "16")
    check("image_start", s["image_start"] == "/tmp/start.png")
    check("image_end", s["image_end"] == "/tmp/end.png")
    check("negative_prompt", s["negative_prompt"] == "blurry")
    check("duration_seconds default", s["duration_seconds"] == 0)
    check("result success", result["success"] is True)


def test_video_no_images():
    """Test video generation without start/end images omits those keys."""
    print("\n--- Video No Images ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    session = FakeSession()
    agent._session = session

    agent.generate_video(prompt="No images", model="wan21_t2v_14B")
    s = session.submitted_tasks[0]
    check("no image_start key", "image_start" not in s)
    check("no image_end key", "image_end" not in s)


def test_audio_settings():
    """Test that generate_audio assembles correct settings."""
    print("\n--- Audio Settings ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    session = FakeSession()
    agent._session = session

    result = agent.generate_audio(
        prompt="Hello world",
        model="qwen3_tts",
        seed=99,
        audio_source="/tmp/voice.wav",
    )

    check("task submitted", len(session.submitted_tasks) == 1)
    s = session.submitted_tasks[0]
    check("model_type", s["model_type"] == "qwen3_tts")
    check("prompt", s["prompt"] == "Hello world")
    check("seed", s["seed"] == 99)
    check("audio_source", s["audio_source"] == "/tmp/voice.wav")
    check("result success", result["success"] is True)


def test_audio_no_source():
    """Test audio without source omits that key."""
    print("\n--- Audio No Source ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    session = FakeSession()
    agent._session = session

    agent.generate_audio(prompt="TTS test")
    s = session.submitted_tasks[0]
    check("no audio_source key", "audio_source" not in s)


def test_batch():
    """Test batch submission."""
    print("\n--- Batch ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    session = FakeSession()
    agent._session = session

    tasks = [
        {"model_type": "z_image", "prompt": "img1"},
        {"model_type": "z_image", "prompt": "img2"},
    ]
    result = agent.generate_batch(tasks)

    check("manifest submitted", len(session.submitted_manifests) == 1)
    check("two tasks in manifest", len(session.submitted_manifests[0]) == 2)
    check("total_tasks=2", result["total_tasks"] == 2)
    check("files count=2", len(result["files"]) == 2)


def test_extra_settings():
    """Test that **extra_settings are forwarded."""
    print("\n--- Extra Settings ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    session = FakeSession()
    agent._session = session

    agent.generate_image(
        prompt="test",
        NAG_scale=2,
        NAG_tau=4.0,
        batch_size=2,
    )
    s = session.submitted_tasks[0]
    check("NAG_scale forwarded", s.get("NAG_scale") == 2)
    check("NAG_tau forwarded", s.get("NAG_tau") == 4.0)
    check("batch_size forwarded", s.get("batch_size") == 2)


def test_verbose_progress(capsys=None):
    """Test that verbose mode processes progress events without error."""
    print("\n--- Verbose Progress ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=True)
    session = FakeSession()
    agent._session = session

    # Should print progress lines without crashing
    result = agent.generate_image(prompt="verbose test")
    check("verbose run success", result["success"] is True)


def test_error_result():
    """Test that errors are properly extracted from result."""
    print("\n--- Error Result ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)

    err_result = FakeGenerationResult(
        success=False,
        generated_files=[],
        errors=[FakeGenerationError(message="CUDA OOM"), FakeGenerationError(message="bad seed")],
        total_tasks=1,
        successful_tasks=0,
        failed_tasks=1,
    )
    error_session = FakeSession()
    error_session.submit_task = lambda s: FakeJob(err_result)
    agent._session = error_session

    result = agent.generate_image(prompt="will fail")
    check("result not success", result["success"] is False)
    check("errors extracted", result["errors"] == ["CUDA OOM", "bad seed"])
    check("failed_tasks=1", result["failed_tasks"] == 1)


def test_close_resets_session():
    """Test that close() sets session to None so it can be reinitialized."""
    print("\n--- Close Resets Session ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    session = FakeSession()
    agent._session = session

    agent.close()
    check("session closed", session.closed is True)
    check("session set to None", agent._session is None)


def test_release_model_idempotent():
    """Test that release_model is safe to call multiple times."""
    print("\n--- Release Model Idempotent ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    session = FakeSession()
    agent._session = session

    agent.release_model()
    check("first release OK", session.closed is True)
    check("session None after release", agent._session is None)

    # Second call should not raise
    agent.release_model()
    check("second release no error", True)


def test_default_settings_file():
    """Test get_default_settings reads the actual settings file."""
    print("\n--- Default Settings File ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent()
    # Don't init session — this method reads file directly
    settings_path = agent._root / "models" / "_settings.json"

    if settings_path.exists():
        settings = agent.get_default_settings()
        check("settings is dict", isinstance(settings, dict))
        check("has prompt key", "prompt" in settings)
        check("has model_type or resolution", "resolution" in settings)
        check("has video_length", "video_length" in settings)
        check("has seed", "seed" in settings)
        check("has activated_loras", "activated_loras" in settings)
    else:
        check("settings file exists", False, f"Missing: {settings_path}")


def test_settings_keys_match_api():
    """Verify that keys produced by the wrapper exist in _settings.json."""
    print("\n--- Settings Keys Match ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    settings_path = agent._root / "models" / "_settings.json"

    if not settings_path.exists():
        check("settings file exists", False)
        return

    with open(settings_path) as f:
        valid_keys = set(json.load(f).keys())

    # Keys that generate_image sets
    image_keys = {
        "model_type", "prompt", "resolution", "num_inference_steps",
        "seed", "guidance_scale", "negative_prompt", "image_mode",
        "activated_loras", "loras_multipliers", "output_filename",
    }
    # Keys that generate_video adds
    video_keys = image_keys | {
        "video_length", "duration_seconds", "flow_shift", "force_fps",
        "image_start", "image_end",
    }
    # Keys that generate_audio uses
    audio_keys = {"model_type", "prompt", "seed", "output_filename", "audio_source"}

    # model_type is not in _settings.json (it's the task type selector, not a setting)
    # That's expected — model_type is added by the wrapper
    for key in image_keys - {"model_type"}:
        check(f"image key '{key}' valid", key in valid_keys, f"not in _settings.json")

    for key in video_keys - {"model_type"}:
        check(f"video key '{key}' valid", key in valid_keys, f"not in _settings.json")

    for key in audio_keys - {"model_type"}:
        check(f"audio key '{key}' valid", key in valid_keys, f"not in _settings.json")


# ------------------------------------------------------------------ #
#  Remote mode tests
# ------------------------------------------------------------------ #

def test_remote_construction():
    """Test construction in remote mode."""
    print("\n--- Remote Construction ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(url="http://localhost:8100")
    check("url stored", agent._url == "http://localhost:8100")
    check("trailing slash stripped", WanGPAgent(url="http://host:8100/")._url == "http://host:8100")
    check("timeout default", agent._timeout == 3600)
    check("session None", agent._session is None)
    check("custom timeout", WanGPAgent(url="http://h:9", timeout=60)._timeout == 60)


def test_remote_ensure_session_noop():
    """_ensure_session is a no-op in remote mode."""
    print("\n--- Remote Ensure Session ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(url="http://localhost:8100")
    agent._ensure_session()
    check("no session created", agent._session is None)


def test_remote_close_noop():
    """close() doesn't crash in remote mode."""
    print("\n--- Remote Close ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(url="http://localhost:8100")
    agent.close()
    check("close no error", True)


def test_download_file_local():
    """download_file returns path unchanged in local mode."""
    print("\n--- Download File Local ---")
    from agent_api import WanGPAgent

    agent = WanGPAgent(verbose=False)
    check("local passthrough", agent.download_file("/tmp/out.jpg") == "/tmp/out.jpg")
    check("local with dest", agent.download_file("/tmp/out.jpg", "/tmp/local.jpg") == "/tmp/out.jpg")


def test_serve_importable():
    """serve() is importable."""
    print("\n--- Serve Function ---")
    from agent_api import serve
    check("serve callable", callable(serve))


def test_remote_roundtrip():
    """Start a mock server, connect a remote client, exercise all endpoints."""
    print("\n--- Remote Roundtrip ---")
    import threading
    import time as _time
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    from agent_api import WanGPAgent

    class MockHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            p = self.path.split('?')[0]
            if p == '/api/health':
                self._ok({"status": "ok"})
            elif p == '/api/models':
                self._ok({"TestFamily": ["test_model"]})
            elif p == '/api/loras':
                self._ok(["lora1.safetensors"])
            elif p == '/api/settings':
                self._ok({"prompt": "", "seed": -1})
            else:
                self._err("not found", 404)

        def do_POST(self):
            p = self.path.split('?')[0]
            n = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(n)) if n else {}
            if p == '/api/generate':
                self._ok({
                    "success": True,
                    "files": ["/tmp/mock.png"],
                    "errors": [],
                    "total_tasks": 1,
                    "successful_tasks": 1,
                    "failed_tasks": 0,
                    "duration_seconds": 1.23,
                })
            elif p == '/api/batch':
                count = len(body) if isinstance(body, list) else 1
                self._ok({
                    "success": True,
                    "files": [f"/tmp/mock_{i}.png" for i in range(count)],
                    "errors": [],
                    "total_tasks": count,
                    "successful_tasks": count,
                    "failed_tasks": 0,
                    "duration_seconds": 2.0,
                })
            elif p == '/api/release':
                self._ok({"ok": True})
            else:
                self._err("not found", 404)

        def _ok(self, data):
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _err(self, msg, code):
            raw = json.dumps({"error": msg}).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    class _TS(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    srv = _TS(("localhost", 18919), MockHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _time.sleep(0.2)

    try:
        agent = WanGPAgent(url="http://localhost:18919")

        health = agent._remote_get('/api/health')
        check("health", health == {"status": "ok"})

        models = agent.list_models()
        check("list_models", models == {"TestFamily": ["test_model"]})

        loras = agent.list_loras("z_image")
        check("list_loras", loras == ["lora1.safetensors"])

        settings = agent.get_default_settings()
        check("settings", "prompt" in settings)

        r = agent.generate_image(prompt="test")
        check("generate_image success", r["success"] is True)
        check("generate_image files", r["files"] == ["/tmp/mock.png"])

        r2 = agent.generate_video(prompt="vid")
        check("generate_video success", r2["success"] is True)

        r3 = agent.generate_audio(prompt="audio")
        check("generate_audio success", r3["success"] is True)

        r4 = agent.generate_batch([{"prompt": "a"}, {"prompt": "b"}])
        check("batch success", r4["success"] is True)
        check("batch count", r4["total_tasks"] == 2)

        agent.release_model()
        check("release ok", True)

        agent.close()
        check("close ok", True)
    finally:
        srv.shutdown()
        srv.server_close()


# ------------------------------------------------------------------ #
#  Run all tests
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("  WanGP Agent API — Plan Test")
    print("=" * 60)

    test_construction()
    test_image_settings()
    test_video_settings()
    test_video_no_images()
    test_audio_settings()
    test_audio_no_source()
    test_batch()
    test_extra_settings()
    test_verbose_progress()
    test_error_result()
    test_close_resets_session()
    test_release_model_idempotent()
    test_default_settings_file()
    test_settings_keys_match_api()
    test_remote_construction()
    test_remote_ensure_session_noop()
    test_remote_close_noop()
    test_download_file_local()
    test_serve_importable()
    test_remote_roundtrip()

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
