#!/usr/bin/env python3
"""
WanGP Agent API Wrapper
=======================
A simplified, agent-friendly interface around WanGP's in-process Python API.

This wrapper provides:
  - Simple function calls for image/video/audio generation
  - Blocking (synchronous) execution with progress logging
  - Model listing and discovery
  - Automatic session management

Usage:
    from agent_api import WanGPAgent

    agent = WanGPAgent()

    # Generate an image
    result = agent.generate_image(
        prompt="A cyberpunk cityscape at sunset",
        model="z_image",
        resolution="1024x1024",
        steps=8,
    )
    print(result["files"])  # List of output file paths

    # Generate a video
    result = agent.generate_video(
        prompt="A cat walking through a garden",
        model="wan21_t2v_14B",
        resolution="832x480",
        steps=30,
        frames=81,
    )
    print(result["files"])

    # List available models
    models = agent.list_models()

Remote Mode:
    # Start the API server (headless, keeps model warm in VRAM):
    python agent_api.py serve --port 8100

    # Connect from another script/agent:
    agent = WanGPAgent(url="http://localhost:8100")
    result = agent.generate_image(prompt="A sunset")
    agent.download_file(result["files"][0], "sunset.jpg")

Important:
    In local mode, this API runs WanGP in-process and CANNOT connect to an
    already-running WanGP web UI — stop the web UI before using local mode.
    For remote mode, start the dedicated API server with 'python agent_api.py serve'.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


# Resolve WanGP root relative to this script
WANGP_ROOT = Path(__file__).resolve().parent


class WanGPAgent:
    """Agent-friendly wrapper around WanGP's Python API."""

    def __init__(
        self,
        root: str | Path | None = None,
        profile: int = 4,
        attention: str = "sage2",
        output_dir: str | Path | None = None,
        verbose: bool = True,
        extra_cli_args: list[str] | None = None,
        url: str | None = None,
        timeout: int = 3600,
    ):
        """
        Initialize the WanGP agent.

        Args:
            root: Path to WanGP installation. Defaults to this script's directory.
            profile: Memory profile (1-5). 4 is default and works with most GPUs.
            attention: Attention mode: "sdpa", "sage", "sage2", "flash".
            output_dir: Override output directory for generated files.
            verbose: Print progress to console.
            extra_cli_args: Additional CLI arguments for WanGP.
            url: URL of a running agent API server (e.g. "http://localhost:8100").
                 When set, all operations are routed via HTTP to the server.
            timeout: HTTP request timeout in seconds for remote mode (default: 3600).
        """
        self._url = url.rstrip('/') if url else None
        self._timeout = timeout
        self._verbose = verbose
        if self._url:
            # Remote mode — no local session needed
            self._root = Path(root).resolve() if root else None
            self._output_dir = Path(output_dir).resolve() if output_dir else None
            self._session = None
            return
        self._root = Path(root or WANGP_ROOT).resolve()
        self._profile = profile
        self._attention = attention
        self._output_dir = Path(output_dir).resolve() if output_dir else None
        self._extra_cli_args = extra_cli_args or []
        self._session = None

    # ------------------------------------------------------------------ #
    #  Remote HTTP helpers
    # ------------------------------------------------------------------ #

    def _remote_get(self, endpoint: str) -> Any:
        """GET request to the remote API server."""
        import urllib.request
        url = f"{self._url}{endpoint}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def _remote_post(self, endpoint: str, data: dict) -> Any:
        """POST request to the remote API server."""
        import urllib.request
        url = f"{self._url}{endpoint}"
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(
            url, data=body, headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    # ------------------------------------------------------------------ #
    #  Session management
    # ------------------------------------------------------------------ #

    def _ensure_session(self):
        """Lazily create the WanGP session on first use."""
        if self._url or self._session is not None:
            return

        # Add WanGP root to path so we can import its modules
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))

        from shared.api import init

        cli_args = [
            "--attention", self._attention,
            "--profile", str(self._profile),
        ]
        cli_args.extend(self._extra_cli_args)

        self._session = init(
            root=self._root,
            cli_args=cli_args,
            output_dir=self._output_dir,
            console_output=self._verbose,
        )

    def _run_task(self, settings: dict[str, Any]) -> dict[str, Any]:
        """
        Submit a generation task and wait for completion.

        Returns a dict with:
            success: bool
            files: list[str] - absolute paths to generated files
            errors: list[str] - error messages if any
            duration_seconds: float - total generation time
        """
        if self._url:
            return self._remote_post('/api/generate', settings)
        self._ensure_session()

        start_time = time.time()
        job = self._session.submit_task(settings)

        # Stream progress events if verbose
        if self._verbose:
            for event in job.events.iter(timeout=0.5):
                if event.kind == "progress":
                    p = event.data
                    step_info = ""
                    if p.current_step is not None and p.total_steps is not None:
                        step_info = f" [{p.current_step}/{p.total_steps}]"
                    print(f"  [{p.progress:3d}%]{step_info} {p.phase}: {p.status}")
                elif event.kind == "error":
                    print(f"  ERROR: {event.data.message}")
        else:
            # Just wait silently
            job.result()

        result = job.result()
        duration = time.time() - start_time

        return {
            "success": result.success,
            "files": result.generated_files,
            "errors": [str(e.message) for e in result.errors],
            "total_tasks": result.total_tasks,
            "successful_tasks": result.successful_tasks,
            "failed_tasks": result.failed_tasks,
            "duration_seconds": round(duration, 2),
        }

    # ------------------------------------------------------------------ #
    #  High-level generation methods
    # ------------------------------------------------------------------ #

    def generate_image(
        self,
        prompt: str,
        model: str = "z_image",
        resolution: str = "1024x1024",
        steps: int = 8,
        seed: int = -1,
        guidance_scale: float = 0,
        negative_prompt: str = "",
        loras: list[str] | None = None,
        loras_multipliers: str = "",
        output_filename: str = "",
        **extra_settings,
    ) -> dict[str, Any]:
        """
        Generate an image.

        Args:
            prompt: Text description of the image to generate.
            model: Model type. Common choices:
                   - "z_image" (Z-Image Turbo, fast, 8 steps)
                   - "z_image_base" (Z-Image Base, 30+ steps, needs guidance>1)
                   - "qwen_image_20B" (Qwen image model)
                   - "flux2_4B" / "flux2_9B" (Flux 2 Klein)
            resolution: Image resolution as "WxH" (e.g. "1024x1024", "1280x720").
            steps: Number of denoising steps. Z-Image Turbo: 4-8, Base: 30-50.
            seed: Random seed. -1 for random.
            guidance_scale: CFG scale. Z-Image Turbo uses 0 (NAG instead).
            negative_prompt: What to avoid in the image.
            loras: List of lora filenames to activate.
            loras_multipliers: Lora strength multipliers (space-separated).
            output_filename: Custom output filename (without extension).
            **extra_settings: Any additional WanGP settings.

        Returns:
            Dict with success, files, errors, duration_seconds.
        """
        settings = {
            "model_type": model,
            "prompt": prompt,
            "resolution": resolution,
            "num_inference_steps": steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "image_mode": 1,
            "activated_loras": loras or [],
            "loras_multipliers": loras_multipliers,
            "output_filename": output_filename,
        }
        settings.update(extra_settings)
        return self._run_task(settings)

    def generate_video(
        self,
        prompt: str,
        model: str = "wan21_t2v_14B",
        resolution: str = "832x480",
        steps: int = 30,
        frames: int = 81,
        seed: int = -1,
        guidance_scale: float = 5.0,
        flow_shift: float = 3.0,
        negative_prompt: str = "",
        fps: int | str = "",
        duration_seconds: float = 0,
        loras: list[str] | None = None,
        loras_multipliers: str = "",
        start_image: str | None = None,
        end_image: str | None = None,
        output_filename: str = "",
        **extra_settings,
    ) -> dict[str, Any]:
        """
        Generate a video.

        Args:
            prompt: Text description of the video to generate.
            model: Model type. Common choices:
                   - "wan21_t2v_14B" (Wan 2.1 Text-to-Video 14B)
                   - "wan21_t2v_1.3B" (Wan 2.1 Text-to-Video 1.3B, faster)
                   - "wan22_t2v_14B" (Wan 2.2 with High/Low noise models)
                   - "wan21_i2v_14B_480P" / "wan21_i2v_14B_720P" (Image-to-Video)
                   - "ltx2_22B_distilled" (LTX-2, fast, with audio)
                   - "ltx2_22B" (LTX-2 non-distilled)
                   - "hunyuan_13B" (Hunyuan Video)
            resolution: Video resolution as "WxH" (e.g. "832x480", "1280x720").
            steps: Number of denoising steps.
            frames: Number of frames (e.g. 81 = ~5s at 16fps for Wan).
            seed: Random seed. -1 for random.
            guidance_scale: CFG scale. Use 1 with lora accelerators.
            flow_shift: Flow shift parameter. Default 3.0 for Wan.
            negative_prompt: What to avoid in the video.
            fps: Force specific FPS (leave "" for model default).
            duration_seconds: Target duration in seconds (0 = use frames).
            loras: List of lora filenames to activate.
            loras_multipliers: Lora strength multipliers.
            start_image: Path to start frame image (for i2v or guided generation).
            end_image: Path to end frame image.
            output_filename: Custom output filename.
            **extra_settings: Any additional WanGP settings.

        Returns:
            Dict with success, files, errors, duration_seconds.
        """
        settings = {
            "model_type": model,
            "prompt": prompt,
            "resolution": resolution,
            "num_inference_steps": steps,
            "video_length": frames,
            "duration_seconds": duration_seconds,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "negative_prompt": negative_prompt,
            "force_fps": str(fps),
            "image_mode": 0,
            "activated_loras": loras or [],
            "loras_multipliers": loras_multipliers,
            "output_filename": output_filename,
        }
        if start_image:
            settings["image_start"] = start_image
        if end_image:
            settings["image_end"] = end_image
        settings.update(extra_settings)
        return self._run_task(settings)

    def generate_audio(
        self,
        prompt: str,
        model: str = "qwen3_tts",
        seed: int = -1,
        audio_source: str | None = None,
        output_filename: str = "",
        **extra_settings,
    ) -> dict[str, Any]:
        """
        Generate audio (TTS, music, etc.).

        Args:
            prompt: Text to speak or music description.
            model: Model type. Common choices:
                   - "qwen3_tts" (Qwen3 Text-to-Speech with voice cloning)
                   - "chatterbox" (ChatterBox TTS)
                   - "ace_step_1.5" (Song generation)
                   - "heartmula" (Song with lyrics)
                   - "index_tts2" (Index TTS 2 with emotions)
                   - "kugel_audio" (Kugel Audio TTS with voice cloning)
            seed: Random seed. -1 for random.
            audio_source: Path to voice sample for cloning.
            output_filename: Custom output filename.
            **extra_settings: Any additional WanGP settings.

        Returns:
            Dict with success, files, errors, duration_seconds.
        """
        settings = {
            "model_type": model,
            "prompt": prompt,
            "seed": seed,
            "output_filename": output_filename,
        }
        if audio_source:
            settings["audio_source"] = audio_source
        settings.update(extra_settings)
        return self._run_task(settings)

    def generate_batch(
        self,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Submit multiple generation tasks as a batch.

        Args:
            tasks: List of settings dicts (same format as individual methods).

        Returns:
            Dict with success, files, errors, duration_seconds.
        """
        if self._url:
            return self._remote_post('/api/batch', tasks)
        self._ensure_session()

        start_time = time.time()
        job = self._session.submit_manifest(tasks)
        result = job.result()
        duration = time.time() - start_time

        return {
            "success": result.success,
            "files": result.generated_files,
            "errors": [str(e.message) for e in result.errors],
            "total_tasks": result.total_tasks,
            "successful_tasks": result.successful_tasks,
            "failed_tasks": result.failed_tasks,
            "duration_seconds": round(duration, 2),
        }

    # ------------------------------------------------------------------ #
    #  Informational methods
    # ------------------------------------------------------------------ #

    def list_models(self) -> dict[str, list[str]]:
        """
        List available model types grouped by family.

        Returns:
            Dict mapping family name to list of model_type strings.
        """
        if self._url:
            return self._remote_get('/api/models')
        self._ensure_session()
        import wgp
        families = {}
        for handler_path in wgp.family_handlers:
            try:
                import importlib
                mod = importlib.import_module(handler_path)
                handler = mod.family_handler
                types = handler.query_supported_types()
                infos = handler.query_family_infos()
                family_name = list(infos.values())[0][1] if infos else handler.query_model_family()
                families[family_name] = types
            except Exception:
                pass
        return families

    def list_loras(self, model_type: str = "z_image") -> list[str]:
        """
        List available lora files for a given model type.

        Args:
            model_type: The model type to list loras for.

        Returns:
            List of lora filenames.
        """
        if self._url:
            import urllib.parse
            return self._remote_get(f'/api/loras?model_type={urllib.parse.quote(model_type)}')
        self._ensure_session()
        import wgp
        try:
            lora_dir = wgp.get_lora_dir(model_type)
            if not os.path.isdir(lora_dir):
                return []
            return sorted([
                f for f in os.listdir(lora_dir)
                if f.endswith(('.safetensors', '.pt', '.pth', '.ckpt', '.lset'))
            ])
        except Exception:
            return []

    def get_default_settings(self) -> dict[str, Any]:
        """
        Get the default settings template.

        Returns:
            Dict of all available settings with their default values.
        """
        if self._url:
            return self._remote_get('/api/settings')
        settings_path = self._root / "models" / "_settings.json"
        if settings_path.exists():
            with open(settings_path, "r") as f:
                return json.load(f)
        return {}

    def release_model(self):
        """Release the currently loaded model from VRAM."""
        if self._url:
            self._remote_post('/api/release', {})
            return
        if self._session is not None:
            self._session.close()
            self._session = None

    def close(self):
        """Release resources."""
        if not self._url:
            self.release_model()

    def download_file(self, remote_path: str, local_path: str | None = None) -> str:
        """
        Download a generated file from the remote server.

        In local mode this simply returns remote_path unchanged.

        Args:
            remote_path: File path from result["files"].
            local_path: Local destination. Defaults to filename in cwd.

        Returns:
            Path to the local file.
        """
        if not self._url:
            return remote_path
        import urllib.request
        import urllib.parse
        url = f"{self._url}/api/file?path={urllib.parse.quote(remote_path)}"
        if local_path is None:
            local_path = os.path.basename(remote_path)
        urllib.request.urlretrieve(url, local_path)
        return local_path


# ------------------------------------------------------------------ #
#  HTTP API Server
# ------------------------------------------------------------------ #

def serve(
    host: str = "localhost",
    port: int = 8100,
    **agent_kwargs,
):
    """
    Start an HTTP API server wrapping WanGPAgent.

    Clients connect with: WanGPAgent(url="http://host:port")

    Endpoints:
        GET  /api/health              Health check
        POST /api/generate            Submit a single generation task
        POST /api/batch               Submit multiple tasks
        GET  /api/models              List model families
        GET  /api/loras?model_type=X  List loras for a model type
        GET  /api/settings            Get default settings template
        POST /api/release             Release loaded model from VRAM
        GET  /api/file?path=X         Download a generated file

    Args:
        host: Bind address (default: "localhost").
        port: Port number (default: 8100).
        **agent_kwargs: Forwarded to WanGPAgent() for local init.
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    import urllib.parse as _up
    import threading

    agent = WanGPAgent(**agent_kwargs)
    gen_lock = threading.Lock()

    # Directories from which files may be served
    _allowed = set()
    if agent._root:
        _allowed.add(os.path.realpath(agent._root / "outputs"))
    if agent._output_dir:
        _allowed.add(os.path.realpath(agent._output_dir))

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            p = self.path.split('?')[0]
            try:
                if p == '/api/health':
                    self._ok({"status": "ok"})
                elif p == '/api/models':
                    self._ok(agent.list_models())
                elif p == '/api/loras':
                    q = self._qs()
                    self._ok(agent.list_loras(q.get('model_type', 'z_image')))
                elif p == '/api/settings':
                    self._ok(agent.get_default_settings())
                elif p == '/api/file':
                    self._serve_file()
                else:
                    self._err("not found", 404)
            except Exception as exc:
                self._err(str(exc), 500)

        def do_POST(self):
            p = self.path.split('?')[0]
            try:
                body = self._body()
                if p == '/api/generate':
                    with gen_lock:
                        self._ok(agent._run_task(body))
                elif p == '/api/batch':
                    with gen_lock:
                        result = agent.generate_batch(body)
                    self._ok(result)
                elif p == '/api/release':
                    with gen_lock:
                        agent.release_model()
                    self._ok({"ok": True})
                else:
                    self._err("not found", 404)
            except Exception as exc:
                self._err(str(exc), 500)

        # --- helpers ---

        def _body(self):
            n = int(self.headers.get('Content-Length', 0))
            return json.loads(self.rfile.read(n)) if n else {}

        def _qs(self):
            if '?' not in self.path:
                return {}
            return dict(_up.parse_qsl(self.path.split('?', 1)[1]))

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

        def _serve_file(self):
            q = self._qs()
            fp = q.get('path', '')
            if not fp or not os.path.isfile(fp):
                return self._err("file not found", 404)
            real = os.path.realpath(fp)
            if not any(real.startswith(d + os.sep) for d in _allowed):
                return self._err("access denied", 403)
            with open(real, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Content-Disposition',
                             f'attachment; filename="{os.path.basename(fp)}"')
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *a):
            if agent._verbose:
                super().log_message(fmt, *a)

    class _Server(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    srv = _Server((host, port), _Handler)
    print(f"WanGP Agent API server on http://{host}:{port}")
    print(f"Connect with: WanGPAgent(url='http://{host}:{port}')")
    print("Press Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        agent.close()
        srv.server_close()


# ------------------------------------------------------------------ #
#  Convenience CLI for quick testing
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WanGP Agent API - Quick Test")
    sub = parser.add_subparsers(dest="command")

    # Image generation
    img_parser = sub.add_parser("image", help="Generate an image")
    img_parser.add_argument("prompt", help="Text prompt")
    img_parser.add_argument("--model", default="z_image", help="Model type")
    img_parser.add_argument("--resolution", default="1024x1024", help="Resolution WxH")
    img_parser.add_argument("--steps", type=int, default=8, help="Denoising steps")
    img_parser.add_argument("--seed", type=int, default=-1, help="Random seed")

    # Video generation
    vid_parser = sub.add_parser("video", help="Generate a video")
    vid_parser.add_argument("prompt", help="Text prompt")
    vid_parser.add_argument("--model", default="wan21_t2v_14B", help="Model type")
    vid_parser.add_argument("--resolution", default="832x480", help="Resolution WxH")
    vid_parser.add_argument("--steps", type=int, default=30, help="Denoising steps")
    vid_parser.add_argument("--frames", type=int, default=81, help="Number of frames")
    vid_parser.add_argument("--seed", type=int, default=-1, help="Random seed")

    # List models
    sub.add_parser("models", help="List available models")

    # List loras
    lora_parser = sub.add_parser("loras", help="List available loras")
    lora_parser.add_argument("--model", default="z_image", help="Model type")

    # API server
    serve_parser = sub.add_parser("serve", help="Start HTTP API server")
    serve_parser.add_argument("--host", default="localhost", help="Bind address")
    serve_parser.add_argument("--port", type=int, default=8100, help="Port")
    serve_parser.add_argument("--profile", type=int, default=4, help="Memory profile")
    serve_parser.add_argument("--attention", default="sage2", help="Attention mode")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "serve":
        serve(
            host=args.host, port=args.port,
            profile=args.profile, attention=args.attention,
        )
        sys.exit(0)

    agent = WanGPAgent()

    if args.command == "image":
        print(f"Generating image: {args.prompt}")
        result = agent.generate_image(
            prompt=args.prompt,
            model=args.model,
            resolution=args.resolution,
            steps=args.steps,
            seed=args.seed,
        )
        print(f"\nResult: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Files: {result['files']}")
        print(f"Duration: {result['duration_seconds']}s")

    elif args.command == "video":
        print(f"Generating video: {args.prompt}")
        result = agent.generate_video(
            prompt=args.prompt,
            model=args.model,
            resolution=args.resolution,
            steps=args.steps,
            frames=args.frames,
            seed=args.seed,
        )
        print(f"\nResult: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Files: {result['files']}")
        print(f"Duration: {result['duration_seconds']}s")

    elif args.command == "models":
        models = agent.list_models()
        for family, types in models.items():
            print(f"\n{family}:")
            for t in types:
                print(f"  - {t}")

    elif args.command == "loras":
        loras = agent.list_loras(args.model)
        print(f"Loras for {args.model}:")
        for l in loras:
            print(f"  - {l}")
