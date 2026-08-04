#!/usr/bin/env python3
"""
WanGP Agent API Wrapper
=======================
A simplified, agent-friendly interface around WanGP's in-process Python API.

This wrapper provides:
  - Simple function calls for image/video/audio generation
  - Late media post-processing, audio remuxing, and audio editing
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

MINIMAX_H3_FL2VA_MODELS = {
    "minimax_h3_fl2va",
    "minimax_h3_fl2va_pruned",
}
MINIMAX_H3_REF2VA_MODELS = {
    "minimax_h3_ref2va",
    "minimax_h3_ref2va_pruned",
}
MINIMAX_H3_MODELS = MINIMAX_H3_FL2VA_MODELS | MINIMAX_H3_REF2VA_MODELS
MINIMAX_H3_TEXT_ENCODER_CONFIGS = {
    "",
    "bf16",
    "int8",
    "nvfp4_awq",
    "gguf_q2_k",
    "gguf_q4_k_m",
}


def _merge_prompt_flags(current: str | None, flags: str) -> str:
    """Append prompt-mode flags once while preserving separators/order."""
    value = str(current or "")
    for flag in flags:
        if flag not in value:
            value += flag
    return value


def _path_list(values: list[str | Path] | None) -> list[str]:
    return [os.fspath(value) for value in (values or [])]


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
        token: str | None = None,
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
            token: Bearer token for remote mode. Falls back to WAN2GP_TOKEN env var.
        """
        self._url = url.rstrip('/') if url else None
        self._timeout = timeout
        self._verbose = verbose
        self._token = token or os.environ.get("WAN2GP_TOKEN") or None
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

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _remote_get(self, endpoint: str) -> Any:
        """GET request to the remote API server."""
        import urllib.request
        url = f"{self._url}{endpoint}"
        req = urllib.request.Request(url, headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def _remote_post(self, endpoint: str, data: Any) -> Any:
        """POST request to the remote API server."""
        import urllib.request
        url = f"{self._url}{endpoint}"
        body = json.dumps(data).encode('utf-8')
        headers = {'Content-Type': 'application/json', **self._auth_headers()}
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def _remote_multipart_file(
        self,
        endpoint: str,
        file_path: str | Path,
        *,
        filename: str | None = None,
    ) -> Any:
        """POST one file using the server's bounded multipart contract."""
        import mimetypes
        import urllib.request
        import uuid

        path = Path(file_path)
        payload = path.read_bytes()
        upload_name = Path(filename or path.name).name.replace('"', "_").replace("\r", "").replace("\n", "")
        mime_type = mimetypes.guess_type(upload_name)[0] or "application/octet-stream"
        boundary = f"----WanGP-{uuid.uuid4().hex}"
        prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{upload_name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
        body = prefix + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            **self._auth_headers(),
        }
        request = urllib.request.Request(
            f"{self._url}{endpoint}", data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return json.loads(response.read())

    def _remote_delete(self, endpoint: str) -> Any:
        """DELETE request to the remote API server."""
        import urllib.request
        url = f"{self._url}{endpoint}"
        req = urllib.request.Request(url, method='DELETE', headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    # ------------------------------------------------------------------ #
    #  Async job API (remote mode)
    # ------------------------------------------------------------------ #

    def submit_job(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Create a job. Returns the job record (status: 'queued')."""
        if not self._url:
            raise RuntimeError("submit_job() requires remote mode (url=...)")
        return self._remote_post('/api/jobs', settings)

    def get_job(self, job_id: str) -> dict[str, Any]:
        if not self._url:
            raise RuntimeError("get_job() requires remote mode (url=...)")
        return self._remote_get(f'/api/jobs/{job_id}')

    def list_jobs(self, limit: int = 50) -> dict[str, Any]:
        if not self._url:
            raise RuntimeError("list_jobs() requires remote mode (url=...)")
        return self._remote_get(f'/api/jobs?limit={limit}')

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        if not self._url:
            raise RuntimeError("cancel_job() requires remote mode (url=...)")
        return self._remote_delete(f'/api/jobs/{job_id}')

    def wait_for_job(self, job_id: str, *, poll_seconds: float = 1.0,
                     timeout: float | None = None) -> dict[str, Any]:
        """Block until a job reaches a terminal state. Polls /api/jobs/:id."""
        terminal = ("completed", "failed", "cancelled")
        deadline = None if timeout is None else time.time() + timeout
        while True:
            rec = self.get_job(job_id)
            if rec.get("status") in terminal:
                return rec
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
            time.sleep(poll_seconds)

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
            submitted = self.submit_job(settings)
            record = self.wait_for_job(submitted["job_id"])
            error = record.get("error")
            success = record.get("status") == "completed"
            return {
                "success": success,
                "files": record.get("files") or [],
                "errors": [str(error)] if error else [],
                "total_tasks": 1,
                "successful_tasks": 1 if success else 0,
                "failed_tasks": 0 if success else 1,
                "duration_seconds": record.get("duration_seconds") or 0,
                "job_id": record.get("job_id"),
            }
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
        reference_images: list[str | Path] | None = None,
        control_image: str | Path | None = None,
        mask_image: str | Path | None = None,
        video_prompt_type: str = "",
        model_mode: int | str | None = None,
        config: str = "",
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
            reference_images: Reference/conditional images, used by models such
                              as Krea 2 Identity Edit and LTX image workflows.
            control_image: Source/control image for image-to-image or inpainting.
            mask_image: Optional mask paired with ``control_image``.
            video_prompt_type: Explicit WanGP media-mode flags. ``I`` is added
                               automatically when reference images are supplied.
            model_mode: Optional model-specific mode, such as a Krea LanPaint mode.
            config: Optional model config variant.
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
        refs = _path_list(reference_images)
        if refs:
            settings["image_refs"] = refs
            video_prompt_type = _merge_prompt_flags(video_prompt_type, "I")
        if control_image is not None:
            settings["image_guide"] = os.fspath(control_image)
            video_prompt_type = _merge_prompt_flags(video_prompt_type, "V")
        if mask_image is not None:
            settings["image_mask"] = os.fspath(mask_image)
            video_prompt_type = _merge_prompt_flags(video_prompt_type, "AG")
        if video_prompt_type:
            settings["video_prompt_type"] = video_prompt_type
        if model_mode is not None:
            settings["model_mode"] = model_mode
        if config:
            settings["config"] = config
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
        reference_images: list[str | Path] | None = None,
        image_prompt_type: str = "",
        video_prompt_type: str = "",
        config: str = "",
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
            reference_images: Reference images for models such as MiniMax H3
                              Ref2VA and LTX-2 MSR.
            image_prompt_type: Explicit start/end/continuation flags. ``S`` and
                               ``E`` are inferred from ``start_image`` and
                               ``end_image``.
            video_prompt_type: Explicit reference/control flags. ``I`` is added
                               when reference images are supplied.
            config: Optional model config, including quantized text encoders.
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
            image_prompt_type = _merge_prompt_flags(image_prompt_type, "S")
        if end_image:
            settings["image_end"] = end_image
            image_prompt_type = _merge_prompt_flags(image_prompt_type, "E")
        refs = _path_list(reference_images)
        if refs:
            settings["image_refs"] = refs
            video_prompt_type = _merge_prompt_flags(video_prompt_type, "I")
        if image_prompt_type:
            settings["image_prompt_type"] = image_prompt_type
        if video_prompt_type:
            settings["video_prompt_type"] = video_prompt_type
        if config:
            settings["config"] = config
        settings.update(extra_settings)
        return self._run_task(settings)

    def generate_minimax_h3(
        self,
        prompt: str,
        model: str = "minimax_h3_fl2va",
        resolution: str = "832x480",
        steps: int = 20,
        frames: int = 124,
        seed: int = -1,
        start_image: str | Path | None = None,
        end_image: str | Path | None = None,
        reference_images: list[str | Path] | None = None,
        reference_videos: list[str | Path] | None = None,
        reference_audios: list[str | Path] | None = None,
        use_reference_video_soundtracks: bool = False,
        text_encoder_config: str = "",
        loras: list[str] | None = None,
        loras_multipliers: str = "",
        output_filename: str = "",
        **extra_settings: Any,
    ) -> dict[str, Any]:
        """Generate native-audio video with a MiniMax H3 v12.41 model.

        FL2VA accepts optional start/end boundary images. Ref2VA instead
        accepts up to nine images, two videos, and two audio references.
        Paths are interpreted on the WanGP server when using remote mode.
        """
        if model not in MINIMAX_H3_MODELS:
            raise ValueError(f"unsupported MiniMax H3 model: {model}")
        if text_encoder_config not in MINIMAX_H3_TEXT_ENCODER_CONFIGS:
            choices = ", ".join(sorted(v for v in MINIMAX_H3_TEXT_ENCODER_CONFIGS if v))
            raise ValueError(f"invalid MiniMax H3 text encoder config; choose one of: {choices}")

        images = _path_list(reference_images)
        videos = _path_list(reference_videos)
        audios = _path_list(reference_audios)
        is_reference_model = model in MINIMAX_H3_REF2VA_MODELS
        if not is_reference_model:
            if images or videos or audios or use_reference_video_soundtracks:
                raise ValueError("MiniMax H3 references require a Ref2VA model")
        else:
            if start_image is not None or end_image is not None:
                raise ValueError("MiniMax H3 Ref2VA does not accept start/end boundary images")
            if len(images) > 9:
                raise ValueError("MiniMax H3 Ref2VA accepts at most 9 reference images")
            if len(videos) > 2:
                raise ValueError("MiniMax H3 Ref2VA accepts at most 2 reference videos")
            if len(audios) > 2:
                raise ValueError("MiniMax H3 Ref2VA accepts at most 2 audio references")
            if use_reference_video_soundtracks and audios:
                raise ValueError("reference audio and reference-video soundtrack modes are mutually exclusive")
            if use_reference_video_soundtracks and not videos:
                raise ValueError("reference-video soundtrack mode requires a reference video")
            audio_count = len(videos) if use_reference_video_soundtracks else len(audios)
            visual_count = len(images) + len(videos)
            if audio_count > visual_count:
                raise ValueError("MiniMax H3 requires at least as many visual references as audio references")
            file_count = visual_count + (0 if use_reference_video_soundtracks else len(audios))
            if file_count == 0:
                raise ValueError("MiniMax H3 Ref2VA requires at least one reference")
            if file_count > 12:
                raise ValueError("MiniMax H3 Ref2VA accepts at most 12 reference files")

        video_prompt_type = ""
        if len(videos) == 1:
            video_prompt_type = "VG"
        elif len(videos) == 2:
            video_prompt_type = "V+G"
        audio_prompt_type = ""
        if use_reference_video_soundtracks:
            audio_prompt_type = "K"
        elif len(audios) == 1:
            audio_prompt_type = "A"
        elif len(audios) == 2:
            audio_prompt_type = "AB"

        if videos:
            extra_settings["video_guide"] = videos[0]
        if len(videos) > 1:
            extra_settings["video_guide2"] = videos[1]
        if audios:
            extra_settings["audio_guide"] = audios[0]
        if len(audios) > 1:
            extra_settings["audio_guide2"] = audios[1]
        if audio_prompt_type:
            extra_settings["audio_prompt_type"] = audio_prompt_type

        return self.generate_video(
            prompt=prompt,
            model=model,
            resolution=resolution,
            steps=steps,
            frames=frames,
            seed=seed,
            guidance_scale=1.0,
            flow_shift=12.0,
            fps=24,
            loras=loras,
            loras_multipliers=loras_multipliers,
            start_image=None if start_image is None else os.fspath(start_image),
            end_image=None if end_image is None else os.fspath(end_image),
            output_filename=output_filename,
            reference_images=images,
            video_prompt_type=video_prompt_type,
            config=text_encoder_config,
            **extra_settings,
        )

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

    def postprocess_media(
        self,
        media_source: str | Path,
        *,
        temporal_upsampling: str = "",
        spatial_upsampling: str = "",
        film_grain_intensity: float = 0,
        film_grain_saturation: float = 0.5,
        seed: int = -1,
        **extra_settings: Any,
    ) -> dict[str, Any]:
        """Late-postprocess an existing image or video."""
        settings = {
            "mode": "edit_postprocessing",
            "prompt": "Media postprocessing",
            "image_mode": 0,
            "video_source": os.fspath(media_source),
            "temporal_upsampling": temporal_upsampling,
            "spatial_upsampling": spatial_upsampling,
            "film_grain_intensity": film_grain_intensity,
            "film_grain_saturation": film_grain_saturation,
            "postprocess_audio": "",
            "repeat_generation": 1,
            "batch_size": 1,
            "seed": seed,
        }
        settings.update(extra_settings)
        settings.pop("model_type", None)
        return self._run_task(settings)

    def remux_audio(
        self,
        video_source: str | Path,
        *,
        postprocess_audio: str,
        audio_source: str | Path | None = None,
        postprocess_audio_prompt: str = "",
        postprocess_audio_neg_prompt: str = "",
        seed: int = -1,
        repeat_generation: int = 1,
        replace_voice_sample: str | Path | None = None,
        replace_voice_sample2: str | Path | None = None,
        **extra_settings: Any,
    ) -> dict[str, Any]:
        """Replace or generate the audio track of an existing video."""
        settings = {
            "mode": "edit_remux",
            "prompt": "Audio remuxing",
            "image_mode": 0,
            "video_source": os.fspath(video_source),
            "postprocess_audio": postprocess_audio,
            "postprocess_audio_prompt": postprocess_audio_prompt,
            "postprocess_audio_neg_prompt": postprocess_audio_neg_prompt,
            "seed": seed,
            "repeat_generation": repeat_generation,
            "audio_source": None if audio_source is None else os.fspath(audio_source),
            "replace_voice_sample": None if replace_voice_sample is None else os.fspath(replace_voice_sample),
            "replace_voice_sample2": None if replace_voice_sample2 is None else os.fspath(replace_voice_sample2),
            "temporal_upsampling": "",
            "spatial_upsampling": "",
            "film_grain_intensity": 0,
            "film_grain_saturation": 0.5,
            "batch_size": 1,
        }
        settings.update(extra_settings)
        settings.pop("model_type", None)
        return self._run_task(settings)

    def postprocess_audio(
        self,
        audio_source: str | Path,
        *,
        postprocess_audio: str,
        replace_voice_sample: str | Path | None = None,
        replace_voice_sample2: str | Path | None = None,
        **extra_settings: Any,
    ) -> dict[str, Any]:
        """Late-postprocess an existing audio file."""
        settings = {
            "mode": "edit_audio",
            "prompt": "Audio postprocessing",
            "image_mode": 0,
            "audio_source": os.fspath(audio_source),
            "postprocess_audio": postprocess_audio,
            "replace_voice_sample": None if replace_voice_sample is None else os.fspath(replace_voice_sample),
            "replace_voice_sample2": None if replace_voice_sample2 is None else os.fspath(replace_voice_sample2),
            "repeat_generation": 1,
            "batch_size": 1,
        }
        settings.update(extra_settings)
        settings.pop("model_type", None)
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

    def discover_models(
        self,
        *,
        family: str | None = None,
        capability: str | None = None,
        input_modality: str | None = None,
        output_modality: str | None = None,
    ) -> dict[str, Any]:
        """Return enriched model records suitable for building an API UI.

        Filters accept comma-separated values. For example, request models
        that consume images and return video with
        ``input_modality="image", output_modality="video"``.
        """
        filters = {
            "family": family,
            "capability": capability,
            "input": input_modality,
            "output": output_modality,
        }
        if self._url:
            import urllib.parse

            query = urllib.parse.urlencode({k: v for k, v in filters.items() if v})
            endpoint = "/api/models" + (f"?{query}" if query else "")
            return self._remote_get(endpoint)

        import agent_api_introspect

        index_data = agent_api_introspect.build_index()
        requested = {
            key: {part.strip().casefold() for part in str(value or "").replace("|", ",").split(",") if part.strip()}
            for key, value in filters.items()
        }
        records = []
        families: dict[str, list[str]] = {}
        for entry in index_data["models"].values():
            metadata = entry.get("api_metadata") or {}
            if requested["family"] and str(entry["family"]).casefold() not in requested["family"]:
                continue
            if requested["capability"] and str(entry["capability"]).casefold() not in requested["capability"]:
                continue
            if requested["input"] and not requested["input"].intersection(str(v).casefold() for v in metadata.get("inputs") or []):
                continue
            if requested["output"] and not requested["output"].intersection(str(v).casefold() for v in metadata.get("outputs") or []):
                continue
            record = agent_api_introspect.public_entry(entry, include_model_def=False)
            records.append(record)
            families.setdefault(entry["family"], []).append(entry["model_type"])
        records.sort(key=lambda record: (record["family"], record["model_type"]))
        return {
            "models": records,
            "families": families,
            "filters": {key: sorted(value) for key, value in requested.items()},
            "errors": index_data.get("errors") or [],
        }

    def get_model(self, model_type: str) -> dict[str, Any] | None:
        """Return one enriched model record and its accepted settings."""
        if self._url:
            import urllib.parse

            return self._remote_get(f"/api/models/{urllib.parse.quote(model_type)}")
        import agent_api_introspect

        entry = agent_api_introspect.get_model_entry(model_type)
        return None if entry is None else agent_api_introspect.public_entry(entry)

    def get_health(self) -> dict[str, Any]:
        """Return dedicated server health, or a compact local-mode status."""
        if self._url:
            return self._remote_get("/api/health")
        return {"status": "ok", "mode": "local", "root": str(self._root)}

    def get_settings_schema(self) -> dict[str, Any]:
        """Return typed registered settings plus freeform WanGP keys."""
        if self._url:
            return self._remote_get("/api/settings/schema")
        import agent_api_introspect

        registered = agent_api_introspect.get_settings_schema()
        registered_keys = {entry["key"] for entry in registered}
        freeform = [
            {"key": key, "default": value, "type": type(value).__name__}
            for key, value in self.get_default_settings().items()
            if key not in registered_keys
        ]
        return {"registered": registered, "freeform": freeform}

    def validate_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Dry-run validate settings without queueing a generation."""
        if self._url:
            return self._remote_post("/api/settings/validate", settings)
        import agent_api_introspect

        model_type = str(settings.get("model_type") or "")
        if not model_type:
            return {"valid": False, "error": "request must include model_type"}
        error = agent_api_introspect.validate_request(model_type, settings)
        return {"valid": error is None, **({"error": error} if error else {})}

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
        req = urllib.request.Request(url, headers=self._auth_headers())
        with urllib.request.urlopen(req, timeout=self._timeout) as resp, open(local_path, 'wb') as out:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        return local_path

    def upload_file(
        self,
        local_path: str | Path,
        *,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Upload an image/video/audio input and return its server-side path.

        Pass the returned ``path`` to ``reference_images``,
        ``reference_videos``, ``reference_audios``, or other media settings.
        In local mode no copy is needed and the resolved local path is returned.
        """
        path = Path(local_path).resolve()
        if not self._url:
            return {
                "upload_id": None,
                "filename": filename or path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
            }
        return self._remote_multipart_file("/api/uploads", path, filename=filename)


# ------------------------------------------------------------------ #
#  HTTP API Server
# ------------------------------------------------------------------ #

def serve(
    host: str = "0.0.0.0",
    port: int = 8100,
    profile: int = 4,
    attention: str = "sage2",
    outputs_root: str | None = None,
    token: str | None = None,
    history_limit: int | None = None,
    cors_origins: str | None = None,
    **_legacy_kwargs,
):
    """
    Start the hardened WanGP Agent API server.

    Delegates to ``agent_api_server.serve`` which provides:
        - async /api/jobs lifecycle (POST/GET/DELETE/SSE)
        - bearer-token auth (env WAN2GP_TOKEN)
        - constrained /api/file (env WAN2GP_OUTPUTS_ROOT)
        - rich /api/health (gpu, queue, version)
        - structured JSON logs with request/job correlation
        - back-compat sync /api/generate and /api/batch with Deprecation header

    Clients connect with: ``WanGPAgent(url="http://host:port", token=...)``.
    """
    from agent_api_server import serve as _serve
    _serve(
        host=host,
        port=port,
        profile=profile,
        attention=attention,
        outputs_root=outputs_root,
        token=token,
        history_limit=history_limit,
        cors_origins=cors_origins,
    )


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
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    serve_parser.add_argument("--port", type=int, default=8100, help="Port")
    serve_parser.add_argument("--profile", type=int, default=4, help="Memory profile")
    serve_parser.add_argument("--attention", default="sage2", help="Attention mode")
    serve_parser.add_argument("--outputs-root", default=None,
                              help="Override outputs root for /api/file (default: <repo>/outputs)")
    serve_parser.add_argument("--token", default=None,
                              help="Bearer token; falls back to WAN2GP_TOKEN env var")
    serve_parser.add_argument("--history-limit", type=int, default=None,
                              help="Number of jobs to retain in SQLite history (default: 200)")
    serve_parser.add_argument("--cors-origins", default=None,
                              help="Comma-separated CORS allow-list (e.g. 'http://localhost:5173') or '*'. "
                                   "Falls back to WAN2GP_CORS_ORIGINS env var. Empty = CORS disabled.")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "serve":
        serve(
            host=args.host, port=args.port,
            profile=args.profile, attention=args.attention,
            outputs_root=args.outputs_root,
            token=args.token,
            history_limit=args.history_limit,
            cors_origins=args.cors_origins,
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
