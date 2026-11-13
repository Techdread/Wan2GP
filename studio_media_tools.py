"""WGS-012: installed-tool media jobs, with no model/weight discovery effects.

The HTTP adapter owns authentication/queue admission. This module accepts only
enumerated operations and local files inside the service's output root.
"""
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

MEDIA_CAPABILITY = "media-processing"
OPERATIONS = {"mp3": ".mp3", "flac": ".flac", "png": ".png", "upscale2x": ".png"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".aif"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv"}
MAX_SOURCE_BYTES = 1024 * 1024 * 1024
MAX_PATH_CHARS = 4096
MAX_AUDIO_SECONDS = 1800
MAX_OUTPUT_PIXELS = 16_777_216
MAX_OUTPUT_SIDE = 8192
MAX_PROBE_BYTES = 1_048_576
MAX_PROCESS_LOG_BYTES = 16384
MAX_MEDIA_STREAMS = 32
PROCESS_STOP_SECONDS = 5
MP3_BITRATE = "320k"
FLAC_COMPRESSION_LEVEL = "8"
UPSCALE_FACTOR = 2
PROCESS_TIMEOUT_SECONDS = 300
PROBE_TIMEOUT_SECONDS = 15
POLL_SECONDS = 0.1
MAX_INITIAL_PROMPT_CHARS = 2048
MAX_MEDIA_REQUEST_BYTES = 16384
INPUT_DEMUXERS = 'wav,mp3,flac,ogg,mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,aac,aiff,png_pipe,jpeg_pipe,webp_pipe,bmp_pipe,gif'
PUBLIC_KEYS = {"operation", "source_path", "source_model_type"}


class MediaError(Exception):
    def __init__(self, code, **params):
        self.code, self.params = code, params
        super().__init__(code)


class MediaCancelled(MediaError):
    def __init__(self):
        super().__init__("media.cancelled")


def _refuse(code, **params):
    raise MediaError(code, **params)


def _inside_file(path, outputs_root):
    if not isinstance(path, str) or not path or len(path) > MAX_PATH_CHARS or "\0" in path:
        _refuse("media.source_invalid")
    try:
        root = Path(outputs_root).resolve(strict=True)
        source = Path(path).resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file():
            _refuse("media.source_forbidden")
        if source.stat().st_size > MAX_SOURCE_BYTES:
            _refuse("media.input_too_large", limit=MAX_SOURCE_BYTES)
        return source
    except (OSError, ValueError):
        _refuse("media.source_invalid")


def validate_media_request(body, outputs_root):
    if not isinstance(body, dict) or set(body) - PUBLIC_KEYS:
        _refuse("media.request_invalid")
    operation = body.get("operation")
    if not isinstance(operation, str) or operation not in OPERATIONS:
        _refuse("media.operation_unsupported")
    source = _inside_file(body.get("source_path"), outputs_root)
    allowed = IMAGE_EXTENSIONS if operation in {"png", "upscale2x"} else AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
    if source.suffix.lower() not in allowed:
        _refuse("media.input_unsupported")
    model = body.get("source_model_type", f"media_{operation}")
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", model):
        _refuse("media.request_invalid")
    return {"operation": operation, "source_path": str(source), "source_model_type": model}


def tool_paths(service_root):
    suffix = ".exe" if os.name == "nt" else ""
    directory = Path(service_root) / "ffmpeg_bins"
    ffmpeg, ffprobe = directory / f"ffmpeg{suffix}", directory / f"ffprobe{suffix}"
    if not ffmpeg.is_file() or not ffprobe.is_file():
        _refuse("media.tools_unavailable")
    return ffmpeg, ffprobe


def _spawn_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


@lru_cache(maxsize=4)
def media_capabilities(service_root):
    try:
        ffmpeg, _ = tool_paths(service_root)
        result = subprocess.run([str(ffmpeg), "-hide_banner", "-encoders"], capture_output=True, timeout=PROBE_TIMEOUT_SECONDS, **_spawn_options())
        encoders = result.stdout.decode("utf-8", errors="replace") if result.returncode == 0 else ""
        operations = ["png", "upscale2x"] if re.search(r"\bpng\b", encoders) else []
        if re.search(r"\blibmp3lame\b", encoders):
            operations.append("mp3")
        if re.search(r"\bflac\b", encoders):
            operations.append("flac")
        return {"operations": operations, "stems": False, "midi": False, "unavailable": {"stems": "media.stems_not_installed", "midi": "media.midi_not_installed"}}
    except (MediaError, OSError, subprocess.SubprocessError):
        return {"operations": [], "stems": False, "midi": False, "unavailable": {"processing": "media.tools_unavailable", "stems": "media.stems_not_installed", "midi": "media.midi_not_installed"}}


def _stop(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=PROCESS_STOP_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=PROCESS_STOP_SECONDS)


def _run(command, timeout, cancelled, progress):
    # File-backed logs avoid deadlocks and unbounded pipe capture. Only a
    # bounded tail is read; paths/decoder prose never become client errors.
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors, **_spawn_options())
        started = time.monotonic()
        progress(0.0)  # No invented percentage from the timeout budget.
        try:
            while process.poll() is None:
                if cancelled():
                    raise MediaCancelled()
                if time.monotonic() - started > timeout:
                    _refuse("media.processing_timeout")
                if os.fstat(output.fileno()).st_size > MAX_PROBE_BYTES or os.fstat(errors.fileno()).st_size > MAX_PROCESS_LOG_BYTES:
                    _refuse("media.metadata_too_large")
                time.sleep(POLL_SECONDS)
            if cancelled():
                raise MediaCancelled()
            if process.returncode != 0:
                _refuse("media.decoder_failed")
            output.seek(0, os.SEEK_END)
            if output.tell() > MAX_PROBE_BYTES:
                _refuse("media.metadata_too_large")
            output.seek(0)
            return output.read(MAX_PROBE_BYTES)
        finally:
            _stop(process)


def _probe(source, ffprobe, cancelled):
    data = _run([str(ffprobe), "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", INPUT_DEMUXERS, "-show_entries", "format=duration,format_name:stream=codec_type,codec_name,width,height,channels,sample_rate", "-of", "json", str(source)], PROBE_TIMEOUT_SECONDS, cancelled, lambda _: None)
    try:
        result = json.loads(data)
    except (ValueError, TypeError):
        _refuse("media.decoder_failed")
    streams = result.get("streams")
    if not isinstance(streams, list) or not streams or len(streams) > MAX_MEDIA_STREAMS:
        _refuse("media.input_unsupported")
    if any(stream.get("codec_name") in {"svg", "svg_pipe"} for stream in streams):
        _refuse("media.input_unsupported")
    return result


def run_media_operation(body, *, outputs_root, service_root, job_id, cancelled, progress):
    request = validate_media_request(body, outputs_root)
    operation = request["operation"]
    if operation not in media_capabilities(str(service_root))["operations"]:
        _refuse("media.operation_unsupported")
    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        _refuse("media.request_invalid")
    if cancelled():
        raise MediaCancelled()
    ffmpeg, ffprobe = tool_paths(service_root)
    source = Path(request["source_path"])
    metadata = _probe(source, ffprobe, cancelled)
    audio = next((stream for stream in metadata["streams"] if stream.get("codec_type") == "audio"), None)
    image = next((stream for stream in metadata["streams"] if stream.get("codec_type") == "video"), None)
    if operation in {"mp3", "flac"}:
        if not audio:
            _refuse("media.no_audio")
        try:
            duration = float(metadata.get("format", {}).get("duration", "nan"))
        except (ValueError, TypeError):
            duration = float("nan")
        if not math.isfinite(duration) or not 0 < duration <= MAX_AUDIO_SECONDS:
            _refuse("media.duration_limit", limit=MAX_AUDIO_SECONDS)
        encoding = ["-map", "0:a:0", "-vn", "-c:a", "libmp3lame", "-b:a", MP3_BITRATE] if operation == "mp3" else ["-map", "0:a:0", "-vn", "-c:a", "flac", "-compression_level", FLAC_COMPRESSION_LEVEL]
    else:
        if not image:
            _refuse("media.no_image")
        width, height = image.get("width", 0), image.get("height", 0)
        factor = UPSCALE_FACTOR if operation == "upscale2x" else 1
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0 or max(width, height) * factor > MAX_OUTPUT_SIDE or width * height * factor * factor > MAX_OUTPUT_PIXELS:
            _refuse("media.image_limit", limit=MAX_OUTPUT_PIXELS)
        encoding = ["-map", "0:v:0", "-frames:v", "1", "-c:v", "png"]
        if operation == "upscale2x":
            encoding += ["-vf", f"scale=iw*{UPSCALE_FACTOR}:ih*{UPSCALE_FACTOR}:flags=lanczos"]
    root = Path(outputs_root).resolve(strict=True)
    media_directory = root / "media"
    directory = media_directory / job_id
    # Check each ancestor before creating anything below it. A symlink or
    # Windows junction must not make even an empty directory outside the root.
    for candidate in (media_directory, directory):
        if candidate.is_symlink() or candidate.resolve() != candidate:
            _refuse("media.source_forbidden")
        candidate.mkdir(exist_ok=True)
        if candidate.resolve() != candidate or not candidate.is_dir():
            _refuse("media.source_forbidden")
    target = directory / f"output{OPERATIONS[operation]}"
    if target.exists() or target.is_symlink():
        _refuse("media.output_taken")
    try:
        command = [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", INPUT_DEMUXERS, "-i", str(source), *encoding, "-n", str(target)]
        _run(command, PROCESS_TIMEOUT_SECONDS, cancelled, progress)
        if not target.is_file() or not target.stat().st_size:
            _refuse("media.output_missing")
        return [str(target)], {"operation": operation, "source_path": str(source)}
    except BaseException:
        if target.parent.resolve().is_relative_to(root):
            target.unlink(missing_ok=True)
        raise


def transcription_options(fields):
    word_value = fields.get("word_timestamps", "false")
    if not isinstance(word_value, str) or word_value not in {"true", "false", "1", "0"}:
        _refuse("transcription.options_invalid")
    context = fields.get("initial_prompt", "")
    if not isinstance(context, str) or len(context) > MAX_INITIAL_PROMPT_CHARS or "\0" in context:
        _refuse("transcription.context_limit", limit=MAX_INITIAL_PROMPT_CHARS)
    return {"word_timestamps": word_value in {"true", "1"}, "initial_prompt": context}


def whisper_option_arguments(request):
    arguments = ["--word_timestamps", "True"] if request.get("word_timestamps") else []
    if request.get("initial_prompt"):
        arguments += [f"--initial_prompt={request['initial_prompt']}"]
    return arguments


def transcript_segments(data):
    result = []
    for segment in data.get("segments") or []:
        item = {"startSec": float(segment.get("start") or 0), "endSec": float(segment.get("end") or 0), "text": str(segment.get("text") or "").strip()}
        if segment.get("words"):
            item["words"] = [{"startSec": float(word["start"]), "endSec": float(word["end"]), "text": str(word.get("word") or "").strip()} for word in segment["words"]]
        times = [item, *item.get("words", [])]
        if any(not math.isfinite(value[key]) for value in times for key in ("startSec", "endSec")) or any(value["startSec"] < 0 or value["endSec"] < value["startSec"] for value in times):
            _refuse("transcription.timing_invalid")
        result.append(item)
    return result
