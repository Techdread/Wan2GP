"""Fine-tuning job requests: the bases we can train, validation, defaults and a
memory estimate. Pure Python (no torch) so the API server and tests can import
it without loading a GPU stack.

A request is POSTed to /api/jobs like any other job:

    {"model_type": "finetune-qwen3.5-4b", "name": "glb-coder-v1",
     "dataset_path": "<path returned by /api/uploads>", "epochs": 1, ...}
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

FINETUNE_PREFIX = "finetune-"
FINETUNE_CAPABILITY = "fine-tuning"

# Text part of each base, from the published configs. `params` is the whole
# checkpoint (vision tower and MTP head included); training loads text only.
BASES: dict[str, dict[str, Any]] = {
    "qwen3.5-0.8b": {"repo": "Qwen/Qwen3.5-0.8B", "params": 0.87e9, "text_gb": 1.6, "hidden": 1024, "layers": 24, "intermediate": 3584},
    "qwen3.5-2b": {"repo": "Qwen/Qwen3.5-2B", "params": 2.27e9, "text_gb": 4.2, "hidden": 2048, "layers": 24, "intermediate": 6144},
    "qwen3.5-4b": {"repo": "Qwen/Qwen3.5-4B", "params": 4.66e9, "text_gb": 8.6, "hidden": 2560, "layers": 32, "intermediate": 9216},
}

GGUF_TYPES = ("q8_0", "bf16", "f16", "none")
VOCAB = 248320
LOSS_CHUNK = 1024  # tokens per slice of the output layer: logits for 16k tokens at once would be 16 GB

DEFAULTS: dict[str, Any] = {
    "epochs": 1.0,
    "learning_rate": 1e-4,
    "lora_rank": 32,
    "lora_alpha": None,       # None: 2 × rank
    "lora_dropout": 0.0,
    "max_length": 16384,      # longer examples are left out, never cut: a cut page teaches cut pages
    "grad_accum": 8,
    "warmup_ratio": 0.03,
    "seed": 1,
    "holdout": 32,            # examples kept back to watch eval loss
    "eval_every": 50,         # optimizer steps
    "save_every": 50,         # optimizer steps; every adapter checkpoint is kept
    "gguf": "q8_0",
    "lmstudio": True,         # copy the GGUF into this machine's LM Studio models folder
    "resume_from": None,      # job id of an earlier fine-tune to continue from its last checkpoint
}

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


def is_finetune_model_type(model_type: Any) -> bool:
    return isinstance(model_type, str) and model_type.startswith(FINETUNE_PREFIX)


def base_for_model_type(model_type: str) -> str | None:
    base = model_type[len(FINETUNE_PREFIX):] if is_finetune_model_type(model_type) else None
    return base if base in BASES else None


def _number(value: Any, name: str, *, lo: float, hi: float, integer: bool = False) -> tuple[Any, str | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None, f"{name} must be a number"
    if integer and int(value) != value:
        return None, f"{name} must be a whole number"
    if not lo <= value <= hi:
        return None, f"{name} must be between {lo:g} and {hi:g}"
    return (int(value) if integer else float(value)), None


def normalize_request(request: Any, *, uploads_root: Path | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Fill in defaults and check every field. Returns (spec, None) or (None, error).

    `uploads_root`, when given, is where `dataset_path` must live: the server
    only trains on files that came in through /api/uploads.
    """
    if not isinstance(request, dict):
        return None, "request must be a JSON object"
    model_type = request.get("model_type")
    base = base_for_model_type(model_type) if isinstance(model_type, str) else None
    if base is None:
        return None, f"unknown fine-tuning base {model_type!r}; one of: " + ", ".join(FINETUNE_PREFIX + b for b in BASES)

    name = request.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return None, "name must be 1–80 characters: lower-case letters, digits, '.', '_' or '-', starting with a letter or digit"

    dataset_path = request.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        return None, "dataset_path is required (upload the .jsonl with /api/uploads first)"
    if not dataset_path.lower().endswith(".jsonl"):
        return None, "dataset_path must be a .jsonl file"
    if uploads_root is not None:
        try:
            resolved = Path(dataset_path).resolve()
            resolved.relative_to(Path(uploads_root).resolve())
        except (ValueError, OSError):
            return None, "dataset_path must be a file uploaded through /api/uploads"

    spec: dict[str, Any] = {"model_type": model_type, "base": base, "repo": BASES[base]["repo"], "name": name, "dataset_path": dataset_path}
    unknown = sorted(set(request) - set(DEFAULTS) - {"model_type", "name", "dataset_path"})
    if unknown:
        return None, f"unknown fine-tuning option(s): {', '.join(unknown)}"

    limits = {
        "epochs": dict(lo=0.01, hi=20),
        "learning_rate": dict(lo=1e-7, hi=1e-2),
        "lora_rank": dict(lo=1, hi=512, integer=True),
        "lora_alpha": dict(lo=1, hi=2048),
        "lora_dropout": dict(lo=0, hi=0.5),
        "max_length": dict(lo=256, hi=131072, integer=True),
        "grad_accum": dict(lo=1, hi=512, integer=True),
        "warmup_ratio": dict(lo=0, hi=0.5),
        "seed": dict(lo=0, hi=2**31 - 1, integer=True),
        "holdout": dict(lo=0, hi=1000, integer=True),
        "eval_every": dict(lo=1, hi=100000, integer=True),
        "save_every": dict(lo=1, hi=100000, integer=True),
    }
    for key, default in DEFAULTS.items():
        value = request.get(key, default)
        if key in limits:
            if value is None and default is None:
                spec[key] = None
                continue
            value, err = _number(value, key, **limits[key])
            if err:
                return None, err
        elif key == "gguf":
            if value not in GGUF_TYPES:
                return None, f"gguf must be one of: {', '.join(GGUF_TYPES)}"
        elif key == "lmstudio":
            if not isinstance(value, bool):
                return None, "lmstudio must be true or false"
        elif key == "resume_from":
            if value is not None and (not isinstance(value, str) or not re.fullmatch(r"j_[0-9A-Z]{26}", value)):
                return None, "resume_from must be the job id of an earlier fine-tune"
        spec[key] = value
    if spec["lora_alpha"] is None:
        spec["lora_alpha"] = 2 * spec["lora_rank"]
    return spec, None


def estimate_vram_gb(base: str, max_length: int, lora_rank: int = 32) -> float:
    """Rough peak memory for LoRA in bf16 with gradient checkpointing at
    batch 1. An estimate to compare machines with, not a promise: it leaves out
    the CUDA context and whatever else holds memory on that GPU."""
    b = BASES[base]
    weights = b["text_gb"]
    # one saved hidden state per layer, plus one layer's recompute (MLP is the peak)
    per_token = 2 * (b["hidden"] * b["layers"] + 4 * b["intermediate"] + 6 * b["hidden"])
    activations = per_token * max_length / 1e9
    linears_per_layer = 7
    lora_params = lora_rank * b["layers"] * linears_per_layer * (b["hidden"] + max(b["hidden"], b["intermediate"]))
    lora = lora_params * 16 / 1e9  # fp32 weights + grads + two Adam moments
    loss_chunk = LOSS_CHUNK * VOCAB * 4 * 2 / 1e9  # one fp32 logits slice and its gradient
    return round(weights + activations + lora + loss_chunk + 1.0, 1)


def describe_bases() -> list[dict[str, Any]]:
    """What /api/finetune advertises: each base with memory estimates at common lengths."""
    out = []
    for key, b in BASES.items():
        out.append({
            "model_type": FINETUNE_PREFIX + key,
            "base": key,
            "repo": b["repo"],
            "params": b["params"],
            "vram_estimate_gb": {str(n): estimate_vram_gb(key, n) for n in (4096, 8192, 16384, 32768)},
        })
    return out
