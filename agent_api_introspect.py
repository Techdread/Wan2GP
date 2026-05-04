#!/usr/bin/env python3
"""
WanGP API — auto-derived model & settings introspection.

Builds, on demand and cached, a complete capability/schema view of every
model under ``defaults/*.json``:

  - Per-model: architecture, family, name, description, URLs, param-count,
    fully-merged ``model_def`` (handler feature flags + JSON overlay), and
    the *applicable* setting schema for that model.
  - Global: full settings schema (key, label, type, min/max/step) for every
    registered setting.

Hand-curation is bounded to the WanGP runtime, not the model count:
  - new ``defaults/<x>.json``    → zero work (auto-discovered, schema derived)
  - new architecture / handler  → wgp.py:family_handlers picks it up;
                                  this module re-reads that list via AST
                                  on each rebuild, so it stays in sync
  - new setting key             → add one ``_add_setting(...)`` to
                                  ``shared/extra_settings.py``; this module
                                  reflects it automatically

Importantly we DO NOT import ``wgp`` (which has heavy side-effects on
import — Gradio app, model registry writes, etc.). We replicate the small
amount of merge logic needed for read-only introspection.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any


WANGP_ROOT = Path(__file__).resolve().parent
DEFAULTS_DIR = WANGP_ROOT / "defaults"
SETTINGS_TEMPLATE = WANGP_ROOT / "models" / "_settings.json"

_BUILD_LOCK = threading.RLock()
_INDEX_CACHE: dict[str, Any] | None = None
_SCHEMA_CACHE: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Family-handler discovery
# ---------------------------------------------------------------------------

def _extract_family_handlers_from_wgp() -> list[str]:
    """AST-parse wgp.py for the ``family_handlers = [...]`` literal.

    Avoids importing wgp. Stays in sync with new architectures automatically.
    Returns an empty list on parse failure (caller should treat as fatal).
    """
    src = (WANGP_ROOT / "wgp.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "family_handlers":
                if isinstance(node.value, ast.List):
                    out = []
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            out.append(elt.value)
                    return out
    return []


def _install_wgp_stub() -> None:
    """Provide a minimal ``wgp`` module so handlers that touch it during
    ``query_supported_types`` (e.g. ltx2's ``_migrate_loras``) don't crash.

    Only installed if ``wgp`` isn't already loaded. The stub returns the
    canonical WanGP lora root so any migration logic sees real paths.
    """
    if "wgp" in sys.modules:
        return
    import types
    stub = types.ModuleType("wgp")
    stub.__file__ = str(WANGP_ROOT / "wgp.py")  # type: ignore[attr-defined]
    stub.get_lora_root = lambda: str(WANGP_ROOT / "loras")  # type: ignore[attr-defined]
    stub.__wgp_stub__ = True  # type: ignore[attr-defined]
    sys.modules["wgp"] = stub


def _load_handlers() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load every family handler. Returns ``(arch_to_handler, load_errors)``."""
    if str(WANGP_ROOT) not in sys.path:
        sys.path.insert(0, str(WANGP_ROOT))
    _install_wgp_stub()
    handler_paths = _extract_family_handlers_from_wgp()
    arch_to_handler: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    for path in handler_paths:
        try:
            mod = importlib.import_module(path)
            handler = mod.family_handler
            try:
                supported = handler.query_supported_types() or []
            except Exception as exc:
                # Some handlers (e.g. ltx2) do bookkeeping in this method.
                # Fall back to AST-extracting the literal list from the
                # handler source — pure introspection, no side effects.
                supported = _ast_extract_supported_types(path) or []
                if not supported:
                    raise
                errors.append({
                    "handler": path,
                    "error": f"query_supported_types raised: {exc}; using AST fallback",
                    "level": "warn",
                })
            for arch in supported:
                arch_to_handler[arch] = handler
        except Exception as exc:  # noqa: BLE001
            errors.append({"handler": path, "error": str(exc)})
    return arch_to_handler, errors


def _ast_extract_supported_types(handler_module_path: str) -> list[str]:
    """Fallback: AST-scan a handler file for the literal list returned by
    ``query_supported_types``. Returns ``[]`` if it can't find a static list.
    """
    rel = handler_module_path.replace(".", "/") + ".py"
    src_path = WANGP_ROOT / rel
    if not src_path.is_file():
        return []
    try:
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "query_supported_types":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.List):
                out = []
                for elt in sub.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.append(elt.value)
                if out:
                    return out
    return []


# ---------------------------------------------------------------------------
# Per-model merge (mirrors wgp.init_model_def + get_default_settings, no I/O)
# ---------------------------------------------------------------------------

def _read_settings_template() -> dict[str, Any]:
    if not SETTINGS_TEMPLATE.is_file():
        return {}
    with open(SETTINGS_TEMPLATE, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_model_def(model_dict: dict[str, Any], handler: Any, arch: str) -> dict[str, Any]:
    """Replicates ``wgp.init_model_def``: handler.query_model_def() ∪ JSON model dict."""
    try:
        extra = handler.query_model_def(arch, model_dict) or {}
    except Exception:
        extra = {}
    merged = dict(extra)
    merged.update(model_dict)
    return merged


def _merge_default_settings(
    *,
    template: dict[str, Any],
    handler: Any,
    arch: str,
    model_def: dict[str, Any],
    json_settings: dict[str, Any],
) -> dict[str, Any]:
    """Replicates ``wgp.get_default_settings`` *without* writing to disk.

    Order: template → handler.update_default_settings() → JSON-file overrides.
    Skips wgp's CLI-arg overlays (--seed/--frames/--steps), which only apply
    to the interactive run.
    """
    ui_defaults: dict[str, Any] = dict(template)
    if handler is not None and hasattr(handler, "update_default_settings"):
        try:
            handler.update_default_settings(arch, model_def, ui_defaults)
        except Exception:
            pass
    if json_settings:
        ui_defaults.update(json_settings)
    return ui_defaults


# ---------------------------------------------------------------------------
# Applicable-setting filter (uses shared.extra_settings)
# ---------------------------------------------------------------------------

def _import_extra_settings():
    if str(WANGP_ROOT) not in sys.path:
        sys.path.insert(0, str(WANGP_ROOT))
    from shared import extra_settings  # type: ignore
    return extra_settings


def _applicable_settings(model_def: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the visible (=applicable) registered settings for a model."""
    extra_settings = _import_extra_settings()
    out: list[dict[str, Any]] = []
    try:
        defs = extra_settings.iter_defs(model_def, only_visible=True)
    except Exception:
        return out
    for key, sd in defs.items():
        out.append({
            "key": key,
            "label": sd.label,
            "type": sd.type,
            "min": sd.min,
            "max": sd.max,
            "step": sd.step,
            "custom": sd.custom,
        })
    return out


# ---------------------------------------------------------------------------
# Capability + heuristics
# ---------------------------------------------------------------------------

_PARAM_COUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*B\b", re.IGNORECASE)


def _param_count_b(name: str, description: str = "") -> float | None:
    """Best-effort param-count in billions, parsed from name then description."""
    for source in (name, description):
        m = _PARAM_COUNT_RE.search(source or "")
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _capability_from_def(model_def: dict[str, Any], arch: str) -> str:
    """Derive coarse capability from feature flags + arch heuristics."""
    if model_def.get("image_outputs"):
        return "image-generation"
    if model_def.get("audio_only"):
        return "audio-generation"
    arch_l = (arch or "").lower()
    if any(t in arch_l for t in ("tts", "ace_step", "chatterbox", "qwen3_tts",
                                  "yue", "heartmula", "kugel", "index_tts")):
        return "audio-generation"
    if any(model_def.get(flag) for flag in
           ("i2v_class", "t2v_class", "vace_class", "multitalk_class",
            "wan_5B_class", "lynx_class", "alpha_class")):
        return "video-generation"
    if any(t in arch_l for t in ("wan", "ltx", "hunyuan", "longcat",
                                  "magi", "video", "ovi", "k5_", "kandinsky")):
        return "video-generation"
    return "image-generation"


def _quant_variants(urls: list[str]) -> list[str]:
    """Pick out quantization tags (bf16, int8, fp4, gguf q*) from filenames."""
    tags: list[str] = []
    for url in urls or []:
        fname = url.rsplit("/", 1)[-1].lower()
        for tag in ("bf16", "fp16", "int8", "int4", "fp4", "fp8",
                    "q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0",
                    "nvfp4", "nunchaku"):
            if tag in fname and tag not in tags:
                tags.append(tag)
    return tags


def _resolution_choices(model_def: dict[str, Any]) -> list[str] | None:
    """Pull the model's allowed resolutions if it declares them."""
    res = model_def.get("resolutions")
    if isinstance(res, list):
        return [str(r) for r in res]
    return None


# ---------------------------------------------------------------------------
# Public: rebuild & accessors
# ---------------------------------------------------------------------------

def _build_one_model(
    model_type: str,
    json_def: dict[str, Any],
    arch_to_handler: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any] | None:
    model_dict = json_def.get("model")
    if not isinstance(model_dict, dict):
        return None
    arch = str(model_dict.get("architecture") or "").strip()
    handler = arch_to_handler.get(arch)
    if handler is None:
        # unknown architecture: still emit what we know
        merged = dict(model_dict)
        defaults = dict(template)
        json_settings = {k: v for k, v in json_def.items() if k != "model"}
        defaults.update(json_settings)
        return {
            "model_type": model_type,
            "architecture": arch,
            "family": "unknown",
            "capability": _capability_from_def(merged, arch),
            "name": merged.get("name", model_type),
            "description": merged.get("description", ""),
            "param_count_b": _param_count_b(merged.get("name", ""), merged.get("description", "")),
            "urls": merged.get("URLs", []) or [],
            "preload_urls": merged.get("preload_URLs", []) or [],
            "quant_variants": _quant_variants(merged.get("URLs", []) or []),
            "model_def": merged,
            "defaults": defaults,
            "applicable_settings": [],
            "resolution_choices": _resolution_choices(merged),
            "handler_loaded": False,
        }
    model_def = _merge_model_def(model_dict, handler, arch)
    json_settings = {k: v for k, v in json_def.items() if k != "model"}
    defaults = _merge_default_settings(
        template=template,
        handler=handler,
        arch=arch,
        model_def=model_def,
        json_settings=json_settings,
    )
    try:
        family = handler.query_model_family()
    except Exception:
        family = "unknown"
    return {
        "model_type": model_type,
        "architecture": arch,
        "family": family,
        "capability": _capability_from_def(model_def, arch),
        "name": model_def.get("name", model_type),
        "description": model_def.get("description", ""),
        "param_count_b": _param_count_b(model_def.get("name", ""), model_def.get("description", "")),
        "urls": model_def.get("URLs", []) or [],
        "preload_urls": model_def.get("preload_URLs", []) or [],
        "quant_variants": _quant_variants(model_def.get("URLs", []) or []),
        "model_def": model_def,
        "defaults": defaults,
        "applicable_settings": _applicable_settings(model_def),
        "resolution_choices": _resolution_choices(model_def),
        "handler_loaded": True,
    }


def build_index(*, force: bool = False) -> dict[str, Any]:
    """Build (or return cached) ``{model_type → enriched entry}`` index."""
    global _INDEX_CACHE
    with _BUILD_LOCK:
        if _INDEX_CACHE is not None and not force:
            return _INDEX_CACHE
        arch_to_handler, errors = _load_handlers()
        template = _read_settings_template()
        index: dict[str, Any] = {}
        if not DEFAULTS_DIR.is_dir():
            _INDEX_CACHE = {"models": {}, "errors": errors + [{"path": str(DEFAULTS_DIR), "error": "missing"}]}
            return _INDEX_CACHE
        for path in sorted(DEFAULTS_DIR.glob("*.json")):
            model_type = path.stem
            try:
                with open(path, "r", encoding="utf-8") as f:
                    json_def = json.load(f)
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            try:
                entry = _build_one_model(model_type, json_def, arch_to_handler, template)
            except Exception as exc:
                errors.append({"model_type": model_type, "error": str(exc)})
                continue
            if entry is not None:
                index[model_type] = entry
        _INDEX_CACHE = {"models": index, "errors": errors}
        return _INDEX_CACHE


def get_settings_schema(*, force: bool = False) -> list[dict[str, Any]]:
    """Return the global registered-setting schema (independent of any model).

    Each entry is ``{key, label, type, min, max, step, custom}`` with values
    resolved against ``model_def=None`` (so callable resolvers get their
    "no model context" branch — typically a permissive default).
    """
    global _SCHEMA_CACHE
    with _BUILD_LOCK:
        if _SCHEMA_CACHE is not None and not force:
            return _SCHEMA_CACHE
        extra_settings = _import_extra_settings()
        out: list[dict[str, Any]] = []
        for key in extra_settings._SETTING_ORDER:
            try:
                sd = extra_settings.get_def(key, None)
            except Exception:
                continue
            out.append({
                "key": key,
                "label": sd.label,
                "type": sd.type,
                "min": sd.min,
                "max": sd.max,
                "step": sd.step,
                "custom": sd.custom,
            })
        _SCHEMA_CACHE = out
        return out


def get_model_entry(model_type: str) -> dict[str, Any] | None:
    return build_index()["models"].get(model_type)


def validate_request(model_type: str, inputs: dict[str, Any]) -> str | None:
    """Validate ``inputs`` against the registered schema for ``model_type``.

    Returns an error string, or ``None`` if valid (or the model is unknown,
    in which case validation is skipped — the runtime will reject it).
    """
    entry = get_model_entry(model_type)
    if entry is None:
        return None
    extra_settings = _import_extra_settings()
    try:
        return extra_settings.validate_inputs(inputs, entry["model_def"]) or None
    except Exception:
        return None


def invalidate_cache() -> None:
    """Forget cached index/schema (e.g. after defaults/*.json changes)."""
    global _INDEX_CACHE, _SCHEMA_CACHE
    with _BUILD_LOCK:
        _INDEX_CACHE = None
        _SCHEMA_CACHE = None


# ---------------------------------------------------------------------------
# JSON-safe trimming (model_def can carry torch dtypes, etc.)
# ---------------------------------------------------------------------------

_JSON_PRIMITIVES = (str, int, float, bool, type(None))


def _make_json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if isinstance(value, _JSON_PRIMITIVES):
        return value
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v, depth=depth + 1) for v in value]
    return repr(value)


def public_entry(entry: dict[str, Any], *, include_model_def: bool = True) -> dict[str, Any]:
    """JSON-safe view of a model entry, suitable for HTTP responses."""
    out = {
        "model_type": entry["model_type"],
        "architecture": entry["architecture"],
        "family": entry["family"],
        "capability": entry["capability"],
        "name": entry["name"],
        "description": entry["description"],
        "param_count_b": entry["param_count_b"],
        "urls": list(entry["urls"]),
        "preload_urls": list(entry["preload_urls"]),
        "quant_variants": list(entry["quant_variants"]),
        "resolution_choices": entry["resolution_choices"],
        "applicable_settings": entry["applicable_settings"],
        "defaults": _make_json_safe(entry["defaults"]),
        "handler_loaded": entry["handler_loaded"],
    }
    if include_model_def:
        out["model_def"] = _make_json_safe(entry["model_def"])
    return out


if __name__ == "__main__":
    # Quick CLI sanity-check: dump a model's enriched view as JSON.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", nargs="?", default=None)
    parser.add_argument("--schema", action="store_true",
                        help="Dump the global setting schema instead.")
    args = parser.parse_args()

    if args.schema:
        print(json.dumps(get_settings_schema(), indent=2, default=str))
        sys.exit(0)

    idx = build_index()
    if args.model_type:
        entry = idx["models"].get(args.model_type)
        if entry is None:
            print(f"unknown model_type: {args.model_type}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(public_entry(entry), indent=2, default=str))
    else:
        print(json.dumps({
            "model_count": len(idx["models"]),
            "errors": idx["errors"],
            "sample": list(idx["models"].keys())[:10],
        }, indent=2, default=str))
