"""Put a finished GGUF where this machine's LM Studio finds it, so the model can
be loaded (over LM Studio's own REST API) and evaluated without copying files
between machines by hand.

LM Studio lists `<models folder>/<publisher>/<model>/<file>.gguf`. The folder is
WAN2GP_LMSTUDIO_MODELS_DIR, else `downloadsFolder` in ~/.lmstudio/settings.json,
else ~/.lmstudio/models.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

PUBLISHER = "wan2gp-finetunes"


def models_dir(env: dict | None = None, home: Path | None = None) -> Path | None:
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    configured = (env.get("WAN2GP_LMSTUDIO_MODELS_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser()
    settings = home / ".lmstudio" / "settings.json"
    try:
        folder = json.loads(settings.read_text(encoding="utf-8")).get("downloadsFolder")
        if isinstance(folder, str) and folder.strip():
            return Path(folder).expanduser()
    except (OSError, ValueError):
        pass
    default = home / ".lmstudio" / "models"
    return default if default.parent.is_dir() else None


def place(gguf: Path, name: str, root: Path) -> tuple[Path, str]:
    """Hard-link (same drive) or copy the GGUF in. Never overwrites: an existing
    file of the same name is left alone and reported. Returns (path, how)."""
    target_dir = root / PUBLISHER / name
    target = target_dir / gguf.name
    if target.exists():
        return target, "exists"
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.link(gguf, target)
        return target, "linked"
    except OSError:
        tmp = target.with_name(target.name + ".partial")
        shutil.copyfile(gguf, tmp)
        os.replace(tmp, target)
        return target, "copied"
