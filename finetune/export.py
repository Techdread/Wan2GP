"""After training: merge the LoRA into the base weights and write the result in
the base checkpoint's own layout (same shards, same tensor names, vision tower
and MTP head copied through untouched), then convert that folder to GGUF with
llama.cpp's converter so LM Studio can run it.

The converter is fetched once at a pinned tag into ckpts/llama.cpp-<tag>/ and
run in this same Python with that checkout's gguf-py first on the path, so the
older `gguf` package Wan2GP itself pins is left alone.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable

LLAMA_CPP_TAG = "v0.9.0"  # first checked release whose converter registers Qwen3_5ForConditionalGeneration
LLAMA_CPP_TARBALL = f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/{LLAMA_CPP_TAG}"

TEXT_PREFIX = "model.language_model."   # base checkpoint names
TRAIN_PREFIX = "model."                 # the same tensors inside Qwen3_5ForCausalLM

COPY_FILES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json",
    "video_preprocessor_config.json", "special_tokens_map.json", "LICENSE",
)


def checkpoint_to_train_name(name: str) -> str | None:
    """Base checkpoint tensor name → the name inside the text-only training model (None: not trained)."""
    if name.startswith(TEXT_PREFIX):
        return TRAIN_PREFIX + name[len(TEXT_PREFIX):]
    return None


def train_to_checkpoint_name(name: str) -> str | None:
    if name == "lm_head.weight":
        return None  # tied to the embeddings: the checkpoint has no separate copy
    if name.startswith(TRAIN_PREFIX):
        return TEXT_PREFIX + name[len(TRAIN_PREFIX):]
    return None


def shard_files(base_dir: Path) -> list[Path]:
    index = base_dir / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        return [base_dir / n for n in names]
    single = base_dir / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"no safetensors weights in {base_dir}")


def write_merged(base_dir: Path, out_dir: Path, merged_text: dict, log: Callable[[str], None]) -> list[Path]:
    """Rewrite each base shard with the merged text tensors swapped in.

    `merged_text` maps training-model names (model.layers.0.mlp.up_proj.weight …)
    to CPU tensors. Every one must find its place in the base checkpoint, and
    every text tensor of the base must have a merged counterpart: a partial
    merge would quietly ship base weights.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=False)
    for name in COPY_FILES:
        src = base_dir / name
        if src.is_file():
            shutil.copyfile(src, out_dir / name)
    index = base_dir / "model.safetensors.index.json"
    if index.is_file():
        shutil.copyfile(index, out_dir / index.name)

    wanted = {train_to_checkpoint_name(k): k for k in merged_text if train_to_checkpoint_name(k)}
    placed: set = set()
    written = []
    for shard in shard_files(base_dir):
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            metadata = f.metadata() or {}
            for name in f.keys():
                if name in wanted:
                    tensors[name] = merged_text[wanted[name]].contiguous()
                    placed.add(name)
                elif name.startswith(TEXT_PREFIX):
                    raise RuntimeError(f"merged weights are missing {name}")
                else:
                    tensors[name] = f.get_tensor(name)
        target = out_dir / shard.name
        save_file(tensors, str(target), metadata={**metadata, "format": "pt"})
        written.append(target)
        log(f"wrote {target.name} ({len(tensors)} tensors)")
    unplaced = sorted(set(wanted) - placed)
    if unplaced:
        raise RuntimeError(f"{len(unplaced)} merged tensor(s) have no place in the base checkpoint, e.g. {unplaced[:3]}")
    return written


def ensure_converter(ckpts_dir: Path, log: Callable[[str], None]) -> Path:
    """The pinned llama.cpp checkout (only the converter's files are used)."""
    root = ckpts_dir / f"llama.cpp-{LLAMA_CPP_TAG}"
    script = root / "convert_hf_to_gguf.py"
    if script.is_file() and (root / "gguf-py").is_dir() and (root / "conversion").is_dir():
        return root
    log(f"fetching llama.cpp {LLAMA_CPP_TAG} converter")
    with urllib.request.urlopen(LLAMA_CPP_TARBALL, timeout=300) as resp:
        data = resp.read()
    partial = root.with_name(root.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    keep = ("convert_hf_to_gguf.py", "conversion/", "gguf-py/")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) < 2 or not parts[1].startswith(keep):
                continue
            rel = Path(parts[1])
            if rel.is_absolute() or ".." in rel.parts or not (member.isfile() or member.isdir()):
                continue
            target = partial / rel
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    if not (partial / "convert_hf_to_gguf.py").is_file():
        raise RuntimeError("llama.cpp tarball had no convert_hf_to_gguf.py")
    if root.exists():
        shutil.rmtree(root)
    os.replace(partial, root)
    return root


def convert_to_gguf(converter_root: Path, merged_dir: Path, outfile: Path, outtype: str,
                    log: Callable[[str], None], should_stop: Callable[[], bool]) -> Path:
    if outfile.exists():
        raise FileExistsError(f"{outfile} already exists")
    partial = outfile.with_name(outfile.name + ".partial")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(converter_root / "gguf-py"), str(converter_root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["NO_LOCAL_GGUF"] = ""
    cmd = [sys.executable, "-u", str(converter_root / "convert_hf_to_gguf.py"), str(merged_dir),
           "--outfile", str(partial), "--outtype", outtype]
    log("converting: " + " ".join(cmd[2:]))
    proc = subprocess.Popen(cmd, cwd=str(converter_root), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        log("gguf: " + line.rstrip())
        if should_stop():
            proc.terminate()
            break
    rc = proc.wait()
    if should_stop():
        raise InterruptedError("stopped during GGUF conversion")
    if rc != 0 or not partial.is_file():
        raise RuntimeError(f"GGUF conversion failed (exit {rc}); see train.log")
    os.replace(partial, outfile)
    return outfile
