"""LoRA fine-tuning of a Qwen3.5 text model, run by the agent API as a job:

    python -u -m finetune.train --job-dir <outputs>/finetune/<job id>

The job folder holds spec.json (written by the server) and receives everything
the run produces:

    checkpoints/step-000050/   LoRA adapter at each save (all kept)
    adapter/                   the final adapter
    merged/                    base + LoRA in the base checkpoint's own layout
    <name>-<id>-Q8_0.gguf      the same, for LM Studio / llama.cpp
    metrics.jsonl              one line per optimizer step and eval
    resume.pt                  optimizer + position, for resume_from (rewritten at each save)
    STOP                       created by the server to ask for a clean stop

It runs in Wan2GP's own Python and uses the repo's vendored Qwen3.5 model code
(shared/prompt_enhancer/qwen3_5), so no second environment is needed.
Progress goes to stdout as `@@ft {json}` lines (see finetune/progress.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

WANGP_ROOT = Path(__file__).resolve().parent.parent
if str(WANGP_ROOT) not in sys.path:
    sys.path.insert(0, str(WANGP_ROOT))

from finetune import data as ftdata  # noqa: E402
from finetune import export, lmstudio  # noqa: E402
from finetune.progress import event_line  # noqa: E402
from finetune.spec import LOSS_CHUNK, estimate_vram_gb  # noqa: E402

EXIT_STOPPED = 3
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                  "in_proj_qkv", "in_proj_z", "out_proj"]
BASE_FILES = ["*.json", "*.safetensors", "*.jinja", "merges.txt", "vocab.json", "LICENSE"]
CHECK_TOKENS = 1024
CHECK_MAX_LOSS = 4.0  # a loaded base model scores ~0.5–1.5 on code; random weights ~12


class Stopped(Exception):
    pass


def emit(event: str, **fields) -> None:
    print(event_line(event, **fields), flush=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def ckpts_dir() -> Path:
    configured = (os.environ.get("WAN2GP_FINETUNE_CACHE") or "").strip()
    if configured:
        return Path(configured).expanduser()
    try:
        cfg = json.loads((WANGP_ROOT / "wgp_config.json").read_text(encoding="utf-8"))
        first = (cfg.get("checkpoints_paths") or ["ckpts"])[0]
    except (OSError, ValueError, IndexError):
        first = "ckpts"
    root = Path(first)
    return (root if root.is_absolute() else WANGP_ROOT / root) / "finetune"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  phases
# --------------------------------------------------------------------------- #

def download_base(spec: dict) -> Path:
    from huggingface_hub import snapshot_download
    target = ckpts_dir() / spec["repo"].replace("/", "__")
    emit("phase", phase="download", message=f"{spec['repo']} → {target}")
    snapshot_download(repo_id=spec["repo"], local_dir=str(target), allow_patterns=BASE_FILES)
    return target


def load_tokenizer(base_dir: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    template = base_dir / "chat_template.jinja"
    if not getattr(tok, "chat_template", None) and template.is_file():
        tok.chat_template = template.read_text(encoding="utf-8")
    if not getattr(tok, "chat_template", None):
        raise RuntimeError(f"no chat template in {base_dir}")
    return tok


def prepare_data(spec: dict, tok):
    import numpy as np
    emit("phase", phase="data", message="tokenizing")

    def render(messages, add_generation_prompt):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def encode(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    def compact(enc):
        return (enc.id, np.asarray(enc.input_ids, dtype=np.int32), np.asarray(enc.labels, dtype=np.int32), enc.trained_tokens)

    with open(spec["dataset_path"], "r", encoding="utf-8") as f:
        kept, report = ftdata.prepare(f, render, encode, spec["max_length"], compact=compact)
    train, holdout = ftdata.split_holdout(kept, spec["holdout"], spec["seed"])
    emit("data", **report.to_dict(), train=len(train), holdout=len(holdout), max_length=spec["max_length"])
    if not train:
        raise RuntimeError(f"no usable examples (skipped: {report.skipped}); try a larger max_length")
    return train, holdout


def build_model(spec: dict, base_dir: Path, device):
    import torch
    from safetensors import safe_open
    from transformers.modeling_utils import no_init_weights

    from shared.prompt_enhancer.qwen3_5 import register_qwen35_config
    from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from shared.prompt_enhancer.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    emit("phase", phase="load", message=f"loading {spec['repo']} (text model)")
    register_qwen35_config()
    raw = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    config = Qwen3_5TextConfig(**raw.get("text_config", raw))
    config._attn_implementation = "sdpa"
    config._prompt_enhancer_safe_legacy = False
    config.use_cache = False

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with no_init_weights(), torch.device(device):
            model = Qwen3_5ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous)

    params = dict(model.named_parameters())
    loaded: set = set()
    for shard in export.shard_files(base_dir):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                target = export.checkpoint_to_train_name(name)
                if target is None:
                    continue  # vision tower, MTP head: not trained, copied through at export
                if target not in params:
                    raise RuntimeError(f"base tensor {name} has no place in the training model")
                tensor = f.get_tensor(name)
                param = params[target]
                if tuple(tensor.shape) != tuple(param.shape):
                    raise RuntimeError(f"{name}: checkpoint shape {tuple(tensor.shape)} ≠ model {tuple(param.shape)}")
                # keep the checkpoint's own precision (e.g. fp32 A_log / norms)
                param.data = tensor.to(device=device, dtype=tensor.dtype if tensor.dtype == torch.float32 else param.dtype)
                loaded.add(target)
    missing = sorted(set(params) - loaded - {"lm_head.weight"})
    if missing:
        raise RuntimeError(f"{len(missing)} model weight(s) not in the checkpoint, e.g. {missing[:3]}")
    model.lm_head.weight = model.model.embed_tokens.weight  # tied, as in the base
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model


def add_lora(model, spec: dict):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=spec["lora_rank"], lora_alpha=spec["lora_alpha"], lora_dropout=spec["lora_dropout"],
                     target_modules=TARGET_MODULES, bias="none")
    peft_model = get_peft_model(model, cfg)
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in trainable)
    log(f"LoRA r={spec['lora_rank']} alpha={spec['lora_alpha']}: {n / 1e6:.1f}M trainable parameters")
    return peft_model, trainable


def loss_sum(peft_model, ids, labels, device):
    """Summed cross-entropy over the learned tokens of one example. The output
    layer is applied a slice at a time (and recomputed in backward), so a 16k
    sequence never holds 16k × vocab logits."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    base = peft_model.get_base_model()
    input_ids = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
    target = torch.as_tensor(labels, dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = base.model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
    hidden, target = hidden[:-1], target[1:]
    keep = (target != ftdata.IGNORE).nonzero().squeeze(1)
    hidden, target = hidden.index_select(0, keep), target.index_select(0, keep)
    weight = base.lm_head.weight

    def piece(h, y):
        return F.cross_entropy((h @ weight.t()).float(), y, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, hidden.shape[0], LOSS_CHUNK):
        total = total + checkpoint(piece, hidden[s:s + LOSS_CHUNK], target[s:s + LOSS_CHUNK], use_reentrant=False)
    return total, int(keep.numel())


def sanity_check(peft_model, trainable, example, device) -> dict:
    """One short forward/backward before hours of training: a mis-loaded
    checkpoint shows up as a loss near ln(vocab) instead of ~1."""
    import torch
    emit("phase", phase="check", message="checking the loaded weights on one example")
    _, ids, labels, _ = example
    ids, labels = ids[:CHECK_TOKENS], labels[:CHECK_TOKENS]
    if not (labels != ftdata.IGNORE).any():
        ids, labels = example[1][-CHECK_TOKENS:], example[2][-CHECK_TOKENS:]
    t0 = time.time()
    total, n = loss_sum(peft_model, ids, labels, device)
    loss = float(total.detach()) / max(1, n)
    (total / max(1, n)).backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in trainable)
    for p in trainable:
        p.grad = None
    result = {"loss": round(loss, 4), "tokens": len(ids), "seconds": round(time.time() - t0, 2), "gradients": grad_ok}
    emit("check", **result)
    if not math.isfinite(loss) or loss > CHECK_MAX_LOSS:
        raise RuntimeError(f"the base model scores loss {loss:.2f} on a training example (expected under {CHECK_MAX_LOSS}): "
                           "the weights did not load correctly, so training was not started")
    if not grad_ok:
        raise RuntimeError("no gradient reached the LoRA weights, so training was not started")
    return result


def lr_at(step: int, total: int, peak: float, warmup_ratio: float) -> float:
    warm = max(1, int(round(total * warmup_ratio)))
    if step < warm:
        return peak * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def evaluate(peft_model, holdout, device) -> float | None:
    import torch
    if not holdout:
        return None
    peft_model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for _, ids, labels, _ in holdout:
            s, n = loss_sum(peft_model, ids, labels, device)
            total += float(s)
            tokens += n
    peft_model.train()
    return total / max(1, tokens)


def save_checkpoint(job_dir: Path, peft_model, optimizer, step: int, extra: dict) -> Path:
    import torch
    path = job_dir / "checkpoints" / f"step-{step:06d}"
    if not path.exists():
        peft_model.save_pretrained(str(path))
    tmp = job_dir / "resume.pt.partial"
    torch.save({"step": step, "optimizer": optimizer.state_dict(), **extra}, tmp)
    os.replace(tmp, job_dir / "resume.pt")
    emit("checkpoint", step=step, path=str(path))
    return path


def load_resume(spec: dict, peft_model, optimizer, dataset_sha: str) -> int:
    """Continue an earlier job: its last adapter checkpoint and optimizer state."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    src = Path(spec["resume_dir"])
    state = torch.load(src / "resume.pt", map_location="cpu", weights_only=False)
    if state.get("dataset_sha256") != dataset_sha:
        raise RuntimeError("resume_from used a different dataset file; start a new fine-tune instead")
    for key in ("base", "lora_rank", "lora_alpha", "max_length", "grad_accum", "epochs", "seed", "holdout"):
        if state.get("spec", {}).get(key) != spec.get(key):
            raise RuntimeError(f"resume_from had {key}={state.get('spec', {}).get(key)!r}, this job {spec.get(key)!r}; settings must match to resume")
    step = int(state["step"])
    ckpt = src / "checkpoints" / f"step-{step:06d}" / "adapter_model.safetensors"
    set_peft_model_state_dict(peft_model, load_file(str(ckpt)))
    optimizer.load_state_dict(state["optimizer"])
    log(f"resumed from {src.name} at step {step}")
    return step


def train(spec: dict, job_dir: Path, peft_model, trainable, train_set, holdout, device, should_stop, dataset_sha) -> int:
    import torch
    visits, total_steps = ftdata.plan_steps(len(train_set), spec["epochs"], spec["grad_accum"])
    emit("plan", examples=len(train_set), visits=visits, total_steps=total_steps, grad_accum=spec["grad_accum"],
         epochs=spec["epochs"], tokens_per_epoch=int(sum(e[3] for e in train_set)))
    optimizer = torch.optim.AdamW(trainable, lr=spec["learning_rate"], betas=(0.9, 0.999), weight_decay=0.0)
    start = load_resume(spec, peft_model, optimizer, dataset_sha) if spec.get("resume_dir") else 0
    emit("phase", phase="train", message=f"{total_steps} optimizer steps")
    peft_model.train()
    n = len(train_set)
    metrics = open(job_dir / "metrics.jsonl", "a", encoding="utf-8")
    extra = {"spec": spec, "dataset_sha256": dataset_sha}
    t_start, tokens_done = time.time(), 0
    step = start
    try:
        for step in range(start, total_steps):
            window = range(step * spec["grad_accum"], min((step + 1) * spec["grad_accum"], visits))
            batch = [train_set[ftdata.epoch_order(n, v // n, spec["seed"])[v % n]] for v in window]
            window_tokens = sum(e[3] for e in batch)
            lr = lr_at(step, total_steps, spec["learning_rate"], spec["warmup_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            step_loss = 0.0
            for _, ids, labels, _ in batch:
                s, _ = loss_sum(peft_model, ids, labels, device)
                (s / max(1, window_tokens)).backward()
                step_loss += float(s.detach())
                tokens_done += len(ids)
                if should_stop():
                    break
            if should_stop():
                optimizer.zero_grad(set_to_none=True)
                raise Stopped()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            done = step + 1
            elapsed = time.time() - t_start
            rate = tokens_done / max(1e-6, elapsed)
            remaining_steps = total_steps - done
            eta = elapsed / max(1, done - start) * remaining_steps
            row = {"step": done, "total_steps": total_steps, "loss": round(step_loss / max(1, window_tokens), 5),
                   "lr": lr, "grad_norm": round(grad_norm, 4), "epoch": round(window.stop / n, 4),
                   "tokens_per_s": round(rate, 1), "eta_s": int(eta),
                   "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            emit("step", **row)
            metrics.write(json.dumps({"kind": "step", **row}) + "\n")
            metrics.flush()
            last = done == total_steps
            if done % spec["eval_every"] == 0 or last:
                ev = evaluate(peft_model, holdout, device)
                if ev is not None:
                    emit("eval", step=done, eval_loss=round(ev, 5))
                    metrics.write(json.dumps({"kind": "eval", "step": done, "eval_loss": ev}) + "\n")
                    metrics.flush()
            if done % spec["save_every"] == 0 or last:
                save_checkpoint(job_dir, peft_model, optimizer, done, extra)
    except Stopped:
        if step > start:
            save_checkpoint(job_dir, peft_model, optimizer, step, extra)
        raise
    finally:
        metrics.close()
    return total_steps


def finish(spec: dict, job_dir: Path, base_dir: Path, peft_model, device, should_stop) -> None:
    import gc
    import torch
    adapter_dir = job_dir / "adapter"
    peft_model.save_pretrained(str(adapter_dir))
    emit("file", kind="adapter", path=str(adapter_dir / "adapter_model.safetensors"))
    emit("file", kind="adapter-config", path=str(adapter_dir / "adapter_config.json"))

    emit("phase", phase="merge", message="merging the LoRA into the base weights")
    merged_model = peft_model.merge_and_unload()
    merged = {k: v.detach().to("cpu", copy=True) for k, v in merged_model.state_dict().items()
              if export.train_to_checkpoint_name(k)}
    del merged_model, peft_model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()
    merged_dir = job_dir / "merged"
    export.write_merged(base_dir, merged_dir, merged, log)
    del merged
    gc.collect()
    emit("file", kind="merged", path=str(merged_dir / "config.json"))

    if spec["gguf"] == "none":
        return
    emit("phase", phase="gguf", message=f"converting to GGUF ({spec['gguf']})")
    converter = export.ensure_converter(ckpts_dir(), log)
    short = job_dir.name[-6:].lower()
    gguf = job_dir / f"{spec['name']}-{short}-{spec['gguf'].upper()}.gguf"
    export.convert_to_gguf(converter, merged_dir, gguf, spec["gguf"], log, should_stop)
    emit("file", kind="gguf", path=str(gguf), bytes=gguf.stat().st_size)

    if spec["lmstudio"]:
        emit("phase", phase="lmstudio", message="adding the model to LM Studio")
        root = lmstudio.models_dir()
        if root is None:
            emit("lmstudio", placed=False, reason="no LM Studio models folder found (set WAN2GP_LMSTUDIO_MODELS_DIR)")
        else:
            target, how = lmstudio.place(gguf, spec["name"], root)
            emit("lmstudio", placed=True, path=str(target), how=how, model=gguf.stem.lower())


def run(job_dir: Path) -> int:
    spec = json.loads((job_dir / "spec.json").read_text(encoding="utf-8"))
    stop_file = job_dir / "STOP"

    def should_stop() -> bool:
        return stop_file.exists()

    emit("phase", phase="starting", message=f"fine-tune {spec['name']} on {spec['repo']}")
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU visible to the trainer")
        device = "cuda"
        free, total = torch.cuda.mem_get_info()
        need = estimate_vram_gb(spec["base"], spec["max_length"], spec["lora_rank"])
        log(f"GPU {torch.cuda.get_device_name(0)}: {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB; estimate {need} GB")
        if free / 1e9 < need:
            log(f"warning: about {need} GB is needed and {free / 1e9:.1f} GB is free. "
                "If this runs out of memory, unload models in LM Studio or other apps on this machine.")
        torch.manual_seed(spec["seed"])

        base_dir = download_base(spec)
        tok = load_tokenizer(base_dir)
        dataset_sha = file_sha256(Path(spec["dataset_path"]))
        train_set, holdout = prepare_data(spec, tok)
        model = build_model(spec, base_dir, device)
        peft_model, trainable = add_lora(model, spec)
        sanity_check(peft_model, trainable, (holdout or train_set)[0], device)
        if should_stop():
            raise Stopped()
        train(spec, job_dir, peft_model, trainable, train_set, holdout, device, should_stop, dataset_sha)
        finish(spec, job_dir, base_dir, peft_model, device, should_stop)
        emit("done")
        return 0
    except (Stopped, InterruptedError):
        emit("phase", phase="done", message="stopped; checkpoints kept")
        return EXIT_STOPPED
    except Exception as exc:  # noqa: BLE001 - every failure must reach the job record
        message = f"{type(exc).__name__}: {exc}"
        if "out of memory" in str(exc).lower():
            message += " (free the GPU: unload LM Studio models on this machine, or lower max_length)"
        emit("error", message=message)
        traceback.print_exc(file=sys.stdout)
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    return run(Path(args.job_dir))


if __name__ == "__main__":
    sys.exit(main())
