"""Fine-tuning jobs: request validation, data preparation, the trainer's event
protocol, LM Studio placement and the job runner (with a fake trainer process).
None of it needs torch or a GPU; the trainer itself is exercised on a node.

Run:  python -m unittest test_agent_api_finetune
"""

import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import agent_api_server as server
from finetune import data as ftdata
from finetune import export, job, lmstudio, progress, spec


def _uploads(root: Path) -> Path:
    d = root / "uploads" / "u_1"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "train.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    return f


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = str(_uploads(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def req(self, **over):
        return {"model_type": "finetune-qwen3.5-4b", "name": "glb-v1", "dataset_path": self.dataset, **over}

    def test_defaults_fill_in(self):
        s, err = spec.normalize_request(self.req(), uploads_root=self.root / "uploads")
        self.assertIsNone(err)
        self.assertEqual(s["base"], "qwen3.5-4b")
        self.assertEqual(s["repo"], "Qwen/Qwen3.5-4B")
        self.assertEqual(s["max_length"], 16384)
        self.assertEqual(s["lora_alpha"], 64)  # 2 × rank
        self.assertEqual(s["gguf"], "q8_0")
        self.assertIs(s["lmstudio"], True)

    def test_refusals(self):
        cases = {
            "unknown fine-tuning base": self.req(model_type="finetune-llama-70b"),
            "name must be": self.req(name="Has Spaces"),
            "dataset_path is required": self.req(dataset_path=""),
            "must be a .jsonl": self.req(dataset_path=self.dataset[:-1]),
            "uploaded through /api/uploads": self.req(dataset_path=str(self.root / "elsewhere.jsonl")),
            "unknown fine-tuning option": self.req(batch_size=4),
            "lora_rank must be a whole number": self.req(lora_rank=8.5),
            "epochs must be between": self.req(epochs=0),
            "max_length must be a number": self.req(max_length="16k"),
            "gguf must be one of": self.req(gguf="q4_k_m"),
            "lmstudio must be true or false": self.req(lmstudio="yes"),
            "resume_from must be": self.req(resume_from="last"),
        }
        for expected, request in cases.items():
            _, err = spec.normalize_request(request, uploads_root=self.root / "uploads")
            self.assertIsNotNone(err, expected)
            self.assertIn(expected, err)

    def test_dataset_path_cannot_escape_uploads(self):
        sneaky = str(self.root / "uploads" / ".." / "secret.jsonl")
        _, err = spec.normalize_request(self.req(dataset_path=sneaky), uploads_root=self.root / "uploads")
        self.assertIn("uploaded through", err)

    def test_estimates_grow_with_length_and_size(self):
        self.assertLess(spec.estimate_vram_gb("qwen3.5-4b", 8192), spec.estimate_vram_gb("qwen3.5-4b", 16384))
        self.assertLess(spec.estimate_vram_gb("qwen3.5-2b", 16384), spec.estimate_vram_gb("qwen3.5-4b", 16384))
        bases = {b["base"] for b in spec.describe_bases()}
        self.assertEqual(bases, set(spec.BASES))

    def test_server_routes_finetune_requests(self):
        self.assertEqual(server.capability_for_model_type("finetune-qwen3.5-4b"), "fine-tuning")
        with patch.dict(os.environ, {"WAN2GP_OUTPUTS_ROOT": str(self.root)}):
            self.assertIsNone(server.validate_job_request(self.req()))
            self.assertIn("name must be", server.validate_job_request(self.req(name="")))
        self.assertIn(".jsonl", server.UPLOAD_EXTENSIONS)


# A stand-in for Qwen's template: same shape (generation prompt ends with
# "<think>\n"; the answer carries the reasoning, then </think>, then content).
def render(messages, add_generation_prompt):
    out = ""
    for m in messages:
        if m["role"] == "assistant":
            out += "<|im_start|>assistant\n<think>\n" + m.get("reasoning_content", "") + "\n</think>\n\n" + m["content"] + "<|im_end|>\n"
        else:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n<think>\n"
    return out


def encode(text):
    return [ord(c) for c in text]


def example(i, answer="<html></html>", reasoning=None):
    a = {"role": "assistant", "content": answer}
    if reasoning:
        a["reasoning_content"] = reasoning
    return json.dumps({"id": f"ex-{i}", "messages": [
        {"role": "system", "content": "sys"}, {"role": "user", "content": f"make {i}"}, a]})


class DataTests(unittest.TestCase):
    def test_only_the_answer_is_learned(self):
        kept, report = ftdata.prepare([example(1, reasoning="plan")], render, encode, 10_000)
        self.assertEqual(report.kept, 1)
        e = kept[0]
        prompt = render([{"role": "system", "content": "sys"}, {"role": "user", "content": "make 1"}], True)
        self.assertEqual(e.labels[: len(prompt)], [ftdata.IGNORE] * len(prompt))
        learned = "".join(chr(t) for t in e.labels if t != ftdata.IGNORE)
        self.assertEqual(learned, "plan\n</think>\n\n<html></html><|im_end|>\n")

    def test_too_long_is_left_out_not_cut(self):
        kept, report = ftdata.prepare([example(1, answer="x" * 500), example(2)], render, encode, 200)
        self.assertEqual([e.id for e in kept], ["ex-2"])
        self.assertEqual(report.skipped, {"longer than max_length": 1})

    def test_bad_lines_are_reported(self):
        lines = ["not json", "[1]", json.dumps({"id": "a", "messages": [{"role": "user", "content": "hi"}]}),
                 json.dumps({"id": "b", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": " "}]}),
                 example(1), example(1)]
        kept, report = ftdata.prepare(lines, render, encode, 10_000)
        self.assertEqual(len(kept), 1)
        self.assertEqual(report.skipped, {"invalid line": 2, "no messages": 1, "no assistant answer": 1, "duplicate id": 1})

    def test_template_mismatch_is_refused(self):
        def odd(messages, gen):
            return render(messages, gen).replace("sys", "SYS") if gen else render(messages, gen)
        _, report = ftdata.prepare([example(1)], odd, encode, 10_000)
        self.assertEqual(report.skipped, {"chat template mismatch": 1})

    def test_holdout_order_and_steps(self):
        items = list(range(100))
        train, held = ftdata.split_holdout(items, 32, seed=1)
        self.assertEqual(len(held), 10)  # capped at 10%
        self.assertEqual(sorted(train + held), items)
        self.assertEqual(ftdata.split_holdout(items, 32, seed=1), (train, held))
        self.assertEqual(ftdata.epoch_order(50, 0, 7), ftdata.epoch_order(50, 0, 7))
        self.assertNotEqual(ftdata.epoch_order(50, 0, 7), ftdata.epoch_order(50, 1, 7))
        self.assertEqual(ftdata.plan_steps(100, 1, 8), (100, 13))
        self.assertEqual(ftdata.plan_steps(100, 0.5, 8), (50, 7))


class ProgressTests(unittest.TestCase):
    def test_events_fold_into_progress(self):
        st = progress.new_state()
        lines = [
            progress.event_line("phase", phase="download"),
            "plain log line",
            progress.event_line("plan", total_steps=10),
            progress.event_line("phase", phase="train"),
            progress.event_line("step", step=5, total_steps=10, loss=1.2),
            progress.event_line("eval", step=5, eval_loss=1.1),
            progress.event_line("checkpoint", step=5, path="/x"),
            progress.event_line("file", kind="gguf", path="/x.gguf"),
            progress.event_line("done"),
        ]
        seen = []
        for line in lines:
            ev = progress.parse_line(line)
            if ev:
                progress.apply_event(st, ev)
                seen.append(st["progress"])
        self.assertEqual(seen, sorted(seen))  # never goes backwards
        self.assertEqual(st["progress"], 1.0)
        self.assertEqual((st["step"], st["total_steps"]), (5, 10))
        self.assertEqual(st["result"]["loss_history"], [[5, 1.2]])
        self.assertEqual(st["result"]["eval_history"], [[5, 1.1]])
        self.assertEqual(st["result"]["files"], [{"kind": "gguf", "path": "/x.gguf"}])
        self.assertIsNone(progress.parse_line("@@ft not json"))

    def test_loss_history_stays_bounded(self):
        st = progress.new_state()
        for i in range(5000):
            progress.apply_event(st, {"event": "step", "step": i, "total_steps": 5000, "loss": 1.0})
        self.assertLessEqual(len(st["result"]["loss_history"]), 2000)
        self.assertEqual(st["result"]["loss_history"][-1][0], 4999)


class LmStudioTests(unittest.TestCase):
    def test_models_dir_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            self.assertIsNone(lmstudio.models_dir(env={}, home=home))
            (home / ".lmstudio").mkdir()
            self.assertEqual(lmstudio.models_dir(env={}, home=home), home / ".lmstudio" / "models")
            (home / ".lmstudio" / "settings.json").write_text(json.dumps({"downloadsFolder": "/big/LM"}))
            self.assertEqual(lmstudio.models_dir(env={}, home=home), Path("/big/LM"))
            self.assertEqual(lmstudio.models_dir(env={"WAN2GP_LMSTUDIO_MODELS_DIR": "/x"}, home=home), Path("/x"))

    def test_place_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            gguf = root / "m-Q8_0.gguf"
            gguf.write_bytes(b"first")
            target, how = lmstudio.place(gguf, "m", root / "lm")
            self.assertIn(how, ("linked", "copied"))
            self.assertEqual(target, root / "lm" / lmstudio.PUBLISHER / "m" / "m-Q8_0.gguf")
            gguf2 = root / "other" / "m-Q8_0.gguf"
            gguf2.parent.mkdir()
            gguf2.write_bytes(b"second")
            _, how = lmstudio.place(gguf2, "m", root / "lm")
            self.assertEqual(how, "exists")
            self.assertEqual(target.read_bytes(), b"first")


class ExportNameTests(unittest.TestCase):
    def test_names_round_trip(self):
        ck = "model.language_model.layers.3.self_attn.q_proj.weight"
        self.assertEqual(export.checkpoint_to_train_name(ck), "model.layers.3.self_attn.q_proj.weight")
        self.assertEqual(export.train_to_checkpoint_name(export.checkpoint_to_train_name(ck)), ck)
        self.assertIsNone(export.checkpoint_to_train_name("model.visual.blocks.0.attn.qkv.weight"))
        self.assertIsNone(export.checkpoint_to_train_name("mtp.fc.weight"))
        self.assertIsNone(export.train_to_checkpoint_name("lm_head.weight"))  # tied in the base

    def test_write_merged_swaps_text_tensors_only(self):
        try:
            import torch
            from safetensors.torch import load_file, save_file
        except ImportError:
            self.skipTest("torch/safetensors not installed here (they are on the GPU nodes)")
        with tempfile.TemporaryDirectory() as d:
            base, out = Path(d) / "base", Path(d) / "out"
            base.mkdir()
            save_file({"model.language_model.layers.0.mlp.up_proj.weight": torch.zeros(2, 2),
                       "model.visual.patch.weight": torch.ones(1)}, str(base / "model.safetensors"))
            (base / "config.json").write_text("{}")
            export.write_merged(base, out, {"model.layers.0.mlp.up_proj.weight": torch.full((2, 2), 3.0)}, lambda m: None)
            t = load_file(str(out / "model.safetensors"))
            self.assertTrue(torch.equal(t["model.language_model.layers.0.mlp.up_proj.weight"], torch.full((2, 2), 3.0)))
            self.assertTrue(torch.equal(t["model.visual.patch.weight"], torch.ones(1)))
            with self.assertRaises(RuntimeError):  # a text tensor without a merged counterpart
                export.write_merged(base, Path(d) / "out2", {}, lambda m: None)


FAKE_TRAINER = textwrap.dedent('''
    import json, sys, time, pathlib
    job = pathlib.Path(sys.argv[sys.argv.index("--job-dir") + 1])
    mode = json.loads((job / "spec.json").read_text())["mode"]
    def ev(**k): print("@@ft " + json.dumps(k), flush=True)
    ev(event="phase", phase="train"); ev(event="plan", total_steps=4)
    if mode == "fail":
        print("Traceback: boom", flush=True); ev(event="error", message="RuntimeError: boom"); sys.exit(1)
    if mode == "stop":
        for i in range(600):
            if (job / "STOP").exists():
                ev(event="checkpoint", step=i, path=str(job)); sys.exit(3)
            ev(event="step", step=1, total_steps=4, loss=2.0); time.sleep(0.05)
        sys.exit(1)
    for i in range(1, 5):
        ev(event="step", step=i, total_steps=4, loss=2.0 / i)
    out = job / "m.gguf"; out.write_text("gguf")
    ev(event="file", kind="gguf", path=str(out)); ev(event="done")
''')


class JobRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.script = self.root / "fake_trainer.py"
        self.script.write_text(FAKE_TRAINER)
        self.updates = []
        self.status = "queued"

    def tearDown(self):
        self.tmp.cleanup()

    def update(self, **fields):
        self.updates.append(fields)
        if "status" in fields:
            self.status = fields["status"]

    def run_fake(self, job_id, mode, **kw):
        return job.run_job(job_id=job_id, spec={"mode": mode, "name": "m"}, outputs_root=self.root,
                           update=self.update, get_status=lambda: self.status,
                           command=[sys.executable, str(self.script), "--job-dir", str(job.job_dir_for(self.root, job_id))],
                           persist_every_s=0, **kw)

    def test_completed_job_records_files_and_progress(self):
        released = []
        status = job.run_job(job_id="j_ok", spec={"mode": "ok", "name": "m"}, outputs_root=self.root,
                             update=self.update, get_status=lambda: self.status, release_vram=lambda: released.append(1),
                             command=[sys.executable, str(self.script), "--job-dir", str(job.job_dir_for(self.root, "j_ok"))],
                             persist_every_s=0)
        self.assertEqual(status, "completed")
        self.assertEqual(released, [1])
        final = self.updates[-1]
        self.assertEqual(final["progress"], 1.0)
        self.assertEqual((final["step"], final["total_steps"]), (4, 4))
        self.assertTrue(any(f.endswith("m.gguf") for f in final["files"]))
        self.assertTrue(any(f.endswith("train.log") for f in final["files"]))
        log = (job.job_dir_for(self.root, "j_ok") / "train.log").read_text()
        self.assertIn('"event": "done"', log)

    def test_failed_job_carries_the_trainers_error(self):
        self.assertEqual(self.run_fake("j_bad", "fail"), "failed")
        self.assertEqual(self.updates[-1]["error"], "RuntimeError: boom")

    def test_cancel_asks_for_a_clean_stop(self):
        def cancel_soon():
            time.sleep(0.5)
            self.status = "cancelling"
        import threading
        threading.Thread(target=cancel_soon, daemon=True).start()
        self.assertEqual(self.run_fake("j_stop", "stop"), "cancelled")
        self.assertTrue((job.job_dir_for(self.root, "j_stop") / "STOP").exists())
        self.assertTrue(self.updates[-1]["result"]["checkpoints"])

    def test_a_job_folder_is_never_reused(self):
        self.run_fake("j_once", "ok")
        with self.assertRaises(FileExistsError):
            self.run_fake("j_once", "ok")

    def test_resume_needs_a_saved_checkpoint(self):
        self.assertIn("no fine-tune job", job.resume_dir_error(self.root, "j_missing"))
        self.run_fake("j_ok2", "ok")
        self.assertIn("saved no checkpoint", job.resume_dir_error(self.root, "j_ok2"))
        (job.job_dir_for(self.root, "j_ok2") / "resume.pt").write_bytes(b"")
        self.assertIsNone(job.resume_dir_error(self.root, "j_ok2"))


if __name__ == "__main__":
    unittest.main()
