# Fine-tuning on a Wan2GP node

The agent API can LoRA fine-tune a small Qwen3.5 model as an ordinary job, so
a GPU machine that already serves images, video and speech can also train. No
extra server and no second Python environment: the trainer runs in Wan2GP's
own venv with the repo's vendored Qwen3.5 model code
(`shared/prompt_enhancer/qwen3_5`), peft and flash-linear-attention, which
Wan2GP already installs.

Fine-Tuning Lab in the DevTools Hub drives this: pick a training set and a
machine, and it uploads the data and submits the job. The API is plain HTTP,
so anything else can drive it too.

## Deploying

1. `git pull` in the Wan2GP folder on the node.
2. Restart the agent API (`start-wan2gp-api.bat` / the scheduled task on
   Windows, `systemctl restart wan2gp-api` on Linux).
3. Check: `GET /api/finetune` answers (older servers return 404).

On its first job the node downloads the base model from Hugging Face (about
9 GB for Qwen3.5-4B) and llama.cpp's GGUF converter (a source tarball, pinned
in `finetune/export.py`) into `<checkpoints path>/finetune/`. Both are kept.

## What fits where

Training holds the bf16 base weights plus activations for one sequence
(gradient checkpointing, batch 1). `GET /api/finetune` lists estimates:

| base | 8k tokens | 16k tokens | 32k tokens |
|---|---|---|---|
| Qwen3.5-0.8B | 5.8 GB | 6.5 GB | 8.0 GB |
| Qwen3.5-2B | 9.3 GB | 10.8 GB | 13.6 GB |
| Qwen3.5-4B | 15.2 GB | 17.4 GB | 21.8 GB |

So a 3090 (24 GB) trains the 4B at 16k, and a 3060 (12 GB) the 2B. Memory
held by other programs counts against this: **unload LM Studio's models on
that machine first.** The job frees Wan2GP's own model before it starts.

Examples longer than `max_length` are left out, never cut: a cut page teaches
the model to stop mid-page. The job reports how many were left out.

## The job

Upload the training data (the dataset's SFT `.jsonl` format: `messages` with
system, user and assistant, the assistant's reasoning optional in
`reasoning_content`):

```
POST /api/uploads   (multipart, field "file", a .jsonl)  → {"path": ...}
```

Then submit:

```json
POST /api/jobs
{
  "model_type": "finetune-qwen3.5-4b",
  "name": "glb-coder-v1",
  "dataset_path": "<path from /api/uploads>",
  "epochs": 1, "learning_rate": 1e-4, "lora_rank": 32,
  "max_length": 16384, "grad_accum": 8,
  "gguf": "q8_0", "lmstudio": true
}
```

All options and defaults are in `finetune/spec.py` and `GET /api/finetune`.
`resume_from: "<job id>"` continues an earlier job from its last checkpoint
with the same data and settings.

Poll `GET /api/jobs/<id>` (or stream `/events`). `progress` covers the whole
job; `step` / `total_steps` count optimizer steps; `result` carries the
phase, the latest loss, speed and ETA, loss and eval-loss history, the data
report and the checkpoints. `DELETE /api/jobs/<id>` asks the trainer to save
a checkpoint and stop (it is terminated if it hasn't stopped after 5 minutes).

The job takes the API's single queue like any generation job: image and video
jobs sent while it trains wait until it finishes.

### Before training starts

The job checks the loaded model on one example. A correctly loaded base scores
a loss around 1 on this data; a mis-loaded one scores near 12. Anything above
4, or no gradient reaching the LoRA weights, fails the job within minutes
rather than after hours.

## What a job leaves behind

Everything is under `<outputs>/finetune/<job id>/`, and nothing is overwritten:

| path | |
|---|---|
| `spec.json` | the settings the job ran with |
| `train.log` | everything the trainer printed |
| `metrics.jsonl` | loss, learning rate, speed per step; eval loss |
| `checkpoints/step-NNNNNN/` | the LoRA adapter at each save (all kept) |
| `adapter/` | the final adapter |
| `merged/` | base + LoRA, in the base checkpoint's own layout |
| `<name>-<id>-Q8_0.gguf` | the merged model for llama.cpp / LM Studio |

With `lmstudio: true` the GGUF is also hard-linked (or copied) into this
machine's LM Studio models folder under `wan2gp-finetunes/<name>/`, so LM
Studio on the same machine can load it straight away. The folder is
`WAN2GP_LMSTUDIO_MODELS_DIR`, else LM Studio's own `downloadsFolder` setting,
else `~/.lmstudio/models`.

## Tests

`python -m unittest test_agent_api_finetune` covers request validation, data
preparation (labels on the answer only, long examples left out), the progress
protocol, LM Studio placement and the job runner with a fake trainer. They
need no GPU. The trainer itself (`finetune/train.py`) only runs on a node.
