# Wan2GP Network API Guide

This is the LAN-facing programmable API around WanGP. Apps on the network
should integrate against this server, **not** against the Gradio web UI on
port 7860.

The server lives in `agent_api_server.py` (delegated to from
`agent_api.py serve`). Hardening details — async jobs, auth, observability —
are tracked in the companion file `Wan2GP-API-HARDENING.md`.

---

## Endpoints

```
GET    /api/health                        — liveness + GPU + queue + version
GET    /api/models                        — list of model_type entries with sizes + capability hints
GET    /api/models/{model_type}           — full enriched entry: defaults, applicable settings, model_def, sizes
GET    /api/loras?model_type=...          — lora files for a model 
GET    /api/settings                      — global default settings template
GET    /api/settings/schema               — typed settings schema (registered + freeform)
POST   /api/settings/validate             — dry-run validate a request without queueing it
POST   /api/uploads                       — upload an image/video/audio input for generation
GET    /api/file?path=...                 — download a file (constrained to outputs root)

POST   /api/jobs                          — create a job (returns immediately, validated)
GET    /api/jobs?limit=N                  — list recent jobs
GET    /api/jobs/{id}                     — status + result
DELETE /api/jobs/{id}                     — cancel
GET    /api/jobs/{id}/events              — Server-Sent Events stream

POST   /api/release                       — release VRAM

# Deprecated (kept for one release):
POST   /api/generate                      — sync wrapper around /api/jobs
POST   /api/batch                         — sync wrapper around /api/jobs
```

Every response except `/api/health` is gated by an optional bearer token
(`WAN2GP_TOKEN`). If the env var is unset, the server runs unauthenticated
and logs a warning on boot.

Every request is tagged with `X-Request-Id` (echoed in JSON logs). Clients
may set their own; otherwise the server assigns one.

---

## Quick start

### Run the server

For one-off testing:

```bash
cd /media/peter/AI/Wan2GP
WAN2GP_TOKEN=mysecret ./venv/bin/python agent_api.py serve \
    --host 0.0.0.0 --port 8100
```

For "always on" via systemd user unit:

```bash
cd /media/peter/AI/Wan2GP
WAN2GP_TOKEN=mysecret ./deploy/install-systemd-unit.sh
```

The installer enables `wan2gp-api.service`, calls `loginctl enable-linger`
so it survives logout, and writes the unit to
`~/.config/systemd/user/wan2gp-api.service`.

### Health check

```bash
curl http://192.168.1.199:8100/api/health
```

```json
{
  "status": "ok",
  "version": "0.6.0",
  "uptime_seconds": 1234,
  "queue": {"running": 0, "queued": 0},
  "current_job_id": null,
  "gpu": {
    "name": "NVIDIA GeForce RTX 3060",
    "vram_total_mb": 12288,
    "vram_used_mb": 4321
  }
}
```

If the GPU is unreachable, `/api/health` returns `503` with
`status: "degraded"` and a short `reason` field. Use this for liveness probes.

---

## Auth

```bash
# All non-health endpoints require:
curl -H 'Authorization: Bearer mysecret' http://192.168.1.199:8100/api/jobs
```

Missing or wrong token → `401`. Health is always public.

There is no per-user identity, no rotation, no scopes. This is a LAN gate.

---

## Async job lifecycle (recommended)

### Upload reference media

Remote jobs reference files by their server-side paths. Upload local images,
videos, or audio first, then use the returned `path` in the job payload:

```bash
curl -X POST http://192.168.1.199:8100/api/uploads \
  -H 'Authorization: Bearer mysecret' \
  -F 'file=@person.png'
```

```json
{
  "upload_id": "u_...",
  "filename": "person.png",
  "path": "/media/peter/AI/Wan2GP/outputs/uploads/u_.../person.png",
  "bytes": 248193,
  "mime_type": "image/png"
}
```

Uploads are authenticated, limited to common image/video/audio extensions,
stored below `WAN2GP_OUTPUTS_ROOT/uploads`, and capped at 1 GB by default.
Set `WAN2GP_UPLOAD_MAX_BYTES` to change that cap.

### 1. Create a job

```bash
curl -X POST http://192.168.1.199:8100/api/jobs \
  -H 'Authorization: Bearer mysecret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "z_image",
    "prompt": "A cinematic fox in a rainy cyberpunk street",
    "resolution": "1024x1024",
    "num_inference_steps": 8,
    "seed": -1
  }'
```

Returns immediately (`202 Accepted`):

```json
{
  "job_id": "j_01HXXXXXXXXXXXXXXXXXXXX",
  "capability": "image-generation",
  "model_type": "z_image",
  "status": "queued",
  "queue_position": 1,
  "progress": 0.0,
  "step": 0,
  "total_steps": 0,
  "created_at": "2026-05-03T19:03:14Z",
  "started_at": null,
  "completed_at": null,
  "request": { ... },
  "files": [],
  "error": null,
  "request_id": "..."
}
```

### 2. Poll status

```bash
curl -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/jobs/j_01HXXXXXXXXXXXXXXXXXXXX
```

`status` transitions: `queued` → `running` → `completed | failed | cancelled`.
While running, `progress` (0.0-1.0), `step`, and `total_steps` advance.

### 3. Stream progress (SSE)

```bash
curl -N -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/jobs/j_01HXXXX/events
```

The connection emits `event: status` lines with the full job record on each
update, plus heartbeat comments every 15s, and closes on terminal status.

### 4. Cancel

```bash
curl -X DELETE -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/jobs/j_01HXXXX
```

- Queued → 200, `status: cancelled`.
- Running → 202, `status: cancelling` (worker stops at next safe step, VRAM released).
- Terminal → 409.

### 5. Download the result

```bash
# files[] in the job record are server-side absolute paths.
curl -L -H 'Authorization: Bearer mysecret' -o fox.jpg \
  "http://192.168.1.199:8100/api/file?path=/media/peter/AI/Wan2GP/outputs/fox.jpg"
```

`/api/file` rejects anything outside `WAN2GP_OUTPUTS_ROOT`
(default `/media/peter/AI/Wan2GP/outputs`). Symlink escapes and `..`
traversal are checked via `realpath` and return `403`.
Responses use the file's media type (`audio/wav`, `audio/mpeg`,
`video/mp4`, and so on), allowing authenticated browser clients to turn the
downloaded Blob into an inline image, video, or audio preview.

---

## Video example

```bash
curl -X POST http://192.168.1.199:8100/api/jobs \
  -H 'Authorization: Bearer mysecret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "wan21_t2v_14B",
    "prompt": "A cat walking through a moonlit garden",
    "resolution": "832x480",
    "num_inference_steps": 30,
    "video_length": 81,
    "guidance_scale": 5.0,
    "flow_shift": 3.0
  }'
```

Then poll `/api/jobs/{id}` until `status: "completed"` and download via
`/api/file`.

---

## WanGP v12.41 models

The v0.6 API exposes normalized `inputs`, `outputs`, `media_inputs`,
`capabilities`, and text-encoder `config_choices` for all models. These are
derived from WanGP's own model metadata rather than a separate hard-coded API
list.

Important v12.41 model IDs:

| Family | Model IDs | API notes |
| --- | --- | --- |
| MiniMax H3 FL2VA | `minimax_h3_fl2va`, `minimax_h3_fl2va_pruned` | Text/start/end image to video with native stereo audio. |
| MiniMax H3 Ref2VA | `minimax_h3_ref2va`, `minimax_h3_ref2va_pruned` | Up to 9 image, 2 video, and 2 audio references. |
| Krea 2 Edit | `krea2_raw_edit`, `krea2_turbo_edit` | Reference editing, masks, inpainting, and outpainting. |
| LTX-2 MSR V2 | `ltx2_22B_msr_v2` | Supply 2–5 reference images; background first. |
| Joy Echo Surgical | `joyai_echo_surgical` | Connected audiovisual shots with Joy memory commands. |
| Shotplan | `shotplan_t2v`, `shotplan_t2v_2_2` | Prompt-relay ranges define planned shots. |

### MiniMax H3 FL2VA payload

```json
{
  "model_type": "minimax_h3_fl2va",
  "prompt": "A cinematic jazz performance with synchronized dialogue and room sound.",
  "resolution": "832x480",
  "num_inference_steps": 20,
  "video_length": 124,
  "image_start": "/server/path/start.png",
  "image_end": "/server/path/end.png",
  "image_prompt_type": "SE",
  "config": "int8"
}
```

Valid H3 text-encoder configs are `bf16`, `int8`, `nvfp4_awq`,
`gguf_q2_k`, and `gguf_q4_k_m`.

### MiniMax H3 Ref2VA payload

```json
{
  "model_type": "minimax_h3_ref2va_pruned",
  "prompt": "Use the referenced person, motion, and voice in a new cinematic shot.",
  "resolution": "832x480",
  "num_inference_steps": 20,
  "video_length": 124,
  "image_refs": ["/server/path/person.png"],
  "video_prompt_type": "IVG",
  "video_guide": "/server/path/motion.mp4",
  "audio_prompt_type": "A",
  "audio_guide": "/server/path/voice.wav"
}
```

Use `audio_prompt_type: "K"` to reuse the soundtrack of the reference
video instead of supplying separate audio. Reference clips must satisfy the
duration/count constraints returned by the model detail endpoint.

### Krea 2 Edit and LTX MSR

Krea 2 Identity Edit accepts reference images through `image_refs` and enables
them with `video_prompt_type: "I"` or `"KI"`. For inpainting, also provide
`image_guide`, `image_mask`, and an appropriate `model_mode` returned by
`GET /api/models/{model_type}`.

LTX MSR V2 uses:

```json
{
  "model_type": "ltx2_22B_msr_v2",
  "prompt": "The referenced subjects together in the referenced environment.",
  "image_refs": ["background.png", "person.png"],
  "video_prompt_type": "KI",
  "resolution": "1280x720",
  "video_length": 145,
  "num_inference_steps": 8
}
```

## Other upstream models and Ideogram 4

Network nodes should not hard-code the old model list. After updating a node,
call `GET /api/models` or `GET /api/models/{model_type}` and build the UI or
task payloads from the returned model definition/defaults. New upstream model
definitions live in `defaults/*.json`; once the node has pulled the latest
repo, `ideogram4` and `ideogram4_nf4` are available as normal `model_type`
values.

### Ideogram 4 model ids

| `model_type` | Quantization | Notes |
| --- | --- | --- |
| `ideogram4` | FP8 | Default Ideogram 4 path. |
| `ideogram4_nf4` | NF4 | Uses the NF4 transformer files and should run with the NF4/bitsandbytes kernels installed. The model definition forces SDPA on lower-capability GPUs. |

Both variants are image-output models. They use the `ideogram4` architecture,
the Flux2 VAE backbone, the Qwen3-VL text encoder, and paired conditional /
unconditional Ideogram 4 transformer files. First use may download the model
assets from Hugging Face, so each network node needs outbound model-download
access or pre-populated `models/` cache files.

### Ideogram 4 generation payload

Ideogram 4 works best with its structured JSON prompt format. Plain text is
accepted, but nodes should either submit serialized JSON in `prompt` or let the
operator use Magic Prompt / the Visual Helper in the main UI to create the JSON
first.

Minimum payload:

```bash
curl -X POST http://192.168.1.199:8100/api/jobs \
  -H 'Authorization: Bearer mysecret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "ideogram4",
    "image_mode": 1,
    "prompt": "{\"aspect_ratio\":\"4:3\",\"high_level_description\":\"A clean poster for a night market noodle stall, readable title text, warm lights, busy street background.\",\"style_description\":{\"medium\":\"graphic_design\",\"aesthetics\":\"polished editorial poster\",\"lighting\":\"warm neon and lantern light\",\"color_palette\":[\"#101820\",\"#FEE715\",\"#E94F37\",\"#FFFFFF\"]},\"compositional_deconstruction\":{\"background\":\"A lively night market street with lanterns, steam, and softly blurred people.\",\"elements\":[{\"type\":\"text\",\"bbox\":[80,120,210,880],\"text\":\"MIDNIGHT NOODLES\",\"desc\":\"Large crisp uppercase title, centered, high contrast, readable from a distance.\"},{\"type\":\"obj\",\"bbox\":[360,260,820,740],\"desc\":\"A steaming bowl of noodles with chopsticks, placed as the main subject.\"}]}}",
    "resolution": "1024x768",
    "batch_size": 1,
    "model_mode": "V4_DEFAULT_20",
    "seed": -1
  }'
```

WanGP settings treat `prompt` as a string. If your node builds the prompt from a
native object, serialize the Ideogram JSON before calling `/api/jobs`:

```json
{
  "model_type": "ideogram4",
  "image_mode": 1,
  "prompt": "{\"aspect_ratio\":\"4:3\",\"high_level_description\":\"A clean poster...\"}",
  "resolution": "1024x768",
  "batch_size": 1,
  "model_mode": "V4_DEFAULT_20",
  "seed": -1
}
```

Supported Ideogram 4 presets:

| `model_mode` | Use |
| --- | --- |
| `V4_QUALITY_48` | Highest quality, slowest. |
| `V4_DEFAULT_20` | Balanced default. |
| `V4_TURBO_12` | Fastest. |

Recommended node behavior:

- set `image_mode` to `1`
- keep `batch_size` at `1` unless the node has enough VRAM for larger batches
- omit `negative_prompt`; Ideogram 4 does not use it
- prefer `model_mode` over legacy `sample_solver` for the preset
- treat safety-filter failures as model/runtime failures, not client retry loops
- use `GET /api/models/ideogram4` or `GET /api/models/ideogram4_nf4` to inspect
  the current defaults before queueing work

### Visual Helper / Magic Wand exposure

The Visual Helper is a browser-side editor for Ideogram 4's JSON prompt format.
It is exposed in the main Web UI through the Magic Wand next to the prompt when
an Ideogram 4 model is selected. It edits/draws bounding boxes and applies the
final JSON back into the prompt field.

Headless network API clients do not receive the Visual Helper UI over
`/api/jobs`; they should send the final prompt JSON in the job payload. Browser
clients embedding the WanGP UI can open the helper with:

```js
window.wangpIdeogram4PromptHelper.openMagicWand();
window.wangpIdeogram4PromptHelper.openMagicWand("advanced");
window.wangpIdeogram4PromptHelper.openMagicWand("wizard");
```

Use Magic Prompt to create the first JSON draft, then Visual Helper to adjust
object and text boxes before queueing the job.

---

## Capability discovery

### List all models (`GET /api/models`)

```bash
curl -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/models
```

Filter server-side with `family`, `capability`, `input`, and `output` query
parameters. Values may be comma-separated:

```bash
curl -H 'Authorization: Bearer mysecret' \
  'http://192.168.1.199:8100/api/models?input=image&output=video'
```

Every field below is **auto-derived**: from `defaults/<model_type>.json`,
the family handler's `query_model_def()` feature flags, and a HEAD-cached
HuggingFace lookup for `size_bytes`. Adding a new model is zero-work —
drop a new `defaults/<x>.json` and it appears here on next call.

```json
{
  "models": [
    {
      "model_type": "z_image",
      "architecture": "z_image",
      "family": "z_image",
      "capability": "image-generation",
      "name": "Z-Image Turbo 6B",
      "description": "Z-Image is a powerful and highly efficient image generation model with 6B parameters...",
      "param_count_b": 6.0,
      "default_resolution": "1024x1024",
      "default_steps": 8,
      "default_video_length": null,
      "size_bytes": 12309879106,
      "size_status": null,
      "quant_variants": ["bf16", "int8"],
      "main_outputs": ["image"],
      "outputs": ["image"],
      "inputs": ["text"],
      "media_inputs": {"image": {"reference": false}, "video": {}, "audio": {}},
      "capabilities": {"text_to_image": true, "reference_images": false},
      "config_label": "Config",
      "config_choices": [],
      "applicable_settings_count": 3,
      "url_count": 2
    },
    {
      "model_type": "ltx2_22B_distilled",
      "architecture": "ltx2_22B",
      "family": "ltx2",
      "capability": "video-generation",
      "name": "LTX-2 2.3 Distilled 1.0 22B",
      "param_count_b": 22.0,
      "default_resolution": "1280x720",
      "default_steps": 8,
      "default_video_length": 241,
      "size_bytes": 37987776440,
      "quant_variants": ["int8"],
      "applicable_settings_count": 18,
      "url_count": 2
    }
  ],
  "families": { "z_image": ["z_image", "z_image_base", ...], "ltx2": [...], ... },
  "errors": []
}
```

- `capability`: one of `image-generation | video-generation | audio-generation` — derived from the model's feature flags (`image_outputs`, `i2v_class`, `t2v_class`, `vace_class`, ...).
- `param_count_b`: parsed from the model name (e.g. `"22B"` in `"LTX-2 2.3 Distilled 1.0 22B"`).
- `size_bytes`: HEAD-resolved size of the primary safetensors file. The first call to `/api/models` blocks briefly (≤4 s) to populate the cache; later calls are instant. Persisted to `~/.wan2gp/model_sizes.json` (override via `WAN2GP_SIZE_CACHE`).
- `size_status`: `null` if cached cleanly, `"pending"` if still resolving, or an HTTP error string.
- `quant_variants`: detected from filenames (`bf16`, `fp16`, `int8`, `fp4`, `q4_k_m`, etc.).
- `inputs` / `outputs`: normalized media modalities used to build model pickers.
- `media_inputs` / `capabilities`: detailed start/end/reference/control and generation-mode support.
- `config_choices`: selectable checkpoint components such as H3 quantized text encoders or PrunaAI VAE.
- `applicable_settings_count`: how many bounded/typed settings apply to this model. Use `GET /api/models/{model_type}` for the full list.

### Single model detail (`GET /api/models/{model_type}`)

```bash
curl -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/models/z_image
```

Returns the full enriched entry — useful for building UIs or for an agent
that wants the precise schema of valid inputs for a specific model:

```json
{
  "model_type": "z_image",
  "architecture": "z_image",
  "family": "z_image",
  "capability": "image-generation",
  "name": "Z-Image Turbo 6B",
  "description": "...",
  "param_count_b": 6.0,
  "urls": ["https://huggingface.co/.../ZImageTurbo_bf16.safetensors", "..."],
  "preload_urls": [],
  "quant_variants": ["bf16", "int8"],
  "resolution_choices": null,
  "api_metadata": {
    "main_outputs": ["image"],
    "outputs": ["image"],
    "inputs": ["text"],
    "media_inputs": {},
    "capabilities": {"text_to_image": true},
    "config_choices": []
  },
  "primary_size_bytes": 12309879106,
  "sizes": [
    {"url": "...", "bytes": 12309879106, "etag": "...", "fetched_at": "..."},
    {"url": "...", "bytes": 6154900000, "etag": "..."}
  ],
  "applicable_settings": [
    {"key": "NAG_scale", "label": "NAG Scale", "type": "number", "min": 1.0, "max": 20.0, "step": 0.01, "custom": false},
    {"key": "NAG_tau",   "label": "NAG Tau",   "type": "number", "min": 1.0, "max": 5.0,  "step": 0.01, "custom": false},
    {"key": "NAG_alpha", "label": "NAG Alpha", "type": "number", "min": 0.0, "max": 2.0,  "step": 0.01, "custom": false}
  ],
  "defaults": { "resolution": "1024x1024", "num_inference_steps": 8, "guidance_scale": 0, ... },
  "model_def": { "image_outputs": true, "guidance_max_phases": 0, "NAG": true, ... },
  "handler_loaded": true
}
```

`applicable_settings` is the set of typed/bounded settings whose visibility
resolver returns true for this model (e.g. NAG only matters for `z_image`,
not `z_image_base`; sliding-window settings only apply when the model
supports it). It's the smallest "extra dials" surface you need to build a
fully-functional UI for any model.

### Settings schema (`GET /api/settings/schema`)

```bash
curl -H 'Authorization: Bearer mysecret' \
  http://192.168.1.199:8100/api/settings/schema
```

```json
{
  "registered": [
    {"key": "guidance_scale", "label": "Guidance (CFG)", "type": "number", "min": 1.0, "max": 20.0, "step": 0.1, "custom": false},
    {"key": "flow_shift",     "label": "Shift Scale",    "type": "number", "min": 1.0, "max": 25.0, "step": 0.1, "custom": false},
    "..."
  ],
  "freeform": [
    {"key": "prompt",            "type": "string",  "default": ""},
    {"key": "seed",              "type": "integer", "default": -1},
    {"key": "resolution",        "type": "string",  "default": "832x480"},
    {"key": "image_start",       "type": "null",    "default": null},
    "..."
  ],
  "note": "registered: typed/bounded settings discovered from shared/extra_settings.py..."
}
```

- `registered`: settings with explicit type/range/step from
  `shared/extra_settings.py`. Adding a new bounded setting in WanGP
  automatically surfaces here.
- `freeform`: every other key in `models/_settings.json`, with a type
  inferred from its default value. Both lists are accepted by `POST /api/jobs`.

### Validate without queueing (`POST /api/settings/validate`)

```bash
curl -X POST -H 'Authorization: Bearer mysecret' \
  -H 'Content-Type: application/json' \
  http://192.168.1.199:8100/api/settings/validate \
  -d '{"model_type": "z_image", "num_inference_steps": 8, "NAG_scale": 999}'
```

```json
{"valid": false, "error": "NAG Scale must be at most 20."}
```

Useful for form validation before submitting a job. `POST /api/jobs` runs
the same validation and returns `400` if it fails.

### Capability summary

A platform adapter can build its capability → model list purely from
`/api/models` without hardcoding any mapping. For finer control (per-model
dials), pull `/api/models/{model_type}` lazily.

---

## Python client

```python
from agent_api import WanGPAgent

agent = WanGPAgent(url="http://192.168.1.199:8100", token="mysecret")

# Discover only models that accept images and produce video.
models = agent.discover_models(input_modality="image", output_modality="video")

# Upload references from the client machine.
person = agent.upload_file("person.png")["path"]
motion = agent.upload_file("motion.mp4")["path"]
voice = agent.upload_file("voice.wav")["path"]

# First-class MiniMax H3 helper.
h3 = agent.generate_minimax_h3(
    prompt="Use the referenced person, movement, and voice in a new shot.",
    model="minimax_h3_ref2va_pruned",
    reference_images=[person],
    reference_videos=[motion],
    reference_audios=[voice],
    text_encoder_config="int8",
)

# Async job API
job = agent.submit_job({
    "model_type": "z_image",
    "prompt": "A cinematic fox in a rainy cyberpunk street",
    "resolution": "1024x1024",
    "num_inference_steps": 8,
})
final = agent.wait_for_job(job["job_id"])
if final["status"] == "completed":
    agent.download_file(final["files"][0], "fox.jpg")

# Convenience wrappers submit through the asynchronous job API and wait for
# the terminal job record.
result = agent.generate_image(
    prompt="A sunset",
    model="z_image",
)
```

---

## Observability

Logs are JSON lines on stdout. Fields: `ts`, `level`, `event`,
`request_id`, `job_id?`, `capability?`, `model_type?`, `duration_ms?`,
plus event-specific fields. Examples:

```
{"ts":"...","event":"server_started","host":"0.0.0.0","port":8100,"auth":true}
{"ts":"...","event":"job_created","job_id":"j_...","request_id":"...","capability":"image-generation","request":{...}}
{"ts":"...","event":"job_started","job_id":"j_...","request_id":"..."}
{"ts":"...","event":"job_completed","job_id":"j_...","duration_ms":12340,"files":1}
{"ts":"...","event":"vram_released","reason":"post_cancel"}
{"ts":"...","event":"request","request_id":"...","method":"POST","path":"/api/jobs","status":202,"duration_ms":405}
```

Prompts are redacted by default. Set `WAN2GP_LOG_PROMPTS=1` to include them
when debugging.

The last 200 jobs are persisted to SQLite at `~/.wan2gp/jobs.sqlite` (override
via `WAN2GP_JOB_DB`). On restart, jobs that were in `running` are marked
`failed` with `error: "server restarted"` so callers don't poll forever.

---

## Configuration reference

| Env var               | Default                              | Purpose                                                      |
| --------------------- | ------------------------------------ | ------------------------------------------------------------ |
| `WAN2GP_TOKEN`        | _(unset)_                            | Bearer token. Unset = unauth (LAN trust mode).               |
| `WAN2GP_OUTPUTS_ROOT` | `<repo>/outputs`                     | Root for `/api/file` realpath check.                         |
| `WAN2GP_JOB_DB`       | `~/.wan2gp/jobs.sqlite`              | Persistent job log path.                                     |
| `WAN2GP_JOB_HISTORY`  | `200`                                | Number of jobs to retain.                                    |
| `WAN2GP_LOG_PROMPTS`  | _(unset)_                            | "1" to include prompt text in JSON logs.                     |
| `WAN2GP_CORS_ORIGINS` | _(unset)_                            | Comma-separated origin allow-list, or `*`. Empty = CORS off. |
| `WAN2GP_SIZE_CACHE`   | `~/.wan2gp/model_sizes.json`         | Persisted HEAD cache of model file sizes.                    |
| `WAN2GP_UPLOAD_MAX_BYTES` | `1073741824` (1 GB)             | Maximum size accepted by `POST /api/uploads`.                |

CLI flags on `agent_api.py serve` mirror the env vars: `--host`, `--port`,
`--profile`, `--attention`, `--token`, `--outputs-root`, `--history-limit`,
`--cors-origins`.

---

## Constraints to be aware of

- **One generation at a time** — the GPU is single-tenant; jobs queue rather
  than run in parallel.
- **Uploaded inputs persist** — `/api/uploads` stores inputs below
  `WAN2GP_OUTPUTS_ROOT/uploads`; remove old upload directories as part of the
  node's normal storage-retention policy.
- **No HTTPS** — terminate TLS in a reverse proxy if you ever expose this off
  LAN.
- **Local in-process mode and the Gradio web UI cannot coexist** — either
  use the dedicated API server or the web UI on `7860`, not both at once.
- **Back-compat shims (`POST /api/generate`, `POST /api/batch`) are deprecated**
  — they still work and return `Deprecation: true` plus
  `Link: </api/jobs>; rel="successor-version"`. Migrate to `/api/jobs`.

---

## Adding a new model

The whole capability + schema layer is auto-derived. Cost of adding a model:

| What you add                               | Work in this API |
| ------------------------------------------ | ---------------- |
| New `defaults/<model_type>.json`           | **Zero** — it appears in `/api/models` on next call. Capability, params, defaults, applicable settings, and size are all derived from the file + the family handler's feature flags. |
| New architecture (e.g. a new `*_handler`)  | **One line** — add it to `wgp.py:family_handlers`. This API re-reads that list via AST on each rebuild. |
| New typed setting                          | **One entry** in `shared/extra_settings.py` (`_add_setting(...)`) — surfaces in `/api/settings/schema` and per-model `applicable_settings` automatically. |

To force a rebuild without restarting (e.g. after dropping a new
`defaults/*.json`), call `agent_api_introspect.invalidate_cache()` from a
Python REPL — the next `/api/models` call rebuilds.

---

## Web UI

The Gradio UI is still on `0.0.0.0:7860` (e.g. `http://192.168.1.199:7860`)
for human use. Don't build other programs against it — it's noisy and not
stable-looking. Use `/api/jobs` instead.
