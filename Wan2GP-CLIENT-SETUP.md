# Wan2GP Client Setup

> Most apps in the hub should not call this server directly — they should go
> through the shared capability router (see
> `01-ai-media-capability-architecture.md`). This doc is for tools that need
> raw access, for debugging, or for writing the platform adapter itself.

How to consume the Wan2GP Agent API from another machine on your LAN.

The server lives on the 3060 host at `http://192.168.1.199:8100`. It runs
as a systemd user unit (`wan2gp-api.service`) and survives reboots.

For the server-side spec, see `Wan2GP-NETWORK-API.md` and
`Wan2GP-API-HARDENING.md`.

---

## 1. Get the token

The server expects a bearer token on every call except `/api/health`. On the
3060 host:

```bash
grep WAN2GP_TOKEN ~/.config/systemd/user/wan2gp-api.service | cut -d= -f3-
```

(`grep` returns `Environment=WAN2GP_TOKEN=<value>`; the `cut` strips both
prefixes.)

Copy that value to the client machine. Treat it like a password — anyone
with it can drive generations on the GPU.

---

## 2. Verify reachability

From the client, no token required:

```bash
curl http://192.168.1.199:8100/api/health
```

Expected: a JSON body with `"status": "ok"`, GPU info, queue counts, and
version. If it hangs or refuses, the issue is the LAN/firewall — not the
API.

---

## 3. First authenticated call

```bash
TOKEN=<paste-token-here>
curl -H "Authorization: Bearer $TOKEN" \
  http://192.168.1.199:8100/api/models | head
```

- `401` → wrong token.
- A list of model entries with `capability` fields → you're in.

---

## 4. Generate from the shell (curl)

The minimum required body for `POST /api/jobs` is just `{"model_type": "..."}`.
Any field you omit (`prompt`, `resolution`, `num_inference_steps`, `seed`,
`negative_prompt`, `image_mode`, `activated_loras`, `loras_multipliers`,
etc.) is filled in by the underlying session from the model's defaults.
For real generations you'll always want at least a prompt; everything else
is optional unless the model needs it.

```bash
TOKEN=<paste-token>
SERVER=http://192.168.1.199:8100

# 1) submit — returns immediately with status "queued" or "running"
JOB=$(curl -s -X POST "$SERVER/api/jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "z_image",
    "prompt": "a fox in a rainy cyberpunk street",
    "resolution": "1024x1024",
    "num_inference_steps": 8
  }' | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
echo "job=$JOB"

# 2) poll until terminal, with a max-wait and progress logging
MAX_WAIT=60          # 60 s for image; raise to 600 for video
ELAPSED=0
LAST_LOG=0
while :; do
  STATE=$(curl -s -H "Authorization: Bearer $TOKEN" "$SERVER/api/jobs/$JOB")
  STATUS=$(echo "$STATE" | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')
  case "$STATUS" in
    completed|failed|cancelled) break ;;
  esac
  if (( ELAPSED >= MAX_WAIT )); then
    echo "timed out after ${MAX_WAIT}s; job is $STATUS" >&2
    exit 1
  fi
  if (( ELAPSED - LAST_LOG >= 30 )); then
    echo "still $STATUS after ${ELAPSED}s..."
    LAST_LOG=$ELAPSED
  fi
  sleep 2
  ELAPSED=$((ELAPSED + 2))
done

# 3) inspect terminal state
echo "$STATE" | python3 -m json.tool

# 4) download the result; URL-encode the path so spaces / & are safe
FILE=$(echo "$STATE" | python3 -c 'import sys,json;print(json.load(sys.stdin)["files"][0])')
curl -L -G \
  -H "Authorization: Bearer $TOKEN" \
  --data-urlencode "path=$FILE" \
  -o out.jpg \
  "$SERVER/api/file"
```

Replace `z_image` with any `model_type` returned by `/api/models`. Video
example body:

```json
{
  "model_type": "wan21_t2v_14B",
  "prompt": "a cat walking through a moonlit garden",
  "resolution": "832x480",
  "num_inference_steps": 30,
  "video_length": 81,
  "guidance_scale": 5.0,
  "flow_shift": 3.0
}
```

### Failed-job shape

When `status` is `"failed"`, `error` is a single string (the joined
generation errors, or the worker exception message). Example:

```json
{
  "job_id": "j_01KQR5NGPED8Y99WWFBVXX4P6P",
  "status": "failed",
  "model_type": "definitely_not_a_real_model",
  "files": [],
  "error": "'NoneType' object has no attribute 'get'",
  "started_at": "2026-05-04T00:21:51Z",
  "completed_at": "2026-05-04T00:21:51Z",
  "duration_seconds": 0.11
}
```

There is no structured error code. Treat `error` as a human-readable
message; for programmatic handling, branch on `status`.

### Listing recent jobs

```bash
curl -H "Authorization: Bearer $TOKEN" "$SERVER/api/jobs?limit=5"
```

```json
{
  "jobs": [
    {"job_id": "j_01...", "status": "completed", "model_type": "z_image", "files": ["..."], ...},
    {"job_id": "j_01...", "status": "cancelled", "model_type": "z_image", "files": [], ...}
  ]
}
```

Newest first. Default limit is 50. Useful for debugging or rebuilding
client-side history after a restart.

---

## 5. Generate from JavaScript (browser or Node)

> Browser-side CORS is supported once the operator adds your origin to
> `WAN2GP_CORS_ORIGINS` — see § Browser apps for setup and the
> token-exposure caveat. The example below works from Node, Electron's
> main process, or any browser running on an allowed origin.

```js
const SERVER = "http://192.168.1.199:8100";
const TOKEN = process.env.WAN2GP_TOKEN;        // server-side / Node only

const auth = { Authorization: `Bearer ${TOKEN}` };

async function generate(body, { maxWaitMs = 60_000 } = {}) {
  // 1) submit
  const create = await fetch(`${SERVER}/api/jobs`, {
    method: "POST",
    headers: { ...auth, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!create.ok) throw new Error(`submit failed: ${create.status}`);
  const { job_id } = await create.json();

  // 2) poll
  const start = Date.now();
  let lastLog = 0;
  while (true) {
    const r = await fetch(`${SERVER}/api/jobs/${job_id}`, { headers: auth });
    if (!r.ok) throw new Error(`poll failed: ${r.status}`);
    const state = await r.json();
    if (["completed", "failed", "cancelled"].includes(state.status)) {
      return state;
    }
    const elapsed = Date.now() - start;
    if (elapsed >= maxWaitMs) {
      throw new Error(`timed out after ${elapsed}ms; job is ${state.status}`);
    }
    if (elapsed - lastLog >= 30_000) {
      console.log(`still ${state.status} after ${Math.round(elapsed / 1000)}s`);
      lastLog = elapsed;
    }
    await new Promise(r => setTimeout(r, 2000));
  }
}

async function downloadAsObjectURL(serverPath) {
  const url = new URL(`${SERVER}/api/file`);
  url.searchParams.set("path", serverPath);    // URLSearchParams encodes for you
  const r = await fetch(url, { headers: auth });
  if (!r.ok) throw new Error(`download failed: ${r.status}`);
  const blob = await r.blob();
  return URL.createObjectURL(blob);            // feed to <img src> / <video src>
}

const final = await generate({
  model_type: "z_image",
  prompt: "a fox in a rainy cyberpunk street",
  resolution: "1024x1024",
  num_inference_steps: 8,
});

if (final.status === "completed") {
  const objectUrl = await downloadAsObjectURL(final.files[0]);
  document.getElementById("preview").src = objectUrl;   // <img id="preview">
  // For videos: <video id="preview" src=…> works the same way.
  // When you're done with the blob: URL.revokeObjectURL(objectUrl)
} else {
  console.error("job failed:", final.error);
}
```

For live progress you can use `EventSource` instead of polling — but
`EventSource` cannot send custom headers, so you would need either
same-origin + cookie auth or the `EventSource polyfill` that supports
headers. SSE format: `event: status` lines whose `data:` payload is the
full job record (same shape as `GET /api/jobs/{id}`), plus `: ping`
heartbeat comments every 15 s, closing on terminal status.

---

## 6. Generate from Python

Copy `agent_api.py` from the server repo to your client (or
`pip install -e` the repo). The wrapper handles HTTP, auth, downloads,
and polling for you.

```python
from agent_api import WanGPAgent

agent = WanGPAgent(
    url="http://192.168.1.199:8100",
    token="<paste-token>",
)

job = agent.submit_job({
    "model_type": "z_image",
    "prompt": "a fox in a rainy cyberpunk street",
    "resolution": "1024x1024",
    "num_inference_steps": 8,
})
final = agent.wait_for_job(job["job_id"], timeout=60)   # raises TimeoutError

if final["status"] == "completed":
    agent.download_file(final["files"][0], "fox.jpg")
else:
    print("job failed:", final.get("error"))
```

You can also export `WAN2GP_TOKEN` in the client's environment and omit
the `token=` argument — the constructor falls back to the env var.

### Convenience wrappers (legacy sync API)

These still work and are easier for one-off scripts; they go through the
deprecated `POST /api/generate` shim:

```python
result = agent.generate_image(prompt="a sunset", model="z_image")
print(result["files"])

result = agent.generate_video(
    prompt="a cat walking",
    model="wan21_t2v_14B",
    resolution="832x480",
    steps=30,
)
```

---

## 7. Live progress (Server-Sent Events)

```bash
curl -N -H "Authorization: Bearer $TOKEN" \
  "$SERVER/api/jobs/$JOB/events"
```

The connection emits `event: status` lines whose `data:` payload is the
full job record (same JSON shape as `GET /api/jobs/{id}`), plus `: ping`
heartbeat comments every 15 s, closing on terminal status.

```
event: status
data: {"job_id": "j_01...", "status": "running", "progress": 0.625, "step": 5, "total_steps": 8, ...}

: ping

event: status
data: {"job_id": "j_01...", "status": "completed", "files": ["..."], ...}
```

The first event always carries the current snapshot.

---

## 8. Cancellation

```bash
curl -X DELETE -H "Authorization: Bearer $TOKEN" \
  "$SERVER/api/jobs/$JOB"
```

- Queued → `200`, `status: cancelled`.
- Running → `202`, `status: cancelling` (worker stops at the next safe step
  and VRAM is released afterward; for jobs cancelled while the model is
  still loading, the transition can take tens of seconds).
- Already terminal → `409`.

In Python:

```python
agent.cancel_job(job_id)
```

---

## Browser apps

CORS is supported. Browser-side dashboards can call the API directly once
the operator has added their origin to the allow-list.

### 1. Configure the server

On the 3060, set `WAN2GP_CORS_ORIGINS` (comma-separated origins, or `*`).
For a Vite/Next dev dashboard at `http://localhost:5173`:

```bash
sudo -u peter sed -i \
  's|^Environment=WAN2GP_CORS_ORIGINS=.*|Environment=WAN2GP_CORS_ORIGINS=http://localhost:5173|' \
  /home/peter/.config/systemd/user/wan2gp-api.service
systemctl --user daemon-reload
systemctl --user restart wan2gp-api
```

Or pass it through the installer:

```bash
WAN2GP_CORS_ORIGINS=http://localhost:5173 ./deploy/install-systemd-unit.sh
```

Verify on the server's stdout / journal:

```
cors:    http://localhost:5173
```

A wildcard (`WAN2GP_CORS_ORIGINS=*`) lets any origin call the server.
Safe in the bearer-token model used here, but obviously broader.

### 2. What the server sends

For every origin in the allow-list:

```
Access-Control-Allow-Origin: http://localhost:5173
Vary: Origin
Access-Control-Expose-Headers: Deprecation, Link, X-Request-Id
```

Preflight (`OPTIONS`) responses also include:

```
Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS
Access-Control-Allow-Headers: Authorization, Content-Type, X-Request-Id
Access-Control-Max-Age: 600
```

`OPTIONS` requests bypass the bearer-token check (browsers cannot attach
auth headers to preflights). Requests from origins not in the allow-list
get a `403` on preflight, blocking the actual call.

### 3. Token exposure — still real

Even with CORS configured, putting a bearer token into browser-side
`fetch()` exposes it in:

- page source / JS bundles
- the network tab of devtools
- browser extensions with host permissions
- any other script on the page (XSS, supply-chain compromise of an npm
  dependency, etc.)

Once leaked, the token drives arbitrary generations on the GPU until it
is rotated — and there is no rotation tooling.

**Acceptable patterns for direct browser access:**

- A dashboard you alone use, on a trusted machine, on a trusted LAN.
- A short-lived token you regenerate often.
- An Electron renderer where the token lives in the main process and is
  injected via IPC, not the renderer's source.

**Don't do direct browser access if:**

- The page is reachable by anyone you don't fully trust.
- The token is shared across multiple users / installs.
- You can route through the platform capability router instead — that
  router is same-origin to the hub apps and holds the token server-side.

### 4. Worked example: dashboard polling

```js
const SERVER = "http://192.168.1.199:8100";
const TOKEN  = localStorage.getItem("wan2gp_token");   // see caveat above

const auth = { Authorization: `Bearer ${TOKEN}` };

async function refreshDashboard() {
  // Anyone can call /api/health (no auth, no CORS allow-list match needed
  // since browsers send Origin only when scripting; preflight is skipped
  // for simple GETs without custom headers).
  const health = await fetch(`${SERVER}/api/health`).then(r => r.json());

  // /api/jobs requires the token; preflight will fire because Authorization
  // is not a CORS-safelisted header.
  const jobs = await fetch(`${SERVER}/api/jobs?limit=20`, { headers: auth })
    .then(r => r.json());

  document.querySelector("#queue").textContent =
    `running: ${health.queue.running}  queued: ${health.queue.queued}`;
  document.querySelector("#gpu").textContent =
    `${health.gpu?.name ?? "?"}  ${health.gpu?.vram_used_mb ?? "?"} / ${health.gpu?.vram_total_mb ?? "?"} MB`;

  const tbody = document.querySelector("#jobs tbody");
  tbody.innerHTML = "";
  for (const j of jobs.jobs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${j.job_id.slice(-8)}</td>
      <td>${j.capability}</td>
      <td>${j.model_type}</td>
      <td>${j.status}</td>
      <td>${j.progress != null ? Math.round(j.progress * 100) + "%" : ""}</td>
      <td>${j.created_at}</td>
      <td>${j.error ?? ""}</td>`;
    tbody.appendChild(tr);
  }
}

// Refresh every 2 s. For an event-driven UI, attach an EventSource to a
// specific /api/jobs/{id}/events stream when a row is selected.
setInterval(refreshDashboard, 2000);
refreshDashboard();
```

Live progress stream from a selected job row:

```js
function streamJob(jobId, onUpdate) {
  // EventSource cannot send Authorization headers. Two ways out:
  //   1) Use the EventSourcePolyfill from `event-source-polyfill` which does.
  //   2) Or, run the dashboard same-origin via a proxy that injects the
  //      header server-side.
  const url = new URL(`${SERVER}/api/jobs/${jobId}/events`);
  const es = new EventSourcePolyfill(url, {
    headers: { Authorization: `Bearer ${TOKEN}` },
  });
  es.addEventListener("status", e => {
    const rec = JSON.parse(e.data);
    onUpdate(rec);
    if (["completed", "failed", "cancelled"].includes(rec.status)) {
      es.close();
    }
  });
  return () => es.close();
}
```

### 5. Cancelling a job from the dashboard

```js
async function cancelJob(jobId) {
  const r = await fetch(`${SERVER}/api/jobs/${jobId}`, {
    method: "DELETE",
    headers: auth,
  });
  if (r.status === 409) {
    console.log("job already terminal");
    return;
  }
  if (!r.ok) throw new Error(`cancel failed: ${r.status}`);
  return r.json();   // returns the updated job record
}
```

---

## Things to remember

- **`Authorization: Bearer <token>`** on every call except `/api/health`.
- **`files[]` are server-side paths** — feed them straight into
  `/api/file?path=...` to download. Don't try to read them locally.
- **URL-encode the `path` parameter** if it might contain spaces, `&`, or
  `#`. `curl --data-urlencode`, `URLSearchParams`, and Python's
  `urllib.parse.quote` all do this for you.
- **Only files under `/media/peter/AI/Wan2GP/outputs` are downloadable.**
  Anything else returns `403`.
- **One generation at a time.** The GPU is single-tenant; jobs queue.
  `queue_position` in the job record tells you where you sit.
- **Server restarts mark in-flight jobs as `failed: server restarted`** so
  your poller doesn't loop forever.
- **`POST /api/generate` and `POST /api/batch` are deprecated.** They still
  work and return `Deprecation: true` plus
  `Link: </api/jobs>; rel="successor-version"`. New code should use
  `/api/jobs`.

---

## Endpoint cheat sheet

| Method   | Path                          | Purpose                                 |
| -------- | ----------------------------- | --------------------------------------- |
| `GET`    | `/api/health`                 | Liveness + GPU + queue (no auth).       |
| `GET`    | `/api/models`                 | Model list with capability hints.       |
| `GET`    | `/api/loras?model_type=...`   | LoRA files for a model.                 |
| `GET`    | `/api/settings`               | Default settings template.              |
| `POST`   | `/api/jobs`                   | Create a job (returns immediately).     |
| `GET`    | `/api/jobs?limit=N`           | List recent jobs (default 50).          |
| `GET`    | `/api/jobs/{id}`              | Status + result.                        |
| `DELETE` | `/api/jobs/{id}`              | Cancel.                                 |
| `GET`    | `/api/jobs/{id}/events`       | SSE progress stream.                    |
| `GET`    | `/api/file?path=...`          | Download a generated file.              |
| `POST`   | `/api/release`                | Release VRAM.                           |
| `POST`   | `/api/generate` *(deprecated)* | Sync wrapper around `/api/jobs`.        |
| `POST`   | `/api/batch` *(deprecated)*    | Sync batch wrapper around `/api/jobs`.  |
