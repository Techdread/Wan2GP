# Wan2GP Windows API Node Setup

This guide is for setting up a **Windows 3090 machine** as a **LAN-callable Wan2GP API node**.

## Goal

Set up Wan2GP so other machines on the network can call it through the hardened API server with a bearer token.

---

## Files to bring over from the working machine

Copy these custom files into the fresh Wan2GP clone on the Windows machine:

- `agent_api.py`
- `agent_api_server.py`

Optional reference docs:

- `Wan2GP-NETWORK-API.md`
- `Wan2GP-API-HARDENING.md`
- `Wan2GP-CLIENT-SETUP.md`
- `Wan2GP-CLIENT-SETUP-REVISION-BRIEF.md`

Do **not** copy:

- `ckpts/`
- `outputs/`
- `__pycache__/`
- `.git/`
- old venvs

Models/checkpoints should be downloaded fresh on the Windows machine.

---

## Recommended install location

Example:

```text
C:\Wan2GP
```

The batch file provided separately assumes this path by default.

---

## Base install

1. Clone Wan2GP fresh onto the Windows machine.
2. Create a fresh Python environment.
3. Install dependencies.
4. Copy in the two custom API files.
5. Download the required checkpoints/models.
6. Start the API server.

---

## Suggested Python / Torch stack

From the Wan2GP installation docs for RTX 30xx:

- Python `3.11.14`
- PyTorch `2.10.0`
- CUDA `13.x` stack as used by Wan2GP docs

If the machine will follow Wan2GP's current recommended path, use the repo's installation instructions as the source of truth.

---

## Minimum runtime requirements

The Windows node needs:

- fresh Wan2GP clone
- fresh Python environment / venv
- Wan2GP dependencies installed
- `agent_api.py`
- `agent_api_server.py`
- downloaded models/checkpoints
- bearer token configured via environment variable

---

## Start command

PowerShell example:

```powershell
$env:WAN2GP_TOKEN="REPLACE_WITH_YOUR_TOKEN"
$env:WAN2GP_OUTPUTS_ROOT="E:\Wan2GP\Outputs"
$env:WAN2GP_JOB_DB="$env:USERPROFILE\.wan2gp\jobs.sqlite"
python .\agent_api.py serve --host 0.0.0.0 --port 8100
```

Batch-file version is in:

- `start-wan2gp-api.bat`

---

## Environment variables

Important variables:

- `WAN2GP_TOKEN` — required bearer token for all non-health endpoints
- `WAN2GP_OUTPUTS_ROOT` — root directory for safe file downloads
- `WAN2GP_JOB_DB` — SQLite job history location
- `WAN2GP_LOG_PROMPTS` — set `0` normally, `1` only for debugging
- `WAN2GP_CORS_ORIGINS` — optional, only if browser-based callers need CORS

Example values:

```text
WAN2GP_TOKEN=your-shared-secret-token
WAN2GP_OUTPUTS_ROOT=C:\Wan2GP\outputs
WAN2GP_JOB_DB=C:\Users\Peter\.wan2gp\jobs.sqlite
WAN2GP_LOG_PROMPTS=0
WAN2GP_CORS_ORIGINS=
```

---

## Windows startup file

A ready batch file has been created:

- `start-wan2gp-api.bat`

Before use, edit:

- `WANGP_DIR`
- `PYTHON_EXE` if needed
- `WAN2GP_TOKEN`

---

## Running at login with Task Scheduler

Recommended approach on Windows:

### Create a scheduled task

- Open **Task Scheduler**
- Create a new task called `Wan2GP API`
- Run whether user is logged on or not
- Run with highest privileges

### Trigger

Use one of:

- **At log on**
- or **At startup**

### Action

Program/script:

```text
C:\Wan2GP\start-wan2gp-api.bat
```

If you store the batch file somewhere else, point to that path instead.

### Start in

```text
C:\Wan2GP
```

---

## Firewall

Allow inbound TCP on the API port if needed:

- Port `8100`

Restrict to LAN/private network only if possible.

---

## Health check

From another machine on the network:

```bash
curl http://WINDOWS_MACHINE_IP:8100/api/health
```

Expected: health should respond without auth.

---

## Auth check

Non-health endpoints should require the bearer token:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" http://WINDOWS_MACHINE_IP:8100/api/models
```

Without token, expected result:

- `401 Unauthorized`

---

## Important note

The dedicated API server is the intended interface for network callers.
Avoid building automation against the Gradio UI.

Also: do not assume the Gradio web UI and this in-process API server should run together on the same runtime instance.

---

## Quick handoff summary for the setup agent

Set up a fresh Wan2GP install on Windows, then copy in:

- `agent_api.py`
- `agent_api_server.py`

Configure:

- `WAN2GP_TOKEN`
- `WAN2GP_OUTPUTS_ROOT`
- `WAN2GP_JOB_DB`

Use:

- `start-wan2gp-api.bat`

Then verify:

1. `GET /api/health` works from LAN
2. `GET /api/models` requires bearer token
3. a test job can be created successfully
