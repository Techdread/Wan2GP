# Wan2GP Dashboard API Settings

Use these settings on the dashboard machine to call the Wan2GP API node running on the RTX 3090 Windows machine.

## API Node

```text
Base URL: http://192.168.1.218:8100
Auth: Bearer token
```

Use the same token value that is set on the Wan2GP server as `WAN2GP_TOKEN`.

Do not include the token in source control or public dashboard config.

## Dashboard Config

Suggested environment variables for the dashboard/client:

```powershell
$env:WAN2GP_API_BASE_URL="http://192.168.1.218:8100"
$env:WAN2GP_API_TOKEN="YOUR_WAN2GP_TOKEN"
```

Equivalent `.env` style:

```env
WAN2GP_API_BASE_URL=http://192.168.1.218:8100
WAN2GP_API_TOKEN=YOUR_WAN2GP_TOKEN
```

## Required Headers

All non-health API calls need:

```http
Authorization: Bearer YOUR_WAN2GP_TOKEN
Content-Type: application/json
```

Health check does not require auth.

## Quick Tests

From the dashboard machine:

```bash
curl http://192.168.1.218:8100/api/health
```

Expected: `status` should be `ok`, and GPU should show the RTX 3090.

```bash
curl -H "Authorization: Bearer YOUR_WAN2GP_TOKEN" http://192.168.1.218:8100/api/models
```

Expected: model list JSON.

Without the bearer token, `/api/models` should return `401 Unauthorized`.

## JavaScript Fetch Example

```js
const apiBaseUrl = process.env.WAN2GP_API_BASE_URL ?? "http://192.168.1.218:8100";
const token = process.env.WAN2GP_API_TOKEN;

async function wan2gpFetch(path, options = {}) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    throw new Error(`Wan2GP API ${response.status}: ${await response.text()}`);
  }

  return response.json();
}

const models = await wan2gpFetch("/api/models");
```

## Creating A Job

Use `POST /api/jobs`. Minimum request needs a `model_type`; other fields depend on the selected model.

```bash
curl -X POST http://192.168.1.218:8100/api/jobs ^
  -H "Authorization: Bearer YOUR_WAN2GP_TOKEN" ^
  -H "Content-Type: application/json" ^
  -d "{\"model_type\":\"z_image\",\"prompt\":\"a cinematic landscape\",\"resolution\":\"1024x1024\",\"num_inference_steps\":8}"
```

Poll job status:

```bash
curl -H "Authorization: Bearer YOUR_WAN2GP_TOKEN" http://192.168.1.218:8100/api/jobs/JOB_ID
```

Stream progress events:

```text
GET /api/jobs/JOB_ID/events
```

## Common Endpoints

```text
GET    /api/health              no auth required
GET    /api/models              list available model types
GET    /api/settings            default settings template
GET    /api/loras?model_type=X  list LoRAs for a model type
POST   /api/jobs                create async generation job
GET    /api/jobs                list recent jobs
GET    /api/jobs/JOB_ID         get job status/result
DELETE /api/jobs/JOB_ID         cancel queued/running job
GET    /api/jobs/JOB_ID/events  server-sent events progress stream
GET    /api/file?path=PATH      download output file under outputs root
POST   /api/release             release loaded model/VRAM
```

## Browser Dashboard CORS

If the dashboard runs in a browser and calls the Wan2GP API directly, set CORS on the Wan2GP server to the dashboard origin, then restart the API server.

Example:

```powershell
[Environment]::SetEnvironmentVariable(
  "WAN2GP_CORS_ORIGINS",
  "http://DASHBOARD_MACHINE_IP:PORT",
  "User"
)
```

For a fully trusted LAN test only, you can use `*`:

```powershell
[Environment]::SetEnvironmentVariable("WAN2GP_CORS_ORIGINS", "*", "User")
```

If the dashboard backend calls Wan2GP server-side, CORS is not needed.

