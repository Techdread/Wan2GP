# Modly Mesh Enhancement API

This service exposes MeshFlow as a mesh enhancement model in the Modly API.
It takes an existing mesh, optionally uses a reference image for visual guidance,
and returns an enhanced `.glb` mesh in the workspace.

## Model

- Model ID: `meshflow/enhance`
- Display name: `MeshFlow - Enhance Mesh`
- Capability: mesh enhancement / mesh-to-mesh generation
- Input: mesh file
- Optional input: reference image
- Output: `.glb` mesh

Supported source mesh formats:

- `.glb`
- `.gltf`
- `.obj`
- `.ply`
- `.stl`

## Discovery

List all available models:

```bash
curl http://localhost:8765/api/model/all
```

Look for:

```json
{
  "id": "meshflow/enhance",
  "name": "MeshFlow - Enhance Mesh",
  "capability": "mesh-enhancement",
  "input": "mesh",
  "inputs": ["mesh", "image"],
  "output": "mesh"
}
```

Fetch the MeshFlow parameter schema:

```bash
curl "http://localhost:8765/api/model/params?model_id=meshflow/enhance"
```

## Submit A Mesh Enhancement Job

Endpoint:

```http
POST /api/generate/from-mesh
```

Multipart fields:

| Field | Required | Description |
| --- | --- | --- |
| `mesh` | yes | Source mesh file. |
| `reference_image` | no | Optional image used for visual guidance. |
| `model_id` | no | Defaults to `meshflow/enhance`. |
| `collection` | no | Workspace collection folder. Defaults to `Default`. |
| `params` | no | JSON string of MeshFlow settings. |

Common `params`:

```json
{
  "steps": 28,
  "guidance_scale": 2.5,
  "num_verts": 4096,
  "seed": 42,
  "dtype": "fp16",
  "compile_models": "auto"
}
```

Submit with `curl`:

```bash
curl -X POST http://localhost:8765/api/generate/from-mesh \
  -F "mesh=@chair.obj" \
  -F "reference_image=@chair_reference.png" \
  -F "model_id=meshflow/enhance" \
  -F "collection=ClientMeshes" \
  -F 'params={"steps":28,"guidance_scale":2.5,"num_verts":4096,"seed":42}'
```

Response:

```json
{
  "job_id": "6c5f604c-bc19-4c44-bf0b-f43acbc5d859"
}
```

## Poll Job Status

```bash
curl http://localhost:8765/api/generate/status/{job_id}
```

Example completed response:

```json
{
  "job_id": "6c5f604c-bc19-4c44-bf0b-f43acbc5d859",
  "status": "done",
  "progress": 100,
  "step": "MeshFlow complete",
  "output_url": "/workspace/ClientMeshes/1781481200_ab12cd34_meshflow.glb"
}
```

Download the result:

```bash
curl -o enhanced.glb "http://localhost:8765/workspace/ClientMeshes/1781481200_ab12cd34_meshflow.glb"
```

## JavaScript Example

```js
async function enhanceMesh({ apiBase, meshFile, referenceImageFile, token }) {
  const form = new FormData();
  form.append("mesh", meshFile);
  if (referenceImageFile) form.append("reference_image", referenceImageFile);
  form.append("model_id", "meshflow/enhance");
  form.append("collection", "ClientMeshes");
  form.append("params", JSON.stringify({
    steps: 28,
    guidance_scale: 2.5,
    num_verts: 4096,
    seed: 42,
    dtype: "fp16",
    compile_models: "auto"
  }));

  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const submit = await fetch(`${apiBase}/api/generate/from-mesh`, {
    method: "POST",
    headers,
    body: form
  });
  if (!submit.ok) throw new Error(await submit.text());
  const { job_id } = await submit.json();

  while (true) {
    const statusRes = await fetch(`${apiBase}/api/generate/status/${job_id}`, { headers });
    const job = await statusRes.json();
    if (job.status === "done") return job;
    if (job.status === "error" || job.status === "cancelled") {
      throw new Error(job.error || `Job ${job.status}`);
    }
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
}
```

## Python Example

```python
import json
import time
import requests


def enhance_mesh(api_base, mesh_path, reference_image_path=None, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    files = {"mesh": open(mesh_path, "rb")}
    if reference_image_path:
        files["reference_image"] = open(reference_image_path, "rb")
    data = {
        "model_id": "meshflow/enhance",
        "collection": "ClientMeshes",
        "params": json.dumps({
            "steps": 28,
            "guidance_scale": 2.5,
            "num_verts": 4096,
            "seed": 42,
            "dtype": "fp16",
            "compile_models": "auto",
        }),
    }

    response = requests.post(
        f"{api_base}/api/generate/from-mesh",
        headers=headers,
        files=files,
        data=data,
        timeout=60,
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]

    while True:
        job = requests.get(
            f"{api_base}/api/generate/status/{job_id}",
            headers=headers,
            timeout=30,
        ).json()
        if job["status"] == "done":
            return job
        if job["status"] in {"error", "cancelled"}:
            raise RuntimeError(job.get("error") or job["status"])
        time.sleep(1.5)
```

## Notes For Clients

- The first run may take longer because MeshFlow loads a large checkpoint and may compile models.
- `reference_image` is only used when explicitly uploaded on this endpoint.
- `num_verts` is an output vertex budget, not a decimation target.
- The service returns workspace URLs. Prefix them with the API origin to download.
- If `MODLY_API_TOKEN` is set on the server, include `Authorization: Bearer <token>` on every non-health request.
