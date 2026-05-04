# Brief: Revise `Wan2GP-CLIENT-SETUP.md`

Hand-off for a fresh Claude agent. Self-contained — you do not need any prior conversation context to act on this.

---

## Task

Revise `docs/ai-platform/Wan2GP-CLIENT-SETUP.md` based on the review below.

## Context you need before editing

- This is the client-side companion to the Wan2GP Agent API server running on the 3060 host at `http://192.168.1.199:8100`.
- Server source: `/media/peter/AI/Wan2GP/agent_api.py` (on the 3060 host — ssh if needed; if you cannot reach it, stop and ask).
- Sibling docs you must read first:
  - `docs/ai-platform/Wan2GP-NETWORK-API.md` — original API description
  - `docs/ai-platform/Wan2GP-API-HARDENING.md` — the spec the server was hardened against
  - `docs/ai-platform/01-ai-media-capability-architecture.md` — the platform layer this server feeds into

---

## Verify before editing

Verify these claims in the current doc against the actual `agent_api.py`. If any are wrong, fix the doc to match reality, not the other way around.

1. The Python wrapper exposes `submit_job`, `wait_for_job`, `cancel_job`, `download_file`, and reads `WAN2GP_TOKEN` from the environment as a fallback. Grep `agent_api.py` for these method names. If they don't exist, either change the doc to use HTTP calls only, or note that the helpers need to be added.
2. SSE on `/api/jobs/:id/events` actually emits `event: status` lines with the full job record and 15-second heartbeat comments. If the server emits a different shape, document what it actually emits.
3. `/api/batch` is genuinely deprecated. If it isn't, remove the deprecation marker from the cheat sheet. If it is, also update `Wan2GP-API-HARDENING.md` to reflect that decision (the hardening spec only deprecated `/api/generate`).
4. The minimum required body for `POST /api/jobs` for an image. The current section 4 example omits `activated_loras`, `loras_multipliers`, `image_mode`, `seed`, `negative_prompt`, etc. that the legacy `Wan2GP-NETWORK-API.md` shows. Either the new endpoint defaults these (say so explicitly) or the example needs them or it will 422.

---

## Required additions

1. **Add a JavaScript / `fetch` section** mirroring section 4 (submit → poll → download). The hub apps are HTML/JS — Python is not the integration target. Show how to turn the downloaded blob into an object URL for `<img>` / `<video>`.

2. **Add a CORS section.** Browser code at any other origin will be blocked unless `agent_api.py` sends `Access-Control-Allow-Origin`. Check whether the server currently sets CORS headers.
   - If yes, document the configuration.
   - If no, document that browser apps must call through a same-origin proxy and add a follow-up note to `Wan2GP-API-HARDENING.md` to add CORS support (configurable allow-list via `WAN2GP_CORS_ORIGINS` env var, default empty).

3. **Add a browser-token warning.** Putting `Authorization: Bearer <token>` in browser-side fetches exposes the token in page source / devtools / extensions. Recommend: Node / Electron / server-side direct calls only; browser apps route through the platform capability layer (which will be a same-origin proxy).

4. **Add a one-paragraph preamble at the very top:**

   > Most apps in the hub should not call this server directly — they should go through the shared capability router (see `01-ai-media-capability-architecture.md`). This doc is for tools that need raw access, for debugging, or for writing the platform adapter itself.

---

## Concrete fixes to existing content

5. Section 4 curl example, line ~83: URL-encode the `path` query parameter. Current code breaks on filenames with spaces or `&`. Use `curl -G --data-urlencode "path=$FILE"` or pipe through `python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.stdin.read().strip()))'`.

6. Section 1 token retrieval: `grep WAN2GP_TOKEN ...` returns `Environment=WAN2GP_TOKEN=xxx`. Reader has to strip the prefix. Replace with either `systemctl --user show wan2gp-api.service -p Environment` or pipe through `cut -d= -f3-`.

7. Section 4 poll loop has no max-wait or backoff — a stuck job loops forever. Add a max-wait (suggest 10 minutes for video, 60 s for image) and a log line every 30 s of waiting.

8. Document the failed-job shape. What's in `final["error"]` — string, or `{code, message}`? Read `agent_api.py` to find out, then show one example.

9. Show one example of `GET /api/jobs?limit=N` — it's in the cheat sheet but never demonstrated. One curl + sample response is enough.

---

## Style notes

- Match the existing doc's voice: short sections, code blocks, terse bullets. No emojis.
- Keep the existing structure where it works — this is a revision, not a rewrite.
- Do not add a "Changes in this revision" section.

---

## When done

- Print a short summary of what you verified vs. what you changed vs. what you punted on.
- If you updated `Wan2GP-API-HARDENING.md` (CORS, batch deprecation), note that explicitly.
- Do not commit. Leave the changes staged for the user to review.
