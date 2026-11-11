# Deploying automix (free, cloud)

The app is a single Flask service that renders a continuous DJ set and streams it as
it renders. It is packaged as a container (`Dockerfile`) that runs on any CPU box. The
recommended free home is a **Hugging Face Spaces "Docker" Space** (2 vCPU / 16 GB RAM /
50 GB disk on the free tier — enough RAM to run demucs on CPU).

## What's in the box

| Concern | How it's handled |
|---|---|
| **Bounded work** | Renders run on a fixed pool (`MAX_CONCURRENT_RENDERS`, default 1: demucs is heavy) behind a bounded line (`MAX_QUEUE_LENGTH`). When both are full, `POST /api/set/start` answers **503** with `Retry-After`. Status responses carry `queue_position`. |
| **Cancellation** | `DELETE /api/set/<sid>` (or `POST /api/set/<sid>/cancel` for `navigator.sendBeacon`). The player calls it on **Stop** and on page unload. A running render unwinds from inside its next callback, frees its slot and deletes its chunk files. Renders whose browser stops polling for `ABANDON_AFTER_SEC` are cancelled too. |
| **Uploads** | `POST /api/upload` (multipart, field `file`, one or more files). Content is sniffed with `ffprobe` against an allow-list of audio containers (the extension is ignored), stored under random server-side names, decoded to WAV at `SAMPLE_RATE` and capped at `PER_TRACK_CAP_SEC`. Per-file cap `MAX_UPLOAD_MB`, per-request/per-set cap `MAX_PLAYLIST_TRACKS`. Unused uploads expire after `UPLOAD_TTL_SEC`. |
| **Memory** | Tracks are decoded lazily, one at a time; the full set is **never** held in RAM — chunks are flushed to disk as they render. Idle sessions and their chunk files are reaped on a TTL. |
| **Disk** | `CACHE_DIR` is capped at `MAX_CACHE_GB`; downloaded tracks are evicted oldest-first (never one a render is using). Stale chunks, uploads and partial downloads are deleted at startup. Uploads answer **507** if the cache is still full. |
| **Robustness** | `MAX_CONTENT_LENGTH`; strict JSON validation with specific 400 messages; every subprocess has a timeout; a track that fails to download/decode or is shorter than `MIN_TRACK_SEC` is **skipped** and listed in the status `skipped: [{source, reason}]`; a set with no usable tracks fails with a clear message. |
| **Security** | Only `youtube.com`/`youtu.be` URLs (SSRF guard); video ids whitelisted to the canonical 11-char form (path-traversal guard); ffmpeg/ffprobe restricted to the `file` protocol; error text sent to clients never contains exception strings or paths; nonce-based CSP (no `unsafe-inline`), `frame-ancestors`, `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy` on every response; per-IP rate limits on start and upload. |
| **Observability** | `logging` with the session id (`rid=`) on every line (`LOG_JSON=true` for JSON lines), `X-Request-ID` response header, render timing (`queue_wait_sec`, `first_chunk_sec`, `wall_sec`, `realtime_x`) per set; `/healthz` (liveness) and `/readyz` (ffmpeg/ffprobe present, cache writable, queue not saturated — 503 otherwise). |
| **Cost** | Demucs is optional (slim image without torch). Demo audio is synthesized, so the demo needs no network and no copyrighted assets. |

### Why exactly one gunicorn worker

Sessions, the render queue, the upload registry and the chunk index are **in-process
memory**. A second worker would have its own copies: a status or chunk request routed
to it would 404, and the render limit would silently double. `gunicorn.conf.py` therefore
hard-codes `workers = 1` with a `gthread` pool (`GUNICORN_THREADS`) for the cheap
status/chunk/upload requests; renders run on the separate render pool. Scale a single
box with `MAX_CONCURRENT_RENDERS` (on a GPU) and threads; scale out with more containers
behind a **sticky** load balancer. `max_requests = 0` keeps the worker from being recycled
(which would drop every session). On `SIGTERM` gunicorn stops accepting, waits up to
`GUNICORN_GRACEFUL_TIMEOUT` for in-flight requests, then cancels all renders and deletes
their chunks.

## Option A — Hugging Face Spaces, free CPU (recommended)

Runs the whole app — **including demucs on CPU** — on the free tier (2 vCPU / 16 GB).
demucs is slow on CPU (~10–30 s per transition); it is on by default when installed —
untick **stems** in the player (or send `use_stems: false`) for fast EQ-only transitions.

> ⚠️ **Do not push the bundled audio to a public Space.** `mixes/` (~1.3 GB) and
> `data/old_mixes/` (~445 MB) are tracked `.wav`s of copyrighted / copyright-derived
> audio and are **not needed at runtime** (the sample is the synthesized loops, generated
> at build). The recipe below sends a clean single-commit tree without them or the heavy
> git history.

1. **Account + token.** Create a free account, then **Settings → Access Tokens → New
   token** (role: *write*). Copy it.
2. **Create the Space:** huggingface.co/new-space → name `automix`, **SDK: Docker**,
   **Hardware: CPU basic (free)**. It reads the `README.md` front-matter
   (`sdk: docker`, `app_port: 7860`) and builds the `Dockerfile`.
3. **Push a clean tree** (from this worktree, after committing your changes):
   ```bash
   # one orphan commit = only current files; no heavy history, no copyrighted audio
   git checkout --orphan space-deploy
   git add -A
   git rm -r --cached --quiet mixes data/old_mixes data/demo   # drop bundled audio (files stay on disk)
   git commit -m "automix: HF Space deploy build"
   git remote add space https://<user>:<HF_WRITE_TOKEN>@huggingface.co/spaces/<user>/automix
   git push -f space space-deploy:main
   git checkout fix/debug-instrumentation-and-bugs   # back to your working branch
   ```
   (`data/demo` is dropped too — the demo loops are regenerated at build by the
   Dockerfile's `python src/demo_samples.py` step, so the Space recreates them itself.)
4. First build takes a few minutes (CPU torch + demucs; the htdemucs weights are
   downloaded into the image at build time so the first stems request isn't slow).
   When it's up, open the Space — **Sample** and **uploads** work immediately, no
   network or keys needed.
5. In **Settings → Variables**, set `TRUST_PROXY_HOPS=1` (Spaces sits behind a proxy;
   without it every visitor shares one rate-limit bucket) and
   `FRAME_ANCESTORS=https://huggingface.co` (the Space page embeds the app in an iframe;
   the default `none` only allows opening the `*.hf.space` URL directly).

### Make YouTube work on a datacenter IP (optional)

YouTube bot-throttles datacenter IPs (all cloud hosts), so server-side downloads often
fail with *"Sign in to confirm you're not a bot."* To mitigate, supply your own cookies:

1. Export cookies for `youtube.com` in Netscape format (e.g. the *Get cookies.txt*
   browser extension).
2. In the Space: **Settings → Variables and secrets → New secret**
   - Name: `YT_COOKIES`
   - Value: paste the entire `cookies.txt` contents
3. Restart. The entrypoint writes the secret to a file and points yt-dlp at it.

If you'd rather disable YouTube entirely (samples only), set a **variable**
`ALLOW_YOUTUBE=false`.

## Option B — any Docker host (Fly.io, Render, a VPS)

```bash
# Full image (includes demucs / CPU torch):
docker build -t automix .
# Or a small, torch-free image (EQ transitions only, ~5x smaller):
docker build -t automix-slim --build-arg INSTALL_STEMS=false .

docker run -p 7860:7860 \
  -e ALLOW_YOUTUBE=true \
  -e YT_COOKIES="$(cat cookies.txt)" \
  automix
# open http://localhost:7860
```

## Configuration (env vars)

All are optional; see `.env.example`.

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `7860` | Listen port (HF Spaces expects 7860). |
| `SAMPLE_RATE` | `44100` | Render rate; uploads/downloads are decoded to it. |
| `CACHE_DIR` | `data/cache/stream` | Chunks, downloaded tracks and uploads. |
| `ALLOW_YOUTUBE` | `true` | `false` disables YouTube ingestion (and the URL field). |
| `ALLOW_UPLOADS` | `true` | `false` disables `/api/upload` (and the drop zone). |
| `MAX_PLAYLIST_TRACKS` | `12` | Tracks per set; also files per upload request. |
| `PER_TRACK_CAP_SEC` | `300` (image: `150`) | Seconds kept per track (download and upload). |
| `MIN_TRACK_SEC` | `30` | Shorter tracks are skipped. |
| `MAX_DOWNLOAD_MB` | `60` | yt-dlp `--max-filesize`. |
| `MAX_UPLOAD_MB` | `60` | Per uploaded file. `MAX_CONTENT_LENGTH` = this × `MAX_PLAYLIST_TRACKS` + 1 MB. |
| `UPLOAD_TTL_SEC` | `3600` | Delete uploads not used for this long. |
| `MAX_CONCURRENT_RENDERS` | `1` | Renders at once. Keep 1 on CPU; raise on a GPU. |
| `MAX_QUEUE_LENGTH` | `4` | Waiting renders; beyond this `/api/set/start` returns 503. |
| `ABANDON_AFTER_SEC` | `120` | Cancel a queued/running render nobody has polled for this long. |
| `SESSION_TTL_SEC` | `1800` | Evict finished/idle sessions (and chunks) after this long. |
| `MAX_SESSIONS` | `24` | Session cap (oldest evicted). |
| `MAX_CACHE_GB` | `5` | `CACHE_DIR` cap; oldest downloaded tracks evicted first. |
| `DOWNLOAD_TIMEOUT_SEC` | `120` | yt-dlp timeout (playlist resolve and each download). |
| `FFMPEG_TIMEOUT_SEC` | `120` | Timeout for every ffmpeg/ffprobe call. |
| `RATE_LIMIT` | `30 per hour` | Per-IP limit on `/api/set/start`. |
| `UPLOAD_RATE_LIMIT` | `120 per hour` | Per-IP limit on `/api/upload`. |
| `TRUST_PROXY_HOPS` | `0` | Proxies in front (HF Spaces/Fly/nginx: `1`) so limits see client IPs. |
| `FRAME_ANCESTORS` | `'none'` | CSP `frame-ancestors`; `https://huggingface.co` for the Spaces embed. |
| `LOG_LEVEL` | `INFO` | App and gunicorn log level. |
| `LOG_JSON` | `false` | JSON log lines (`ts`, `level`, `logger`, `rid`, `msg`). |
| `YT_COOKIES` | — | (secret) cookies.txt contents for yt-dlp. |
| `YT_COOKIES_FILE` | — | Path to a cookies.txt (set automatically from `YT_COOKIES`). |
| `GUNICORN_THREADS` | `8` | Request threads in the single worker. |
| `GUNICORN_TIMEOUT` | `180` | Per-request timeout (uploads; renders are not requests). |
| `GUNICORN_GRACEFUL_TIMEOUT` | `30` | Drain time on SIGTERM. |
| `TORCH_HOME` | `/opt/torch` (image) | Where the baked-in htdemucs weights live. |

> **Single worker only** — see *Why exactly one gunicorn worker* above. The worker
> count is fixed in `gunicorn.conf.py`, not configurable.

## Operations

- **Health:** the image has a Docker `HEALTHCHECK` on `/healthz` (liveness). Point
  readiness probes at `/readyz`: it returns 503 with `checks` when ffmpeg/ffprobe is
  missing, the cache is not writable, or the render queue is full.
- **Logs:** one line per event, tagged with `rid=<session id>`; the gunicorn access log
  carries the same id. Per set you get `set queued`, `render start` (with
  `queue_wait_sec`), `first chunk after Ns`, `render done … wall_sec realtime_x`, or
  `render cancelled` / `render error`. Startup logs missing binaries and how many stale
  cache files it deleted.
- **API errors** are JSON `{"error": "..."}` with a human-readable message: 400
  (validation), 403 (source disabled), 404 (unknown/expired set), 413 (upload too
  large), 415 (not audio), 429 (rate limit), 503 (queue full), 507 (cache full).
- **Tests:** `pip install -r requirements-dev.txt && python -m pytest tests/test_server.py -q`.

## Running on a GPU (web)

A GPU only accelerates **demucs** (the **stems** toggle). The rest of the mix
is CPU DSP. demucs auto-selects CUDA when a GPU is visible — no code change, just a
CUDA torch build (`--build-arg TORCH_VARIANT=cu121`). Pick one of three paths:

### Path 1 — persistent GPU box (least rewrite, paid)
Keep this exact Flask app; run it on GPU hardware.

- **Hugging Face Spaces (Docker) + GPU hardware.** Deploy as in Option A, then
  **Settings → Hardware → pick a GPU** (e.g. *Nvidia T4 small*, ~US$0.40/hr; billed
  while running — set a **sleep timer** so an idle Space stops). Rebuild the Space with
  build arg `TORCH_VARIANT=cu121` (Settings → Variables: add `TORCH_VARIANT=cu121`).
- **Fly.io GPU / RunPod / Lambda / a GPU VPS.** `docker build --build-arg TORCH_VARIANT=cu121 -t automix . && docker run --gpus all -p 7860:7860 automix`.

There is **no free persistent GPU** — a server holding a GPU 24/7 always costs money.
Mitigate with idle-sleep so you only pay while it's actually mixing.

### Path 2 — free GPU via HF ZeroGPU (needs a Gradio-SDK Space)
ZeroGPU (free, dynamically-allocated A100) is only available to **Gradio/Streamlit SDK**
Spaces, **not** Docker Spaces. To use it while keeping this custom streaming UI:

1. Create a **Gradio SDK** Space.
2. In `app.py`, build a Gradio app, then **mount this Flask/streaming API onto it**
   (`gradio.mount_gradio_app` / a FastAPI sub-app) so the chunk endpoints + cassette UI
   still serve.
3. Move the demucs call into a function decorated with `@spaces.GPU` (import `spaces`);
   ZeroGPU attaches a GPU only for that call. demucs-per-transition is short, so this
   fits the ZeroGPU model well and burns little quota.

Free, but GPU access is **quota-limited** and queued under load. This is a real port
(a day's work), not a flag flip.

### Path 3 — keep app on free CPU, offload only demucs to serverless GPU
Best cost/scale: the Flask app stays on a free CPU box; only stem separation is sent to
a pay-per-second GPU that scales to zero.

- **Modal** (~US$30/mo free credits) or **Replicate / RunPod serverless**.
- Wrap `StemSeparator.separate_stems` (`src/stem_separator.py`) so it sends the short
  transition segment to a remote GPU function running htdemucs and returns the stems.
- At low volume this is effectively free and there's no idle GPU cost.

### Which should I pick?
- **Just want it live, cheaply, and HQ stems are optional** → free CPU Space (Option A).
  CPU demucs works, just slow; or leave HQ off (EQ transitions sound great).
- **Want fast stems and will pay a little** → **Path 1** (HF GPU hardware + `cu121`, with
  idle-sleep). Least effort.
- **Want fast stems for free** → **Path 2** (ZeroGPU; accept the Gradio port + quota).
- **Want scale-to-zero, pay-per-use** → **Path 3** (Modal/Replicate offload).
