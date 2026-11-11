# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# automix — Club AutoMixer.
#
#   CPU  (free CPU box / HF Spaces Docker, 16 GB):   docker build .
#   GPU  (paid GPU host / HF Spaces GPU hardware):   docker build --build-arg TORCH_VARIANT=cu121 .
#   slim (no stems, no torch, ~5x smaller image):    docker build --build-arg INSTALL_STEMS=false .
#
# Stems = demucs = torch. GPU only accelerates the demucs step (transitions);
# demucs auto-selects CUDA when a GPU is visible (src/utils.get_best_device).
# The torch CUDA wheel bundles its own CUDA libs, so a plain python base works as
# long as the host exposes an NVIDIA driver (HF GPU hardware / nvidia-container-runtime).
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# System deps: ffmpeg + ffprobe (upload sniffing/decoding, yt-dlp transcode), libsndfile (soundfile), git (demucs),
# libgomp1 (OpenMP runtime that torch's CPU build needs to import).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps (cached layer: only re-runs when requirements change) ------
COPY requirements.txt requirements-stems.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    # librosa imports pkg_resources at runtime; setuptools >=81 removed it.
    && pip install --no-cache-dir "setuptools<81"

# TORCH_VARIANT: "cpu" (default, small, CPU-only) or "cu121" (CUDA 12.1 GPU build).
ARG INSTALL_STEMS=true
ARG TORCH_VARIANT=cpu
RUN if [ "$INSTALL_STEMS" = "true" ]; then \
        pip install --no-cache-dir -r requirements-stems.txt \
            --extra-index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" && \
        pip install --no-cache-dir --no-deps \
            "git+https://github.com/adefossez/demucs.git" && \
        # a stems dep must not silently bump numpy to 2.x (breaks the numba/librosa ABI)
        pip install --no-cache-dir "numpy==1.26.4" ; \
    fi

# Bake the htdemucs weights (~80 MB, 955717e8-8726e21a.th) into the image so the
# first stems request doesn't stall on a download. TORCH_HOME is a fixed,
# world-readable path used at build AND run time (the runtime user is not root).
ENV TORCH_HOME=/opt/torch
RUN if [ "$INSTALL_STEMS" = "true" ]; then \
        python -c "from demucs.pretrained import get_model; get_model('htdemucs')" && \
        chmod -R a+rX /opt/torch ; \
    fi

# --- App code + non-root runtime user (Hugging Face Spaces runs as uid 1000) -
RUN useradd -m -u 1000 user
COPY --chown=user:user . /app
# /app itself was created by WORKDIR as root; COPY --chown only owns the copied
# files, not the dir. Give the whole tree to the runtime user so the build-time
# demo step AND runtime writes (CACHE_DIR=/app/data/cache, chunk files) succeed.
RUN mkdir -p /app/data && chown -R user:user /app
USER user
# Hosted demo: keep per-track audio short so a free CPU box stays within RAM/CPU.
ENV PER_TRACK_CAP_SEC=150
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1

# Pre-generate the royalty-free demo loops into the image (idempotent at runtime).
# Run as a standalone script (NOT `python -m src...`) so it needs only numpy+soundfile
# and never imports the heavy src package (torch/numba/demucs) during the build.
RUN python src/demo_samples.py data/demo

EXPOSE 7860
# Liveness probe (slim image has no curl). /readyz is the readiness endpoint for
# orchestrators that distinguish the two; it turns 503 while the render queue is full.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','7860'), timeout=4)" || exit 1
# gunicorn gets SIGTERM; graceful_timeout (gunicorn.conf.py) must fit inside this.
STOPSIGNAL SIGTERM
ENTRYPOINT ["./docker-entrypoint.sh"]
