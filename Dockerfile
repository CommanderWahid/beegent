# syntax=docker/dockerfile:1
#
# One image, one process. `api/runner.py` holds the run lock and the live log buffer in memory,
# so a second worker or replica would get its own lock and answer for runs it never performed.
# Do not add --workers, and do not scale this beyond one container.

# --- stage 1: build the Angular UI -------------------------------------------------------------
# Node 22 matches CI (.github/workflows/ci.yml).
FROM node:22-slim AS ui

WORKDIR /src/ui
# Dependencies first, so a source-only change does not reinstall them.
COPY ui/package.json ui/package-lock.json ./
RUN npm ci

# angular.json pulls beegent_logo.svg from ../assets, so the build needs the repo root layout.
COPY assets/ /src/assets/
COPY ui/ /src/ui/
RUN npx ng build

# --- stage 2: the application ------------------------------------------------------------------
FROM python:3.12-slim

# Unbuffered, or the streamed run log arrives in `docker logs` in lumps.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY beegent/ ./beegent/
COPY api/ ./api/

# Installed EDITABLE on purpose. api/main.py locates the UI relative to its own file
# (`Path(__file__).parent.parent / "ui" / "dist" / ...`), so the package has to stay at /app
# rather than being copied into site-packages. This also gives us the beegent-ui entry point.
RUN pip install -e ".[api,embeddings]"

COPY --from=ui /src/ui/dist/ui/browser ./ui/dist/ui/browser

# Everything the app writes lives under /data: the SQLite store, the preview cache, the model
# cache, and candidate_list.json - which run.py writes to the working directory.
RUN useradd --create-home --uid 10001 beegent \
    && mkdir -p /data \
    && chown -R beegent:beegent /data /app
USER beegent
WORKDIR /data

ENV BEEGENT_HOST=0.0.0.0 \
    BEEGENT_PORT=8000 \
    BEEGENT_DB=/data/memory.db \
    PREVIEW_DIR=/data/previews \
    HF_HOME=/data/.cache

EXPOSE 8000

# /api/config resolves the backend and the per-role models without one network or LLM call.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/config', timeout=3)"

# main() runs connector.validate() before serving, so a missing key or an unreachable endpoint
# exits here rather than failing later mid-run. In Docker that reads as a restart loop; it is
# the backend configuration that is wrong, not the image.
CMD ["beegent-ui"]
