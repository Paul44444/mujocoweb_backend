#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BACKEND_DIRECTORY"

if [[ -f .env ]]; then
    set -a
    # This file is local and owner-controlled; it is never committed.
    source .env
    set +a
fi

# Use the NVIDIA GPU locally by default and keep MuJoCo/PyOpenGL consistent.
# Start with `MUJOCO_GL=osmesa ./run-local.sh` for a CPU-only fallback.
RENDERING_PLATFORM="${MUJOCO_GL:-egl}"
if [[ "$RENDERING_PLATFORM" != "egl" && "$RENDERING_PLATFORM" != "osmesa" ]]; then
    printf 'Unsupported MUJOCO_GL=%s (use egl or osmesa)\n' "$RENDERING_PLATFORM" >&2
    exit 2
fi
export MUJOCO_GL="$RENDERING_PLATFORM"
export PYOPENGL_PLATFORM="$RENDERING_PLATFORM"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export SIMULATION_WIDTH="${SIMULATION_WIDTH:-960}"
export SIMULATION_HEIGHT="${SIMULATION_HEIGHT:-720}"
export JPEG_QUALITY="${JPEG_QUALITY:-85}"
export STREAM_FPS="${STREAM_FPS:-30}"

PYTHON_EXECUTABLE="${MUJOCOWEB_PYTHON:-/home/paul/miniconda3/envs/dapg38/bin/python}"
exec "$PYTHON_EXECUTABLE" -m uvicorn server:app \
    --host 127.0.0.1 \
    --port "${PORT:-8000}"
