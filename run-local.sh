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

# Force one consistent headless CPU renderer. Inheriting MUJOCO_GL=egl while
# setting PyOpenGL to OSMesa makes MuJoCo fail during import.
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PYTHON_EXECUTABLE="${MUJOCOWEB_PYTHON:-/home/paul/miniconda3/envs/dapg38/bin/python}"
exec "$PYTHON_EXECUTABLE" -m uvicorn server:app \
    --host 127.0.0.1 \
    --port "${PORT:-8000}"
