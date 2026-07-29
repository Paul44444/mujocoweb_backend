FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

# Force headless CPU-based OpenGL rendering
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Keep native math libraries from creating a large thread pool on a
# memory-constrained Render instance.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libosmesa6 \
    libosmesa6-dev \
    libegl1 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# RoboHive declares an unqualified "torch" dependency. Install the CPU-only
# wheel first so pip does not pull a much larger CUDA build.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
