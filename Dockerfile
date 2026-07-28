FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]