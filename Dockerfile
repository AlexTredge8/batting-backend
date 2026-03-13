FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    # X11 / XCB — required by opencv even in headless mode
    libxcb1 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-render0 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libxi6 \
    libxtst6 \
    # OpenGL — mediapipe links against these even on CPU-only builds
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    # GLib / threading — mediapipe + opencv runtime
    libglib2.0-0 \
    libgomp1 \
    # Video processing
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p results

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
