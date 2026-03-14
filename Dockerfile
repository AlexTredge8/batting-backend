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

# Validate the FULL import chain at build time: if any lib or module is broken
# the Build fails here with a clear error rather than a silent deploy failure.
# This also confirms /health is actually registered on the app object.
RUN python -c "
from api import app
routes = [r.path for r in app.routes]
print('Registered routes:', routes)
assert '/health' in routes, '/health route missing from app!'
print('Build validation OK')
"

# Use exec-form with explicit sh so \${PORT:-8000} is always shell-expanded.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port \${PORT:-8000}"]
