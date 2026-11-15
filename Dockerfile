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

# Optional RTMPose pose fallback (docs/phase5/rtmpose_fallback.md). Off unless the build arg
# INSTALL_RTMPOSE=1 is set (on Railway: a service variable of that name). Installs rtmlib +
# onnxruntime and bakes the two model files (~150 MB) into the image so there is no download at
# request time. The fallback itself still needs POSE_FALLBACK=rtmpose at run time.
ARG INSTALL_RTMPOSE=0
RUN if [ "$INSTALL_RTMPOSE" = "1" ]; then \
        pip install --no-cache-dir --no-deps rtmlib==0.0.16 && \
        pip install --no-cache-dir onnxruntime==1.23.2 tqdm && \
        python -c "from rtmlib import BodyWithFeet; BodyWithFeet(mode='balanced', backend='onnxruntime', device='cpu')"; \
    fi

COPY . .

RUN mkdir -p results

# Validate full import chain at build time — Docker parses multiline RUN as
# instructions so keep this as a single line.
RUN python -c "from api import app; routes=[r.path for r in app.routes]; print('routes:', routes); assert '/health' in routes, 'MISSING /health'"

CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
