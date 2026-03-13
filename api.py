"""
BattingIQ Phase 2 — FastAPI Wrapper
=====================================
Endpoints:
  GET  /health   — liveness probe
  POST /analyse  — upload video, run full pipeline, return JSON report
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from run_analysis import run_full_analysis

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="BattingIQ API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "BattingIQ API", "version": "2.0.0"}


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    """
    Upload a batting video and receive a full BattingIQ analysis report.

    Returns JSON with: battingiq_score, score_band, pillars, priority_fix,
    development_notes, phases, metadata.
    """
    # Validate file type
    suffix = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: mp4, mov, avi, mkv, m4v",
        )

    job_id = uuid.uuid4().hex
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / f"input{suffix}"

    try:
        # Save uploaded file
        with open(video_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        # Run analysis pipeline
        output_dir = job_dir / "output"
        report = run_full_analysis(str(video_path), output_dir=str(output_dir))

        report["job_id"] = job_id
        return JSONResponse(content=report)

    except Exception as exc:
        # Clean up on failure
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await file.close()
