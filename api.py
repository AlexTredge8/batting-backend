"""
BattingIQ Phase 2 — FastAPI Wrapper
=====================================
Endpoints:
  GET  /health   — liveness probe
  POST /analyse  — upload video, run full pipeline, return JSON report
"""

import json
import os
import shutil
import traceback
import uuid
from pathlib import Path
from urllib import request as urlrequest

import psutil
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

from inline_media import file_to_data_url
from media_storage import download_result_file, result_redirect_url, storage_config, upload_tree
from run_analysis import run_full_analysis

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="BattingIQ API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://battingiq.lovable.app",
        "https://*.lovable.app",
        "https://*.lovableproject.com",
        "http://localhost:3000",
        "http://localhost:8080",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _download_video_url(video_url: str, destination: Path) -> None:
    """Download a remote video into destination."""
    req = urlrequest.Request(
        video_url,
        headers={"User-Agent": "BattingIQ/2.0"},
    )
    with urlrequest.urlopen(req, timeout=60) as response, open(destination, "wb") as fh:
        shutil.copyfileobj(response, fh)


def _build_analysis_response(report: dict, job_id: str, output_dir: Path) -> dict:
    """Convert internal report paths to public API response fields."""
    def _to_url(abs_path):
        if not abs_path:
            return None
        p = Path(abs_path)
        if not p.exists():
            return None
        rel = p.relative_to(RESULTS_DIR)
        return f"/results/{rel}"

    def _to_storyboard_frame(frame: dict) -> dict:
        public = dict(frame)
        frame_path = public.pop("path", None)
        public["url"] = _to_url(frame_path)
        public["data_url"] = file_to_data_url(frame_path)
        return public

    annotated_video_path = report.pop("_annotated_video", None)
    storyboard_path = report.pop("_storyboard", None)
    annotated_video_url = _to_url(annotated_video_path)
    storyboard_url = _to_url(storyboard_path)
    storyboard_frames = report.pop("_storyboard_frames", [])
    public_storyboard_frames = [
        _to_storyboard_frame(frame)
        for frame in storyboard_frames
        if isinstance(frame, dict)
    ]

    try:
        storage_summary = upload_tree(output_dir, RESULTS_DIR)
    except Exception as storage_exc:
        storage_summary = {
            "enabled": True,
            "status": "failed",
            "uploaded_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "error": str(storage_exc),
        }
    report.setdefault("metadata", {})
    report["metadata"]["media_storage"] = storage_summary
    report["storyboard_frames"] = public_storyboard_frames
    report["job_id"] = job_id
    report["annotated_video_url"] = annotated_video_url
    report["storyboard_url"] = storyboard_url
    return report


def _parse_anchor_frames_json(anchor_frames_json: Optional[str]) -> dict[str, int | None] | None:
    if anchor_frames_json in (None, ""):
        return None

    try:
        parsed = json.loads(anchor_frames_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid anchor_frames_json: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="anchor_frames_json must be a JSON object")

    normalized: dict[str, int | None] = {}
    for key, value in parsed.items():
        if value is None or value == "":
            normalized[str(key)] = None
            continue
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Anchor frame '{key}' must be an integer or null",
            ) from exc

    return normalized


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "BattingIQ API", "version": "2.0.0"}


@app.get("/diag")
def diag():
    """Live diagnostic — memory, disk, Python env. Hit this to understand failures."""
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    media_storage = storage_config()
    return {
        "memory": {
            "total_mb": round(vm.total / 1e6),
            "available_mb": round(vm.available / 1e6),
            "used_mb": round(vm.used / 1e6),
            "percent": vm.percent,
        },
        "disk": {
            "total_gb": round(disk.total / 1e9, 1),
            "free_gb": round(disk.free / 1e9, 1),
        },
        "env": {
            "PORT": os.environ.get("PORT", "not set"),
            "results_dir_exists": RESULTS_DIR.exists(),
            "media_storage": {
                "configured": bool(media_storage["enabled"]),
                "bucket": media_storage["bucket"],
                "prefix": media_storage["prefix"],
                "base_url_present": bool(media_storage["base_url"]),
                "service_key_present": bool(media_storage["service_key"]),
            },
        },
    }


@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    angle: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    consent: Optional[str] = Form(None),
    handedness: Optional[str] = Form(None),
    contact_frame: Optional[int] = Form(None),
    anchor_frames_json: Optional[str] = Form(None),
):
    """
    Upload a batting video and receive a full BattingIQ analysis report.

    Args:
        handedness: "right" or "left". Defaults to "right" if not provided.

    Returns JSON with: battingiq_score, score_band, handedness, pillars,
    priority_fix, development_notes, phases, metadata.
    """
    # Validate file type
    suffix = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: mp4, mov, avi, mkv, m4v",
        )

    job_id = uuid.uuid4().hex
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / f"input{suffix}"

    def _mem_mb():
        return round(psutil.virtual_memory().available / 1e6)

    try:
        anchor_frames = _parse_anchor_frames_json(anchor_frames_json)

        # Save uploaded file
        with open(video_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        file_mb = round(video_path.stat().st_size / 1e6, 1)
        # Resolve handedness
        h = (handedness or "").strip().lower()
        h_source = "api" if h in ("right", "left") else "default"
        if h not in ("right", "left"):
            h = None  # let pipeline use default
        print(f"[analyse] job={job_id} file={file_mb}MB suffix={suffix} handedness={h or 'default'} mem_avail={_mem_mb()}MB")

        # Run analysis pipeline
        output_dir = job_dir / "output"
        print(f"[analyse] starting pipeline mem_avail={_mem_mb()}MB")
        report = run_full_analysis(
            str(video_path),
            output_dir=str(output_dir),
            handedness=h,
            handedness_source=h_source,
            contact_frame=contact_frame,
            anchor_frames=anchor_frames,
        )
        print(f"[analyse] pipeline done mem_avail={_mem_mb()}MB")

        annotated_files = sorted(output_dir.glob("*_battingiq_annotated.mp4"))
        annotated_video_url = None
        if annotated_files:
            annotated_video_url = f"/results/{job_id}/output/{annotated_files[0].name}"

        report["job_id"] = job_id
        report["annotated_video_url"] = annotated_video_url
        return JSONResponse(content=report)

    except Exception as exc:
        # Clean up on failure
        shutil.rmtree(job_dir, ignore_errors=True)
        tb = traceback.format_exc()
        print(f"[analyse] FAILED job={job_id} mem_avail={_mem_mb()}MB\n{tb}")
        raise HTTPException(status_code=500, detail=f"{exc}\n\n{tb}")
    finally:
        await file.close()


@app.post("/analyse-from-url")
def analyse_from_url(
    video_url: str = Form(...),
    handedness: Optional[str] = Form(None),
    contact_frame: Optional[int] = Form(None),
    anchor_frames_json: Optional[str] = Form(None),
):
    """
    Re-analyse an existing remote video URL, optionally with a resolved contact frame.

    Intended for the admin/manual-review flow where the original upload already exists
    in storage and the coach wants the report/storyboard regenerated from a corrected
    contact frame.
    """
    suffix = Path(video_url).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}:
        suffix = ".mp4"

    job_id = uuid.uuid4().hex
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / f"input{suffix}"

    h = (handedness or "").strip().lower()
    h_source = "api" if h in ("right", "left") else "default"
    if h not in ("right", "left"):
        h = None

    try:
        anchor_frames = _parse_anchor_frames_json(anchor_frames_json)
        _download_video_url(video_url, video_path)
        output_dir = job_dir / "output"
        report = run_full_analysis(
            str(video_path),
            output_dir=str(output_dir),
            handedness=h,
            handedness_source=h_source,
            contact_frame=contact_frame,
            anchor_frames=anchor_frames,
        )
        return JSONResponse(content=_build_analysis_response(report, job_id, output_dir))
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        tb = traceback.format_exc()
        print(f"[analyse-from-url] FAILED job={job_id}\n{tb}")
        raise HTTPException(status_code=500, detail=f"{exc}\n\n{tb}")


@app.get("/results/{job_id}/{file_path:path}")
def get_result_file(job_id: str, file_path: str):
    """Serve generated files for a completed analysis job."""
    job_root = (RESULTS_DIR / job_id).resolve()
    requested = (job_root / file_path).resolve()

    if not str(requested).startswith(str(job_root)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(requested))
