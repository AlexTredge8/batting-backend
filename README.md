# BattingIQ Backend

Cricket batting technique analysis from a single phone clip filmed from the bowler's end.
The service extracts MediaPipe pose landmarks, detects six anchor moments
(Setup → Backlift starts → Hands peak → Front foot down → Contact → Follow-through),
evaluates the active coaching rules (`coaching_rules.py`, `config.py`) and returns a
0–100 BattingIQ score across four pillars (Access, Tracking, Stability, Flow), an
annotated H.264 video and six storyboard stills.

Deployed on Railway (`Dockerfile`, `railway.json`, `start.py`). Frontend: Lovable.

## Pipeline

```
POST /analyse
  → run_full_analysis()            run_analysis.py
      → load_reference_baseline()  reference/reference_baseline.json (absolute path)
      → extract_poses()            pose_extractor.py  — mp.tasks PoseLandmarker (heavy), every frame, source resolution
      → calculate_metrics()        metrics_calculator.py
      → detect_phases()            phase_detector.py  — audio-first contact, contact-anchored hands peak, stillness-anchored setup
      → run_all_rules()            coaching_rules.py  — active: A5, T2, S4, F4 (others suspended/deleted, see RULES_CHANGELOG.md)
      → build_scores()             scorer.py
      → annotate_video()           video_annotator.py — draws the analysis landmarks, no second pose model
      → generate_storyboard()      six stills + strip
  → JSON (+ media served from /results/{job_id}/…)
```

**Processing mode.** Production runs the *same* full-rate, source-resolution path that every
calibration batch was measured on (`PROCESSING_MODE=full_rate_calibrated`). The previous
subsampled path (every 2nd frame, 640px) moved anchors by up to 40 frames and changed rule
measurements on the reference clip; it remains available as `FAST_MODE=1` for emergencies only
and is flagged in the response `warnings`.

## Run locally

```bash
pip install -r requirements-dev.txt      # includes pytest, httpx, a static ffmpeg
python run_analysis.py test_batting.mov --output-dir output/
uvicorn api:app --reload --port 8000
```

ffmpeg must be on `PATH` for audio-based contact detection (the Dockerfile installs it;
`imageio-ffmpeg` provides a static binary for local runs).

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/diag` | memory, disk, processing mode, ffmpeg availability, storage config |
| POST | `/analyse` | multipart upload → full analysis |
| POST | `/analyse-from-url` | re-analyse a stored video (optionally with validated anchor frames) |
| GET | `/results/{job_id}/{path}` | annotated video / storyboard files |

`POST /analyse` form fields: `file` (mp4/mov/avi/mkv/m4v/webm), optional `handedness`
(`right`|`left`, default right), `contact_frame`, `anchor_frames_json`, plus `angle`, `name`,
`email`, `consent` (accepted, not used by the pipeline). Clips longer than
`MAX_VIDEO_DURATION_S` (30s) are rejected with `422 {"detail": {"error": "video_too_long", …}}`.

Response (top level):

| Field | Meaning |
|---|---|
| `battingiq_score`, `score_band` | 0–100 and band |
| `pillars.{access,tracking,stability,flow}` | `score`, `max`, `status`, `faults[]`, `positives[]` |
| `priority_fix`, `development_notes` | coaching output |
| `phases.*` | anchors: `frame` (metric index), `original_frame` (video frame), `ms`, confidence, contact diagnostics |
| `analysis_quality` | `reliable`, `audio_available`, `contact_method`, `contact_confidence`, `low_confidence_anchors`, `detection_rate`, `processing_mode`, `baseline_status` |
| `warnings[]` | plain-English notices safe to show the user (no audio, low contact confidence, fast mode, …) |
| `annotated_video_url`, `storyboard_url` | relative URLs under `/results/` |
| `storyboard_frames` | object keyed `setup`, `hands_start_up`, `front_foot_down`, `hands_peak`, `contact`, `follow_through`. Each value: `{available, label, image (data:image/jpeg URI, skeleton drawn, 400px), image_url, data_url, url, frame (original video frame), timestamp_ms, …}` |
| `metadata.storyboard_frame_items[]` | full-size annotated stills by URL: `label`, `url`, `original_frame_idx`, `timestamp_ms`, selection diagnostics |
| `metadata` | everything else: anchor frames/confidence, rule measurements, detector versions, media/storage status |

## Configuration (environment)

| Variable | Default | Effect |
|---|---|---|
| `FAST_MODE` / `LOCAL_MODE` | `0` / `1` | `FAST_MODE=1` re-enables the uncalibrated subsampled path |
| `MAX_VIDEO_DURATION_S` | `30` | duration guard |
| `USE_SETUP_V4`, `USE_HANDS_PEAK_V3`, `USE_AUDIO_CONTACT`, … | see `config.py` | detector versions |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET` | unset | optional media persistence |

## Calibration workflow

Ground truth lives in `anchor_truth.csv` (manually validated anchor frames) and
`coach_ground_truth_from_screenshot.csv` (coach pillar scores) for 17 videos, with a permanent
held-out split in `heldout_split.csv` (see `heldout_discipline.md`). `batch_calibration_compare.py`
runs the full set locally and against Railway and emits tier/pillar concordance, drift and rule-health
reports. Strategy, task tracks and the change history are in `claude.md`, `ORCHESTRATOR.md`,
`latest_developments.md` and `RULES_CHANGELOG.md`.

## Tests

```bash
python -m pytest tests -q                                   # fast unit + API contract tests
BATTINGIQ_E2E=1 python -m pytest tests/test_reference_video_e2e.py -q   # MediaPipe end-to-end on test_batting.mov
```

The end-to-end test pins the calibrated anchors for the reference clip
(setup 60, hands start 61, front foot 77, hands peak 73, contact 81, follow-through 109) and
fails if extraction, detection or the processing mode moves them.
