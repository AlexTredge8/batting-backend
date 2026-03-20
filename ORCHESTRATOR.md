# ORCHESTRATOR.md — BattingIQ Backend Living Memory

> This file is maintained by the orchestrator Claude thread. Update it after every significant change.
> Last updated: 2026-03-20

---

## What This Repo Is

**BattingIQ Backend** — a Python video analysis service that accepts cricket batting videos, runs MediaPipe pose extraction, computes 30+ coaching metrics per frame, runs 21 rule-based coaching checks across 4 pillars, and returns a scored JSON report + annotated video + storyboard.

**Deployed on:** Railway.app
**Tech stack:** Python 3.10 + FastAPI + MediaPipe 0.10.9 + OpenCV + NumPy
**Python version:** 3.10.13 (pinned in `.python-version`)

---

## Directory Structure

```
batting-backend/
├── start.py                    # Entry point: reads PORT, starts uvicorn on api:app
├── api.py                      # FastAPI app — 4 routes, CORS, job management
├── run_analysis.py             # Pipeline orchestrator: run_full_analysis(video_path, output_dir)
├── pose_extractor.py           # MediaPipe BlazePose → list[FramePose]
├── metrics_calculator.py       # Per-frame metrics → list[FrameMetrics]
├── phase_detector.py           # State machine → PhaseResult (7 phases)
├── coaching_rules.py           # 21 coaching rules → dict[pillar: list[Fault]]
├── scorer.py                   # Pillar scores + BattingIQ score → BattingIQResult
├── report_generator.py         # JSON report assembly + console print
├── video_annotator.py          # Frame overlays + storyboard PNG
├── reference_builder.py        # Calibrates baseline from gold-standard video
├── models.py                   # Dataclasses: FramePose, FrameMetrics, PhaseResult, Fault, etc.
├── config.py                   # ALL tunable thresholds & parameters
├── requirements.txt
├── Dockerfile
├── Procfile
├── railway.json
├── reference/
│   └── reference_baseline.json # Gold-standard calibration (used by all rules)
├── output/                     # Sample outputs from test_batting.mov
└── test_batting.mov            # Gold-standard reference video
```

**Legacy (Phase 1, unused by API):** `analyse.py`, `poc_analyse.py`

---

## API Routes

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/health` | Liveness probe |
| GET | `/diag` | Memory/disk/env diagnostic |
| POST | `/analyse` | Main: upload video → full analysis |
| GET | `/results/{job_id}/{file_path}` | Download annotated video / storyboard |

`POST /analyse` accepts multipart form: `file` (video), `angle?`, `name?`, `email?`, `consent?`
Returns: `battingiq_score`, `score_band`, `pillars`, `priority_fix`, `development_notes`, `phases`, `metadata`, `job_id`, `annotated_video_url`, `storyboard_url`

**CORS allowed origins:** `https://battingiq.lovable.app`, `https://*.lovable.app`, `https://*.lovableproject.com`, `http://localhost:3000`, `http://localhost:8080`

---

## Data Flow

```
POST /analyse
  → run_full_analysis()
      → load_reference_baseline()
      → extract_poses()         # MediaPipe BlazePose, 33 landmarks/frame
      → calculate_metrics()     # 30+ metrics, gap-fill, smooth, velocities
      → detect_phases()         # 7-state machine on wrist velocity
      → run_all_rules()         # 21 rules × 4 pillars vs baseline
      → build_scores()          # Pillar scores (0-25) → BattingIQ (0-100)
      → save_json_report()
      → annotate_video()        # Frame overlays (no MediaPipe re-run)
      → generate_storyboard()   # 6-panel PNG
  → JSONResponse with URLs
```

---

## Pipeline Modules

### pose_extractor.py
- `extract_poses(video_path, verbose=True)` → `(list[FramePose], video_meta)`
- Subsamples to 15 fps max, downscales to 640px width (OOM prevention)
- model_complexity=1 (was 2, downgraded for memory)

### metrics_calculator.py
- `calculate_metrics(frame_poses, fps)` → `list[FrameMetrics]`
- Steps: raw calc → gap-fill → velocities → smooth (3-frame window) → re-calc velocities
- **Y-axis is inverted**: smaller Y = higher in frame

### phase_detector.py
- `detect_phases(metrics, fps)` → `PhaseResult`
- 7 phases: SETUP → BACKLIFT_STARTS → HANDS_PEAK → FRONT_FOOT_DOWN → CONTACT → FOLLOW_THROUGH → UNKNOWN
- Detects via wrist velocity sign reversals + threshold crossings
- Seeds from first 15 detected frames; constrains peak search to first 45% of stroke

### coaching_rules.py
- `run_all_rules(metrics, phases, baseline)` → `dict[pillar: list[Fault]]`
- **4 Pillars, 21 Rules:**
  - Stability (S1–S4, max 25): weight transfer, post-contact stability, hip drift, body rotation
  - Tracking (T1–T5, max 25): head line, early head move, head stillness, eye level, setup composure
  - Access (A1–A6, max 25): bat path, elbow angle, compression window, torso lean, shoulder lag, early opening
  - Flow (F1–F6, max 25): peak/FFD sync, jerky accel, timing range, pause at peak, mid-swing hitch, follow-through

### scorer.py
- `build_scores(fault_map, phases, baseline, video_meta)` → `BattingIQResult`
- Pillar score = 25 − sum(fault deductions)
- Traffic lights: Green ≥20, Amber 12–19, Red <12
- BattingIQ = sum of 4 pillars (0–100)
- Bands: Excellent 85–100, Good 70–84, Developing 55–69, Work Needed 40–54, Fundamentals <40
- Priority fix: highest deduction in lowest pillar (tiebreak: stability > tracking > access > flow)

### models.py — Key Types
- `RawLandmark`: x, y, z, visibility
- `FramePose`: frame_idx, timestamp, landmarks (None if undetected)
- `FrameMetrics`: 30+ fields per frame
- `PhaseResult`: labels[], setup_end, backlift_start, hands_peak, front_foot_down, contact, follow_through_start, timing diffs
- `Fault`: rule_id, fault, deduction, detail, feedback
- `PillarScore`: score, max, status, faults[], positives[]
- `BattingIQResult`: all of the above assembled

### config.py
- Single source of truth for ALL thresholds
- FRONT_SIDE = "left" (right-handed batter from bowler's end)
- Phase detection: wrist rise threshold, consecutive frame counts, velocity window sizes
- Metric smoothing: 3-frame window
- Scoring: pillar max (25), traffic light thresholds (20/12), score bands

---

## Reference Baseline

- Location: `reference/reference_baseline.json`
- Built by: `python reference_builder.py test_batting.mov`
- All 21 coaching rules scale thresholds relative to this baseline
- **Risk:** If missing, `run_analysis.py` rebuilds from input video (circular — not ideal)

---

## Known Fragile Areas

1. **Phase detection** — relies on wrist velocity sign reversals; noisy video/lighting can confuse it. Mitigated by 3–5 frame smoothing.
2. **Gap filling** — forward-fills missing detections with last-known value. Long gaps (>10 frames) can propagate stale data.
3. **Y-axis inversion** — MediaPipe Y: smaller = higher. Phase/metric code must account for this everywhere.
4. **Frame index vs subsampled index** — video annotator uses original frame count; phases detected on 15fps subsampled frames. Off-by-one risk if indexing mixes these.
5. **Reference baseline dependency** — all rules silently degrade if baseline is wrong/missing.
6. **Coaching rules fail silently** — each rule wrapped in try-catch; failure logs a warning but skips that fault.
7. **No auth** — API is fully public. Results are accessible by anyone who knows the job_id.

---

## Conventions to Follow

- **Module naming:** `lowercase_with_underscores.py`
- **Classes/dataclasses:** PascalCase
- **Functions:** `lowercase_with_underscores()`
- **Private helpers:** `_leading_underscore()`
- **Constants:** `UPPER_CASE`
- **Type hints** on all function signatures
- **f-strings** for string formatting
- **Docstrings** on public functions only
- **config.py** is the only place to add new thresholds — never hardcode in rule/metric files
- **models.py** is the only place to add new data structures

---

## Deployment

- `start.py` reads `PORT` env var (default 8000), runs `uvicorn api:app --host 0.0.0.0`
- `railway.json`: Dockerfile builder, `python start.py` start command, `/health` check (300s timeout), ON_FAILURE restart (max 3)
- Dockerfile validates imports at build time: `python -c "from api import app"`
- No database — stateless, file-based job outputs in `results/{job_id}/`
- No authentication

---

## Decisions Made

*(none yet — populate as decisions are made in this thread)*

---

## Current State

- Codebase fully explored on 2026-03-20
- No tests written; manual validation via `test_batting.mov`
- OOM crashes previously fixed: model_complexity 2→1, frame downscaling, skip-frame processing, no MediaPipe re-run in annotator
- Path traversal hardened in `get_result_file()` (commit 8ad17be)
- IndexError fix in `_fill_gaps()` (commit 4a49f77)
