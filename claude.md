# BattingIQ — Claude Code Permanent Context

## What this system does

BattingIQ analyses cricket batting technique from video. A user uploads a video; the pipeline extracts pose landmarks, detects 6 batting phases (anchors), evaluates the active coaching rules (21 defined, 4 active: A5, T2, S4, F4), and returns a score out of 100 across 4 pillars: **Access, Tracking, Stability, Flow** (each out of 25). Output includes pillar breakdown, coaching feedback, and storyboard keyframes.

**Core problem being solved:** score compression — near-perfect and beginner technique score within 5 points of each other. The goal is 40–60 point separation across skill tiers.

-----

## Architecture

### Backend (Railway — FastAPI)

|File                          |Role                                                         |
|------------------------------|-------------------------------------------------------------|
|`pose_extractor.py`           |MediaPipe BlazePose landmark extraction                      |
|`phase_detector.py`           |Detects 6 anchor frames (Setup → Follow Through)             |
|`metrics_calculator.py`       |Computes measurements from landmarks                         |
|`coaching_rules.py`           |Rules engine; v2.2.0 has 4 active (A5, T2, S4, F4), 13 suspended, 4 deleted|
|`config.py`                   |All thresholds, rule flags, pillar weights                   |
|`scorer.py`                   |Aggregates rule deductions into pillar and overall scores    |
|`video_annotator.py`          |Renders MediaPipe overlay video                              |
|`api.py`                      |FastAPI: `POST /analyse`, `GET /health`                      |
|`run_analysis.py`             |Local single-video pipeline                                  |
|`batch_calibration_compare.py`|Batch runner; compares system vs manual scores               |

### Key data files

- `coach_ground_truth_from_screenshot.csv` — 18 videos, Alex's 4 pillar scores + overall, tier labels (`Female_Average.mp4` has filmed_correctly = N, so batches use 17). `coach_notes` is empty. (`ground_truth_scores.csv` and `annotation.csv` do not exist; the batch falls back to the screenshot CSV.)
- `anchor_truth.csv` — manually validated anchor frames for 17 videos (not `Female_Average.mp4`)
- `heldout_split.csv` — tuning / heldout / reference split
- `clip_sheet.csv` — per-clip capture facts and context (Phase 0)
- `baselines/phase0_<date>/` — the Phase 0 baseline table for the four-rule scorer

### Frontend (Lovable)

- Result shape: `result.analysis.battingiq_score`, `.pillar_scores`, `.coaching_points`, `.storyboard_frames`
- Backend `/analyse` response now also carries `analysis_quality` (audio/contact/anchor reliability, processing mode) and `warnings` (plain-English, safe to show to the user). `storyboard_frames` is an object keyed by phase (setup, hands_start_up, front_foot_down, hands_peak, contact, follow_through); each value is `{available, label, image (data URI), url, frame, timestamp_ms}`. Full-size stills are in `metadata.storyboard_frame_items` by URL.
- Backend URL: `https://web-production-e9c26.up.railway.app`


**Current stage (2026-11-14): Phase 0 of the coach-agreement plan (baseline and freeze).**
- Active rules: A5, T2, S4, F4 (4 active, one per pillar). Source of truth: `config.PILLAR_RULE_MAX_DEDUCTION`.
- Suspended rules: A1, A2, A3, A4, T1, S1, S2, S3, F1, F2, F3, F5, F6 (13 suspended).
- Deleted rules: A6, T3, T4, T5.
- `RULES_VERSION = "2.2.0"`; see `RULES_CHANGELOG.md` for the 13–15 May 2026 changes.
- Pillar weights: Option B3. Each pillar is 25, and its deduction is normalised by the active rule capacity. With one rule per pillar, a pillar is 25 whenever its rule does not fire, is suppressed, or cannot measure. Phase 6 replaces this with "not assessed".
- Plan: `docs/phase0/README.md` and the plan doc "Batting IQ: plan to prove agreement with expert coaches".

### Skill tiers (ground truth)

Beginner → Average → Good Club → Elite (target monotonic ordering)

-----

## Strict rules — do NOT do these unless explicitly told

1. **Do not tune any threshold** unless the specific task says to.
1. **Do not modify rule logic or detector logic** in diagnostic tasks (D-track).
1. **Do not change suspension status** of any rule unless explicitly directed.
1. **Do not delete the v1 detector** when adding v2 — keep as fallback with config switch.
1. **Do not start P-track tasks** (storyboard, video playback, soft launch) until convergence exit.
1. **Do not bundle multiple task IDs** into one Codex thread.
1. **Do not run R2 until R1 (hub decision) exists.**
1. **Do not run F3 until F2 (hub design note) exists.**
1. **Production runs the calibrated full-rate path** (`PROCESSING_MODE=full_rate_calibrated`, frame_step=1, source resolution, mp.tasks heavy landmarker). Do not re-introduce frame subsampling or downscaling in production: on `test_batting.mov` the old fast path moved setup by −40 and hands_start_up by −35 original frames and changed A5/A2/F4 measurements (see config.py). `FAST_MODE=1` exists only as an emergency capacity fallback and is flagged in `analysis_quality.warnings`.

-----

## How tasks are structured

Tasks are organised into tracks. Each track has a dependency chain.

|Track|Purpose                                             |Key IDs                   |
|-----|----------------------------------------------------|--------------------------|
|**D**|Diagnostics — measure before changing anything      |D0→D1→D2→D3→D4            |
|**R**|Rules — threshold retune or rule redesign           |R1(hub)→R2→R3→R4→R5       |
|**F**|Frame detection — anchor detector improvements      |F1→F2(hub)→F3→F4→F5→F6    |
|**C**|Convergence — tight loop until success criterion met|C1→C2(hub)→C3→loop        |
|**T**|Ground truth — dataset expansion                    |T1(manual)→T2→T3→T4       |
|**P**|Product — frontend features                         |P1→P2→P3→P4→P5            |
|**G**|Governance — ongoing validation rituals             |G1(monthly), G2(quarterly)|

**Hub vs Codex:** Hub = reasoning/decisions in the strategy chat. Codex = one concrete task, one concrete output. Never reason in Codex; never execute in the hub.

-----

## Output format for every Codex task

Every task deliverable must contain:

1. Paths to all output files produced.
1. Key summary content pasted inline (not just file paths).
1. Any unexpected errors or blockers encountered.
1. Final step: **update LATEST_DEVELOPMENTS.md** with current stage, last completed task, next task.

When a batch run is produced, always include per-tier mean scores and whether monotonic ordering holds.

-----

## Success criterion

**Superseded (Nov 2026).** The old criterion was ±5 on overall and all four pillars for all 17 videos, Elite mean ≥ 90, and monotonic tier order. It is replaced because it rates the player, not the shot, and a fixed ±5 is probably tighter than coaches manage between themselves.

The current targets are provisional until Phase 2 measures the coach-to-coach ceiling. They apply to 20 locked test clips from unseen players, run fully automatically:

- System error is no more than 1.25 × coach-to-coach error, overall and per pillar.
- No ability band has a mean bias above 5 points.
- Priority-fix match is within 10 percentage points of the coach-to-coach rate.
- At least 80% of clips that meet the capture standard get a score.
- The same clip gives the same score.

Tier order is a sanity check only. Scope: a front-foot drive, filmed from the bowler's end, that passes the input checks.

-----

## Held-out validation set

Three videos are locked as held-out, plus one reference clip (see below). They are processed in every batch but must never be used to inform threshold or rule decisions.

| Filename | Tier |
|---|---|
| Drive_Average.MOV | Average |
| Drive_Beginner 2.MOV | Beginner |
| AlexT_Ondrive Good.MOV | Good Club |

**Reference clip, not held out:** `Viv P_Offdrive Elite.mov` (split value `reference`). It is the same file as `test_batting.mov` (md5 `0a1b3a74e1f123a97085c2fafbaa12ea`), the development clip that the e2e test pins, so it cannot detect overfitting.

The Phase 4 player-level split replaces this split: today AlexT_Ondrive Good is held out while Alex T_Straightdrive Good is in tuning, so the same batter sits on both sides.

**Rules:**
- Tune against `tuning_metrics.txt` only.
- Read `heldout_metrics.txt` only after a change is committed, to check for regression.
- Every R-track and F-track task deliverable must report both tuning MAE and held-out MAE.
- See `heldout_discipline.md` for full rules.