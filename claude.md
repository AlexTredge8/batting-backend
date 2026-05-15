# BattingIQ — Claude Code Permanent Context

## What this system does

BattingIQ analyses cricket batting technique from video. A user uploads a video; the pipeline extracts pose landmarks, detects 6 batting phases (anchors), fires 21 coaching rules, and returns a score out of 100 across 4 pillars: **Access, Tracking, Stability, Flow** (each out of 25). Output includes pillar breakdown, coaching feedback, and storyboard keyframes.

**Core problem being solved:** score compression — near-perfect and beginner technique score within 5 points of each other. The goal is 40–60 point separation across skill tiers.

-----

## Architecture

### Backend (Railway — FastAPI)

|File                          |Role                                                         |
|------------------------------|-------------------------------------------------------------|
|`pose_extractor.py`           |MediaPipe BlazePose landmark extraction                      |
|`phase_detector.py`           |Detects 6 anchor frames (Setup → Follow Through)             |
|`metrics_calculator.py`       |Computes measurements from landmarks                         |
|`coaching_rules.py`           |Rules engine; v2.0.0 has 11 active, 6 suspended, 4 deleted|
|`config.py`                   |All thresholds, rule flags, pillar weights                   |
|`scorer.py`                   |Aggregates rule deductions into pillar and overall scores    |
|`video_annotator.py`          |Renders MediaPipe overlay video                              |
|`api.py`                      |FastAPI: `POST /analyse`, `GET /health`                      |
|`run_analysis.py`             |Local single-video pipeline                                  |
|`batch_calibration_compare.py`|Batch runner; compares system vs manual scores               |

### Key data files

- `ground_truth_scores.csv` — 17 videos, 4 pillar scores per video, tier labels
- `anchor_truth.csv` — manually validated anchor frames for all 17 videos
- `annotation.csv` — Alex’s per-pillar scoring reasoning (T1 task output)

### Frontend (Lovable)

- Result shape: `result.analysis.battingiq_score`, `.pillar_scores`, `.coaching_points`, `.storyboard_frames`
- Backend URL: `https://web-production-e9c26.up.railway.app`


**Current stage (2026-04-29): R4-partial + R3 complete.**
- Active rules: A1, A3, A5, T2, S2, S4, F1, F3, F4, F5, F6 (11 active).
- Suspended rules: A2, A4, T1, S1, S3, F2 (6 suspended; F2 is suspended, not deleted).
- Deleted rules: A6, T3, T4, T5.
- Pillar weights: Option B3 — Access/Tracking/Stability/Flow remain 25 each, but deductions are normalised by active rule capacity per pillar.

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
1. **Railway optimisations** (model_complexity=1, frame_step=2, 640px cap) are production settings — do not change them. Local calibration uses full quality.

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

All 17 ground-truth videos within ±5 on overall score AND all 4 pillars vs Alex’s manual scores. Elite mean ≥ 90. Tier ordering monotonic. Pipeline deterministic.

-----

## Held-out validation set

Four videos are permanently locked as held-out (one per tier). They are processed in every batch but must never be used to inform threshold or rule decisions.

| Filename | Tier |
|---|---|
| Drive_Average.MOV | Average |
| Drive_Beginner 2.MOV | Beginner |
| Viv P_Offdrive Elite.mov | Elite |
| AlexT_Ondrive Good.MOV | Good Club |

**Rules:**
- Tune against `tuning_metrics.txt` only.
- Read `heldout_metrics.txt` only after a change is committed, to check for regression.
- Every R-track and F-track task deliverable must report both tuning MAE and held-out MAE.
- See `heldout_discipline.md` for full rules.