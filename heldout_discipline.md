# BattingIQ — Held-out Validation Discipline

## Purpose

Three videos are locked as a held-out validation set, and one is a reference clip. They are processed during every batch run but must never be used to inform threshold choices, rule redesigns, or anchor-detector changes. Their purpose is to detect overfitting: if tuning-set performance improves but held-out performance does not, the change is overfit.

## Held-out videos

| Filename | Tier |
|---|---|
| Drive_Average.MOV | Average |
| Drive_Beginner 2.MOV | Beginner |
| AlexT_Ondrive Good.MOV | Good Club |

These are defined in `heldout_split.csv` (one row per video, `split` column = `heldout`).

**Reference clip (2026-11-14):** `Viv P_Offdrive Elite.mov` has `split = reference`. It is byte-for-byte the same file as `test_batting.mov` (md5 `0a1b3a74e1f123a97085c2fafbaa12ea`), the development clip whose anchors the e2e test pins. It cannot detect overfitting, so it is reported in the detailed CSVs and left out of both the tuning and held-out summaries.

**To be replaced (Phase 4):** this split is by clip, not by player. AlexT_Ondrive Good is held out while Alex T_Straightdrive Good is in tuning, so the same batter sits on both sides. Phase 4 draws a player-level split: 40 development clips and 20 locked clips from unseen players.

## Rules

1. **Threshold and rule changes are informed by tuning-set metrics only.** Do not inspect held-out per-video scores, per-pillar scores, or rule firings when deciding whether to raise or lower a threshold.

2. **Held-out metrics are read-after-commit only.** After committing a rule or threshold change, read `heldout_metrics.txt` to check for regression. Do not iterate on held-out performance.

3. **Never move a video from held-out to tuning.** The split is permanent until the dataset grows beyond 25 videos (T4 task), at which point a new random split is drawn.

4. **Both sets are processed in every batch.** `batch_calibration_compare.py` reads `heldout_split.csv` automatically when it is present alongside the script. Outputs `tuning_summary.csv`, `heldout_summary.csv`, `tuning_metrics.txt`, `heldout_metrics.txt` in every batch output directory.

5. **Report both in task outputs.** Every R-track or F-track task deliverable must include tuning-set MAE and held-out MAE side by side.

## Split file format

`heldout_split.csv` columns: `filename`, `tier`, `split` (`tuning` | `heldout` | `reference`).

The batch script resolves filenames case-insensitively and also matches on stem (filename without extension) so `.MOV` vs `.mov` differences are handled automatically.
