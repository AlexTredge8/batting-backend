# P0-A — Phase 0 baseline: four-rule scorer against coach scores

Paste everything below the line into a new Codex thread on the Mac. One task only. Do not bundle P0-B to P0-E into this thread.

---

## Goal

Measure the current scorer. Do not change it. Make one dated table that gives per-clip overall and pillar error against Alex's coach scores for the 17 anchored clips, in two modes:

- **auto**: the anchors are detected automatically. This is what production does.
- **validated**: the six anchors come from `anchor_truth.csv`.

Also record: whether each pillar score came from a measurement or from a default of 25, whether the same clip gives the same score twice, and the error of a "constant" scorer that gives every clip the average coach score.

## Context you need

- Repo: `https://github.com/AlexTredge8/batting-backend`, `main` at commit `50e48da`. The copy already on this Mac is an old March commit (`e8ed377`). **Do not use it and do not change it.**
- Active rules at `50e48da`: A5 (Access), T2 (Tracking), S4 (Stability), F4 (Flow). One rule per pillar. Source of truth: `config.PILLAR_RULE_MAX_DEDUCTION`. `RULES_VERSION` is `"2.0.0"`.
- Coach scores: `coach_ground_truth_from_screenshot.csv` (18 rows). `Female_Average.mp4` has `filmed_correctly = N`, so the batch skips it. 17 clips remain. These are the same 17 clips as `anchor_truth.csv`.
- Use the `tier` column in the CSV, not the tier in the filename. They differ for two clips: `Ollie P_Offdrive Average.mov` is `Good Club` and `Straight Drive_Beginner.mp4` is `Average`.
- Clips: `~/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:/` (four tier folders, 18 clips). The colons are part of the POSIX path: macOS Finder shows a `/` in a folder name as `:`. Quote the path. If it is not there, list `~/Documents/01 Work/03 BattingIQ/03 Working Folders/` and find the folder that holds the four tier folders. Do not copy the clips into the repo.
- Batch runner: `batch_calibration_compare.py`. Flags used here: `--ground-truth`, `--video-dir`, `--anchor-truth`, `--dual-mode`, `--skip-railway`, `--export-measurements`, `--output-dir`. Without `--allow-cached-local-fallback`, every clip is analysed fresh. Keep it that way.
- `calibration_output/` is in `.gitignore`. The committed table must go in `baselines/`, which is not ignored.
- Audio contact needs `ffmpeg` on `PATH`. `audio_contact.py` calls `shutil.which("ffmpeg")`. The `imageio-ffmpeg` package does **not** put ffmpeg on `PATH`. If ffmpeg is missing, every clip falls back to pose-only contact (about 30 frames of error against about 2 with audio) and the auto numbers are not the production numbers.

## Tasks

### 1. Fresh clone and environment

1. Clone into a new folder: `git clone https://github.com/AlexTredge8/batting-backend.git ~/Documents/battingiq-phase0`, then `cd` into it.
2. `git checkout -b phase0-baseline 50e48da`. Record `git rev-parse HEAD`. If `main` on GitHub is now ahead of `50e48da`, tell me its head commit, but still work from `50e48da`.
3. Make a Python 3.10 virtual environment in `.venv310/` (in `.gitignore`). Use `python3.10`, or `uv` or `pyenv` if one is already installed. If no Python 3.10 is available, stop and tell me. Do not install a system Python.
4. `pip install -r requirements-dev.txt`.
5. Make sure `ffmpeg` is on `PATH`. If `which ffmpeg` returns nothing, link the `imageio-ffmpeg` binary into the venv: `ln -s "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .venv310/bin/ffmpeg`. Then run `ffmpeg -version` and record the path.

### 2. Pre-flight checks. Stop and report if any check fails.

1. `python -m pytest tests -q` passes. Expected: 44 passed, 5 skipped.
2. `BATTINGIQ_E2E=1 python -m pytest tests/test_reference_video_e2e.py -q` passes. Expected: 5 passed. This proves that this Mac reproduces the calibrated anchors on `test_batting.mov` (60/61/77/73/81/109).
3. All 17 non-skipped clips in the coach CSV match a file in the clip folder.
4. `md5 "…/Viv P_Offdrive Elite.mov"` gives `0a1b3a74e1f123a97085c2fafbaa12ea`, the same as `test_batting.mov`.
5. Record each clip's audio-stream count. Use `ffprobe` if present; otherwise parse `ffmpeg -hide_banner -i <clip>` stderr for `Audio:`. Expect `Ollie P_Offdrive Average.mov` to have no audio track. Any other clip with no audio is a finding: report it, but do not stop.

### 3. Stop counting the reference clip as held out (its own commit)

`Viv P_Offdrive Elite.mov` is byte-for-byte the same file as `test_batting.mov`. `test_batting.mov` is the development clip: the e2e test pins its anchors and the drift fix was measured on it. So it cannot test for overfitting.

- In `heldout_split.csv`, change the `split` value of `Viv P_Offdrive Elite.mov` from `heldout` to `reference`. Change no other row. Do not choose a replacement clip.
- The batch reads this file without code changes. A `reference` row appears in the detailed CSVs and is left out of both `tuning_*` and `heldout_*` summaries. Check this after the run.
- Commit: `Mark Viv P_Offdrive Elite as reference clip, not held out (same file as test_batting.mov)`.

### 4. Run the batches (local only)

Set `RUN=calibration_output/phase0_$(date +%F)` and `VIDS=<the clip folder>`.

```bash
python batch_calibration_compare.py \
  --ground-truth coach_ground_truth_from_screenshot.csv \
  --video-dir "$VIDS" \
  --anchor-truth anchor_truth.csv \
  --dual-mode --skip-railway --export-measurements \
  --output-dir "$RUN"

python batch_calibration_compare.py \
  --ground-truth coach_ground_truth_from_screenshot.csv \
  --video-dir "$VIDS" \
  --skip-railway --export-measurements \
  --output-dir "$RUN/auto_repeat"
```

The first command writes `$RUN/auto/`, `$RUN/validated/` and `$RUN/dual_run_manifest.json`. The second command is a repeat of the auto pass, for the determinism check. Each clip's report is at `<mode>/local_runs/<stem>/<stem>_battingiq.json`.

Check the log of every mode: 17 processed, 17 local successes, `Cached local fallbacks used: 0`. If not, stop and report.

### 5. Left-hander check (no code change)

`Cam T_LH Offdrive Elite.mov` is a left-hander. The batch never passes `handedness`, so every past batch scored it as a right-hander. By my reading, the four active rules use handedness-independent quantities: `_shoulder_openness` and `_hip_openness` use `abs()`, `head_offset` is nose x minus hip-mid x, and F4 uses wrist velocity. So I expect no change. Test this:

- In a short throwaway script (do not commit it), call `run_analysis.run_full_analysis(path, output_dir=…, handedness="left", handedness_source="manual")` for this clip, once in auto mode and once with its `anchor_truth.csv` row as `anchor_frames`. Write the output under `$RUN/lh_check/`.
- Compare with the right-handed result from step 4: overall, four pillars, six anchors, and the four active-rule measurements.
- Write `handedness_check.csv` (one row per mode). If anything differs, report which field and by how much.

### 6. Write `phase0_baseline_table.py` (new file, read-only on outputs)

This script reads the run outputs only. It must not import or run the pipeline.

CLI: `--run-dir`, `--out-dir`, `--ground-truth`, `--anchor-truth`, `--split`, `--video-dir`.

**`baseline_table.csv`**: one row per clip per mode (auto, validated). That is 34 rows. Columns, in this order:

1. Identity: `run_date`, `git_commit`, `rules_version` (`config.RULES_VERSION`; reading `config` constants is allowed), `active_rules` (keys of `PILLAR_RULE_MAX_DEDUCTION`, e.g. `A5;T2;S4;F4`), `mode`, `filename`, `tier` (from the CSV), `tier_in_filename`, `split` (tuning / heldout / reference), `handedness_used`.
2. Coach: `coach_overall`, `coach_access`, `coach_tracking`, `coach_stability`, `coach_flow`.
3. System: `sys_overall`, `sys_access`, `sys_tracking`, `sys_stability`, `sys_flow`. Take the pillar scores from the report JSON. Check that the overall score equals the pillar sum.
4. Error (system minus coach): `err_overall`, `err_access`, `err_tracking`, `err_stability`, `err_flow`.
5. Evidence per pillar. Map pillar to rule from `PILLAR_RULE_MAX_DEDUCTION`, and rule to measurement key: A5 → `A5_shoulder_hip_gap_deg`, T2 → `T2_early_head_change`, S4 → `S4_post_contact_rotation_deg`, F4 → `F4_pause_frames`. For each pillar, give `<pillar>_rule`, `<pillar>_measurement` (from `metadata.rule_measurements`), `<pillar>_deduction` (sum of that rule's fault deductions, else 0) and `<pillar>_status`, where:
   - `suppressed`: the rule id is in `metadata.rule_evaluation.suppressed_rule_ids`;
   - `not_measured`: the measurement is null;
   - `fault`: the rule produced a fault;
   - `no_fault`: measured, no fault. The pillar then scores 25.
6. Anchors: for each of the six anchors (`setup_frame`, `hands_start_up_frame`, `front_foot_down_frame`, `hands_peak_frame`, `contact_frame`, `follow_through_frame`), give `<anchor>_sys` (`metadata.anchor_frames.<anchor>.original_frame`), `<anchor>_truth` (from `anchor_truth.csv`), `<anchor>_delta` (system minus truth) and `<anchor>_conf` (`metadata.anchor_confidence`).
7. Contact and capture: `contact_method` and `contact_confidence` (`metadata.phase_diagnostics`), `audio_streams`, `fps`, `width`, `height`, `total_frames`, `duration_s`, `detection_rate`.
8. Coaching output: `priority_fix_rule`.
9. `repeat_identical` (auto rows only): Y if the auto and auto_repeat runs match exactly on overall, the four pillars, the six anchor frames and the four active measurements; else N. Leave it blank on validated rows.

**`baseline_summary.md`**: put a header first, then a section for each mode, with the numbers in tables.

- **Header**: run date, commit, `RULES_VERSION`, active rules, Python version, MediaPipe version, macOS version, chip, ffmpeg path, clip count, and the exclusion of `Female_Average.mp4` with the reason.
- **Error**: mean absolute error, overall and per pillar. Give it for all 17 clips and for the 16 without the reference clip.
- **Bias**: mean signed error (system minus coach) for each CSV tier, and for each coach-score band (coach overall < 40, 40–59, 60–79, ≥ 80), with n per group. The plan's target uses the bands. Mark every group whose mean bias is above 5 points.
- **Constant benchmark**: predict each clip's overall and pillar scores as the mean coach score of the other 16 clips. Give its mean absolute error beside the system's, overall and per pillar. State plainly whether the system beats it, for each pillar and overall.
- **Rank agreement**: Spearman correlation between system overall and coach overall.
- **Within ±5 overall**: the count, for continuity with past reports only.
- **Tier means and monotonic order**: as a sanity check only.
- **Free 25s**: count pillar results by status. Count the `no_fault` 25s where the coach pillar score is 15 or less. List those clips.
- **Auto against validated**: mean |difference| in overall score between the two modes, and the three clips with the largest difference.
- **Determinism**: N of 17 identical. List any field that varied.
- **Priority fix**: count of clips per system priority-fix rule. State that there is no coach priority fix to compare against (`coach_notes` is empty).
- **Ceiling**: state "coach-to-coach error not yet measured (Phase 2)". Do not compare with any target.
- **Left-hander check**: one line with the result of task 5.

**`determinism.csv`**: one row per clip, the compared fields from auto and auto_repeat, and the match flag.

### 7. Assemble the committed folder

Write all of this into `baselines/phase0_<YYYY-MM-DD>/`:

- `baseline_table.csv`, `baseline_summary.md`, `determinism.csv`, `handedness_check.csv`;
- `run_manifest.json`: the batch's `dual_run_manifest.json` plus `rules_version`, `active_rules`, the environment fields from the summary header, the clip-folder path and the `md5` of every clip;
- `raw/<mode>/` for `auto`, `validated` and `auto_repeat`: copy each mode's `battingiq_calibration_comparison_detailed.csv`, `battingiq_calibration_tier_summary.csv`, `battingiq_calibration_metrics.txt`, `rule_measurements.csv`, the `tuning_*` and `heldout_*` files and `batch_calibration.log`;
- `reports/<mode>/<stem>_battingiq.json`: the per-clip report JSONs, about 15 KB each.

Do **not** commit videos, annotated MP4s, storyboard PNGs or anything else from `local_runs/`.

## Constraints

- Measurement only. Do not change scoring, rules, thresholds, detectors, `config.py`, `batch_calibration_compare.py` or any existing script. The only edit to an existing file is the one row in `heldout_split.csv`, plus `latest_developments.md` at the end.
- No Railway calls. No cached fallback. No network calls apart from the clone and `pip`.
- Do not send clips or frames to any external service.
- Do not tune anything, even if the numbers look fixable. Report what you see.
- Commit on branch `phase0-baseline`. Do not push. Alex pushes.

## Deliverable (paste back into the hub chat)

1. `git rev-parse HEAD` before and after, and the GitHub `main` head if it is past `50e48da`.
2. Pre-flight results: test counts, e2e result, ffmpeg path, clip match count, Viv md5, and the audio-stream count per clip.
3. The full `baseline_summary.md`, inline.
4. The left-hander check result.
5. The paths of every committed file.
6. Every warning, error or surprise, including the run time per mode.

## Final step

Append to `latest_developments.md`:

- one line under **Completed task log**: `P0-A | <date> | baselines/phase0_<date>/ — four-rule baseline, auto + validated, 17 clips; auto MAE <x>, validated MAE <y>; determinism <n>/17; Viv P_Offdrive Elite marked reference`;
- set **Current stage** to `Phase 0 baseline recorded`.

Commit as `P0-A: four-rule baseline table (auto + validated, 17 clips)`. Then print the exact `git push -u origin phase0-baseline` command for Alex.
