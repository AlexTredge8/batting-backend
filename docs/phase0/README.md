# Phase 0: baseline and freeze

The work that does not need the clips is built, tested and on branch `claude/batting-iq-phase-0-375kjr`. The clips exist only on Alex's Mac, so one command there finishes the job.

## Status

| Item | State |
|---|---|
| Docs match code: A5, T2, S4, F4; `RULES_VERSION` 2.2.0; the 13–15 May changes logged | Done. Scoring unchanged: pytest, the e2e test and the reference-clip score (92; 17/25/25/25) are all the same |
| Viv P_Offdrive Elite no longer counted as held out (`split = reference`) | Done |
| Baseline table: per-clip overall and pillar error, auto and validated, determinism, constant benchmark | Script done (`phase0_baseline_table.py`). Run on the Mac |
| Left-hander check | Script done (`phase0_lh_check.py`). Run on the Mac |
| Clip sheet for all 18 clips | Skeleton committed (`clip_sheet.csv`). Measured columns: Mac run. Judgement columns: Alex |
| Wrong-plane test | Script done (`phase0_wrong_plane.py`). Run on the Mac |
| Lovable upload check (audio, resolution, bitrate) | Manual, 20 minutes: [P0-E](P0-E_upload_path_check.md) |

## The one command (Mac, about 1.5 hours, unattended)

```bash
git clone -b claude/batting-iq-phase-0-375kjr https://github.com/AlexTredge8/batting-backend.git ~/Documents/battingiq-phase0
cd ~/Documents/battingiq-phase0 && ./phase0_mac_run.sh
```

If the clips are not at `~/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:`, pass the folder as the first argument. The script:

1. Sets up Python 3.10 in `.venv310/`. If there is no Python 3.10, it stops and says `brew install python@3.10`.
2. Puts ffmpeg on `PATH`. Without it, every clip silently falls back to pose-only contact.
3. Pre-flight: runs the unit tests and the reference-clip e2e test, checks that all 18 clips are found, and checks the Viv md5.
4. Runs the batch twice: auto + validated, then an auto repeat. 17 clips, local only, no Railway, no cached results.
5. Runs the left-hander check, the baseline table, the clip-sheet measured columns and pose cache, and the wrong-plane test.
6. Commits the results to `baselines/phase0_<date>/`, plus `clip_sheet.csv`, and pushes. Only CSV, JSON and markdown are committed. No video, frames or landmarks.

It stops with `STOP: <reason>` at the first failed check. It was rehearsed end to end in the cloud on 18 copies of the reference clip.

## Then Alex (about 20 minutes)

Fill the judgement columns of `clip_sheet.csv`: `player_id` (P01, …), `handedness`, `feed`, `line`, `venue`, `camera_position`, `others_in_frame` and `flags_confirmed`. Keep real names, ages and junior status in `player_key_PRIVATE.csv`, which git ignores. Commit and push.

## Findings so far (13–14 Nov, from the code and the reference clip)

- No active rule reads `reference_baseline.json`. Only suspended S1 and F3, the measurement export and the development-notes text read it.
- The batch never passed `handedness`. On the reference clip, re-running with `handedness=left` changes nothing in the four active rules. The Mac run checks the real left-hander.
- The T2, S4 and F4 thresholds were revised on 14 May against tier labels on these same 17 clips. No batch result existed for the four-rule set until now.
- The reference clip holds 129 frames in 3.69 s of container time (about 35 fps), but reports 30 fps. Frame-based timings can be about 15% off on such files. The clip sheet flags every clip with a frame-rate mismatch.
- Wrong-plane, reference clip only (n = 1, not a result): the forearms at hands peak (F4, ratio 0.34) and the shoulder and hip lines after contact (S4, 0.48) are strongly foreshortened. A2 is foreshortened (0.61); A4 and S1 are not (0.88). MediaPipe image z was too noisy to use, so the verdict rests on the 2D foreshortening ratio.
