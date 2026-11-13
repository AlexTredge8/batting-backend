# P0-C — Clip sheet: measured columns (Codex) plus judgement columns (Alex)

Run after P0-A. Paste everything below the line into a new Codex thread, in the `~/Documents/battingiq-phase0` clone, on branch `phase0-baseline`.

---

## Goal

Make `clip_sheet.csv` for all 18 clips. Codex fills the columns that the files can measure. Alex fills the columns that need a person who knows the clips. Also save a pose cache that P0-D reuses.

## Privacy rule (the repo is public)

Several clips appear to show junior players. The committed sheet holds pseudonymous player IDs only (`P01`, `P02`, …). Real names, ages, junior status and consent records stay in Alex's private folder, in `player_key_PRIVATE.csv` next to the clips. **Never commit that file.** Add `*_PRIVATE.csv` to `.gitignore`.

## Tasks

1. For each of the 18 clips in the clip folder (the same path as in P0-A), measure:
   - `ffprobe` data: `duration_s`, `avg_fps`, `nb_frames`, `codec`, `width` and `height` after rotation, `rotation`, `bitrate_kbps`, `audio_streams`, `audio_codec`, `audio_sample_rate`;
   - the `md5` of the file;
   - from MediaPipe, with `pose_extractor.extract_poses`, which is the production path: `pose_detection_rate`, and `batter_height_px`, the median over the 11 frames that end at the validated `setup_frame` in `anchor_truth.csv` of (lower ankle y − nose y) × frame height. For `Female_Average.mp4`, which has no anchor row, use the first 15 detected frames and add the flag `no_anchor_row`. Also give `batter_height_pct`, the same value as a percentage of frame height.

   Save the per-frame landmarks (x, y, z, visibility for all 33 landmarks) to `calibration_output/pose_cache/<stem>.npz`, together with the fps and the frame size. This folder is git-ignored. Do not commit it.
2. Write `clip_sheet.csv` at the repo root. Use the columns below in this order. Fill the measured columns. Leave the Alex columns blank, apart from the pre-fills shown.

| Column | Who | Notes |
|---|---|---|
| `filename`, `tier_csv`, `coach_overall`, `split` | Codex | from the coach CSV and `heldout_split.csv` |
| measured columns from task 1 | Codex | |
| `player_id` | Alex | `P01`…; the same batter gets the same ID |
| `scorer_in_clip` | Alex | `Alex`, `Cam`, `Viv` or blank; nobody may score themselves in Phase 2 |
| `handedness` | Alex | pre-fill `left` for `Cam T_LH Offdrive Elite.mov`; confirm the rest |
| `drive_type` | Alex | off, straight or on; pre-fill from the filename where it names one |
| `feed` | Alex | machine, bowler, throw-down or other |
| `line` | Alex | off, middle or leg |
| `venue` | Alex | indoor net lane, sports hall or outdoor nets |
| `camera_position` | Alex | behind the bowler, or angle (estimate the degrees) |
| `others_in_frame` | Alex | Y or N |
| `flags_hub` | Codex | pre-fill from the list below |
| `flags_confirmed` | Alex | Y, N or corrected text |

   `flags_hub` pre-fill (a first reading from one frame per clip; for Alex to confirm):
   - `Viv P_Offdrive Elite.mov`: same file as the reference clip
   - `Alex T_Straightdrive Good.MOV`: under 2 s
   - `Ollie P_Offdrive Average.mov`: no audio; two figures; small; black bars
   - `Drive_Average.MOV`: angle; others in shot
   - `Female_Average.mp4`: small; filmed_correctly = N
   - `Drive_Beginner 2.MOV`: angle; others in shot
   - `Drive_Beginner 3.MOV`: angle; under 2 s
   - `Drive_beginner.MOV`: angle; under 2 s; 42 frames
   - `Straight Drive_Beginner.mp4`: small

   Add `under 2 s` wherever `duration_s` < 2. Add `small` wherever `batter_height_pct` < 25.
3. Write `clip_sheet_notes.md`. Compare the measured batter heights with the hub's one-frame estimates (Cam T_LH 1,400; Cam T_Offdrive Elite 215; Cam T_Ondrive 1,000; Viv 260; Alex T_Straight 290; AlexT_Ondrive 270; Cam T_Offdrive Good 380; Ollie 140; Dan 235; Drive_Average 2 215; Drive_Average 370; Female 145; Drive_Beginner 2 390; Drive_Beginner 3 510; Drive_Beginner 4 295; Drive_beginner 470; Oscar 235; Straight Drive_Beginner 135). List every clip where the two differ by more than 25%.

## Constraints

- Read-only on the pipeline. No change to any existing file except `.gitignore`.
- No clips, frames or landmarks leave the Mac.
- Commit `clip_sheet.csv`, `clip_sheet_notes.md` and the `.gitignore` line as `P0-C: clip sheet (measured columns)`. Do not push.

## Deliverable

- `clip_sheet.csv`, inline.
- The height comparison.
- Any clip whose fps, rotation or audio differs from the P0-A `baseline_table.csv`.

## Then Alex

Fill the Alex columns in `clip_sheet.csv` (about 20 minutes). Keep names in `player_key_PRIVATE.csv`. Commit as `P0-C: clip sheet (Alex columns)`.
