# Phase 0: baseline and freeze

Run these in order. Each Codex prompt goes in its own thread. Bring the deliverable back to the hub chat before you start the next one.

| # | What | Who | Est. time | Depends on |
|---|---|---|---|---|
| P0-A | [Four-rule baseline, auto + validated, 17 clips](P0-A_baseline_prompt.md) | Codex on the Mac | about 1.5 h, mostly unattended | — |
| P0-B | [Docs match code; RULES_VERSION 2.2.0](P0-B_docs_match_code_prompt.md) | Codex | 30 min | P0-A |
| P0-C | [Clip sheet: measured columns, then Alex's columns](P0-C_clip_sheet_prompt.md) | Codex + Alex | 45 min + 20 min | P0-A |
| P0-D | [Wrong-plane hypothesis test](P0-D_wrong_plane_prompt.md) | Codex | 1 h | P0-A, P0-C |
| P0-E | [Production upload check (audio, resolution, bitrate)](P0-E_upload_path_check.md) | Alex, no Codex | 20 min | P0-A |

Phase 0 is done when the repo holds the dated baseline table, the clip sheet covers all 18 clips, and the docs and the code name the same four rules (A5, T2, S4, F4).

## Checked in the hub before these were written (13 Nov 2026, at `50e48da`)

- `pytest tests` gives 44 passed and 5 skipped. The e2e reference test passes, with anchors 60/61/77/73/81/109.
- A dual-mode dry run on `test_batting.mov`, renamed as the Viv clip, works with the flags in P0-A. Auto and validated both score 92 (coach 94). Access is 17 (A5 fires) and the other pillars are 25 with no fault.
- The batch never passes `handedness`, so the left-hander has always been scored as a right-hander. The four active rules use handedness-independent quantities. P0-A checks this.
- No active rule reads `reference_baseline.json`. Only suspended S1, the measurement export and the development-notes text read it.
- `audio_contact.py` finds ffmpeg only on `PATH`. `imageio-ffmpeg` does not put it there.
- `calibration_output/` is git-ignored, so P0-A commits its table to `baselines/`.
- The `tier` column in the coach CSV differs from the filename for two clips: Ollie P is `Good Club` and Straight Drive_Beginner is `Average`.
