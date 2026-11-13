# P0-E — What the production upload delivers to the backend (Alex, about 20 minutes, no Codex)

## Why

All calibration ran on the original phone files. Production receives what the Lovable frontend uploads. If that upload drops the audio track, contact detection falls back to pose-only contact (about 30 frames of error against about 2 frames) and every other anchor follows. If it scales or re-encodes the video, landmark noise rises. Validation must run on what production sees.

## Steps

1. Use `AlexT_Ondrive Good.MOV`. It is your own clip, so no consent issue, and it has a P0-A row to compare with. Before you upload it, note its file size, duration, resolution and fps (Finder, Get Info).
2. Upload it through the live Lovable app on your phone, the way a user would. If you have time, also upload it from a desktop browser.
3. In the browser developer tools (Network tab), find the `POST /analyse` request:
   - **Request**: the uploaded part's filename and content type (`.webm`? `.mp4`?) and its size. Also check whether a `handedness` field is sent.
   - **Response JSON**: copy `analysis_quality` (look for `audio_available`, `contact_method`, `processing_mode`), `warnings`, and from `metadata`: `width`, `height`, `fps`, `total_frames`, `duration_s`, `detection_rate`. Copy `battingiq_score`, the four pillar scores and `handedness_source`.
4. In the Railway logs for that minute, copy the line that starts `[analyse] job=`. It gives the file size in MB, the suffix and the probe result.
5. Work out the bitrate: file MB × 8 ÷ duration, in Mbit/s.

## Bring back

The size, resolution, fps, duration and audio values, from the original file and from the uploaded file, side by side, plus the Railway log line and the scores.

I will compare them with the P0-A local auto row for this clip:

- Audio lost: a launch blocker, fixed in the frontend.
- Resolution or fps changed: Phase 3 sets the capture standard on the uploaded form, not the original.
- Score differs from the local run while the inputs match: production drift. That needs one Codex check.
