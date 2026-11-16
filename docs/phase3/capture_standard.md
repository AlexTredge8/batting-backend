# Capture standard and filming gate (Phase 3)

Status: draft, 16 Nov 2026 (tilt check changed to the camera roll from the background later the same day). The plan (Section 5) says the gate defines what footage is "allowed". This page records the two filming checks added on 16 Nov 2026, what they mean for users, and what they mean for the Phase 5 worse-filming test (test 2). The other gate checks are in `phase3_gate.py` (version 0, provisional thresholds).

## The standard (what we ask players to do)

- Phone fixed on a stand or propped up, not hand-held.
- Behind the bowler's stumps, in line with them, at about chest height.
- **Phone held level**: upright, not at an angle.
- **Back camera, normal video at 1080p** (720p at the least). No slow motion until it has been tested.
- Whole body in the picture from set-up to the end of the follow-through.
- Sound on. One ball per clip.
- Upload the original video from the camera roll, not a copy sent through a messaging app (these are usually shrunk).

## Two filming checks added 16 Nov 2026

Both are **warnings**, not refusals: the clip is still scored and the score is unchanged. Code: `run_analysis._capture_checks`. Thresholds: `config.py`.

| Check | Threshold (`config.py`) | What the user sees |
|---|---|---|
| Resolution | under 720 px on the short side (`CAPTURE_MIN_SHORT_SIDE_PX = 720`) | "This video was filmed at low resolution (… pixels on its short side), so body positions are measured less precisely. Use the phone's normal camera at 1080p (720p at the least) and upload the original video, not a copy sent through a messaging app." |
| Phone level | the camera roll, from the upright lines in the background, more than 3 degrees (`CAPTURE_MAX_TILT_DEG = 3.0`); no warning when the roll is unknown | "The phone looks tilted: upright lines in the background lean … degrees in the picture, which moves the sideways measurements. Hold the phone level, or rest it on something flat." |

The camera roll (`capture_geometry.camera_roll_deg`): 5 frames from the first 40% of the clip are re-read (chosen among frames with a pose), the batter is masked out (his pose box padded by 20%), and the long straight lines (Canny edges, `cv2.HoughLinesP`, each refined by a least-squares fit; at least 15% of the frame height, within 20 degrees of vertical) give the roll as their length-weighted median angle from vertical. It is "ok" with at least 6 distinct lines whose spread (IQR) is at most 3 degrees, otherwise "unknown" and the check is silent. Near-horizontal lines (crease, floor edges) are used only when there are too few upright ones, and the response says so (`roll_source: horizontal_fallback`). It costs about 0.2 s on a 480 px clip and 0.4 s at 1080p. The Phase 5 measurements use the same estimate (v8: rotated by the roll only when it is "ok").

The response's `analysis_quality.capture` block carries `short_side_px`, `camera_roll_deg` (null unless "ok"), `roll_confidence`, `roll_lines`, `roll_source`, and `setup_axis_tilt_deg`. The last is the batter's standing axis at set-up (ankle midpoint to shoulder midpoint, 1000 to 700 ms before contact, `metrics_calculator.standing_axis_tilt_deg`), kept for information only: it holds the batter's lean as well as any roll.

**Why not the batter's standing axis (the first version, 16 Nov 2026).** The first tilt check used the standing axis. From the pose alone a leaning batter and a tilted phone are the same thing: on the reference clip (Viv P_Offdrive Elite) the camera is level, but the batter's axis is 7.0 degrees off vertical at set-up (his head over off stump, which is coached), so the warning fired on a correctly filmed elite clip. With the roll from the background it does not: the reference clip reads +0.2 degrees ("ok", 17 lines), and the live analysis gives only the low-resolution warning (score unchanged, 92, Excellent).

Why these two: on the Phase 5 worse-filming test (`baselines/phase5_2026-11-16_v6/degrade_test.md`), a 360 px copy changed nearly every measurement by more than a quarter of the spread between clips, and a 3 degree roll broke every sideways measurement (M02, M04, M10, M11, M15, M16, M18). A 3 degree roll shifts the head against the feet by about 5% of body length.

## What "allowed" footage is

A clip under 720 px on its short side, or with a camera roll of more than 3 degrees (from the background, confidence "ok"), is **outside allowed footage**: the app warns, and the measurements are not held to give the same answer on it. Today that is a warning, so such clips are still scored; when the gate moves from warnings to refusals, these are the clips it refuses.

What follows for Phase 5 test 2:

- The registered **lowres** variant (`scale=-2:360`: 360 px tall) is refused footage. All 18 calibration clips are portrait, so 360 px tall is about 204 px on the short side, far under the 720 px minimum. A measurement failing on it shows that refused footage is unreliable, which is why it is refused; it does not show that the measurement fails on allowed footage.
- The **tilt** variant (3 degrees) sits at the edge of allowed footage (the warning starts above 3 degrees), so it remains a fair test. On the reference clip the estimator reads it at +3.45 degrees (the original at +0.24: probably perspective, see below), so there it would just get the warning.
- The **lowfps** variant (half the frame rate) is not covered by either check and remains a fair test.

The proposal to re-register test 2 with a variant at the gate's minimum is in `docs/phase5/README.md` (decision for Alex, 16 Nov 2026). Until Alex agrees, the 360 px variant stays the registered test.

## Known limits (read before these become refusals)

- **The roll needs upright lines in the background.** A plain wall, an open field or a net with no long straight edges gives "unknown", and then there is no tilt check (and Phase 5 does not rotate the clip). How often that happens will be seen in `roll_confidence` on the 18 clips (the v8 run). Until then the 3 degree threshold should not become a refusal.
- **Perspective.** A phone pitched up or down makes upright lines away from the centre of the picture lean, and no roll removes that. On the reference clip all the lines are in the left fifth of the picture and the estimate has an offset of about 0.4 degrees (+3 and -3 degree copies read +3.45 and -2.61).
- **The bowler is not masked.** Only the batter is; the bowler running through the first 40% of the clip adds a few lines, which the median and the spread rule are meant to absorb.
- The first version's caveat (a leaning batter looks like a tilted phone) no longer applies to the warning; it still applies to `setup_axis_tilt_deg`, which is why that is reported for information only.
- **Several calibration clips are under 720 px themselves.** By the clip sheet, 7 of the 18 are under 720 px on the short side (Viv P_Offdrive Elite, Cam T_Offdrive Elite and Cam T_Offdrive Good at 480; Oscar V_Beginner at 480; Female_Average and Straight Drive_Beginner at 464; Ollie P_Offdrive Average at 384). They get the resolution warning. This fits the plan's finding that 8 of the 18 clips would fail the input gate: they serve the coach ceiling but are not a fair test of the system.
- The reference clip (`test_batting.mov`, 480 x 848) gets the resolution warning only (it got both with the first, standing-axis version). Its score is unchanged (92, Excellent).
