# Capture standard and filming gate (Phase 3)

Status: draft, 16 Nov 2026. The plan (Section 5) says the gate defines what footage is "allowed". This page records the two filming checks added on 16 Nov 2026, what they mean for users, and what they mean for the Phase 5 worse-filming test (test 2). The other gate checks are in `phase3_gate.py` (version 0, provisional thresholds).

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
| Phone level | the batter's standing axis at set-up more than 3 degrees from vertical (`CAPTURE_MAX_TILT_DEG = 3.0`) | "The phone looks tilted: the batter's standing line at set-up is … degrees off upright in the picture, which moves the sideways measurements. Hold the phone level, or rest it on something flat. If the batter was really leaning to one side, ignore this." |

The standing axis is the line from the midpoint of the two ankles to the midpoint of the two shoulders, median over 1000 to 700 ms before contact, in pixels (`metrics_calculator.standing_axis_tilt_deg`). The Phase 5 measurements use the same estimate (v7). The response's `analysis_quality.capture` block now carries `short_side_px` and `setup_axis_tilt_deg`.

Why these two: on the Phase 5 worse-filming test (`baselines/phase5_2026-11-16_v6/degrade_test.md`), a 360 px copy changed nearly every measurement by more than a quarter of the spread between clips, and a 3 degree roll broke every sideways measurement (M02, M04, M10, M11, M15, M16, M18). A 3 degree roll shifts the head against the feet by about 5% of body length.

## What "allowed" footage is

A clip under 720 px on its short side, or with the batter's standing axis more than 3 degrees off vertical, is **outside allowed footage**: the app warns, and the measurements are not held to give the same answer on it. Today that is a warning, so such clips are still scored; when the gate moves from warnings to refusals, these are the clips it refuses.

What follows for Phase 5 test 2:

- The registered **lowres** variant (`scale=-2:360`: 360 px tall) is refused footage. All 18 calibration clips are portrait, so 360 px tall is about 204 px on the short side, far under the 720 px minimum. A measurement failing on it shows that refused footage is unreliable, which is why it is refused; it does not show that the measurement fails on allowed footage.
- The **tilt** variant (3 degrees) sits exactly at the edge of allowed footage (the warning starts above 3 degrees), so it remains a fair test.
- The **lowfps** variant (half the frame rate) is not covered by either check and remains a fair test.

The proposal to re-register test 2 with a variant at the gate's minimum is in `docs/phase5/README.md` (decision for Alex, 16 Nov 2026). Until Alex agrees, the 360 px variant stays the registered test.

## Known limits (read before these become refusals)

- **A leaning batter looks like a tilted phone.** From the pose alone the two cannot be told apart. On the reference clip (Viv P_Offdrive Elite), the camera is level (the long vertical edges in the background, 31 lines, have a median tilt of 0.0 degrees), yet the batter's standing axis at set-up is 7.0 degrees off vertical (shoulders towards the off side of the feet: his head sits over off stump). The tilt warning fires on it. That is a false alarm on a correctly filmed elite clip, so the 3 degree threshold should not become a refusal until the spread of `setup_axis_tilt_deg` on the 18 clips (written by the v7 run, `checkpoint_measurements.csv`) has been looked at, and perhaps not without a background-based check of the phone's roll.
- **Several calibration clips are under 720 px themselves.** By the clip sheet, 7 of the 18 are under 720 px on the short side (Viv P_Offdrive Elite, Cam T_Offdrive Elite and Cam T_Offdrive Good at 480; Oscar V_Beginner at 480; Female_Average and Straight Drive_Beginner at 464; Ollie P_Offdrive Average at 384). They get the resolution warning. This fits the plan's finding that 8 of the 18 clips would fail the input gate: they serve the coach ceiling but are not a fair test of the system.
- The reference clip (`test_batting.mov`, 480 x 848) gets both warnings. Its score is unchanged (92, Excellent).
