# Phase 5: checkpoint measurements

One measurement per rubric checkpoint (`docs/phase1/rubric_v0.json`), each tied to the fault a tip would name. Code: `phase5_checkpoints.py`. Entry tests: `phase5_entry_tests.py`. Unit tests: `tests/test_phase5_checkpoints.py`.

## Rules for every measurement

- Time windows in milliseconds, converted with the clip's frame rate. No fixed frame counts.
- Windows anchored on contact wherever possible. Automatic contact is out by a median 1 frame; automatic set-up by 6 and follow-through by 8.
- Distances in body lengths (BL: nose-to-ankle height from 1000 to 700 ms before contact), shown as % BL.
- Left-handers are mirrored, so every clip reads as a right-hander. After the mirror the off side is camera-left. The feet confirm this on the reference clip (toes point camera-left at set-up); `toes_point_off = no` flags a clip where the handedness or camera position is probably wrong.
- The hands are the top (front) wrist. From behind the bowler the bottom wrist is hidden most of the time (visibility median 0.12 on the reference clip).
- Points under 0.2 visibility are dropped; positions are smoothed with a 5-frame moving median.
- From v8, every landmark is rotated by minus the camera roll, estimated from the long upright lines in the background with the batter masked out (`capture_geometry.camera_roll_deg`), and only when that estimate's confidence is "ok"; otherwise nothing is rotated. So a phone held off level does not move the sideways measurements, and a batter's real lean is kept. Each row records `camera_roll_deg`, `roll_confidence`, `roll_source`, `roll_lines`, `levelled_by`, and the batter's standing axis at set-up (`setup_axis_tilt_deg`, for information only). v7 levelled to the batter's standing axis instead, which also removed a real lean (see v8 below).
- A blank value always carries a reason.

## The measurements

| Meas. | Checkpoint | Fault | Pillar | What it measures | Better is |
|---|---|---|---|---|---|
| M01 | C01 | F01 | tracking | Head movement (SD) from 1000 to 700 ms before contact (the set-up) | lower |
| M02 | C02 | F02 | stability | Sideways offset of hips from the middle of the feet at set-up | lower |
| M03 | C03 | F03 | access | Top hand's sideways travel towards gully, relative to the shoulders, set-up to top of backlift (highest point 450–100 ms before contact) | lower |
| M04 | C04 | F04 | tracking | Head towards the off side relative to the front foot at foot-down | higher |
| M05 | C05 | F05 | access | Front foot's sideways step towards the drive's side (on drives: leg side) | higher |
| M06 | C06 | F06 | stability | Front-knee angle around contact (10th percentile of a smoothed series, ±200 ms) | lower |
| M07 | C07 | F07 | access | Shoulder-line width 100 ms before contact over its width at set-up | lower |
| M08 | C08 | F08 | access | Hands across the line: top hand's sideways travel away from the drive's side, relative to the shoulders, 200 ms before to 100 ms after contact | lower |
| M09 | C09 | F09 | flow | Longest pause of the top hand in the 450 ms before contact (smoothed speed under 15% of its 90th percentile), ms, timed between frames | lower |
| M10 | C10 | F10 | tracking | Head towards the off side relative to the hips at contact | higher |
| M11 | C11 | F11 | stability | Sideways gap between hips and front foot, 400 to 700 ms after contact | lower |
| M12 | C12 | F12 | flow | Highest the top hand gets above the shoulders, 150 to 700 ms after contact | higher |
| M13 | C12 | F12 | flow | Swing carries through: fastest upward hand speed after contact over fastest downward speed before it (vertical only) | higher |
| M14 | C09 | F09 | flow | Front-foot landing against the start of the downswing (Weissensteiner 2011): gap in ms between the front ankle stopping (vertical speed under 15% of its peak, 450 ms before contact to contact) and the top hand's highest point (450–100 ms before contact); stored as the absolute gap | lower (closer to zero) |
| M15 | C01 | F01 | tracking | Head over the base at set-up (Radford): sideways offset of the nose from the middle of the two ankles, 1000 to 700 ms before contact; stored as the absolute offset | lower |
| M16 | C16 | F16 | stability | Weight over the front foot at contact: sideways gap between the hip-shoulder midpoint and the front ankle, median over contact ±70 ms | lower |
| M17 | C13 | F13 | flow | Trigger (Taliep 2007): largest sideways movement of either ankle from 1000 to 450 ms before contact, before the stride | higher |

"Better is" was registered on 14 Nov 2026 before any clip other than the reference clip was measured. For M14–M17 it was registered on 15 Nov 2026 before any clip, the reference clip included, was measured with them. The entry tests check it; they do not choose it. C13 and C16 are Cam's checkpoints in rubric v0.2 (`docs/phase1/rubric_v0_2.json`): test 1 for M16 and M17 needs checkpoint ratings with C13 and C16 columns, and until then reads "not enough ratings".

## Entry tests (thresholds fixed 14 Nov 2026)

0. Coverage: a value on at least 80% of clips.
1. Tracks the coaches: Spearman at least 0.30 with the checkpoint rating, on at least 10 rated clips. Model ratings make this provisional; it is final only on human coach ratings.
2. Two-phone: median difference between two phones filming the same ball at most 0.25 of the between-clip spread (IQR). Pending the Phase 3 net session.
3. Automatic key frames: Spearman at least 0.80 with the validated-key-frame values, and median gap at most 0.25 of the IQR.

A pillar needs two passing measurements to be scored; otherwise it shows "not assessed".

## Run it on the clips (Mac)

```
cd ~/Documents/battingiq-phase0 && git pull && ./phase5_mac_run.sh
```

It reuses the Phase 0 pose cache, writes `baselines/phase5_<date>/`, and pushes only the tables.

## What the coaches need to supply

Test 1 needs checkpoint ratings (2 / 1 / 0 / X per checkpoint) from Cam and Alex on the same clips. The model's ratings stand in until then. The quick check gives the main fault per clip, which is a cross-check, not a substitute.

## Log

**v1, run 15 Nov 2026 (`baselines/phase5_2026-11-15/`).** No measurement passed. Two causes, neither a verdict on the coaching idea:

- Test 1 could not discriminate. On most checkpoints the model gave 11–13 of its 14 clips the same rating, so its ratings cannot confirm or reject anything. From v2, test 1 needs two rating levels with 3+ clips each, otherwise "not decided". This makes the test stricter. The v1 table is kept as it was.
- Test 3 failed for every measurement keyed to set-up, hands-start-up, hands-peak or follow-through (M01, M03, M06, M08, M09, M11, M12), and passed every one keyed to contact or foot-down (M02, M04, M05, M07, M10). The plan says to anchor on contact; v1 did not. The automatic follow-through is out by a median 8 frames.

**v2, 15 Nov 2026.** The seven failed measurements are re-anchored on contact, with windows set from the validated key frames (set-up a median 964 ms before contact, hands start 467 ms, top of backlift 231 ms, follow-through 560 ms after). Body length and the feet check use the same contact-anchored set-up window. Directions ("better is") and thresholds are unchanged. The five that passed test 3 are unchanged in definition.

Not changed after seeing the data: the direction of any measurement. On the v1 run, against Alex's pillar scores, M03, M04 and M06 ran opposite to the registered direction (rho −0.60, −0.49, −0.49). Flipping them on these 17 clips would fit noise and the venue/camera confound (Addendum A). They are tested as registered, on human checkpoint ratings, and then on new clips.

**v2 result, 15 Nov 2026 (`baselines/phase5_2026-11-15_v2/`).** Nine of twelve now hold up with automatic key frames (v1: five). Still failing that test: M06 front knee (rho 0.58), M08 downswing path (−0.30) and M09 pause (0.63): all three are dropped from v2 as specified and need a redesign. Robust and awaiting coach checkpoint ratings: access M03, M05, M07; tracking M01, M04, M10; stability M02, M11; flow M12. Flow has one candidate, so it cannot reach the two the plan needs without a new flow measurement. The verdict column now treats a model-only disagreement on test 1 as provisional ("model disagrees"), not as a drop: model ratings cannot drop a measurement, as they cannot pass one.

**v3, 15 Nov 2026.** Rework, not removal: Alex asked for failing measurements to be improved rather than dropped. Each change targets the automatic-key-frame test (consistency), not agreement with coaches, so it does not fit the coach scores.

- M06: the single smallest knee angle was set by one bad pose frame on three beginner clips (e.g. 133° against 170°). Now the 10th percentile of a 3-frame-median series over ±200 ms.
- M08: v2's bow was measured from a detected top of backlift that moved several frames when contact moved one. Now a fixed contact window: hands' sideways travel against the shoulders, signed away from the drive's side (across the line).
- M09: v2 values came in whole frames (0, 33, 67 ms...), so ranks jumped; and on the reference clip one bad pose frame at contact (797 px/s against <200) set the peak and doubled the pause. Now 3-frame-median speed, 90th-percentile peak, crossings timed between frames, and stillness already under way when the window opens is not counted as a pause.
- M13 (new, flow): the swing carries through contact. A first version (total hand speed at contact) read ~0 on the reference clip because the hands travel towards the camera at contact (wrong plane); replaced before any clip run with the vertical down-and-up, which this camera sees.
- Geometry from the feet: on four clips (Cam T_Offdrive Elite, Cam T_Offdrive Good, Cam T_Ondrive Elite, Dan P_Offdrive Average) the toes point the wrong way for a right-hander filmed from the bowler's end, so the "top hand" was the hidden bottom hand. When the feet disagree with the clip sheet, the clip is now mirrored and sides swapped (`geometry_source`). Cause still to confirm with Alex: left-handed batter, mirrored video, or camera behind the batter.

**v3 result, 15 Nov 2026 (`baselines/phase5_2026-11-15_v3/`).** 12 of 13 hold up with automatic key frames (v2: 9). M06 and M08 are fixed by the rework. Still failing: M09 (rho 0.30: three beginner clips jump between 0 and 120–145 ms) and M13 on coverage (76%: blank on exactly the four clips the feet flipped).

**v4, 15 Nov 2026.** M09 becomes total slow time in the 400 ms before contact, timed between frames, so a one-frame shift moves it by at most one frame's worth (v3's longest-run rule was all-or-nothing at the window edge). "The hands" become the top (front) wrist unless the other wrist is clearly more visible across the clip. This removes the dependence on left/right labels, which a mirrored or behind-the-batter clip can swap. Directions and thresholds unchanged.

**Key frames (`baselines/phase5_anchor_candidates_2026-11-15/`).** On the 17 clips, with automatic contact and the gap taken from the other clips: contact + typical gap beats the current detectors for hands-start-up, foot-down, hands-peak and follow-through (within 3 frames: 71/76/76/47% against 59/59/53/18%). For set-up, the local pose refinement is best (mean error 6.6 frames, 43% within 3, against 7.9 and 29%). Not shipped: it changes the live T2/F4 measurements, so it needs a before/after batch.

**Handedness confirmed (15 Nov 2026).** Alex confirmed that the four clips the feet flagged (Cam T_Offdrive Elite, Cam T_Offdrive Good, Cam T_Ondrive Elite, Dan P_Offdrive Average) are left-handed batters. `clip_sheet.csv` now says so. The feet check found all four with no false alarms on the other 14, which supports the live feet-based handedness (rules 2.3.1). Phase 0 scored these four as right-handers; the four active rules gave identical results either way on the left-handed clip checked then. Phase 5 v1 and v2 measured them with the wrong hand; v3 onwards is correct.

**v5, registered 15 Nov 2026, before any clip was run with these definitions.** Four measurements added from the research in `docs/phase1/coaching_sources.md`. M01–M13 are unchanged in definition, direction and ID. The definitions, windows and directions below were written into `phase5_checkpoints.py` and this log, and committed, before the reference clip or any other clip was measured with them; the synthetic test figure is the only thing they had been run on.

- M14 (C09 / F09, flow), front-foot landing against the start of the downswing. This is the best-supported skill marker: skilled batters land the front foot as the downswing starts (Weissensteiner et al. 2011; Sarpeshkar et al. 2017). Both events are detected near contact without the legacy detectors. The landing is the first moment in the 450 ms before contact, after the front ankle's peak 3-frame-median vertical speed, when that speed falls below 15% of the peak; it is timed between frames. The downswing start is the top hand's highest point 450 to 100 ms before contact (the same rule as M03). Signed, downswing start minus landing is negative when the hands start down before the stride is complete. Registered direction: closer to zero is better, so the value stored is the absolute gap in ms, with higher worse.
- M15 (C01 / F01, tracking), head over the base at set-up. Radford puts the head over off stump or middle-and-off, with the eyes inside off. Measured as the nose's sideways offset from the middle of the two ankles, median over the set-up window, in % BL, with the off side positive. The entry tests do not allow an unknown sign, so the registered direction is higher worse on the absolute offset: the head should be over the base. The signed offset stays in the code (`_head_offset_setup`) for later analysis; it is not tested.
- M16 (C16 / F16, stability), weight over the front foot at contact. Measured as the sideways gap between the hip-shoulder midpoint (the mean of both hips and both shoulders, standing in for the centre of mass) and the front ankle, median over contact ±70 ms, in % BL. Higher is worse.
- M17 (C13 / F13, flow), trigger present. Taliep et al. 2007 found a trigger in 70% of skilled and 20% of less-skilled batters. Measured as the largest sideways movement (range) of either ankle from 1000 to 450 ms before contact, before the stride, in % BL. Higher is better.

The entry-test thresholds are unchanged. `phase5_entry_tests.py` now reads any checkpoint column (C01–C16), so ratings on rubric v0.2 reach M16 and M17. `MEASUREMENT_SET` is now v5.

**v5 on the reference clip, 15 Nov 2026 (after the registration above; no definition changed).** Viv P_Offdrive Elite, validated key frames (automatic in brackets): M14 blank, "front foot still moving at contact" (blank with automatic key frames too); M15 17.3% BL (17.1); M16 1.3% BL (1.4); M17 2.9% BL (2.2). M01–M13 are unchanged from v4. Three things to watch on the full run, logged now so they are not read as fitting later:

- M14: from behind the bowler the front ankle moves only about 10 px up and down over the whole stride (about 5% BL), and its image height keeps creeping down until about one frame before contact as the weight goes on. Its vertical speed never drops under 15% of the peak before contact, so the landing is not found. The validated foot-down is 5 frames before contact. A version that centres the 3-frame median at the window's edge finds a landing at frame 81.3, 4 frames after the validated foot-down; it goes blank again when contact moves one frame earlier. It was tried on this clip only and **not** adopted. If M14 is blank on most clips it fails coverage (test 0), and a redesign will need its own registration before any clip run. On this clip the front knee stops dropping at frame 77, the validated foot-down; that is a lead from one clip, not evidence.
- M15: the elite reference batter's head is 17% BL to the **off** side of the middle of his feet at set-up. That fits Radford's "head over off stump" with the feet on middle or leg, so a fixed off-side offset may be the coached position rather than a fault. The registered direction (absolute offset, higher worse) stands and the entry tests will judge it. If it fails test 1, this is the likely reason, and the signed offset (`_head_offset_setup`) is the version to register next.
- M17: on this clip the window (1000 to 450 ms before contact) catches the start of the stride. The front ankle starts moving about 530 ms before contact, so most of the 2.9% is stride, not trigger. Research puts the stride start at about 580 ms on average, so on many clips the window's end will overlap the stride.

**v6, 16 Nov 2026 (registered before any clip run).** Added M18 (C17/F17, tracking, lower is better). It measures how well the head, hands and feet line up sideways at set-up: the spread of the nose, the hands and the ankle midpoint, 1000 to 700 ms before contact, as % BL. It follows Alex's correction of the Radford point: "hands, feet and head level over off stump; it is about line". The stumps are not tracked, so M18 measures the three lining up with each other, not their position against off stump. M15 (head over the base) is unchanged.

**v7, registered 16 Nov 2026, before the 18 clips were run with it.** Motivated by the v6 worse-filming result (`baselines/phase5_2026-11-16_v6/degrade_test.md`): the 3 degree tilt variant broke every sideways measurement (M02, M04, M10, M11, M15, M16, M18; change 0.43 to 0.98 of the spread between clips). That is expected: a 3 degree roll shifts the head against the feet by about 5% BL. This is a robustness change for test 2, not a change to fit the coaches. IDs, definitions, directions and thresholds are unchanged.

- The change: before any measurement (body length included), every landmark is rotated about the median ankle midpoint of the set-up window so that the batter's standing axis at set-up is vertical (`level_clip` in `phase5_checkpoints.py`). The axis runs from the ankle midpoint to the shoulder midpoint, median over 1000 to 700 ms before contact (the contact-anchored set-up window, `_setup_window`). It is applied after the left-hander mirror; the result is the same either way round, as the axis is mirrored with the picture. If the axis cannot be found, the clip is measured as it is and `setup_axis_tilt_deg` is blank. `MEASUREMENT_SET` is now v7. Setting `LEVEL_TO_STANDING_AXIS = False` reproduces v6.
- Unit tests: a synthetic figure rolled by +3 or -3 degrees gives the same M02, M04, M10, M11, M15, M16 and M18 as the level figure (within 0.3% BL; in practice the same to 3 decimals), where v6 moved them by 1.8 to 5.3% BL; a rolled left-hander still mirrors the right-hander exactly.
- What was seen before registering, stated so it is not read as fitting later: the axis angle alone (not the measurements) was computed on the reference clip, at -7.0 degrees, and the tilt variant's at -4.1 (so the variant adds 2.9 degrees, as it should). The reference camera is level (the long vertical edges in the background have a median tilt of 0.0 degrees), so the -7.0 is the batter's own lean: shoulders over the off side of the feet at set-up, his head over off stump. **So v7 also removes a real sideways lean at set-up.** From the pose alone a lean and a rolled phone are the same thing. M02, M15 and M18 were partly measuring that lean, so v7 changes what they read, not only how steadily they read it.

**v7 on the reference clip, 16 Nov 2026 (after the registration above; no definition changed).** Viv P_Offdrive Elite, validated key frames, the tilt variant made locally with `phase5_degrade_test.py` (production pose path). "Change" is |tilt - original| as a share of the v6 between-clip IQR (one clip, so this is a check, not test 2):

| Meas. | v6 original | v6 tilt | v6 change | v7 original | v7 tilt | v7 change |
|---|---|---|---|---|---|---|
| M02 | 6.1 | 9.4 | 0.68 | 13.0 | 13.4 | 0.09 |
| M04 | 17.6 | 13.4 | 0.50 | 6.1 | 6.7 | 0.08 |
| M10 | 31.6 | 28.5 | 0.60 | 27.0 | 25.7 | 0.25 |
| M11 | 4.5 | 7.5 | 1.07 | 11.2 | 11.4 | 0.04 |
| M15 | 17.3 | 12.4 | 0.46 | 4.8 | 5.1 | 0.03 |
| M16 | 1.3 | 5.4 | 0.95 | 10.1 | 10.4 | 0.08 |
| M18 | 17.3 | 14.7 | 0.23 | 9.5 | 11.0 | 0.13 |

The roll is removed: all seven are under 0.25 on this clip, M10 only just (0.247); in v6 only M18 was (0.23). The other measurements change little (M13 0.57 to 0.39; M01, M03, M05, M07, M08 and M17 at 0.08 or under either way; M06, M09 and M12 unchanged). Three things to watch on the full run:

- The values move a long way on this clip because its 7 degree lean is rotated out: M15 (head over the base) from 17.3 to 4.8% BL, M16 from 1.3 to 10.1, M04 from 17.6 to 6.1. The between-clip spread may shrink as well, which would raise the change/spread share even where the change itself is smaller.
- The axis is estimated from the pose, so pose noise now reaches every sideways measurement. On the 360 px copy the axis came out at -5.1 degrees against -7.0, and M02, M11, M16 and M18 moved more under v7 than v6 (change/v6 IQR 0.91 to 2.07, 0.08 to 0.76, 0.94 to 2.21, 0.17 to 0.49). On the half-frame-rate copy everything is under 0.25 with v7 except M09 (1.51, unchanged from v6) and M13 (blank on that copy); v6 also had M04 at 0.30.
- `setup_axis_tilt_deg` per clip shows how much real lean there is across the 18 clips. If batters differ by several degrees, v7 removes a real difference between them.

**Filming gate and test 2 (16 Nov 2026).** The plan says the gate defines what footage is "allowed". Two filming checks were added to the live analysis as warnings (`docs/phase3/capture_standard.md`): under 720 px on the short side, and the batter's standing axis at set-up more than 3 degrees off vertical. Clips failing either (after the warning) are outside allowed footage. The registered lowres variant (360 px tall: about 204 px on the short side of these portrait clips) is therefore refused footage, and failing on it says the footage is unreliable, not that the measurement is. The tilt variant (3 degrees) sits at the edge of allowed footage and lowfps is not covered by the gate, so both stay fair tests. Caveat: on the reference clip the tilt check fires (-7.0 degrees, a real lean with a level camera), so the 3 degree threshold needs the v7 spread of `setup_axis_tilt_deg` before it can become a refusal.

**Decision for Alex (proposed 16 Nov 2026, not adopted).** Re-register test 2's lowres variant at the gate's minimum, 720 px on the short side, in place of 360 px tall. It is implemented behind a flag, `phase5_degrade_test.py --lowres-height`; the default stays 360 for the record, and nothing changes without Alex's yes. Because all 18 clips are portrait, 720 px on the short side is `--lowres-height 1280` (`--lowres-height 720` would give about 406 px, still refused). The variant is named `lowres1280` with its own pose cache, and never scales a clip up: 7 of the 18 clips are already under 720 px on the short side (all under 1280 tall), so for them the variant is the same size, compressed harder (CRF 32). Options for those 7: keep them (a compression-only test) or leave them out of test 2 as outside allowed footage. The same pass rule applies (median change at most 0.25 of the IQR). If Alex says yes, the command is:

```
cd ~/Documents/battingiq-phase0 && git pull && PATH="$PWD/.venv310/bin:$PATH" .venv310/bin/python phase5_degrade_test.py --video-dir "$HOME/Documents/01 Work/03 BattingIQ/03 Working Folders/battingiq-groundtruth:/videos:" --out baselines/phase5_$(date +%F)_v7_lowres1280 --variants lowres --lowres-height 1280
```

**v8, registered 16 Nov 2026, before the 18 clips were run with it.** Motivated by the v7 finding above that the batter's standing axis contains real lean: on the reference clip the camera is level but the batter's axis is -7.0 degrees at set-up (his head over off stump, which is coached), so v7 removed real technique (M15, head over the base, went from 17.3 to 4.8% BL) and the live tilt warning was a false alarm. v8 estimates the phone's roll from the background instead of from the batter. IDs, definitions, directions and thresholds of M01 to M18 are unchanged; only the levelling changes.

- The roll (`capture_geometry.camera_roll_deg`, version `roll_v1`): 5 frames evenly spaced over the first 40% of the clip; the batter masked out (his pose bounding box in that frame, padded by 20% on each side) and a 2% border; Canny edges and `cv2.HoughLinesP`, each segment refined by a least-squares fit to its edge pixels; long lines only (at least 15% of the frame height) within 20 degrees of vertical. The roll is the length-weighted median angle from vertical (positive = the top of an upright line is camera-right, the same sign as `setup_axis_tilt_deg`). Confidence is "ok" with at least 6 distinct lines (segments of the same edge, in one frame or across frames, count once) and a length-weighted IQR of at most 3 degrees; otherwise "unknown" and no roll is given. Only if there are fewer than 6 upright lines, near-horizontal lines (crease, floor edges; at least 15% of the frame width, within 20 degrees of horizontal) are used with the same rule, and `roll_source` says `horizontal_fallback`: perspective tilts horizontal lines that are not square to the camera, so it is the weaker estimate.
- The change: before any measurement (body length included), every landmark is rotated about the frame centre by minus the roll, only when its confidence is "ok"; otherwise the clip is measured as filmed (as in v6). It is applied after the left-hander mirror, with the roll's sign flipped for a mirrored clip; `camera_roll_deg` in the table is in that measured picture, so `setup_axis_tilt_deg - camera_roll_deg` is the batter's own lean. `MEASUREMENT_SET` is now v8. `LEVEL_MODE = "standing_axis"` reproduces v7 and `"none"` reproduces v6 (`measure_clip(..., level=...)` per call).
- The roll needs the video. `phase5_checkpoints.py --video-dir <clips>` reads it from `<stem>.roll.json` next to each pose cache, or computes it from the clip and writes that file; with neither, the clip is not rotated and `roll_source` is `none`. `phase5_mac_next.sh` passes the clip folder. The worse-filming test passes `--video-dir` for the originals and for the variants, so each video is levelled by the roll found in its own picture: the tilt variant by what the estimator finds (about +3 degrees), not by the 3 degrees the test put in.
- Unit tests (`tests/test_capture_geometry.py`, `tests/test_phase5_checkpoints.py`): drawn pictures rolled by 0, +3, -3 and +1.5 degrees are recovered within 0.3 degrees; a leaning "batter" inside the mask does not move the roll; too few lines, or lines that disagree, give "unknown". A synthetic figure rolled by +-3 degrees with that roll gives the same sideways measurements as the level figure (within 0.05% BL); a leaning figure with a level camera keeps its v6 values exactly (v7 moved them by more than 2% BL); an "unknown" roll rotates nothing; a rolled left-hander still mirrors the right-hander.
- What was seen before registering, stated so it is not read as fitting later. The estimator was built on the reference clip and two rotated copies made locally with `phase5_degrade_test.make_variant` (the registered +3 degree tilt variant, and the same with -3). With Hough endpoints only it gave 0.00, +3.46 and -2.49 degrees. Hough endpoints sit on whole pixels, so long edges a fraction of a degree off upright snapped to exactly 0; the least-squares refinement was added after seeing this, and gave **+0.24 (17 lines, IQR 0.59), +3.45 (10 lines) and -2.61 (7 lines)**, all "ok": the copies move by +3.21 and -2.85 degrees against the original. Nothing else was tuned on these numbers. On this clip every line found is in the left fifth of the picture (x 10 to 105 px of 480); the remaining offset of about 0.4 degrees (both copies sit 0.2 to 0.4 degrees to the positive side) is probably perspective: a phone pitched slightly up or down makes upright lines away from the centre lean, which no roll removes. The measurement values below were looked at only after the estimator was fixed.

**v8 on the reference clip, 16 Nov 2026 (after the registration above; no definition changed).** Viv P_Offdrive Elite, validated key frames, production pose path on each copy. "Change" is |copy - original| as a share of the v6 between-clip IQR (one clip, so a check, not test 2). The +3 copy is the registered tilt variant; the -3 copy is a local extra check, not part of test 2:

| Meas. | v6 original | v8 original | v8 +3 | v8 -3 | v6 change +3 | v7 change +3 | v8 change +3 | v8 change -3 |
|---|---|---|---|---|---|---|---|---|
| M02 | 6.1 | 5.9 | 6.0 | 6.2 | 0.68 | 0.09 | 0.02 | 0.05 |
| M04 | 17.5 | 17.9 | 19.1 | 18.5 | 0.50 | 0.08 | 0.15 | 0.07 |
| M10 | 31.6 | 31.7 | 30.8 | 31.2 | 0.60 | 0.25 | 0.18 | 0.10 |
| M11 | 4.5 | 4.2 | 4.2 | 3.4 | 1.07 | 0.04 | 0.02 | 0.29 |
| M15 | 17.3 | 17.7 | 18.5 | 18.2 | 0.46 | 0.03 | 0.08 | 0.05 |
| M16 | 1.3 | 1.0 | 1.3 | 0.7 | 0.95 | 0.08 | 0.07 | 0.05 |
| M18 | 17.3 | 17.7 | 18.5 | 18.2 | 0.23 | 0.13 | 0.08 | 0.05 |

- The batter's lean is kept: on the original the v8 values are within 0.5% BL of the v6 values (the clip is rotated by only 0.24 degrees), e.g. M15 17.7 against 17.3, where v7 read 4.8.
- The roll is removed: on the registered +3 copy all seven sideways measurements are at 0.18 or under (v6: 0.23 to 1.07). On the -3 copy all are at 0.10 or under except M11 at 0.29.
- The others: M01, M05, M07, M08 and M17 at 0.07 or under on both copies; M03 0.13 and 0.26; M06 0.12 and 0.08; M12 0.16 and 0.23; M09 0.18 and 0.61 (a speed, which no rotation changes: pose noise on the copies, the same in v6 and v7); M13 0.68 and 0.88 (vertical hand speed; v6 0.57 and 0.88, v7 0.39 on the +3 copy). M14 is blank on this clip.
- `setup_axis_tilt_deg` is -7.05 on the original, -4.13 on the +3 copy and -10.04 on the -3 copy; minus the roll, the batter's own lean reads -7.3, -7.6 and -7.4 degrees.

**Filming gate after v8 (16 Nov 2026).** The live tilt warning (`run_analysis._capture_checks`) now uses the same camera roll (5 frames re-read, the batter masked by the pose; about 0.2 s on this 480 px clip and 0.4 s on a 1080p copy), with the same 3 degree threshold, and is silent when the roll is "unknown". On the reference clip the live analysis reads a roll of +0.2 degrees ("ok", 17 lines) and gives only the low-resolution warning; the score is unchanged (92, Excellent). The tilt variant (+3) reads +3.45 with the estimator, so it would get the warning: with the 0.4 degree perspective offset above, the registered tilt variant sits just outside the gate on this clip rather than exactly at its edge. See `docs/phase3/capture_standard.md`.

