# Phase 5: checkpoint measurements

One measurement per rubric checkpoint (`docs/phase1/rubric_v0.json`), each tied to the fault a tip would name. Code: `phase5_checkpoints.py`. Entry tests: `phase5_entry_tests.py`. Unit tests: `tests/test_phase5_checkpoints.py`.

## Rules for every measurement

- Time windows in milliseconds, converted with the clip's frame rate. No fixed frame counts.
- Windows anchored on contact wherever possible. Automatic contact is out by a median 1 frame; automatic set-up by 6 and follow-through by 8.
- Distances in body lengths (BL: nose-to-ankle height from 1000 to 700 ms before contact), shown as % BL.
- Left-handers are mirrored, so every clip reads as a right-hander. After the mirror the off side is camera-left. The feet confirm this on the reference clip (toes point camera-left at set-up); `toes_point_off = no` flags a clip where the handedness or camera position is probably wrong.
- The hands are the top (front) wrist. From behind the bowler the bottom wrist is hidden most of the time (visibility median 0.12 on the reference clip).
- Points under 0.2 visibility are dropped; positions are smoothed with a 5-frame moving median.
- A blank value always carries a reason.

## The twelve

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

"Better is" was registered on 14 Nov 2026 before any clip other than the reference clip was measured. The entry tests check it; they do not choose it.

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
