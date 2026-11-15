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
| M06 | C06 | F06 | stability | Smallest front-knee angle, 150 ms before to 100 ms after contact | lower |
| M07 | C07 | F07 | access | Shoulder-line width 100 ms before contact over its width at set-up | lower |
| M08 | C08 | F08 | access | Largest sideways bow in the top hand's path, top of backlift to contact | lower |
| M09 | C09 | F09 | flow | Longest pause of the top hand (under 15% of peak speed) from the start of the swing to contact, ms | lower |
| M10 | C10 | F10 | tracking | Head towards the off side relative to the hips at contact | higher |
| M11 | C11 | F11 | stability | Sideways gap between hips and front foot, 400 to 700 ms after contact | lower |
| M12 | C12 | F12 | flow | Highest the top hand gets above the shoulders, 150 to 700 ms after contact | higher |

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
