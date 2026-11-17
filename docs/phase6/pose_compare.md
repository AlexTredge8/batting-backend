# You vs best practice (pose comparison)

Code: `pose_reference.py` (builds the reference), `pose_compare.py` (the comparison and the pictures). Tests: `tests/test_pose_compare.py`. Mac command: `pose_reference_build_mac.sh`. Frontend prompt: `docs/lovable/pose_compare_prompt.md`.

**Status (17 Nov 2026): built and tested, switched OFF.** The live app is unchanged until Alex says go. The reference figure in the repo is **provisional**: it is built from the one bundled reference clip (`test_batting.mov`), so it is not yet an average. Alex rebuilds it from all the Elite clips on his Mac (below) before it is switched on.

## What it does

For the player's priority fix, it shows them their own body position at the moment that matters, with a "best practice" stick figure laid over it, in the style of golf apps such as GolfFix.

1. **Choose the fault.** The same steps the results page uses to choose its one priority fix (`docs/lovable/results_tips_and_sharing_prompt.md`, section 3). Today the live app names a rule, not a rubric fault, so the rule is mapped as a **stopgap**: T2 → F01, A5 → F07, S4 → F12, F4 → F09 (the four rules that run today; the dormant rules follow the frontend's table). If there is no priority fix, the pillar that lost the most points gives the fault. If no fault was found anywhere, it shows "nothing major to fix" (F00).
2. **Choose the moment and the joints.** Each fault has a key moment, the joints that matter, the point the figures are lined up on, and the Phase 5 measurement that belongs to it (table below).
3. **Normalise the player** as the reference was normalised:
   - left-handers mirrored, so they read as a right-hander;
   - the camera roll removed (from the background's upright lines);
   - the origin is the middle of the ankles at set-up;
   - the unit is the body length (nose to ankle at set-up).
4. **Line up the reference on the player.** The figure is scaled to the player's own body length and placed on the player's own set-up ankles. It is shifted so the fault's alignment point matches (for example the hips for "head over the ball"). Then it is turned back into the picture as filmed (roll put back, mirror undone), so the ghost lands on the player's real frame.
5. **Measure.** The fault's Phase 5 measurement is computed with the same code as the measurement table (`phase5_checkpoints.py`, v8). It is compared with the range of good drives: the 10th to 90th percentile of the Elite and Good Club clips, validated key frames. Only the worse side counts as "outside": a value better than the whole range is "inside".
6. **Draw two pictures:**
   - **Overlay:** the player's own frame with the face blurred, their skeleton in orange, and the reference as a translucent white ghost. Each joint that differs by more than 5% of their height has a yellow ring and an arrow to where the reference has it. The picture is cropped around the two figures and is 480 px wide.
   - **Side by side:** the two stick figures on a plain background, at the same scale, with the same joints ringed.
7. **One sentence on what to change**, from a per-fault template. For example: "At contact your head is 9% of your height to the off side of your hips; good drives are between 16% and 31% of your height to the off side of your hips. Keep your head over the ball and meet it close to your front pad."

It never changes a score or any existing report field. Any error is caught and recorded in the block, and the analysis carries on.

## Faults, moments and joints

| Fault | Moment | Joints that matter (first = main) | Lined up on | Measurement |
|---|---|---|---|---|
| F01 head not still at set-up | set-up | head | set-up ankles | M01 |
| F02 unbalanced stance | set-up | hips, head | set-up ankles | M02 |
| F17 head, hands, feet not lined up | set-up | head, hands, both ankles | set-up ankles | M18 |
| F13 no rhythm / trigger | set-up | both ankles, hands | set-up ankles | M17 |
| F03 hands out away from the body | top of backlift | hands, elbows | shoulders | M03 |
| F19 ball too short to drive | none | none | none | not available: shot choice, not body position |
| F15 moves across, not out | front foot down | front ankle, head, front knee | back ankle | none (pose only) |
| F04 head and shoulder do not lead | front foot down | head, front shoulder | front ankle | M04 |
| F05 front foot away from the line | front foot down | front ankle, head, hips | back ankle | M05 |
| F06 front leg stiff | contact | front knee, front hip, front ankle | front ankle | M06 |
| F18 back foot stays put | contact | back knee, back ankle, back hip | front ankle | none (pose only) |
| F16 centre of mass stays back | contact | hips, shoulders | front ankle | M16 |
| F07 front shoulder opens early | 100 ms before contact | shoulders, hips | hips | M07 |
| F14 (retired, now F07) | as F07 | as F07 | as F07 | M07 |
| F08 bat across the line | contact | hands, elbows | shoulders | M08 |
| F09 pause or hitch | top of backlift | hands, front elbow | shoulders | M09 |
| F10 head falls away at contact | contact | head, front shoulder | hips | M10 |
| F11 off balance at the finish | finish | hips, head | front ankle | M11 |
| F12 hands stop at contact | finish | hands, elbows | shoulders | M12 |
| F00 nothing major to fix | contact | none | hips | none |

The moments are anchored on contact wherever possible, because automatic contact is the most accurate key frame:
- **set-up:** the median over 1000 to 700 ms before contact; the frame shown is 850 ms before contact;
- **top of the backlift:** the top hand's highest point 450 to 100 ms before contact;
- **front foot down:** the key frame;
- **100 ms before contact;**
- **contact;**
- **finish:** 550 ms after contact.

A joint that is the alignment point is never marked (it matches by construction). At most three joints are marked.

## Privacy

- **The reference is an anonymised average stick figure.** `reference/pose_reference_v1.json` holds only:
  - normalised joint coordinates (body lengths from the set-up ankle midpoint), each the median across clips;
  - the measurement ranges (10th percentile, 90th percentile and median, with the number of clips).

  It holds no names, no clip names, no frames or images, and no per-clip values. A test checks this, and the Mac command refuses to push a file that contains a clip name.
- **It is only anonymous with enough clips.** With fewer than 3 clips the figure is close to one batter's own, so the file says `"anonymised": false, "provisional": true`. The repo copy is like that today: one clip, the bundled reference clip, which is already in the repo. Rebuild it on all the Elite clips before any player sees it.
- **The player's own frame is shown only to them.** It is part of their own result, like the storyboard. The face is blurred before anything is drawn: a pixelated, blurred oval sized from the shoulders and ears, as in the offline coach pack. If the head points are not seen in that frame, it still blurs where the model places the head, rather than show a face. No picture is stored by this feature beyond the report itself.
- No paid service is used. Everything runs in the analysis on the server.

## How to switch it on

1. **Rebuild the reference on the Mac** (one command, under a minute; Elite clips only):
   ```
   cd ~/Documents/battingiq-phase0 && git pull && ./pose_reference_build_mac.sh
   ```
   - To add the Good Club clips to the figure: `./pose_reference_build_mac.sh --include-good-club`.
   - It uses `calibration_output/pose_cache`, `anchor_truth.csv`, `clip_sheet.csv` and the newest Phase 5 table for the current measurement set.
   - It commits and pushes only `reference/pose_reference_v1.json`. It should say `anonymised=True, provisional=False`.
2. **Ask Alex.** The figure comes from Elite clips, and the ranges from Elite and Good Club clips, all on validated key frames. Alex should look at the three example pictures from the rebuilt reference before it goes live.
3. **Turn it on:** set the environment variable `POSE_COMPARE=on` on the server (Railway → the service → Variables), or change the default in `config.py`. To turn it off again, set `POSE_COMPARE=off` or remove the variable. Off is the default: with it off the report has no `pose_comparison` key at all.
4. Then add the card to the app with `docs/lovable/pose_compare_prompt.md`. The card hides itself when the field is missing.

Cost: about 0.1 s and one extra frame read per analysis; the two JPEGs add about 50 to 75 KB of base64 to the report.

## JSON for the frontend: `report.pose_comparison`

Present only when `POSE_COMPARE=on`. Two shapes.

**Available:**

```json
{
  "version": "pose_compare_v1",
  "available": true,
  "nothing_to_fix": false,
  "fault": "F10",
  "fault_text": "Head falls away at contact, or the ball is met away from the pad",
  "tip": "Keep your head over the ball and meet it close to your front pad, so you stay in control and keep it on the ground.",
  "source_rule": "T1",
  "fault_source": "stopgap: live rule T1 mapped to rubric fault F10",
  "moment": "contact",
  "moment_label": "Contact",
  "frame": 82,
  "timestamp_ms": 2733.3,
  "measurement": {
    "id": "M10",
    "name": "Head over the ball at contact",
    "value": 9.42,
    "unit": "% BL, off side +",
    "unit_text": "% of your height",
    "direction": "higher_better",
    "reference_range": [16.247, 30.958],
    "reference_median": 28.715,
    "reference_n_clips": 8,
    "status": "outside",
    "position": "below"
  },
  "joints": [
    {"name": "nose", "label": "head", "player_xy": [0.5126, 0.2851], "reference_xy": [0.4163, 0.2868],
     "difference_pct_height": 22.3, "highlighted": true},
    {"name": "front_shoulder", "label": "front shoulder", "player_xy": [0.47, 0.32], "reference_xy": [0.4705, 0.3201],
     "difference_pct_height": 0.2, "highlighted": false}
  ],
  "change_text": "At contact your head is 9% of your height to the off side of your hips; good drives are between 16% and 31% of your height to the off side of your hips. Keep your head over the ball and meet it close to your front pad.",
  "alignment": {"align_on": "hips", "shift_body_lengths": [0.0, 0.0], "body_length_px": 209.3, "mirrored": false,
                "camera_roll_removed_deg": 0.24},
  "reference": {"version": "pose_reference_v1", "provisional": true, "anonymised": false, "n_clips": 1, "status": "PROVISIONAL: ..."},
  "privacy": {"face_blurred": true, "note": "The player's own frame, face blurred; the reference is an averaged stick figure."},
  "images": {
    "mime_type": "image/jpeg",
    "overlay_data_url": "data:image/jpeg;base64,...",
    "overlay_size": [480, 674],
    "side_by_side_data_url": "data:image/jpeg;base64,...",
    "side_by_side_size": [480, 394]
  }
}
```

Field notes:
- `measurement` is `null` for a pose-only fault (F15, F18) and for F00. `measurement.status` is `"outside"` (worse than good drives), `"inside"` (within the range, or better than it) or `"unknown"` (the value could not be measured; then `value` is `null` and `blank_reason` says why). `position` is `"below"`, `"within"` or `"above"` the range, for drawing the bar.
- `reference_range` is `[low, high]` in the measurement's own unit. `unit_text` is the plain-English unit for display. `% BL` is shown as "% of your height", where height means nose-to-ankle at set-up.
- `player_xy` and `reference_xy` are fractions of the video frame's width and height, as filmed (0 to 1, top-left origin), at `frame`.
- `images.overlay_data_url` is `null` if the video frame could not be read. The side-by-side picture is always there when `available` is true.
- `reference.provisional === true` means the reference is not yet an average of enough clips. Both pictures then carry the words "Reference: provisional".
- `nothing_to_fix: true` (F00): `joints` is empty, `measurement` is `null`, and `change_text` is the F00 tip. The pictures show the player with the ghost at contact and nothing marked.

**Not available:**

```json
{"version": "pose_compare_v1", "available": false, "fault": "F19", "reason": "This is about choosing which ball to drive, not about body position, so there is no pose to compare."}
```

Reasons include:
- shot choice (F19);
- the reference has not been built;
- the set-up is not in the clip (it needs 1 s before contact);
- no pose at the key frame;
- key frames missing;
- a live rule with no fault mapped.

An unexpected failure adds `"error"`. The frontend shows nothing in all of these cases (or, at most, `reason` in small print).

## Known limits

- **Provisional reference.** One clip today, so the ghost is that batter's figure. The Mac rebuild fixes this.
- **The ranges are wide.** They come from 8 clips (Elite and Good Club). So "inside" is common, and the sentence then says so plainly rather than inventing a fault. The ranges come from the v8 measurement table (`baselines/phase5_2026-11-16_v8`), which matches the live measurement code. The brief named the v6 table; v8 is the same measurements with camera-roll levelling, which is what the live comparison computes.
- **The fault mapping is a stopgap.** A5 → F07 and S4 → F12 are the nearest rubric faults, not the same measurements. When the live app names rubric faults itself (`priority_fault`), that is used first.
- **The live app does not know the drive type.** M05 and M08 are read as off drives.
- **The 2D picture from behind the bowler** cannot see depth along the pitch. The comparison is about sideways and up-and-down positions, the same limits as the Phase 5 measurements.
