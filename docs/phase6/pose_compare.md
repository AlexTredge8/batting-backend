# Fault card: cartoon pictures, your number, and your video (pose comparison)

Code: `silhouettes.py` (draws the cartoon pictures), `pose_reference.py` (builds the reference ranges), `pose_compare.py` (the comparison, the card data and the "On your video" picture). Pictures: `assets/silhouettes/`. Tests: `tests/test_silhouettes.py`, `tests/test_pose_compare.py`. Mac command: `pose_reference_build_mac.sh`. Frontend prompt: `docs/lovable/pose_compare_prompt.md`. Design spec: `docs/phase6/annotation_benchmark.md`.

**17 Nov 2026, version 2 (`pose_compare_v2`):** after Alex's review, the card no longer leads with a ghost skeleton drawn on the player. It leads with a fixed, exaggerated **cartoon pair** for the fault ("Not this" and "This", in the spirit of golf apps such as GolfFix, but our own cricket drawing), then the one-line change, then the player's number on a bar. The player's own frame is now a secondary "On your video" picture with **one** marker, behind a toggle. The side-by-side stick-figure picture is gone. Scores and every other report field are unchanged; with the switch off the report still has no `pose_comparison` key.

**Status: built and tested, switched OFF.** The live app is unchanged until Alex says go. The reference figure in the repo is **provisional**: it is built from the one bundled reference clip (`test_batting.mov`), so it is not yet an average. Alex rebuilds it from all the Elite clips on his Mac (below) before it is switched on.

## What it does

For the player's priority fix, it shows the principle with a fixed cartoon pair ("Not this" / "This"), one line on what to change, the player's own number against good drives, and, as an optional extra, their own frame with one marker.

1. **Choose the fault.** The same steps the results page uses to choose its one priority fix (`docs/lovable/results_tips_and_sharing_prompt.md`, section 3). Today the live app names a rule, not a rubric fault, so the rule is mapped as a **stopgap**: T2 → F01, A5 → F07, S4 → F12, F4 → F09 (the four rules that run today; the dormant rules follow the frontend's table). If there is no priority fix, the pillar that lost the most points gives the fault. If no fault was found anywhere, it shows "nothing major to fix" (F00).
2. **Choose the moment and the joints.** Each fault has a key moment, the joints that matter, the point the figures are lined up on, and the Phase 5 measurement that belongs to it (table below).
3. **Normalise the player** as the reference was normalised:
   - left-handers mirrored, so they read as a right-hander;
   - the camera roll removed (from the background's upright lines);
   - the origin is the middle of the ankles at set-up;
   - the unit is the body length (nose to ankle at set-up).
4. **Line up the reference on the player.** The figure is scaled to the player's own body length and placed on the player's own set-up ankles. It is shifted so the fault's alignment point matches (for example the hips for "head over the ball"). Then it is turned back into the picture as filmed (roll put back, mirror undone), so the reference lands on the player's real frame. (Since version 2 the reference figure is not drawn; the aligned joints are kept in the JSON as data.)
5. **Measure.** The fault's Phase 5 measurement is computed with the same code as the measurement table (`phase5_checkpoints.py`, v8). It is compared with the range of good drives: the 10th to 90th percentile of the Elite and Good Club clips, validated key frames. Only the worse side counts as "outside": a value better than the whole range is "inside".
6. **Attach the cartoon pair** for the fault (`silhouette`, below). It is a fixed picture, the same for every player, so it is shown even when the player's own comparison cannot be made (for example F19, or a clip with no set-up).
7. **Draw "On your video"** (secondary, behind a toggle in the app): the player's own frame with the face blurred, cropped to the fault area (head and shoulders, the legs, or the whole body), with only the fault's one key cue: a blue ring with a white halo on the main joint and, where the cartoon uses a reference line, the same dashed line (for example a plumb line down from the head for F10, the shoulder line for F07, a vertical line through the feet for F17). No ghost skeleton, no arrows, no text in the picture. 480 px wide.
8. **One sentence on what to change**, from a per-fault template (`change_text`), plus a short one-line instruction for the card (`what_to_change`). For example: "At contact your head is 9% of your height to the off side of your hips; good drives are between 16% and 31% of your height to the off side of your hips. Keep your head over the ball and meet it close to your front pad."

It never changes a score or any existing report field. Any error is caught and recorded in the block, and the analysis carries on.

## The cartoon pictures (silhouettes)

`silhouettes.py` draws a simple, friendly, gender-neutral cartoon batter (helmet with a visor band, no face; pads, gloves, bat; slate body with the far limbs one step lighter) in flat SVG shapes. Every figure is a **parameter set** (a few joint angles, the `Pose` class), not a hand drawing: the "Not this" figure is the "This" figure with one or two angles pushed about twice as far as a typical real fault. Nothing in the pictures comes from a player or a video; they contain no per-player data and no images of people.

- **Views** (from the benchmark's per-fault table): front-on as our camera sees it (from behind the bowler, off side on the left for a right-hander) for sideways faults; side-on (from point, bowler to the right, shown by a chevron at the ground's edge) for F06, F16, F18, F12, F09, F13; from above (footprints) for F15; a pitch diagram for F19.
- **One cue per picture:** a dashed ink line (plumb line from the head, the off-stump line, the shoulder line, the ball's line, the eye line), the body part that matters filled in the accent colour, a dotted trail on "Not this" (never an arrow), and one thick arrow on "This" only. No numbers or angle arcs.
- **Colour is never the only cue:** vermilion `#D55E00` for "Not this" and blue `#0072B2` for "This" (colour-blind safe), plus a cross / tick badge and the words "Not this" / "This". Each panel carries its own off-white background, so it reads on light and dark pages.
- **Files** in `assets/silhouettes/`: `<fault>_fault.svg` and `<fault>_correct.svg` (single panels, **no text**, so the app can label them and mirror them for a left-hander), `<fault>_pair.svg` (both panels with the badges, labels and captions baked in, for right-handers and for review), PNG copies of each (palette PNGs about 3x phone size, made with cairosvg at build time), `index.json` (`{fault: {view, moment, caption_fault, caption_correct, cue, alt, files}}`) and `contact_sheet.png` / `.svg` (all 20 pairs, for Alex). F00 has two "This" panels (`F00_correct.svg` at contact and `F00_correct_finish.svg` at the finish) and no fault panel; F14 points to F07.
- **Captions** are 2 to 6 words, from the rubric and `fault_tips_v0_3.json`.
- **Rebuild** after changing a pose: `python silhouettes.py` (add `--no-png` if cairosvg is not installed). A test fails if the committed SVGs differ from what the code draws.
- **Serving: inline, nothing to host.** The backend puts the two single panels into the report as SVG data URLs (about 8 KB each, smaller than today's JPEGs), the same way the pictures already arrive. This is the simplest route: no backend route, no CORS, no copying files into Lovable, and the pictures always match the backend's fault ids. The static files stay in `assets/silhouettes/` for anything else (for example a "fault library" page later: copy the folder into the Lovable project's `public/silhouettes/`).
- **Status: DRAFT.** Alex signs off every pair before it goes live (coaching content, like the drills). The contact sheet is the review sheet.

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
- **The cartoon pictures show no one.** They are drawn by code from joint angles; no player, no video frame and no real person is in them.
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
5. **Alex signs off the cartoon pairs** (`assets/silhouettes/contact_sheet.png`) before the card goes live.

Cost: about 0.1 s and one extra frame read per analysis; the "On your video" JPEG adds about 30 to 45 KB of base64 to the report and the two cartoon panels about 20 KB (version 1's two JPEGs added 50 to 75 KB).

## JSON for the frontend: `report.pose_comparison`

Present only when `POSE_COMPARE=on`. Two shapes.

**Available:**

```json
{
  "version": "pose_compare_v2",
  "available": true,
  "nothing_to_fix": false,
  "fault": "F10",
  "fault_text": "Head falls away at contact, or the ball is met away from the pad",
  "tip": "Keep your head over the ball and meet it close to your front pad, so you stay in control and keep it on the ground.",
  "what_to_change": "Keep your head over the ball and meet it close to your front pad.",
  "silhouette": {
    "version": "silhouettes_v1",
    "fault": "F10",
    "view": "front_on",
    "moment": "contact",
    "drawn_moment": "contact",
    "label_fault": "Not this",
    "label_correct": "This",
    "caption_fault": "Head falls away",
    "caption_correct": "Head over the ball",
    "cue": {"type": "line", "what": "vertical line from the head down to the ground",
            "fault": "the line misses the ball: ...", "correct": "the line runs through the ball, met next to the front pad"},
    "alt": "Two cartoon batters at contact, seen from the bowler's end. Not this: ... This: ...",
    "mirror": false,
    "pair_svg_file": "F10_pair.svg",
    "fault_svg_file": "F10_fault.svg",
    "correct_svg_file": "F10_correct.svg",
    "fault_svg_data_url": "data:image/svg+xml;base64,...",
    "correct_svg_data_url": "data:image/svg+xml;base64,..."
  },
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
     "difference_pct_height": 22.3, "highlighted": true}
  ],
  "change_text": "At contact your head is 9% of your height to the off side of your hips; good drives are between 16% and 31% of your height to the off side of your hips. Keep your head over the ball and meet it close to your front pad.",
  "alignment": {"align_on": "hips", "shift_body_lengths": [0.0, 0.0], "body_length_px": 209.3, "mirrored": false,
                "camera_roll_removed_deg": 0.24},
  "reference": {"version": "pose_reference_v1", "provisional": true, "anonymised": false, "n_clips": 1, "status": "PROVISIONAL: ..."},
  "privacy": {"face_blurred": true, "note": "The player's own frame, face blurred, with one marker; ..."},
  "images": {
    "mime_type": "image/jpeg",
    "overlay_label": "On your video",
    "overlay_data_url": "data:image/jpeg;base64,...",
    "overlay_size": [480, 640],
    "overlay_cue": {"ring": "nose", "line": {"type": "plumb", "joints": ["nose"]}}
  }
}
```

Field notes:
- `silhouette` is the fixed cartoon pair; the card leads with it. `fault_svg_data_url` and `correct_svg_data_url` are text-free single panels: the app shows them side by side with `label_fault` / `label_correct` and the captions as its own text, and flips both horizontally when `mirror` is `true` (a left-hander). `pair_svg_file` names the same pair with the words baked in (right-handers only; for review or a static page). `view` is `front_on`, `side_on`, `top_down` or `pitch_side`; when it is not `front_on` the app adds a small "Seen from the side" (or "From above") tag, because the player's video is front-on. For F00, `fault_svg_data_url` is `null`, and `correct_finish_svg_data_url` with `caption_correct_finish` is a second "This" panel. For F14, the F07 pair is sent with `"shown_for": "F14"`. `silhouette` is `null` only if the asset files are missing.
- `what_to_change` is the one-line instruction under the pair. `change_text` is the longer sentence with the player's number.
- `measurement` is `null` for a pose-only fault (F15, F18) and for F00. `measurement.status` is `"outside"` (worse than good drives), `"inside"` (within the range, or better than it) or `"unknown"` (the value could not be measured; then `value` is `null` and `blank_reason` says why). `position` is `"below"`, `"within"` or `"above"` the range, for drawing the bar.
- `reference_range` is `[low, high]` in the measurement's own unit. `unit_text` is the plain-English unit for display. `% BL` is shown as "% of your height", where height means nose-to-ankle at set-up.
- `player_xy` and `reference_xy` are fractions of the video frame's width and height, as filmed (0 to 1, top-left origin), at `frame`. They are data only; the reference figure is no longer drawn.
- `images.overlay_data_url` is the "On your video" picture; `null` if the video frame could not be read. `images.overlay_cue` says what was drawn on it (`ring`: the joint ringed, or `null`; `line`: `plumb`, `vertical` or `segment` with its joints, or `null`). The picture has no words in it; `overlay_label` is the label for the app to show.
- `reference.provisional === true` means the reference ranges are not yet built from enough clips. This is no longer stamped on any picture; the coach view and logs can say so.
- `nothing_to_fix: true` (F00): `joints` is empty, `measurement` is `null`, `change_text` is the F00 tip, the cartoon pair shows the clean drive at contact and at the finish, and the video picture has no marker.

**Not available** (the player's own comparison could not be made). The cartoon pair, the tip and `what_to_change` still come with it whenever the fault is known, so the card can still teach the principle:

```json
{"version": "pose_compare_v2", "available": false, "fault": "F19",
 "reason": "This is about choosing which ball to drive, not about body position, so there is no pose to compare.",
 "fault_text": "Drives a ball that is too short to drive", "tip": "...", "what_to_change": "Drive only when the ball is pitched up.",
 "silhouette": {"fault": "F19", "view": "pitch_side", "...": "..."}}
```

Reasons include:
- shot choice (F19);
- the reference has not been built;
- the set-up is not in the clip (it needs 1 s before contact);
- no pose at the key frame;
- key frames missing;
- a live rule with no fault mapped (then there is no fault, so no `silhouette`).

An unexpected failure adds `"error"` (and no `silhouette`). The frontend shows the card whenever `silhouette` is present, with only the parts that came; with no `silhouette` it shows nothing.

## Known limits

- **Provisional reference.** One clip today, so the ranges are from that set-up. The Mac rebuild fixes this. The reference figure is no longer drawn for players.
- **The cartoons are drafts.** Drawn procedurally from the benchmark's spec, not yet signed off by Alex. The side-on pictures (F06, F16, F18, F12, F09, F13) do not match the player's front-on video, which the "Seen from the side" tag says.
- **The ranges are wide.** They come from 8 clips (Elite and Good Club). So "inside" is common, and the sentence then says so plainly rather than inventing a fault. The ranges come from the v8 measurement table (`baselines/phase5_2026-11-16_v8`), which matches the live measurement code. The brief named the v6 table; v8 is the same measurements with camera-roll levelling, which is what the live comparison computes.
- **The fault mapping is a stopgap.** A5 → F07 and S4 → F12 are the nearest rubric faults, not the same measurements. When the live app names rubric faults itself (`priority_fault`), that is used first.
- **The live app does not know the drive type.** M05 and M08 are read as off drives.
- **The 2D picture from behind the bowler** cannot see depth along the pitch. The comparison is about sideways and up-and-down positions, the same limits as the Phase 5 measurements.
