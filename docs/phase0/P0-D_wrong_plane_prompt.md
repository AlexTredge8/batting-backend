# P0-D — Test the wrong-plane hypothesis

Run after P0-A and P0-C. Paste everything below the line into a new Codex thread, in the `~/Documents/battingiq-phase0` clone, on branch `phase0-baseline`.

---

## Goal

Find out whether the suspended measurements A2, A3, A4, S1 and S3 fail because they measure movement toward the camera. The camera is at the bowler's end, so the image shows width and height but not depth along the pitch. Run the same test on the four active measurements (A5, T2, S4, F4) as comparators. Measurement only.

## Inputs

- Pose cache: `calibration_output/pose_cache/<stem>.npz`, from P0-C. If it is missing, extract the poses with `pose_extractor.extract_poses`.
- Validated anchors: `anchor_truth.csv` (17 clips). Use the validated anchors only, so that detection error does not mix with geometry.
- Measurement values and coach scores: `baselines/phase0_<date>/raw/validated/rule_measurements.csv`, and `coach_ground_truth_from_screenshot.csv`.
- Handedness and camera position: `clip_sheet.csv` (P0-C). If the Alex columns are still blank, run anyway and mark the camera-position split as pending.

## What each measurement looks at

Take the frames and landmarks from the code, not from this list. `coaching_rules.collect_rule_measurements` is the source of truth. My reading of it:

| Rule | Measurement | Landmarks | Frames |
|---|---|---|---|
| A2 | front elbow angle | front shoulder, elbow, wrist | contact window |
| A3 | compression frames | both wrists (vertical velocity) | contact − window to +15 |
| A4 | torso lean | shoulder midpoint, hip midpoint | contact window |
| S1 | hip shift | hip centre | setup → contact window |
| S3 | hip drift frames | hip centre, front ankle | backlift start → follow-through + 5 |
| A5 | shoulder–hip gap | both shoulders, both hips | contact window |
| S4 | post-contact rotation | both shoulders, both hips | contact → contact + S4 frames |
| T2 | early head change | nose, hip midpoint | setup and backlift windows |
| F4 | pause frames | both wrists (vertical velocity) | hands peak − 2 to + 4 |

## Indices (pre-registered; do not change them after you see the data)

Use pixel coordinates: x × width and y × height, after rotation.

1. **Foreshortening ratio**, for each segment a measurement uses (for example upper arm, forearm, torso, shoulder line, hip line): the 2D segment length at the measured frames, divided by the 95th-percentile 2D length of that segment from setup to follow-through. Take the median over the measured frames, then the median over the 17 clips. A low ratio means that the segment points into the camera.
2. **Depth share of motion**, for each landmark a measurement tracks: Σ|Δz| / (Σ|Δx| + Σ|Δy| + Σ|Δz|) over the measured frames. Use MediaPipe's normalised z, scaled by the frame width. Label this index as rough: MediaPipe's image z is an estimate.
3. **Agreement with coaches**: the Spearman ρ between the measurement (validated mode, 17 clips) and the coach score for its pillar, and between the measurement and the coach overall score. Give the 95% bootstrap interval (2,000 resamples, seed 0).

## Verdict per measurement

- **Supported**: the median foreshortening ratio is below 0.70, or the median depth share is 0.40 or more, AND |ρ| with the coach pillar score is below 0.30.
- **Not supported**: the foreshortening ratio is 0.85 or more AND the depth share is below 0.25.
- **Inconclusive**: everything else.

State that n = 17 makes all of this exploratory.

## Also report

- The indices split by `camera_position` (behind the bowler, or angle), as a description only. Tier is confounded with venue, so no test.
- For A5 and S4: `_shoulder_openness` is atan2(|dx|, |dy|). When the shoulders are side-on to a bowler's-end camera, both dx and dy are small, so the angle can be unstable. Report the median 2D shoulder-line length at contact in pixels, and its ratio to batter height (from `clip_sheet.csv`).

## Outputs

Write these to `baselines/phase0_<date>/wrong_plane/`:

- `wrong_plane_by_clip.csv`: clip × measurement, with every index;
- `wrong_plane_summary.csv`: one row per measurement, with the medians, ρ, the interval and the verdict;
- `wrong_plane_report.md`: one page. The verdict table comes first. Then one paragraph per measurement. Then the limits.

Put the analysis in a new script, `phase0_wrong_plane.py`. It must not import the scorer or the rules. It may import `metrics_calculator` and `pose_extractor` to rebuild the same quantities.

## Constraints

- No change to any existing file.
- No data leaves the Mac.
- Commit as `P0-D: wrong-plane test (A2, A3, A4, S1, S3 vs active rules)`. Do not push.

## Deliverable

`wrong_plane_report.md`, inline, and any measurement that you could not rebuild exactly from the code, with the reason.
