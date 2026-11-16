# Phase 3 clip check with Gemini (2026-11-16)

Model gemini-3.1-pro-preview, 8 frames/s, temperature 0, 2 runs per clip, thinking low, uncropped.
Requests sent 36 of 36; spent about GBP 0.25 (cap 0.80; tokens: prompt 85000, answer 5780, thinking 1010).
Clips answered 18, both runs 18; errors 0.

Agreement = share of answers (both runs) that match the ground truth. Consistency = share of clips where run 1 and run 2 gave the same answer. Fields whose only ground truth is the hub's 14 Nov guesses (feed, venue, camera position, others in frame) are **unconfirmed**: their agreement is with a guess, not a fact.
Reliable enough for an automatic check = agreement >= 0.90 and consistency >= 0.90; soft prompt = >= 0.75 and >= 0.85; at least 5 answers scored.

| field | ground truth | agreement | excl. hub guesses | consistency | verdict |
|---|---|---|---|---|---|
| shot_type | by construction (18) | 0.94 (n=36) | 0.94 (n=36) | 1.00 (n=18) | reliable enough for an automatic check; only one right answer in this set, so this measures false alarms only, not whether it catches the problem |
| drive_direction | filename (10); hub guess, not confirmed (8) | 0.39 (n=36) | 0.40 (n=20) | 1.00 (n=18) | not reliable |
| camera_position | hub guess, not confirmed (18) | 0.67 (n=36) | n/a (n=0) | 1.00 (n=18) | provisional (ground truth is a hub guess): not reliable |
| batter_handedness | confirmed (18) | 0.72 (n=36) | 0.72 (n=36) | 1.00 (n=18) | not reliable |
| feed | hub guess, not confirmed (18) | 0.56 (n=36) | n/a (n=0) | 1.00 (n=18) | provisional (ground truth is a hub guess): not reliable |
| venue | hub guess, not confirmed (18) | 0.94 (n=36) | n/a (n=0) | 1.00 (n=18) | provisional (ground truth is a hub guess): reliable enough for an automatic check |
| whole_batter_visible | none | n/a (n=0) | n/a (n=0) | 1.00 (n=18) | consistent, but accuracy unknown (no ground truth) |
| phone_level | none | n/a (n=0) | n/a (n=0) | 1.00 (n=18) | consistent, but accuracy unknown (no ground truth) |
| number_of_people_visible | hub guess, not confirmed (17); known (1) | 0.83 (n=36) | 1.00 (n=2) | 1.00 (n=18) | provisional (ground truth is a hub guess): use only as a soft prompt, not a block |
| multiple_shots_in_clip | by construction (18) | 0.94 (n=36) | 0.94 (n=36) | 1.00 (n=18) | reliable enough for an automatic check; only one right answer in this set, so this measures false alarms only, not whether it catches the problem |
| batter_size | measured (16) | 0.56 (n=32) | 0.56 (n=32) | 1.00 (n=18) | not reliable |
| usable_for_analysis | confirmed (16); known (2) | 0.83 (n=36) | 0.83 (n=36) | 1.00 (n=18) | use only as a soft prompt, not a block |

Answer counts and clips that disagreed with the ground truth are in field_summary.csv; every answer next to its ground truth is in comparison.csv.

## Notes on the ground truth

- camera_position: the hub's 'angle (~20-30 deg)' clips are expected to be answered 'other' (the prompt asks for 'other' when the camera is more than about 15 degrees off the line of the stumps).
- number_of_people_visible: compared as '1' or '2 or more'. Ollie P (two people) is a known fact; the rest use the hub's others_in_frame guess.
- batter_size: bands from the measured batter_height_pct (nose to ankle, so a little under full height): small < 20%, medium < 40%, large above. Gate v0 asks for at least 20%. Ollie P and Drive_Beginner 4 have no usable measurement and are left out.
- usable_for_analysis: 'no' for Ollie P (two people), Drive_Beginner 4 (poor pose detection) and Female_Average (coach: not filmed correctly); 'yes' for the clips the coach scored as filmed correctly.
- shot_type and multiple_shots_in_clip have one right answer for every clip (all are single front-foot drives), so they can show false alarms but not whether the model catches a wrong shot or a long clip.
- whole_batter_visible and phone_level have no ground truth: only consistency is measured.

## Recommendation

- Use in the live gate as an automatic check: none of the fields
- Use only as a soft prompt (ask the player to confirm, never block): number_of_people_visible, multiple_shots_in_clip, usable_for_analysis
- Do not use: camera_position, batter_handedness, whole_batter_visible, phone_level, batter_size

Gate checks that already have a measured source (batter size from pose, clip length, sound, resolution, frame rate) should stay measured; a model answer is only worth adding where the pipeline cannot measure the thing at all (camera angle, other people in frame, a tilted phone). With 18 clips and 2 runs, one wrong answer moves agreement by about 3 points: treat these as a first look, to be re-checked on the Phase 3 net-session footage before any check blocks a real upload.
