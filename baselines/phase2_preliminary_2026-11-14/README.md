# Phase 2, preliminary (14 Nov 2026)

What exists so far: Alex's original scores (18 clips), Alex's left-or-right session (40 pairs) and the model as a fourth coach (14 clips; 3 were reference clips, 1 had no frames). Cam's earlier scores were never saved (2 clips in the backend); his quick check is pending.

## Model as a fourth coach (14 clips, against Alex's original scores)

| Scorer | Mean abs error, overall | Rank agreement (Spearman) |
|---|---|---|
| Model (Claude, stills + rubric + 3 reference clips) | 13.1 | 0.77 |
| Current four-rule scorer (Phase 0, auto) | 19.3 | 0.71 |
| Constant (average coach score) | 20.5 | n/a |

- Pillars, model against Alex: access 3.5, tracking 2.5, stability 4.4, flow 4.1 points.
- Bias +4.4 overall, but not uniform: beginners +19 to +35, Elite −7 to −17. A leave-one-out linear stretch does not fix it (15.9), so the low end is mis-ordered, not just squashed. The model says "batter small and distant" on almost every clip.

## Alex's left-or-right session

- 39 of 40 pairs agree with Alex's original scores; the pairwise ranking correlates 0.94 with them. One coach is highly self-consistent, and the swipe method measures what the scores measure.

Files: `model_vs_alex.csv`, `alex_swipe_pairs.csv`. The keys that map codes to clips stay off the repo.

## Cam's left-or-right session (15 Nov)

- 34 of 40 pairs agree with Alex's original scores. Of the 6 that differ, 2 are near-ties (49 v 50, 79 v 80), 3 are within the same tier, and 2 put Cam's own on-drive above the other Elite clips.
- Cam's ranking against Alex's scores: Spearman 0.87. Against the model (14 clips): 0.87.
- Alex and Cam combined (80 judgements): scale separation reliability 0.74 (Alex alone: 0.19). Combined ranking against Alex's scores 0.95; against the current four-rule scorer 0.62.
- Two qualified coaches order the 18 drives almost the same way. The four-rule scorer is the outlier.

Files: `cam_swipe_pairs.csv`, `swipe_scale_alex_cam.csv`, `swipe_scale_cam_only.csv`.

## Cam's quick check (15 Nov): overall score and main fault per clip

- **Order:** Cam's scores rank the 17 clips almost exactly as Alex's do (Spearman 0.95), and as his own swipe answers do (0.94). Against the model: 0.85. Against the four-rule scorer: 0.66.
- **Scale:** Cam uses a much wider range. Tier means, Alex / Cam: Beginner 27 / 4, Average 58 / 23, Good Club 76 / 74, Elite 90 / 96. The coaches disagree on the numbers for weak drives, not on the order. The model (Beginner 57) and the four-rule scorer (58) go the other way: both put beginners far above where either coach does.
- **Main fault:** Cam names "no real fault" (F00) for all four Elite clips and Alex T_Straightdrive Good. The model always names a fault and matched Cam's main fault on 0 of 13 clips. For the tips, the app must be able to say "nothing major to fix".
- Mean absolute gap to Cam: Alex 16.8, model 26.5, four-rule scorer 31.7.

Implications: (1) agreement should be judged on order first, and the overall number calibrated separately (Phase 6), because two qualified coaches differ by 15–35 points on weak drives; (2) a correct ordering of beginners is the biggest gap in both the model and the current scorer; (3) a "no real fault" outcome is needed.

File: `cam_quick_check.csv`.
