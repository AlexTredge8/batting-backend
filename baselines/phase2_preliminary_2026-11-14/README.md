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
