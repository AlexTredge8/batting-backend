# Video-model judge v2: closed (16 Nov 2026)

The same three experiments ran on 15 and 16 Nov (the second run was by mistake) and gave near-identical results, so they are reproducible:

| Experiment | Spearman vs Alex | vs Cam |
|---|---|---|
| Pairwise + crop (55–56 of 70 pairs) | 0.35 / 0.36 | 0.38 / 0.39 |
| Absolute + crop, 8 references | 0.68 / 0.68 | 0.67 / 0.67 |
| Checkpoints + crop | 0.34 / 0.34 | 0.29 / 0.29 |

For comparison: the current four-rule scorer scores 0.66, and the coaches agree with each other at 0.87–0.95.

**Decision:** the video model is dropped as a judge and as an extra rater. Total Gemini spend across all runs was about GBP 6.40, over the GBP 5 cap Alex approved; the overrun came from the accidental second run. `phase5_video_judge_v2_mac.sh` is no longer in any instructions.
