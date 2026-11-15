# Contact-anchored key frames against the current detectors

17 clips; automatic contact throughout; prior gaps left one clip out. Error = |frame − validated frame|.

| Key frame | Current: mean (median) | Contact + gap: mean (median) | Refined: mean (median), found | Within 3 frames: current / prior / refined |
|---|---|---|---|---|
| setup_frame | 7.9 (6) | 7.4 (7) | 6.6 (5), 14/17 | 29% / 35% / 43% |
| hands_start_up_frame | 3.8 (3) | 2.8 (3) | 3.7 (4), 7/17 | 59% / 71% / 43% |
| front_foot_down_frame | 3.3 (3) | 1.9 (1) | 2.7 (2), 14/17 | 59% / 76% / 71% |
| hands_peak_frame | 4.6 (3) | 2.1 (2) | 3.9 (3), 16/17 | 53% / 76% / 56% |
| follow_through_frame | 10.8 (8) | 7.8 (5) | 9.6 (10), 16/17 | 18% / 47% / 31% |

A detector is worth shipping only if it beats the current one on mean AND on the share within 3 frames.
Shipping it changes the live T2 / F4 measurements, so a Phase 0 batch runs before and after.
