# Rules Changelog

## 2.1.0 — 2026-05-11

- Suspended S2 (Post-contact instability): inverted after F5b detection improvements.
  Fire rate Elite > Beginner. Root cause: thresholds calibrated on incorrect HP frames.
- Suspended F5 (Mid-downswing hitch): inverted after F5b detection improvements.
  Fire rate Elite > Beginner.
- Suspended F1 (HP/FFD desync): fires 4/4 Elite auto, 0/4 validated. FFD MAE 7.0f
  corrupts sync measurement. Redesign needed once FFD detection tightened.
- Suspended F3 (Timing ratio): backlift_to_contact_frames produces negative values on
  3/4 Elite in auto mode (HP detected after contact). Root cause: backlift_start inherits
  setup_frame errors (MAE 26.4f). Metric is invalid until setup/backlift detection stable.
- Active rules after this change: A1, A3, A5, T2, S4, F4, F6 (7 active).
- Suspended rules cumulative: A2, A4, T1, S1, S2, S3, F1, F2, F3, F5 (10 suspended).
- Elite Overall MAE trajectory: 14.00 (post-F5b) → 10.75 (S2/F5 susp.) → 3.25 (F1/F3 susp.)
- Elite mean score: 79.5 → 89.0. All 4 Elite within ±5 on overall.

## 2.0.0 — 2026-04-29

- Deleted A6 from active rule evaluation: downstream consequence, not independent; inverted in validated mode.
- Deleted T3, T4, and T5 from active rule evaluation: eye/head tracking proxies are not measurable from side-on phone video.
- Suspended S1: hip/stance shift is a proxy-of-proxy for balance; measurement remains available for calibration.
- Rebalanced pillar scoring using Option B3 from `calibration_output/f3c_refined_full/rebalance_options.txt`: four 25-point pillars remain, but each pillar deduction is normalised by active rule capacity.
- Active rules after cleanup: A1, A3, A5, T2, S2, S4, F1, F3, F4, F5, F6.
- Suspended rules after cleanup: A2, A4, T1, S1, S3, F2.
- Deleted rules after cleanup: A6, T3, T4, T5.
