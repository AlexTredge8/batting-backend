# Rules Changelog

## 2.3.1 — 2026-11-15

- Handedness from the feet. When the upload gives no batting hand, the pipeline reads which way the toes point in the stance (from behind the bowler, a right-hander's toes point camera-left) and uses that. When a hand is given and the feet disagree, the given hand is kept and a plain-English warning suggests the video is mirrored or filmed from behind the batter. `analysis_quality.handedness_check` reports it. Found on 4 of the 18 calibration clips (Phase 5 v3).
- Scores: the four active rules gave identical results for right- and left-handed geometry on the left-handed clip (Phase 0 handedness check), so no score change is expected. Rule logic and thresholds are unchanged.

## 2.3.0 — 2026-11-14

- Unknown pillars no longer score full marks. A pillar whose rule could not measure (`not_measured`) or was suppressed for a low-confidence anchor (`suppressed`) now takes the average of the measured pillars, rounded, and is flagged `imputed` with its `raw_score` kept. If no pillar was measured, scores are left unchanged and `no_pillar_assessed` is set. Decided by Alex.
- A pillar measured with no fault found keeps its score. On the Phase 0 clips, averaging those as well raised the overall mean error from 19.6 to 26.8 points.
- Effect on the 17 Phase 0 clips: none. Every pillar there was measured. The change protects live uploads where a key moment cannot be found.
- Rule logic and thresholds are unchanged.

## 2.2.0 — 2026-11-14 (records changes made 13–15 May 2026)

These changes were in the code from 13–15 May 2026 but were not logged. Dates and reasons are from the comments in `config.py` and `coaching_rules.py`. No rule logic or threshold changed on 2026-11-14. Only `RULES_VERSION` moved, from 2.0.0 to 2.2.0.

- 2026-05-13 — Suspended A3 (compression frames). D5 rule health: discrimination score 0.45; fire rate Beginner 0.40, Elite 0.25. "Adding noise, not signal."
- 2026-05-13 — `MEDIAPIPE_STATIC_IMAGE_MODE = True` (frame-by-frame inference). A cross-session determinism failure was confirmed in D4, from temporal-tracking state on session init.
- 2026-05-14 — Suspended A1 (wrist spread / bat path). Camera-angle-sensitive: on-drives show the wrists converging, which looks the same as the fault. Cam T_Ondrive Elite took the maximum 10-point deduction. D7 rule health: flat (discrimination −0.85, inverted on Elite).
- 2026-05-14 — T2 revised: minimum threshold raised to 0.038, bands 0.038 / 0.055 / 0.080, deductions 3 / 6 / 9. Reason given: Good Club values of 0.025–0.037 all fired under the old threshold.
- 2026-05-14 — S4 revised: sign convention fixed (positive = forward rotation that continues after contact). Bands 5 / 12 / 22 degrees, deductions 4 / 8 / 12.
- 2026-05-14 — F4 revised: minimum threshold raised from 3 to 5 pause frames, because Elite and Good Club plateau at 3. Deductions 6 / 6 / 12.
- 2026-05-14 — F6 re-suspended: duration-based F6 was flat after follow-through v2.
- 2026-05-14 — Setup detector v4 (hands-peak-anchored stillness) restored as the default (`USE_SETUP_V4=True`). The v2 min-velocity detector was rejected: MAE 28.94 against 8.29 for v4.
- 2026-05-15 — F6 (velocity-based v2) kept suspended, "keep deployment on the calibrated D7/R1 rule set". It is not in the active flow capacity.
- Active rules: **A5, T2, S4, F4** (4 active, one per pillar; `config.PILLAR_RULE_MAX_DEDUCTION`).
- Suspended: A1, A2, A3, A4, T1, S1, S2, S3, F1, F2, F3, F5, F6 (13).
- Deleted: A6, T3, T4, T5.
- Note: the D7 and R1 outputs these comments cite are not in the repo. The T2, S4 and F4 thresholds were set against tier labels on the same 17 clips. No batch result for this four-rule set was on record before the Phase 0 baseline (`baselines/`).

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
