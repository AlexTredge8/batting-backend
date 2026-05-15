# BattingIQ — Latest Developments (Working Memory)

*Update this file at the end of every Codex task.*
*Last updated: 11 May 2026*

-----

## Current stage

**A2 UNSUSPENSION ANALYSIS (13 May 2026). RECOMMENDATION: KEEP SUSPENDED. Distribution is non-monotonic in both auto and validated modes (Average mean 141° > Elite 127° > Beginner 114° > Good Club 105°). No threshold achieves ≥80% Beginner fire AND ≤25% Elite AND ≤50% Good Club simultaneously. Metric is contact-frame-sensitive (mean |auto−val| delta 13.3°, max 51.4°). Best-case gap gain: +1.1 pts — not worth the Good Club false-positive. Redesign prerequisites: contact MAE <1.0f, consider two-sided rule (<90° too bent AND >160° over-extended), n≥25 videos. Output: calibration_output/a2_unsuspension_analysis.txt**

**D4 DETERMINISM CONFIRMED (11 May 2026). 3 videos × 3 consecutive runs in single Python session — all ✓ PASS. Scores, all 6 anchor frames, anchor confidence, and per-rule deductions identical across every run. Audio onset frame (Drive_beginner: frame 17, strength 1.2887) stable. Root cause of historical swings confirmed as codebase split (Playground vs Playground 2), not non-determinism. Output: calibration_output/d4_determinism/d4_determinism_report.txt**

**F1+F3 SUSPENDED (11 May 2026). Elite Overall MAE: 10.75 → 3.25 (−7.50). All 4 Elite within ±5 on overall. Elite mean score 79.5 → 89.0. ±5 overall: 6/17 (↑2). ±5 all 4 pillars: 2/17 (↑2). Tier ordering monotonic (Beginner 72.8 < Average 73.0 < Good Club 81.2 < Elite 89.0) but Beginner–Average gap is only 0.2 pts — score compression on lower tiers is now the dominant problem. Flow pillar MAE: Elite 9.25→1.25 (−8.0), Good Club 6.50→2.00 (−4.5), Average 8.50→4.50 (−4.0), Beginner worsened 4.40→10.20 (+5.8 — F1/F3 were incorrectly penalising Average/Good Club/Elite but providing some cover for Beginners). Active rules now: A1, A3, A5, T2, S4, F4, F6 (7 active). Output: calibration_output/f1_f3_suspended/f1_f3_suspension_check.txt**

**S2+F5 SUSPENDED (11 May 2026). Both rules confirmed inverted (Elite > Beginner fire rate) after F5b. Suspension recovered Elite Overall MAE 14.0→10.75 (−3.25). Remaining Elite deficit vs pre-F5b: 9.5 pts — not attributable to S2/F5; root cause is F5b HP cascade into Flow/Stability rules on Elite videos. Good Club Stability MAE regressed +2.0 (S2 was legitimately firing there). ±5 overall: 4/17 (↑1). Tier ordering monotonic. Active rules now: A1, A3, A5, T2, S4, F1, F3, F4, F6 (9 active). Suspended: A2, A4, T1, S1, S2, S3, F2, F5 (8 suspended). Output: calibration_output/s2_f5_suspended/s2_f5_suspension_check.txt**

**D2+D3 POST-F5b COMPLETE (10 May 2026). Diagnostics regenerated against post-F5b codebase (USE_HANDS_PEAK_V3=True). Key findings: Beginner Overall MAE improved substantially (55.8→36.4, −19.4); Elite Overall MAE regressed sharply (1.25→14.0, +12.75) — primary F5b side-effect requiring investigation. ±5 overall count: 8→3/17 (regression). Rule health: S4/A5/F6 promoted to Healthy (were Sparse); F4 promoted Flat→Healthy; A1 demoted Healthy→Flat; S2+F5 newly Inverted (were Sparse). Hub decision warranted before next F-track or R-track.**

**F5b hands_peak v3 (contact-anchored backward search) ADOPTED (09 May 2026). Overall HP MAE: v1=22.6 → v3=4.8 (−17.8 frames, 79% improvement). ±5 frame hit rate: 12%→75%. 14 improved, 0 same, 2 regressed (both ≤2 frames). FFD cascade improves from 20.5→8.1 (−12.4 frames) as bonus. Contact MAE unchanged at 2.1 (audio-driven). Config: `USE_HANDS_PEAK_V1=False`, `USE_HANDS_PEAK_V3=True` (production defaults). Setup detector remains at v1 (MAE 26.4 — still the dominant anchor bottleneck). 4/17 videos have v3 confidence="low" (Drive_Average, Drive_Average 2, Drive_Beginner 2, Ollie P_Offdrive Average — all have small audio contact errors that shift the window boundary).**

-----

## Last completed task

**F5b — hands_peak v3 contact-anchored backward search (9 May 2026)**
Implemented `_find_hands_peak_v3_contact_anchored` in phase_detector.py behind `USE_HANDS_PEAK_V3` flag. Algorithm: bootstrap v1 HP for contact detection only; run audio contact first; then backward-search `[contact − 15, contact − 4]` frames for wrist-height minimum. Confidence "low" if audio failed, contact_confidence=="low", or candidate at window edge. Key implementation fix: bypassed `max(hp, backlift_start + 2)` guard for v3 (guard assumes forward search; backlift_auto frequently after true HP causing clamp to wrong value). Key results:
- hands_peak MAE: v1=22.6 → v3=4.8 (−17.8 frames, 79% improvement). All tiers improved.
- ±5 frame hit rate: 12%→75%. ±2 frame hit rate: 6%→18%.
- Elite: 22.0→3.2; Good Club: 32.5→3.5; Average: 22.0→7.5; Beginner: 13.8→5.0.
- 14 improved, 0 same, 2 regressed (Drive_Beginner 4: +2, Viv P Elite: +1 — both trivial).
- FFD cascade: 20.5→8.1 (−12.4 frames improvement as bonus; driven by better HP → better FFD window).
- Contact MAE: 2.1→2.1 (unchanged — sanity check passes; audio-driven contact is independent).
- Score delta (auto vs validated): v3=10.8 vs v1=0.0 (v1 delta artifact — v1 batch had no dual-mode separation; v3 score delta comparable to v2=10.5).
- v3 confidence="low" in 4/17: Drive_Average, Drive_Average 2, Drive_Beginner 2, Ollie P_Offdrive Average. All have small audio contact errors shifting the window boundary slightly.
- 5 AT EDGE alerts in comparison (all expected — contact error of 1–3 frames puts true HP at window boundary).
- VERDICT: ADOPT. `USE_HANDS_PEAK_V1=False`, `USE_HANDS_PEAK_V3=True` are production defaults.
- Outputs: `calibration_output/f5b_hands_peak_v3_contact_anchored/f5b_v3_comparison.txt`, auto + validated batch runs

**F4.5 — setup detector v3 motion-onset (9 May 2026)**
Implemented `_find_setup_end_v3_motion_onset` in phase_detector.py behind USE_SETUP_V1 flag. Algorithm: last frame in search window where rolling |wrist_velocity_y| < 0.005 for ≥5 consecutive frames AND velocity exceeds threshold within next 10 frames (motion onset). Spec windows (30%/50%) insufficient for dataset (setup spans 8%–68% of video); widened to 70%/85%. Spec 2× motion threshold dropped to 1× (slow Beginner backlift barely exceeds stillness threshold). Key results:
- setup MAE: v1=26.4, v3=28.1 — v3 net +1.7 frames WORSE.
- 8 improved / 8 regressed. Largest improvement: Cam T_Offdrive Good (err 114→19, −95 frames). Largest regression: Ollie P_Offdrive Average (err 0→87, +87 frames — v3 picks a late still candidate in a video that has only 10 frames of pre-motion).
- Contact cascade: audio-independent (unchanged for 16/17 videos). ONE video (Ollie P_Offdrive Average) has audio failure → pose fallback → corrupted setup cascades to contact (Good Club contact MAE: 0.8→26.0).
- FFD cascade: slight net improvement overall (20.5→19.4, −1.1 frames). Beginner improved (12→7.2). Good Club large improvement (45.2→30.5). Elite worsened (10.5→22.8).
- Score delta (auto vs validated): v1=10.2, v3=13.2 (v3 worse, driven by Ollie P regression).
- Hands_peak_confidence: 0 low in both (USE_HANDS_PEAK_V1=True returns always "high").
- USE_SETUP_V1 reverted to True (v1 remains default). V3 code is in phase_detector.py as _find_setup_end_v3_motion_onset, accessible via USE_SETUP_V1=False.
- Outputs: `calibration_output/f4_5_setup_v2/f4_5_setup_comparison.txt`, `drift_summary.txt`; `calibration_output/f4_5_setup_v1_baseline/` (v1 auto reference)

**F4 — hands_peak v2 re-evaluation (9 May 2026)**
Re-ran hands_peak v1 vs v2 comparison on post-R3/R4/F3c codebase. Key findings:
- hands_peak MAE: v1=22.2, v2=26.8 — v2 is 4.6 frames worse overall (same conclusion as original F4 run).
- Elite regressed most: v1=22.0, v2=38.0 (+16 frames). Worst regression: Viv P_Offdrive Elite (+29), Cam T_Ondrive Elite (+24).
- Contact cascade: ZERO — audio detector (F3c) has decoupled contact from hands_peak entirely. Both v1 and v2 give contact MAE=2.0 frames.
- FFD cascade: negligible (v1=20.7, v2=20.6).
- Score delta (auto vs validated): v1=9.9, v2=10.5 — v2 marginally worse.
- Confidence distribution: 10/17 high, 7/17 low. Low-confidence videos are early-backlift batters (Drive_beginner, Drive_Beginner 2/4, Alex T_Straightdrive Good, Cam T_Offdrive Good, Straight Drive_Beginner, Drive_Average).
- USE_HANDS_PEAK_V1 should be set to True until v2 window is redesigned. Root cause: v2 window (setup+5% → setup+45%) is too wide for Elite (fast backlift), placing candidate too late in the swing.
- Outputs: `calibration_output/f4_hands_peak_v2/f4_comparison.txt`, `drift_summary.txt`, `drift_report.csv`

**R4/R3 cleanup — rule deletion, S1 suspension, pillar rebalance (29 April 2026)**
Deleted A6/T3/T4/T5 from active evaluation, suspended S1, added `RULES_VERSION = "2.0.0"`, and implemented Option B3 normalised 25-point pillars. F2 remains suspended, not deleted. Dual-mode batch completed 17/17 into `calibration_output/r4_r3_cleanup/`. Auto tier means: Beginner 70.4, Average 76.0, Good Club 66.8, Elite 75.8; not monotonic and Elite auto remains below 80.
- Outputs: `calibration_output/f3c_refined_full/rebalance_options.txt`, `calibration_output/r4_r3_cleanup/cleanup_summary.txt`, `RULES_CHANGELOG.md`

**F3c refined — N-aware audio contact detector (28 April 2026)**
Replaced `AUDIO_MULTI_ONSET_STRATEGY` with two N-aware config keys. N=2 → last (bat-on-ball); N≥3 → loudest (avoids follow-through misfire). Cam T_LH regression resolved: N=3 case now uses `audio_loudest_of_3`, correctly picks frame 52 (err=1). All v2 wins preserved. Full batch run: contact MAE 30.29→2.06 frames overall, Elite 42.00→0.75 frames. First batch with audio detection live in production flow. Worst remaining anchor: hands_peak (MAE=27.38). Score compression unchanged — that is an R-track problem.
- Outputs: `calibration_output/f3c_audio_last_onset/f3c_refined_comparison.txt`, `f3c_refined_summary.txt`; `calibration_output/f3c_refined_full/` (dual-mode batch, 17/17 videos)

**F3c — audio contact detector last-onset fix (28 April 2026)**
Fixed bowling machine video detection in `audio_contact.py`. New approach: group onsets into distinct events using `AUDIO_MIN_ONSET_GAP_SECONDS=0.15`, then when `AUDIO_MULTI_ONSET_STRATEGY="last"` and N≥2 distinct events, select the final event (bat-on-ball). New config keys added to `config.py`: `AUDIO_TEMPORAL_FILTER_START/END` (exposing existing 20%/85% window), `AUDIO_MIN_ONSET_GAP_SECONDS=0.15`, `AUDIO_MULTI_ONSET_STRATEGY="last"`. Diagnostics include `audio_onset_count` and `audio_strategy_used` in returned metadata.
- Pre-check confirmed: `Cam T_Offdrive Elite.mp4` had 2 onsets (frame 109=machine, frame 126=bat); V1 picked 109 (err=16), V2 picks 126 (err=1; truth=125)
- 3 of 4 Elite videos improved by 15–19 frames each; heldout `Viv P_Offdrive Elite` improved err 19→1
- 1 regression: `Cam T_LH Offdrive Elite.mov` — 3 distinct events; last (frame 83) is follow-through not contact; V1 was correct (frame 52, err=1) — resolved in refined
- Output: `calibration_output/f3c_audio_last_onset/f3c_audio_comparison.txt`

**R2 — A3 + F6 rule fixes (25 April 2026)**
Fixed A3 logic bug (unreachable else → both branches returned 0; now collapses to two bands). Fixed F6 anchor-sensitivity (was `follow_through_start`, now `contact_frame + CONTACT_WINDOW_FRAMES + 3`). Ran validated-only batch into `calibration_output/r2_a3_f6_fix/`. Key results:
- A3: fires on 8/17 videos (was 0/17); Elite correctly zero; Beginner 1.80 vs Good Club 2.25 (marginal residual inversion)
- F6: Average inversion resolved (3.50→1.00); Good Club residual 0.10 pts above Beginner (2.50 vs 2.40)
- Overall MAE: 20.0→18.6 (net small improvement)
- Beginner mean did not decrease (+1.6 pts) due to pose re-run S1 variance outweighing A3/F6 gains
- Large unexplained score swings (e.g. Elite +16 with A3=0, F6=0) confirm D4 determinism check is urgent

**D5/R-prep — Validated rule audit (24 April 2026)**
Generated validated_rule_audit.csv and validated_rule_audit_summary.txt from D1 validated runs. Key findings:
- All 5 Beginner videos over-scored by 24–53 pts with validated anchors (rule problem, not detection)
- Tracking gap is structural: T2 is the only active rule, max possible deduction ~9 pts, but manual tracking scores reach 3/25
- 9/13 active rules are inverted or fire more on elite/good-club than beginners: S1, S2, F3, F6, A5, A6, F1, A3(dead)
- A3 fires zero across ALL 17 videos (completely dead rule)
- S1 most damaging inversion: deducts 8.25 pts on Elite vs 4.40 on Beginner
- Drive_beginner has +53 overall gap: stability fires zero rules despite 18pt pillar gap
- Top 3 changes to reduce Beginner error: A3 activate/retune, A6 lower threshold, S2 lower threshold

**F4 — setup detector v2 (24 April 2026)**
Implemented a sustained multi-landmark setup detector in `phase_detector.py` behind `USE_SETUP_V2`, added `SETUP_MIN_LANDMARK_CHANGES` / `SETUP_SUSTAINED_FRAMES`, and ran fresh auto-only v1 vs v2 batches on all 17 calibration videos. Key results:
- setup MAE: v1=27.18 → v2=29.06 (net worse overall)
- Beginner setup improved strongly (24.0 → 17.8 MAE) but Average worsened (27.0 → 35.5), Good Club worsened slightly (38.0 → 38.5), and Elite worsened materially (20.5 → 27.25)
- setup hit rate did not improve overall: within 2 frames stayed 5.9%, within 5 frames stayed 11.8%
- worst setup regressions: `Cam T_Ondrive Elite.mov` (+36 abs-error), `Drive_Average.MOV` (+34), `Ollie P_Offdrive Average.mov` (+32), `Alex T_Straightdrive Good.MOV` (+24)
- cascade effect was mixed: `front_foot_down` MAE 20.59 → 21.00 with slightly better within-2/within-5, `contact` MAE 30.29 → 29.88, `hands_peak` MAE 26.76 → 29.59 (worse)
- Recommendation: revert to v1 / keep `USE_SETUP_V2=False` until setup v3 is redesigned

**F4 — hands_peak detector v2 (24 April 2026)**
Implemented adaptive position-minimum detector (v2) in phase_detector.py behind USE_HANDS_PEAK_V1 flag. Evaluated against D1 auto pass. Key results:
- hands_peak MAE: v1=22.1 → v2=26.8 (net worse overall; better on Average, worse on Elite/Good Club)
- 8 improved, 9 regressed; worst regression: Viv P Elite (+29 frames, was perfect at 1)
- Auto/validated score delta improved: 14.9 → 9.7 (Elite: 16.5 → 8.0, Beginner: 21.4 → 10.8)
- 7/17 videos flagged low confidence (wrist never rises 0.03 above setup — all have late setup detection)
- Root cause: v2 window anchored to auto-detected setup_end which is itself unreliable
- Recommendation: revert to v1 in production (USE_HANDS_PEAK_V1=True default) until setup detector is fixed

**D3 — Rule health audit (24 April 2026)**
Generated `rule_health.csv` and `rule_health_summary.txt` from D1 auto runs. Key findings:
- Auto mode: 1 healthy, 3 flat, 0 inverted, 9 sparse, 8 suspended.
- Flat rules: `T2`, `F1`, `F4`. Healthy rule: `A1`.
- `T2` looked healthy in validated runs but flat in auto runs, marking it as a likely detection-noise rule.
- Code reality differs from the prompt text: current `coaching_rules.py` contains 13 active rules and 8 suspended rules, not 14 active / 7 suspended.

-----

## Next immediate tasks

**F-track summary (as of 09 May 2026):**
- Setup v2 (multi-landmark, Apr-24): REVERT — worse overall.
- Setup v3 (motion-onset, May-09 F4.5): REVERT — +1.7 frames worse; large regression on Ollie P (short pre-motion).
- Hands_peak v2 (adaptive position-min, Apr-24 + May-09 F4): REVERT — +4.6 frames worse; Elite badly regressed.
- **Hands_peak v3 (contact-anchored, May-09 F5b): ADOPTED — −17.8 frames better (22.6→4.8 MAE).**
- Setup remains at v1 (`USE_SETUP_V1=True`). Hands_peak is now v3 (`USE_HANDS_PEAK_V3=True`, `USE_HANDS_PEAK_V1=False`).

**Remaining anchor error budget (current defaults):**
- Setup MAE: 26.4 frames (worst anchor — dominant bottleneck; 15/17 within-5 fails)
- Hands_peak MAE: 4.8 frames (v3 — 12/16 within-5 hits)
- Contact MAE: 2.1 frames (audio-driven, ~fixed)
- FFD MAE: 8.1 frames (improved with v3 HP — was 20.5)

**Candidate next steps (prioritised for hub decision):**

1. **F5 (new): setup v4 — address short-pre-motion guard.**
   V3 failed because Ollie P_Offdrive Average has only 10 frames before motion starts. A v4 could apply a minimum pre-motion guard or use a forward-looking stillness anchor. Requires hub spec before Codex implementation.

2. **F5b follow-up: widen backward search window.**
   4/17 v3 confidence="low" (all have audio contact errors of ~3 frames placing true HP at boundary). Trying `SEARCH_BEFORE_CONTACT=17` or `MIN_OFFSET=3` may recover 1–2 low-confidence videos. Small task; may not warrant a full F-track ID.

3. **R5 (R-track): address remaining auto-validated score gap.**
   Score delta is 10.8 pts overall with v3 HP. HP errors no longer driving this — remaining delta is likely from setup cascade errors on FFD-sensitive rules (F1, F3, F4). Could target after setup v4.

4. **C-track: convergence loop.**
   Accept current detector quality and run a scoring calibration loop to tighten the auto vs validated score delta (currently 10.8 pts overall, 15.8 pts Average). Requires C1 hub decision first.

All R-track and F-track deliverables must report tuning-set MAE and held-out MAE side by side.

-----

## Open questions / risks

- **Score compression root cause is unmeasured.** Current hypothesis: binary rules + auto-detection frame error. But per-pillar concordance and drift have never been quantified. D-track exists to answer this before any tuning.
- **Auto vs validated drift unknown.** All prior calibration used manual anchor runs. The product ships auto-detected. These may behave differently across tiers — this is the central risk.
- **F6/F1/A3 retune is queued but not confirmed.** R1 (hub decision after diagnostics) will decide whether the retune is the right move or whether frame detection should be fixed first.
- **R4/R3 cleanup changed the rule surface.** Current `coaching_rules.py` contains 11 active rules, 6 suspended rules, and 4 deleted rules. Active: A1, A3, A5, T2, S2, S4, F1, F3, F4, F5, F6. Suspended: A2, A4, T1, S1, S3, F2. Deleted: A6, T3, T4, T5. Pillars remain 25 each under Option B3, with deduction normalisation by active rule capacity.
- **D1 baseline contaminated.** D1 ran from old "Playground 2" codebase. All D1-vs-R2 comparisons are invalid. D1-new required before rule-change comparisons can be trusted.
- **Railway vs local drift.** Production pipeline runs compressed settings. Local calibration runs full quality. Drift between these has not been measured. Will need validation after local calibration converges.

-----

## Known issues

- **Score compression:** Elite ≈ 84, Beginner ≈ 79. Target: 40–60 point separation. Root cause not yet confirmed by diagnostic data.
- **Contact frame detection suspected.** Wrong contact frame corrupts metrics, coaching rules, and storyboard output simultaneously.
- **No held-out split yet.** All 17 videos are currently used for both tuning and evaluation — this creates overfitting risk. T4 addresses once dataset exceeds 25 videos.

-----

## Completed task log

*(Append each completed task ID, date, and one-line output summary here.)*

|Task|Date|Output|
|----|----|------|
|—   |—   |—     |
|S2+F5 suspend|2026-05-11|calibration_output/s2_f5_suspended/s2_f5_suspension_check.txt — Elite MAE 14.0→10.75 (−3.25); ±5 overall 4/17 (↑1); tier ordering monotonic; Good Club Stability regressed +2.0; 9.5 pts Elite deficit remains unexplained|
|D2+D3 post-F5b|2026-05-10|calibration_output/diagnostics_post_f5b/ — 17/17 processed; Beginner Overall MAE 55.8→36.4 (−19.4); Elite Overall MAE 1.25→14.0 (+12.75 regression); ±5 overall 8→3/17; S4/A5/F6/F4 now Healthy; S2+F5 newly Inverted; A1 Healthy→Flat|
|F5b|2026-05-09|calibration_output/f5b_hands_peak_v3_contact_anchored/f5b_v3_comparison.txt — HP v3 contact-anchored; ADOPT: v3=4.8 vs v1=22.6 (−17.8 frames, 79%); 14 improved/0 same/2 regressed (≤2 frames); FFD 20.5→8.1; contact unchanged 2.1; `USE_HANDS_PEAK_V3=True` is new default|
|F4.5|2026-05-09|calibration_output/f4_5_setup_v2/f4_5_setup_comparison.txt — setup v3 motion-onset; REVERT: v3=28.1 vs v1=26.4 (+1.7 worse); 8 improved/8 regressed; Ollie P regression +87 (short pre-motion); contact cascade +25.2 for Good Club (audio fails on Ollie P); FFD net improved −1.1|
|F4 re-eval|2026-05-09|calibration_output/f4_hands_peak_v2/f4_comparison.txt — hands_peak v2 on post-F3c codebase; REVERT: v2=26.8 vs v1=22.2 (+4.6 worse); contact cascade zero (audio); score delta v1=9.9 v2=10.5|
|R4/R3 cleanup|2026-04-29|calibration_output/r4_r3_cleanup/cleanup_summary.txt + f3c_refined_full/rebalance_options.txt — deleted A6/T3/T4/T5, suspended S1, implemented Option B3 normalised 25-point pillars; auto Elite mean 75.8, not monotonic|
|F3c rule health refresh|2026-04-28|calibration_output/f3c_refined_full/rule_health_diff.txt + rule_health_summary_refresh.txt + concordance_refresh.txt — 3 rules moved to healthy in auto (A1, S4, F6); 8 remain broken: 3 other-anchor sensitive, 5 concept-level; Flow Beginner gap unchanged at 22.5pts|
|F3c refined|2026-04-28|f3c_refined_comparison.txt + f3c_refined_summary.txt + f3c_refined_full/ — N-aware audio detector; contact MAE 30.29→2.06; Elite 42→0.75; 0 regressions; full batch 17/17|
|F3c|2026-04-28|calibration_output/f3c_audio_last_onset/f3c_audio_comparison.txt — audio last-onset fix; overall contact MAE 7.25→5.12; Elite 13.25→8.00; 1 regression (Cam T_LH, 3-event case) — resolved in refined|
|D3-new|2026-04-28|rule_health.csv + rule_health_summary.txt — 8 healthy, 4 flat, 1 inverted (A6), 0 sparse; A6 inversion is concept-level; 8/13 rules detector-sensitive|
|D2-new|2026-04-28|concordance_report/summary/ranking/bias.csv — validated MAE: Beg 36.2, Avg 13.3, GC 7.7, Eli 8.3; tracking worst pillar for Beg+Avg; Elite access/stability worse in validated|
|D1-new|2026-04-27|calibration_output/d1_new_baseline/ — 17/17 auto + validated; auto MAE 15.12, validated MAE 12.00; contact_frame drift MAE 30.3 frames; 8/17 score-sensitive; drift + split summaries emitted|
|HO1 |2026-04-25|heldout_split.csv created; batch_calibration_compare.py emits tuning/heldout split outputs; heldout_discipline.md + CLAUDE.md updated|
|D4  |2026-04-25|determinism_test/determinism_report.txt — PASS all 3 videos; run1=run2=run3 byte-identical; D1 discrepancy caused by old codebase (Playground 2), not non-determinism|
|R2  |2026-04-25|calibration_output/r2_a3_f6_fix/ — A3 logic bug fixed (8/17 now fire); F6 Average inversion resolved; overall MAE 20.0→18.6; D4 blocking clean interpretation|
|D3|24 April 2026|Generated `calibration_output/diagnostics_d1_drift/auto/rule_health.csv` and `rule_health_summary.txt`; auto run had 1 healthy, 3 flat, 0 inverted, 9 sparse, and 8 suspended rules, with `T2` identified as healthy on validated but flat on auto.|
|D5  |2026-04-24|validated_rule_audit.csv + validated_rule_audit_summary.txt — all 5 Beginners over-scored 24–53pts; T2 structural gap in Tracking; 9/13 active rules inverted or dead|
|F4  |2026-04-24|calibration_output/f4_hands_peak_v2/f4_comparison.txt — v2 net worse on hp MAE (22.1→26.8) but score delta improves (14.9→9.7); revert recommended|
|F4 setup v2|2026-04-24|calibration_output/f4_setup_v2/f4_setup_v2_comparison.txt — multi-landmark setup detector improved Beginner setup accuracy but regressed Average/Good Club/Elite, worsening setup MAE overall (27.18→29.06); revert recommended.|
|D2  |2026-04-24|concordance_report.csv, concordance_summary.csv, concordance_ranking.txt, concordance_bias.csv|
|Beginner rule breakdown|2026-05-11|calibration_output/beginner_rule_breakdown/ — Beginner-only auto+validated fresh local runs (5/5 each, Railway skipped); validated raw active deduction mean 24.8/61 (40.7%); highest missed opportunity S4/F4 tie at 7.2 pts/video, selected S4 as threshold candidate; auto inverted active rules vs Elite: A1, F4|
|S4+F6 retune|2026-05-11|calibration_output/s4_f6_retune/ — tightened S4 severity and F6 follow-through thresholds only; all-17 auto run 17/17 fresh local; Beginner mean score 72.8→69.6 (−3.2), Elite 89.0→89.0 (+0.0); overall ±5 4/17→5/17; T2 gate analysis recommends relaxing any-anchor-low suppression, no gate change implemented|
|T2 gate relax|2026-05-11|calibration_output/t2_gate_relax/ — changed only T2 suppression gate in anchor_accuracy.py from any low dependency to both setup_frame and hands_start_up_frame low; all-17 auto run 17/17 fresh local; Beginner mean 69.6→62.6 vs s4_f6_retune, tier ordering monotonic restored; no T2 threshold/rule changes|
