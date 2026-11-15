# Phase 6: score and diagnosis (offline prototype)

Code: `phase6_model.py`. Tests: `tests/test_phase6_model.py`. Latest run: `baselines/phase6_prototype_2026-11-15/report.md`.

**Status (15 Nov 2026): offline only.** Nothing in the live app imports it. `scorer.py`, `run_analysis.py` and the `config.py` defaults are unchanged. On the 17 clips it does **not** beat the current four-rule scorer, and its priority fix does not match Cam's. Treat it as tested machinery waiting for better measurements and more clips, not as a result.

## Design

**1. Score order first, then the number.** The two coaches agree on order (Spearman 0.95) but differ by 15 to 35 points on weak drives (Phase 2). So the model learns order from comparisons. A separate step maps it to 0 to 100.

**2. Training data: pairs.**
- Every within-coach pair implied by Alex's overall scores and by Cam's quick-check scores (ties skipped).
- Both coaches' swipe pairs.

Each row holds the difference in features between the two clips and which one the coach preferred.

**3. Features.**
- Phase 5 measurements, turned so that higher = better. The direction is the one registered before any clip was measured.
- Standardised with the training clips' median and IQR, and clipped at ±3.
- A missing value counts as no difference.
- Only measurements that pass the automatic-key-frame test (test 3) and coverage are allowed.
- The primary set is the best one per pillar on test 3: M05 (access), M01 (tracking), M11 (stability) and M12 (flow). The rule uses consistency only, never agreement with the coaches, so the choice cannot fit the coach scores.

**4. Model.** A linear Bradley–Terry model with features: P(i beats j) = sigmoid(w · (z_i − z_j)).
- L2 penalty λ = 0.1 on the mean log-loss, and no intercept.
- Weights are constrained to be ≥ 0, because every feature is already oriented so that higher = better.
- Unconstrained fits use scikit-learn when it is installed. Constrained fits use a small numpy projected-gradient solver. Both minimise the same objective, and a test checks that they agree.

**5. Evaluation.**
- Leave one **player** out (`clip_sheet.csv` player_id). A held-out player appears in no training pair and in no scaling statistic. The code asserts this on every fold.
- Reported: Spearman of the out-of-fold strength with Alex and with Cam, and accuracy on held-out pairs.
- 95% intervals come from a bootstrap over players.
- Benchmarks: the four-rule scorer (Phase 0, auto) and a constant scorer. The model-as-fourth-coach is shown for context, as is a "venue only" scorer that shows the confound.

**6. Calibration.**
- The target is the mean of the two coaches' z-scored scores, rescaled to Alex's mean and SD.
- An isotonic map takes strength to that target. In the evaluation, the map is fitted on the training players only.

**7. Diagnosis.**
- Walk `faults_in_chain_order` in `docs/phase1/rubric_v0_2.json` and name the first checkpoint that fails.
- Only checkpoints with a test-3-passing measurement take part: F01–F08, F10, F11 and F12.
- A checkpoint fails when its measurement is worse than 75% of the Good Club and Elite clips. This threshold is **provisional** and is to be replaced by thresholds fitted to the coaches' checkpoint ratings.
- If nothing fails, the answer is F00, "nothing major to fix".
- The tip text comes from `docs/phase6/fault_tips_draft.json`.

## Result on 17 clips (see the report for intervals)

| | Spearman vs Alex | vs Cam | Held-out pairs |
|---|---|---|---|
| Model (out of fold) | 0.20 | 0.14 | 62% |
| Four-rule scorer | 0.66 | 0.66 | 71% |
| Constant | n/a | n/a | 50% |
| Venue only (confound) | 0.87 | 0.87 | 77% |

The intervals are wide, from about −0.6 to 0.7 for the model. The difference from the four-rule scorer is not significant either way.

The diagnosis matched Cam's main fault on 0 of 16 clips. By chance, about 1 would be expected.

Validated key frames barely help (0.27 / 0.23), so the limit is the measurements, not the key frames. Some measurements also run against their registered direction: M02, M04 and M06. These are not flipped, because flipping them on 17 clips would fit noise and the venue.

## Rerun

```
python phase6_model.py --out baselines/phase6_prototype_<date>
python phase6_model.py --measurements baselines/<phase5 run>/checkpoint_measurements.csv \
                       --entry-tests baselines/<phase5 run>/entry_tests.csv --out ...
```

New measurements (for example Phase 5 v5's M14–M17) enter automatically once they pass test 3 and coverage. New coach scores or swipe pairs enter through the same input files.

## How it would plug into the live app later (not now)

Only once it beats the four-rule scorer on held-out players, on clips that mix venues within each tier:

1. Freeze a fitted model as a small JSON file holding the feature list, the medians and IQRs, the weights, the isotonic knots, the thresholds and the chain. Put a version on the file and keep it in `reference/`.
2. Add a flag such as `config.PHASE6_SCORER = False`, default off.
3. With the flag on, `run_analysis.py` would compute the Phase 5 measurements, which are already contact-anchored, and work out the strength. The calibrated score and the priority fault would go in new response fields.
4. The current `battingiq_score` stays the shipped number until a before/after batch and the coaches sign off.
5. A pillar with fewer than two passing measurements shows "not assessed".
6. A clip with too many missing features gets no score, rather than a guess at the median.
7. The tip and the "nothing major to fix" path are already in the Lovable result shape (`coaching_points`). No frontend change is needed beyond showing F00.

Nothing in this list has been done. It needs explicit approval because it changes live behaviour.
