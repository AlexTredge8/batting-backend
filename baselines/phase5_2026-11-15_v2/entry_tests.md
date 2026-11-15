# Phase 5 entry tests

Checkpoint ratings from: model. Model ratings only, so test 1 is provisional.
Thresholds were fixed on 14 Nov 2026 before the clips were measured (see phase5_entry_tests.py).

| Meas. | Checkpoint | Pillar | Coverage | Tracks rating (rho) | Auto anchors (rho, gap/IQR) | Two-phone | Verdict |
|---|---|---|---|---|---|---|---|
| M01 | C01 (F01) | tracking | 94% | 0.57: not decided: rater gives nearly every clip the same rating | 0.91, 0.06: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M02 | C02 (F02) | stability | 94% | -0.23: not decided: rater gives nearly every clip the same rating | 0.91, 0.14: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M03 | C03 (F03) | access | 82% | 0.52: not decided: rater gives nearly every clip the same rating | 0.96, 0.02: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M04 | C04 (F04) | tracking | 100% | -0.45: fail (provisional: model ratings) | 0.96, 0.06: pass | pending (Phase 3 net session) | fail |
| M05 | C05 (F05) | access | 94% | -0.48: not decided: rater gives nearly every clip the same rating | 0.95, 0.07: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M06 | C06 (F06) | stability | 100% | -0.3: not decided: rater gives nearly every clip the same rating | 0.58, 0.05: fail | pending (Phase 3 net session) | fail |
| M07 | C07 (F07) | access | 94% | n/a: not decided: rater gives nearly every clip the same rating | 0.86, 0.24: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M08 | C08 (F08) | access | 88% | 0.35: pass (provisional: model ratings) | -0.3, 0.51: fail | pending (Phase 3 net session) | fail |
| M09 | C09 (F09) | flow | 88% | -0.2: fail (provisional: model ratings) | 0.63, 0.00: fail | pending (Phase 3 net session) | fail |
| M10 | C10 (F10) | tracking | 100% | -0.11: fail (provisional: model ratings) | 0.86, 0.03: pass | pending (Phase 3 net session) | fail |
| M11 | C11 (F11) | stability | 88% | -0.06: not decided: rater gives nearly every clip the same rating | 0.99, 0.01: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M12 | C12 (F12) | flow | 94% | 0.63: pass (provisional: model ratings) | 0.99, 0.00: pass | pending (Phase 3 net session) | pass so far (two-phone pending) |

## Pillars

- access: 0 passing (none); 3 robust and awaiting coach ratings (M03, M05, M07) -> could reach 2 once coaches rate checkpoints
- tracking: 0 passing (none); 1 robust and awaiting coach ratings (M01) -> NOT ASSESSED (needs 2)
- stability: 0 passing (none); 2 robust and awaiting coach ratings (M02, M11) -> could reach 2 once coaches rate checkpoints
- flow: 1 passing (M12); 0 robust and awaiting coach ratings (none) -> NOT ASSESSED (needs 2)
