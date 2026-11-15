# Phase 5 entry tests

Checkpoint ratings from: model. Model ratings only, so test 1 is provisional.
Thresholds were fixed on 14 Nov 2026 before the clips were measured (see phase5_entry_tests.py).

| Meas. | Checkpoint | Pillar | Coverage | Tracks rating (rho) | Auto anchors (rho, gap/IQR) | Two-phone | Verdict |
|---|---|---|---|---|---|---|---|
| M01 | C01 (F01) | tracking | 94% | 0.11: not decided: rater gives nearly every clip the same rating | 0.91, 0.10: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M02 | C02 (F02) | stability | 100% | -0.38: not decided: rater gives nearly every clip the same rating | 0.89, 0.11: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M03 | C03 (F03) | access | 100% | 0.17: not decided: rater gives nearly every clip the same rating | 0.9, 0.05: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M04 | C04 (F04) | tracking | 100% | -0.41: fail (provisional: model ratings) | 0.88, 0.23: pass | pending (Phase 3 net session) | robust; model disagrees (provisional) |
| M05 | C05 (F05) | access | 100% | -0.22: not decided: rater gives nearly every clip the same rating | 0.97, 0.05: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M06 | C06 (F06) | stability | 100% | -0.35: not decided: rater gives nearly every clip the same rating | 0.86, 0.00: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M07 | C07 (F07) | access | 100% | 0.03: not decided: rater gives nearly every clip the same rating | 0.93, 0.12: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M08 | C08 (F08) | access | 100% | 0.33: pass (provisional: model ratings) | 0.85, 0.08: pass | pending (Phase 3 net session) | pass so far (two-phone pending) |
| M09 | C09 (F09) | flow | 100% | 0.11: fail (provisional: model ratings) | 0.7, 0.02: fail | pending (Phase 3 net session) | fail |
| M10 | C10 (F10) | tracking | 100% | 0.39: pass (provisional: model ratings) | 0.82, 0.14: pass | pending (Phase 3 net session) | pass so far (two-phone pending) |
| M11 | C11 (F11) | stability | 94% | -0.06: not decided: rater gives nearly every clip the same rating | 0.99, 0.01: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M12 | C12 (F12) | flow | 100% | 0.45: pass (provisional: model ratings) | 0.96, 0.00: pass | pending (Phase 3 net session) | pass so far (two-phone pending) |
| M13 | C12 (F12) | flow | 71% | 0.19: fail (provisional: model ratings) | 0.93, 0.00: pass | pending (Phase 3 net session) | fail |

## Pillars

- access: 0 passing (none); 4 robust and awaiting coach ratings (M03, M05, M07, M08) -> could reach 2 once coaches rate checkpoints
- tracking: 0 passing (none); 3 robust and awaiting coach ratings (M01, M04, M10) -> could reach 2 once coaches rate checkpoints
- stability: 0 passing (none); 3 robust and awaiting coach ratings (M02, M06, M11) -> could reach 2 once coaches rate checkpoints
- flow: 0 passing (none); 1 robust and awaiting coach ratings (M12) -> NOT ASSESSED (needs 2)
