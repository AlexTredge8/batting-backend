# Phase 5 entry tests

Checkpoint ratings from: model. Model ratings only, so test 1 is provisional.
Thresholds were fixed on 14 Nov 2026 before the clips were measured (see phase5_entry_tests.py).

| Meas. | Checkpoint | Pillar | Coverage | Tracks rating (rho) | Auto anchors (rho, gap/IQR) | Two-phone | Verdict |
|---|---|---|---|---|---|---|---|
| M01 | C01 (F01) | tracking | 94% | 0.57: not decided: rater gives nearly every clip the same rating | 0.91, 0.06: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M02 | C02 (F02) | stability | 94% | -0.23: not decided: rater gives nearly every clip the same rating | 0.91, 0.14: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M03 | C03 (F03) | access | 100% | 0.17: not decided: rater gives nearly every clip the same rating | 0.96, 0.02: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M04 | C04 (F04) | tracking | 100% | -0.24: fail (provisional: model ratings) | 0.9, 0.13: pass | pending (Phase 3 net session) | robust; model disagrees (provisional) |
| M05 | C05 (F05) | access | 94% | -0.48: not decided: rater gives nearly every clip the same rating | 0.97, 0.08: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M06 | C06 (F06) | stability | 100% | -0.2: not decided: rater gives nearly every clip the same rating | 0.86, 0.14: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M07 | C07 (F07) | access | 94% | n/a: not decided: rater gives nearly every clip the same rating | 0.86, 0.24: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M08 | C08 (F08) | access | 94% | 0.32: pass (provisional: model ratings) | 0.92, 0.06: pass | pending (Phase 3 net session) | pass so far (two-phone pending) |
| M09 | C09 (F09) | flow | 94% | -0.1: fail (provisional: model ratings) | 0.3, 0.08: fail | pending (Phase 3 net session) | fail |
| M10 | C10 (F10) | tracking | 100% | 0.25: fail (provisional: model ratings) | 0.8, 0.16: pass | pending (Phase 3 net session) | robust; model disagrees (provisional) |
| M11 | C11 (F11) | stability | 94% | -0.06: not decided: rater gives nearly every clip the same rating | 0.97, 0.03: pass | pending (Phase 3 net session) | robust; awaiting coach ratings |
| M12 | C12 (F12) | flow | 100% | 0.54: pass (provisional: model ratings) | 0.99, 0.00: pass | pending (Phase 3 net session) | pass so far (two-phone pending) |
| M13 | C12 (F12) | flow | 76% | 0.19: fail (provisional: model ratings) | 0.9, 0.00: pass | pending (Phase 3 net session) | fail |

## Pillars

- access: 0 passing (none); 4 robust and awaiting coach ratings (M03, M05, M07, M08) -> could reach 2 once coaches rate checkpoints
- tracking: 0 passing (none); 3 robust and awaiting coach ratings (M01, M04, M10) -> could reach 2 once coaches rate checkpoints
- stability: 0 passing (none); 3 robust and awaiting coach ratings (M02, M06, M11) -> could reach 2 once coaches rate checkpoints
- flow: 0 passing (none); 1 robust and awaiting coach ratings (M12) -> NOT ASSESSED (needs 2)
