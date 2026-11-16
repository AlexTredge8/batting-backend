# Phase 5 entry tests

Checkpoint ratings from: model. Model ratings only, so test 1 is provisional.
Thresholds were fixed on 14 Nov 2026 before the clips were measured (see phase5_entry_tests.py).

| Meas. | Checkpoint | Pillar | Coverage | Tracks rating (rho) | Auto anchors (rho, gap/IQR) | Worse filming | Verdict |
|---|---|---|---|---|---|---|---|
| M01 | C01 (F01) | tracking | 94% | 0.57: not decided: rater gives nearly every clip the same rating | 0.91, 0.06: pass | fail (lowres) | fail |
| M02 | C02 (F02) | stability | 94% | -0.31: not decided: rater gives nearly every clip the same rating | 0.9, 0.16: pass | fail (lowres, tilt) | fail |
| M03 | C03 (F03) | access | 100% | 0.17: not decided: rater gives nearly every clip the same rating | 0.96, 0.02: pass | fail (lowres, lowfps) | fail |
| M04 | C04 (F04) | tracking | 100% | -0.32: fail (provisional: model ratings) | 0.88, 0.13: pass | fail (lowres, tilt) | fail |
| M05 | C05 (F05) | access | 94% | -0.48: not decided: rater gives nearly every clip the same rating | 0.97, 0.07: pass | fail (lowres) | fail |
| M06 | C06 (F06) | stability | 100% | -0.2: not decided: rater gives nearly every clip the same rating | 0.86, 0.14: pass | fail (lowres) | fail |
| M07 | C07 (F07) | access | 94% | n/a: not decided: rater gives nearly every clip the same rating | 0.87, 0.24: pass | fail (lowres, lowfps) | fail |
| M08 | C08 (F08) | access | 94% | 0.32: pass (provisional: model ratings) | 0.92, 0.06: pass | fail (lowres) | fail |
| M09 | C09 (F09) | flow | 94% | -0.1: fail (provisional: model ratings) | 0.69, 0.16: fail | fail (lowres, lowfps, tilt) | fail |
| M10 | C10 (F10) | tracking | 100% | 0.25: fail (provisional: model ratings) | 0.88, 0.16: pass | fail (lowres, tilt) | fail |
| M11 | C11 (F11) | stability | 94% | -0.06: not decided: rater gives nearly every clip the same rating | 0.98, 0.02: pass | fail (lowres, tilt) | fail |
| M12 | C12 (F12) | flow | 100% | 0.54: pass (provisional: model ratings) | 1.0, 0.00: pass | fail (lowres) | fail |
| M13 | C12 (F12) | flow | 82% | 0.25: fail (provisional: model ratings) | 0.91, 0.00: pass | fail (lowres) | fail |
| M14 | C09 (F09) | flow | 53% | -0.72: not enough ratings | 0.93, 0.00: pass | fail (lowres, lowfps, tilt) | fail |
| M15 | C01 (F01) | tracking | 100% | 0.1: not decided: rater gives nearly every clip the same rating | 0.98, 0.01: pass | fail (tilt) | fail |
| M16 | C16 (F16) | stability | 100% | n/a: not enough ratings | 0.87, 0.22: pass | fail (lowres, tilt) | fail |
| M17 | C13 (F13) | flow | 100% | n/a: not enough ratings | 0.93, 0.04: pass | fail (lowres) | fail |
| M18 | C17 (F17) | tracking | 94% | n/a: not enough ratings | 0.99, 0.01: pass | fail (lowres, tilt) | fail |

## Pillars

- access: 0 passing (none); 0 robust and awaiting coach ratings (none) -> NOT ASSESSED (needs 2)
- tracking: 0 passing (none); 0 robust and awaiting coach ratings (none) -> NOT ASSESSED (needs 2)
- stability: 0 passing (none); 0 robust and awaiting coach ratings (none) -> NOT ASSESSED (needs 2)
- flow: 0 passing (none); 0 robust and awaiting coach ratings (none) -> NOT ASSESSED (needs 2)
