# Wrong-plane test (Phase 0)

Clips: 17 with validated anchors. n is small, so all of this is exploratory.
Verdict rules were fixed before the data was seen (see the script docstring).

| Rule | Status | Pillar | Foreshortening ratio | Depth share (descriptive) | ρ vs coach pillar [95% CI] | ρ vs overall | Verdict |
|---|---|---|---|---|---|---|---|
| A2 | suspended | access | 0.59 | 0.56 | 0.01 [-0.54, 0.54] | 0.02 | supported |
| A3 | suspended | access | 0.69 | 0.61 | 0.26 [-0.20, 0.64] | 0.33 | supported |
| A4 | suspended | access | 0.87 | 0.82 | 0.47 [-0.05, 0.74] | 0.46 | not supported |
| S1 | suspended | stability | 0.62 | n/a | 0.46 [-0.07, 0.84] | 0.43 | inconclusive |
| S3 | suspended | stability | 0.81 | 0.74 | 0.38 [-0.01, 0.75] | 0.33 | inconclusive |
| A5 | active | access | 0.80 | 0.71 | -0.18 [-0.66, 0.36] | -0.15 | inconclusive |
| S4 | active | stability | 0.80 | 0.79 | -0.03 [-0.56, 0.53] | 0.02 | inconclusive |
| T2 | active | tracking | 0.97 | 0.66 | -0.30 [-0.67, 0.23] | -0.33 | not supported |
| F4 | active | flow | 0.30 | 0.81 | -0.09 [-0.52, 0.37] | -0.04 | supported |

Foreshortening ratio by camera position (descriptive; tier is confounded with venue):

- A2: pending: 0.59 (n=17)
- A3: pending: 0.69 (n=17)
- A4: pending: 0.87 (n=17)
- S1: pending: 0.62 (n=17)
- S3: pending: 0.81 (n=17)
- A5: pending: 0.80 (n=17)
- S4: pending: 0.80 (n=17)
- T2: pending: 0.97 (n=17)
- F4: pending: 0.30 (n=17)

Shoulder line at the measured frames (A5, S4 use atan2(|dx|, |dy|), which is unstable when the shoulders point at the camera): median 42 px.

Limits: MediaPipe image z is a rough depth estimate; the ratio uses each clip as its own reference, so a segment that never faces the camera has no unforeshortened length to compare with.
