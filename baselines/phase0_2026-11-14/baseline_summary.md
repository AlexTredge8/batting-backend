# Phase 0 baseline — 2026-11-14

- Commit: `6d43a94e6f003b47fd05287b72b2cad461d473ae`; RULES_VERSION 2.2.0; active rules A5, T2, S4, F4
- Environment: Python 3.10.20, MediaPipe 0.10.9, macOS-26.5.2-arm64-arm-64bit (arm64), ffmpeg /opt/homebrew/bin/ffmpeg
- Clips: 17. Excluded: Female_Average.mp4 (filmed_correctly = N; no anchor truth).
- Reference clip (not held out): Viv P_Offdrive Elite.mov.
- Coach-to-coach error: not yet measured (Phase 2). No comparison with targets is made here.

## Mode: auto

**Mean absolute error vs coach (points)**

| Set | n | Overall | Access | Tracking | Stability | Flow |
|---|---|---|---|---|---|---|
| all clips | 17 | 19.65 | 6.88 | 7.29 | 8.94 | 8.41 |
| without reference clip | 16 | 20.75 | 6.88 | 7.62 | 9.44 | 8.81 |

**Against a constant scorer (each clip predicted as the mean coach score of the other clips)**

| Measure | System MAE | Constant MAE | Verdict |
|---|---|---|---|
| Overall | 19.65 | 21.74 | beats |
| Access | 6.88 | 5.70 | does not beat |
| Tracking | 7.29 | 4.60 | does not beat |
| Stability | 8.94 | 5.70 | does not beat |
| Flow | 8.41 | 5.88 | does not beat |

**Mean signed bias, overall (system minus coach)**

| Group | n | Bias | Flag |
|---|---|---|---|
| tier Beginner | 5 | 31.00 | ABOVE 5 |
| tier Average | 4 | 15.50 | ABOVE 5 |
| tier Good Club | 4 | 7.25 | ABOVE 5 |
| tier Elite | 4 | 3.50 |  |
| coach band <40 | 4 | 30.25 | ABOVE 5 |
| coach band 40-59 | 4 | 24.00 | ABOVE 5 |
| coach band 60-79 | 4 | 6.25 | ABOVE 5 |
| coach band >=80 | 5 | 3.60 |  |

- Spearman rank correlation, system vs coach overall: 0.66 (n = 17)
- Within ±5 overall (continuity only): 5/17
- Tier means (sanity check only): Beginner 62.8, Average 73.2, Good Club 83.5, Elite 93.8; monotonic: yes
- Pillar results by status: fault 25, no_fault 43
- Free 25s (no_fault while coach pillar ≤ 15): 14: Drive_Average 2.MOV (tracking, coach 15); Drive_Average.MOV (tracking, coach 14); Drive_Average.MOV (flow, coach 15); Drive_Beginner 2.MOV (stability, coach 12); Drive_Beginner 2.MOV (flow, coach 12); Drive_Beginner 3.MOV (stability, coach 6); Drive_Beginner 4.MOV (flow, coach 7); Drive_beginner.MOV (tracking, coach 3); Drive_beginner.MOV (stability, coach 7); Oscar V_Beginner.mp4 (stability, coach 8); Oscar V_Beginner.mp4 (flow, coach 9); Straight Drive_Beginner.mp4 (access, coach 13); Straight Drive_Beginner.mp4 (tracking, coach 14); Straight Drive_Beginner.mp4 (stability, coach 12)
- System priority fix per clip: F4 5, none 4, T2 3, S4 3, A5 2. No coach priority fix exists to compare against (coach_notes is empty).

## Mode: validated

**Mean absolute error vs coach (points)**

| Set | n | Overall | Access | Tracking | Stability | Flow |
|---|---|---|---|---|---|---|
| all clips | 17 | 25.76 | 8.88 | 7.82 | 8.35 | 8.82 |
| without reference clip | 16 | 27.25 | 9.00 | 8.19 | 8.81 | 9.25 |

**Against a constant scorer (each clip predicted as the mean coach score of the other clips)**

| Measure | System MAE | Constant MAE | Verdict |
|---|---|---|---|
| Overall | 25.76 | 21.74 | does not beat |
| Access | 8.88 | 5.70 | does not beat |
| Tracking | 7.82 | 4.60 | does not beat |
| Stability | 8.35 | 5.70 | does not beat |
| Flow | 8.82 | 5.88 | does not beat |

**Mean signed bias, overall (system minus coach)**

| Group | n | Bias | Flag |
|---|---|---|---|
| tier Beginner | 5 | 42.60 | ABOVE 5 |
| tier Average | 4 | 34.00 | ABOVE 5 |
| tier Good Club | 4 | 5.25 | ABOVE 5 |
| tier Elite | 4 | 3.50 |  |
| coach band <40 | 4 | 42.75 | ABOVE 5 |
| coach band 40-59 | 4 | 36.25 | ABOVE 5 |
| coach band 60-79 | 4 | 10.50 | ABOVE 5 |
| coach band >=80 | 5 | 5.20 | ABOVE 5 |

- Spearman rank correlation, system vs coach overall: 0.23 (n = 17)
- Within ±5 overall (continuity only): 2/17
- Tier means (sanity check only): Beginner 74.4, Average 91.8, Good Club 81.5, Elite 93.8; monotonic: no
- Pillar results by status: fault 17, no_fault 51
- Free 25s (no_fault while coach pillar ≤ 15): 22: Dan P_Offdrive Average.MOV (access, coach 15); Drive_Average 2.MOV (access, coach 13); Drive_Average 2.MOV (tracking, coach 15); Drive_Average.MOV (access, coach 13); Drive_Average.MOV (tracking, coach 14); Drive_Average.MOV (flow, coach 15); Drive_Beginner 2.MOV (tracking, coach 14); Drive_Beginner 2.MOV (stability, coach 12); Drive_Beginner 2.MOV (flow, coach 12); Drive_Beginner 3.MOV (flow, coach 6); Drive_Beginner 4.MOV (stability, coach 9); Drive_beginner.MOV (access, coach 2); Drive_beginner.MOV (tracking, coach 3); Drive_beginner.MOV (stability, coach 7); Drive_beginner.MOV (flow, coach 6); Oscar V_Beginner.mp4 (access, coach 5); Oscar V_Beginner.mp4 (stability, coach 8); Oscar V_Beginner.mp4 (flow, coach 9); Straight Drive_Beginner.mp4 (access, coach 13); Straight Drive_Beginner.mp4 (tracking, coach 14); Straight Drive_Beginner.mp4 (stability, coach 12); Straight Drive_Beginner.mp4 (flow, coach 10)
- System priority fix per clip: none 6, T2 5, A5 3, S4 2, F4 1. No coach priority fix exists to compare against (coach_notes is empty).

## Auto against validated

- Mean |overall difference|: 10.12 points (n = 17)
- Largest: Dan P_Offdrive Average.MOV (33), Drive_Beginner 4.MOV (30), Straight Drive_Beginner.mp4 (25)

## Determinism

- 11/17 clips identical across auto and auto_repeat. Fields that varied: A5, F4, S4, T2, access, flow, follow_through_frame, front_foot_down_frame, hands_peak_frame, hands_start_up_frame, overall, setup_frame, stability, tracking.

## Left-hander check

- 2 runs of the left-handed clip with handedness=left; no field differs from the right-handed run.

