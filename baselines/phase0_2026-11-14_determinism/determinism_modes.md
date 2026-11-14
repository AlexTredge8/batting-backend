# Determinism by pose running mode

Each clip run 3 times, each run in a fresh process, automatic anchors.

| Clip | Mode | Identical | Max landmark drift | Overall per run |
|---|---|---|---|---|
| Drive_Average 2.MOV | video | Y | 0.00e+00 | 84/84/84 |
| Drive_Average 2.MOV | image | Y | 0.00e+00 | 83/83/83 |
| Drive_Average.MOV | video | Y | 0.00e+00 | 92/92/92 |
| Drive_Average.MOV | image | Y | 0.00e+00 | 92/92/92 |
| Drive_Beginner 2.MOV | video | Y | 0.00e+00 | 84/84/84 |
| Drive_Beginner 2.MOV | image | Y | 0.00e+00 | 59/59/59 |
| Drive_Beginner 3.MOV | video | Y | 0.00e+00 | 42/42/42 |
| Drive_Beginner 3.MOV | image | Y | 0.00e+00 | 75/75/75 |
| Drive_Beginner 4.MOV | video | Y | 0.00e+00 | 85/85/85 |
| Drive_Beginner 4.MOV | image | Y | 0.00e+00 | 75/75/75 |
| Drive_beginner.MOV | video | Y | 0.00e+00 | 27/27/27 |
| Drive_beginner.MOV | image | Y | 0.00e+00 | 71/71/71 |
| Viv P_Offdrive Elite.mov | video | Y | 0.00e+00 | 92/92/92 |
| Viv P_Offdrive Elite.mov | image | Y | 0.00e+00 | 100/100/100 |
| Cam T_LH Offdrive Elite.mov | video | Y | 0.00e+00 | 100/100/100 |
| Cam T_LH Offdrive Elite.mov | image | Y | 0.00e+00 | 92/92/92 |

- video: 8/8 clips identical across runs.

- image: 8/8 clips identical across runs.

Image mode against video mode (run 1 of each):

| Clip | Overall video → image | Largest anchor move (frames) |
|---|---|---|
| Drive_Average 2.MOV | 84 → 83 | 34 |
| Drive_Average.MOV | 92 → 92 | 38 |
| Drive_Beginner 2.MOV | 84 → 59 | 27 |
| Drive_Beginner 3.MOV | 42 → 75 | 23 |
| Drive_Beginner 4.MOV | 85 → 75 | 2 |
| Drive_beginner.MOV | 27 → 71 | 1 |
| Viv P_Offdrive Elite.mov | 92 → 100 | 26 |
| Cam T_LH Offdrive Elite.mov | 100 → 92 | 27 |
