# Video-model judge v2: checkpoints + crop (2026-11-16)

Model gemini-3.1-pro-preview, 8 frames/s, temperature 0, thinking low.
Requests sent 60 of 60; spent about GBP 0.42 (cap 1.00; tokens: prompt 110829, answer 3656, thinking 12915; priced at USD 2.5/15.0 per million in/out, GBP 0.8/USD).

- Spearman vs Alex: 0.34 (n=15); vs Cam: 0.29 (n=14)
- Clip-pair order agreement (model order vs coach order over all clip pairs): Alex 0.61 (105 pairs), Cam 0.59 (90 pairs)
- Clips rated 15 (15 with all 12 checkpoints); checkpoint score = share of available marks, used for the Spearman above
- All ratings: 2: 130, 1: 40, 0: 6, X: 4 (2 yes, 1 partly, 0 no, X cannot see)
- Main fault (earliest checkpoint rated no) = Cam's main fault: 0.21 of 14 clips; counting 'partly' when nothing is rated no: 0.00; = Cam's main or second fault: 0.21; Cam's main-fault checkpoint rated no: 0.00

| checkpoint | yes (2) | partly (1) | no (0) | cannot see (X) |
|---|---|---|---|---|
| C01 | 15 | 0 | 0 | 0 |
| C02 | 1 | 14 | 0 | 0 |
| C03 | 15 | 0 | 0 | 0 |
| C04 | 14 | 0 | 0 | 1 |
| C05 | 13 | 1 | 0 | 1 |
| C06 | 7 | 8 | 0 | 0 |
| C07 | 14 | 0 | 1 | 0 |
| C08 | 2 | 12 | 0 | 1 |
| C09 | 15 | 0 | 0 | 0 |
| C10 | 10 | 4 | 0 | 1 |
| C11 | 13 | 0 | 2 | 0 |
| C12 | 11 | 1 | 3 | 0 |

Crop: 15 of 15 clips cropped around the batter (crop keeps 5%-43% of the frame area); pad 25%, long side up to 720 px. Boxes in crop_boxes.csv.
