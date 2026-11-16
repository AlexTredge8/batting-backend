# Video-model judge v2: pairwise + crop (2026-11-16)

Model gemini-3.1-pro-preview, 8 frames/s, temperature 0, seed 20261115.
Requests sent 55 of 70; spent about GBP 1.43 (cap 1.50; tokens: prompt 204250, answer 6161, thinking 79353; priced at USD 2.5/15.0 per million in/out, GBP 0.8/USD).

**Stopped by the cost guard before all requests were sent.**

- Spearman vs Alex: 0.36 (n=15); vs Cam: 0.39 (n=14)
- Clip-pair order agreement (model order vs coach order over all clip pairs): Alex 0.63 (105 pairs), Cam 0.66 (90 pairs)
- Pairs answered 55 of 55 sent; Bradley-Terry scale separation reliability 0.14
- Each pick vs Alex's score order: 0.56 of 55 (high confidence only: 0.56 of 55); vs Cam: 0.57 of 47
- Position bias: drive A (shown first) picked in 0.55 of answers (z = 0.67; A/B order was random per pair, so a bias adds noise but not a direction)

Crop: 15 of 15 clips cropped around the batter (crop keeps 5%-43% of the frame area); pad 25%, long side up to 720 px. Boxes in crop_boxes.csv.
