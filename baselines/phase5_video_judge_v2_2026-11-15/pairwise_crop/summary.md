# Video-model judge v2: pairwise + crop (2026-11-15)

Model gemini-3.1-pro-preview, 8 frames/s, temperature 0, seed 20261115.
Requests sent 56 of 70; spent about GBP 1.47 (cap 1.50; tokens: prompt 208305, answer 6264, thinking 81354; priced at USD 2.5/15.0 per million in/out, GBP 0.8/USD).

**Stopped by the cost guard before all requests were sent.**

- Spearman vs Alex: 0.35 (n=15); vs Cam: 0.38 (n=14)
- Clip-pair order agreement (model order vs coach order over all clip pairs): Alex 0.62 (105 pairs), Cam 0.64 (90 pairs)
- Pairs answered 56 of 56 sent; Bradley-Terry scale separation reliability 0.16
- Each pick vs Alex's score order: 0.55 of 56 (high confidence only: 0.55 of 56); vs Cam: 0.57 of 47
- Position bias: drive A (shown first) picked in 0.54 of answers (z = 0.53; A/B order was random per pair, so a bias adds noise but not a direction)

Crop: 15 of 15 clips cropped around the batter (crop keeps 5%-43% of the frame area); pad 25%, long side up to 720 px. Boxes in crop_boxes.csv.
