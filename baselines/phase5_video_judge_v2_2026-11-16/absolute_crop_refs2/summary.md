# Video-model judge v2: absolute + crop (2026-11-16)

Model gemini-3.1-pro-preview, 8 frames/s, temperature 0, refs per tier 2.
Requests sent 15 of 15; spent about GBP 0.75 (cap 0.90; tokens: prompt 186985, answer 4199, thinking 27223; priced at USD 2.5/15.0 per million in/out, GBP 0.8/USD).

- Spearman vs Alex: 0.68 (n=15); vs Cam: 0.67 (n=14)
- Clip-pair order agreement (model order vs coach order over all clip pairs): Alex 0.73 (105 pairs), Cam 0.74 (90 pairs)
- Clips scored 15; overall score histogram: 0-19: 1, 20-39: 5, 40-59: 0, 60-79: 0, 80-99: 9

Crop: 18 of 18 clips cropped around the batter (crop keeps 4%-43% of the frame area); pad 25%, long side up to 720 px. Boxes in crop_boxes.csv.
