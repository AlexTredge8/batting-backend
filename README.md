# BattingIQ Phase 1 — Proof of Concept

Extracts pose data from a cricket batting video filmed from the bowler's end. Uses MediaPipe BlazePose to track 33 body landmarks and calculates cricket-specific metrics.

## What it measures (from bowler's end view)

* **Shoulder openness** — is the batter side-on or opening up to the bowler?
* **Head offset** — is the head staying central or falling to off side?
* **Stance width** — how wide is the base?
* **Wrist height** — proxy for bat position (backlift, contact point)

## Setup

```bash
# Clone and enter the project
cd battingiq-poc

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python analyse.py path/to/your_batting_video.mp4
```

## Output

The script creates an `output/` folder containing:

* `annotated_video.mp4` — your video with pose skeleton overlay and live metrics
* `analysis.json` — frame-by-frame landmark data and summary statistics

## Filming tips for best results

* Film from behind the bowler's arm (bowler's end)
* Batter should be fully visible in frame (head to feet)
* Good lighting (avoid strong backlight)
* Keep camera steady (tripod ideal, but steady hand works)
* Film at 60fps if possible (phone slow-mo is great for this)

## Next steps

Once this POC works, Phase 2 adds:

* Phase detection (setup → backlift → stride → contact → follow-through)
* Rule-based coaching feedback
* Cricket-specific annotations on the video
