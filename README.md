# BattingIQ Backend - Phase 1 Proof of Concept

Processes a cricket batting video (filmed from bowler's end) and extracts pose-based coaching measurements using MediaPipe BlazePose.

## What it does

1. Runs pose detection on every frame of your video
2. Extracts coaching-relevant measurements:
   - **Head lateral offset** — is the head falling to off side?
   - **Shoulder openness** — is the batter opening up too early?
   - **Stance width** — base stability
   - **Knee bend angles** — power position
   - **Wrist height** — hand/bat position tracking
3. Outputs an annotated video with pose overlay
4. Outputs a JSON file with frame-by-frame + summary data

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python poc_analyse.py path/to/your/batting_video.mp4
```

## Output

Creates an `output/` folder containing:
- `yourfile_annotated.mp4` — video with pose skeleton drawn on
- `yourfile_analysis.json` — all measurements + summary stats

## Camera Angle

This PoC is designed for **bowler's end** filming (camera behind the bowler, looking at the batter). This is the most natural filming angle for users.

## Next Steps

- Phase 2: Add rule-based coaching feedback logic
- Phase 3: Wrap in FastAPI service
- Phase 4: Connect to Lovable frontend
