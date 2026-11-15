# Video-model judge, run 1 (15 Nov 2026)

Model gemini-3.1-pro-preview (gemini-2.5-pro is closed to new users), paid tier, temperature 0, trimmed face-blurred clips from the swipe file, rubric v0, three reference clips with Alex's scores (Viv 94, Dan 67, Drive_beginner 18). 15 clips x 2 runs, about GBP 0.62.

- **Consistent but uninformative.** Both runs identical. Spearman with Alex's scores 0.10, with Cam's 0.19. 12 of 15 clips scored 92–96 with "no real fault", including all three Beginner clips (coaches: Alex 26–35, Cam 0–10). 157 of 180 checkpoint ratings were "done well".
- **Likely cause:** Gemini samples video at 1 frame per second by default. The clips last 1.4–10.5 s (mean 3.0 s) and the shot about 0.5 s, so the model saw 2–4 frames and often none of the shot. Its written reasons describe generic good technique.
- **Run 2** samples at 8 frames per second (`--video-fps 8`, about GBP 0.8). If it still cannot order the clips, the video judge is dropped.
