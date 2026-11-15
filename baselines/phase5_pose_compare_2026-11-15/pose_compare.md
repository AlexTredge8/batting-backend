# Pose model comparison: MediaPipe against RTMPose

RTMPose mode: balanced. Same clips, same key frames, same measurements and tests.

| | MediaPipe (current) | RTMPose |
|---|---|---|
| Clips | 18 | 18 |
| Mean detection rate | 95% | 100% |
| Worst clip | Drive_Beginner 4 41% | Alex T_Straightdrive Good 100% |
| Set-up jitter (median, % body length per frame; lower is steadier) | 0.48 | 0.60 |
| Nose jumps over 25% body length (all clips) | 1 | 0 |
| Measurements passing the automatic-key-frame test (of 13) | 12 | 12 |
| Mean measurement coverage | 95% | 97% |

Per clip: `pose_quality.csv`. Per measurement: `mediapipe/entry_tests.md`, `rtmpose/entry_tests.md`.
Key frames come from the MediaPipe pipeline in both columns, so this compares landmarks, not detectors.
