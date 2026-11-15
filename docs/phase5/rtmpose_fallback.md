# RTMPose fallback for pose extraction

Status: built 15 Nov 2026, **off by default**. Production still runs MediaPipe only until `POSE_FALLBACK=rtmpose` is set on Railway.

Why: `baselines/phase5_pose_compare_2026-11-15/pose_compare.md` found that RTMPose (rtmlib `BodyWithFeet`, balanced mode) fixes MediaPipe's two capture failures on the 18 graded clips. On Drive_Beginner 4, MediaPipe found the batter in 41% of frames and RTMPose in 100%. On Ollie P, MediaPipe tracked the wrong person (87% of frames). On the other clips MediaPipe is slightly steadier (set-up jitter 0.48 against 0.60 % body length per frame), so RTMPose is used only as a fallback.

## What it does

`pose_extractor.extract_poses` always runs MediaPipe first. With `POSE_FALLBACK=rtmpose` it then checks two triggers:

1. **Detection rate**: MediaPipe found a pose in fewer than `POSE_FALLBACK_MIN_DETECTION` (0.90) of the processed frames.
2. **Jump rule**: the nose moves more than `POSE_FALLBACK_JUMP_BL` (0.25) body lengths between consecutive frames more than `POSE_FALLBACK_MAX_JUMPS` (1) times. Body length is the median ankle-to-nose height. Points under 0.2 visibility are dropped, and a frame without a pose never counts as a jump. These are the same definitions `phase5_pose_compare.py` used.

If either trigger fires, the clip is re-extracted with RTMPose on the same frames (same `frame_step`, scaling and orientation handling). The batter is the tallest person in the first frame with a detection, then the nearest skeleton from frame to frame, as in the comparison. The result is a list of `FramePose` objects in the usual format:

- The 21 Halpe-26 points that have a MediaPipe equivalent are mapped to their MediaPipe slots (body, heels and big toes). `x` and `y` are normalised to the frame, `z = 0` (RTMPose has no depth). The only code that reads `z` is a 15% "foot planted" term in the storyboard's choice of the front-foot-down still. With RTMPose that term is the same for every candidate frame, so it has no effect. No rule reads `z`, and `visibility` is the RTMPose score clipped to 0–1.
- The 12 MediaPipe slots RTMPose has no point for (inner and outer eyes, mouth, fingers) are filled from the neighbouring point at visibility 0. Every frame therefore has 33 finite landmarks, and visibility-gated code such as the annotator skips them.

RTMPose replaces MediaPipe only if it is not worse: if its detection rate is lower or it has more nose jumps, MediaPipe is kept. If rtmlib or onnxruntime is not installed, or RTMPose fails in any way, MediaPipe is kept and the reason is recorded. rtmlib is imported only when a fallback actually runs, so the server starts without it. The model is loaded once per process.

`POSE_FALLBACK=rtmpose_always` forces RTMPose on every clip. It is for testing and timing only, because none of the thresholds were calibrated on RTMPose landmarks.

### What the report records

`metadata.pose_backend`: `"mediapipe"` or `"rtmpose_fallback"`. It is also copied to `analysis_quality.pose_backend`.

`metadata.pose_fallback`:

| Field | Meaning |
|---|---|
| `mode` | `off`, `rtmpose` or `rtmpose_always` |
| `triggered`, `reason` | whether a trigger fired and which one (e.g. `detection_rate 40.9% below 90%`), or `not triggered` |
| `detection_rate_mediapipe`, `detection_rate_rtmpose` | % of processed frames with a pose |
| `nose_jumps_mediapipe`, `nose_jumps_rtmpose`, `body_length_px` | the jump rule's inputs |
| `error` | why MediaPipe was kept after a trigger fired (not installed, RTMPose failed, or RTMPose was not better) |
| `rtmpose_seconds`, `rtmpose_ms_per_frame`, `rtmpose_cpu_ms_per_frame` | cost of the RTMPose pass |

When RTMPose is used, `metadata.detection_rate`, `frames_with_pose` and `frames_processed` describe the RTMPose poses. `metadata.pose_seconds_mediapipe` is always recorded. `GET /diag` shows `processing.pose_fallback`: the mode, the thresholds, whether rtmlib and onnxruntime are installed, which model files are cached, and whether the model is loaded. It reports these without importing rtmlib.

### Which graded clips it would catch

From the per-clip numbers in `baselines/phase5_pose_compare_2026-11-15/pose_quality.csv`:

- The detection trigger fires on exactly the two failures: Drive_Beginner 4 (40.9%) and Ollie P (87.2%). The next lowest clip is Dan P at 94.8%.
- The jump rule fires on none. No clip has more than one jump (Ollie P has one), so the wrong-person clip is caught by its detection rate.

The real clips are not on this machine, so this comes from the baseline table and was not re-run through the server. That table counts a frame as detected when the nose is visible (≥ 0.2), while the server counts any frame with a pose, so the two rates can differ slightly. To confirm on the Mac:

```
POSE_FALLBACK=rtmpose python run_analysis.py "<clips>/Drive_Beginner 4.MOV"
POSE_FALLBACK=rtmpose python run_analysis.py "<clips>/Ollie P_Offdrive Average.mov"
```

Look for `Pose backend: RTMPose` in the output and `pose_backend` in the JSON.

## Measured cost (reference clip, this machine)

`test_batting.mov` is 480×848 at 30 fps, with 110 decoded frames. Measurements were taken on 15 Nov 2026 on the 4-vCPU Linux sandbox, with a clean Python 3.10 venv: `requirements.txt` plus the recipe below. Each figure is the range over two runs.

| | Default (MediaPipe) | Forced RTMPose (`rtmpose_always`) |
|---|---|---|
| Pose extraction | 7.1–7.7 s (≈ 66 ms/frame) | MediaPipe 7.4–7.7 s, then RTMPose 29.4 s (**267–272 ms/frame wall**, 1.05 s CPU/frame over 4 threads) |
| Whole analysis (wall) | 10.7–10.9 s | 40.8–40.9 s |
| Whole analysis (CPU) | 10.8–11.2 s | 127.7–128.4 s |
| Peak memory (RSS) | 536 MB | 882–914 MB (**+350 to +380 MB**) |

RTMPose speed by thread count, measured by limiting ONNX Runtime's threads:

| ONNX Runtime threads | ms per frame | 110-frame clip |
|---|---|---|
| 1 | 658 | 72 s |
| 2 | 394 | 43 s |
| 4 | 263 | 29 s |

Railway's share of CPU decides which row applies. Plan for 30–75 s extra on a clip that falls back, on top of the MediaPipe pass that already ran.

Memory: loading the two ONNX models adds about 300 MB (a process grows from 145 MB to about 450 MB). They stay loaded for the life of the process after the first fallback.

Models and packages:

- Two models (balanced mode): YOLOX-m person detector, 94.2 MB zip (101 MB .onnx), and RTMPose-m Halpe-26, 52.1 MB zip (56 MB .onnx). That is **146 MB to download, 157 MB on disk**, in `~/.cache/rtmlib/hub/checkpoints`. The files come from download.openmmlab.com; rtmlib falls back to a Hugging Face mirror if that host is down.
- Packages: onnxruntime 1.23.2 (about 52 MB installed), rtmlib 0.0.16 (0.6 MB) and tqdm. The Docker image grows by about 210 MB.

Cold start:

- **Models baked into the image** (recommended): the first clip that falls back pays a 0.4 s model load, and later clips pay nothing.
- **Models not baked in**: the first clip that falls back after every deploy or restart also downloads 146 MB. This took 33 s here, and Railway's network will differ. Railway's container disk does not persist between deploys.
- **Fallback not triggered or off**: no cost. rtmlib is never imported.

## Result on the reference clip, forced through RTMPose

| | Default (MediaPipe) | `POSE_FALLBACK=rtmpose_always` |
|---|---|---|
| Score | **92**, Excellent | **92**, Excellent |
| Pillars (access / tracking / stability / flow) | 17 / 25 / 25 / 25 | 17 / 25 / 25 / 25 |
| Faults | A5 shoulders lagging (−3): gap −28.7° | A5 shoulders lagging (−3): gap −39.5° |
| Detection rate / nose jumps | 100% / 0 | 100% / 0 (MediaPipe also 100% / 0) |
| setup | 60 | 61 |
| hands_start_up | 61 | 64 |
| front_foot_down | 77 | 77 |
| hands_peak | 73 | 72 |
| contact | 81 | 81 |
| follow_through | 109 | 102 |
| Whole analysis | 10.8 s | 40.9 s |

With `POSE_FALLBACK=rtmpose` (the real fallback mode), the reference clip does not trigger: 100% detection and no jumps. It gives exactly the default result, and the RTMPose pass does not run.

Read this as follows. The score and the faults are unchanged. The key frames stay within 3 frames, except follow-through, which comes 7 frames earlier. The A5 measurement moves by 11°, because RTMPose places the hips and shoulders differently from MediaPipe (Halpe and COCO hip points sit higher and wider). Every rule threshold was calibrated on MediaPipe, so a clip scored on RTMPose landmarks is less comparable with the others. The trade is still worth it for the clips the fallback is meant for, where MediaPipe lost the batter or followed the wrong person.

**Install rtmlib with `--no-deps`.** A plain `pip install rtmlib` also installs `opencv-python` (5.0) and numpy 2.x over the pinned `opencv-python-headless` 4.8 and numpy 1.26. In a development venv set up that way, the default MediaPipe anchors on the reference clip moved by a frame (61/62/108 instead of 60/61/109), and the forced RTMPose run scored 85 instead of 92. The clean `--no-deps` venv reproduces the pinned anchors exactly. rtmlib needs only `cv2`, which the existing OpenCV packages already provide.

## How to enable it on Railway (later; not enabled)

1. **Build the image with the packages and models.** In the Railway service, open Variables and add `INSTALL_RTMPOSE=1`. The Dockerfile declares `ARG INSTALL_RTMPOSE=0`. Railway passes service variables to the build as build arguments, so with this variable set the build runs:
   ```
   pip install --no-cache-dir --no-deps rtmlib==0.0.16
   pip install --no-cache-dir onnxruntime==1.23.2 tqdm
   python -c "from rtmlib import BodyWithFeet; BodyWithFeet(mode='balanced', backend='onnxruntime', device='cpu')"
   ```
   The last line downloads the two models into the image, so there is no download at request time. The build log should show two `load ... .onnx with onnxruntime backend` lines.
2. **Turn the fallback on.** Add `POSE_FALLBACK=rtmpose`. Optional variables, each shown with its default: `POSE_FALLBACK_MIN_DETECTION=0.90`, `POSE_FALLBACK_JUMP_BL=0.25`, `POSE_FALLBACK_MAX_JUMPS=1`, `POSE_FALLBACK_RTM_MODE=balanced`. If `POSE_FALLBACK_RTM_MODE` is changed to `lightweight` or `performance`, the build in step 1 only bakes the balanced models, and the other mode's models are downloaded on first use.
3. **Redeploy**, then check `GET /diag`. `processing.pose_fallback` should show `"mode": "rtmpose"`, `"rtmlib_installed": true`, `"onnxruntime_installed": true` and the two `.onnx` files under `model_files_cached`.
4. **Check memory headroom.** Each analysis that falls back needs about 400 MB more than today: the models (about 300 MB, loaded once) plus working memory. `/diag` and the `[analyse]` log lines show memory in use. If the service is near its limit, raise it before step 2.
5. **Expect slower fallback clips.** A clip that falls back takes 30–75 s longer, depending on Railway's CPU share. The `/analyse` upload route runs the analysis inside an async handler, so other requests to the same worker wait during that time. Clips that do not trigger the fallback are unchanged.
6. **Verify on a known-bad clip.** Upload Drive_Beginner 4. The response metadata should show `"pose_backend": "rtmpose_fallback"`, with `pose_fallback.reason` naming the detection rate.

To switch it off, set `POSE_FALLBACK=off` or delete the variable. This needs no rebuild, because the packages are harmless when unused. To remove the packages as well, delete `INSTALL_RTMPOSE` and redeploy.

Do not add rtmlib to `requirements.txt` as a normal line. pip cannot apply `--no-deps` to a single requirement, so it would pull `opencv-python` and numpy 2 into the production image (see above).

## Running it locally

```
pip install --no-deps rtmlib==0.0.16 && pip install onnxruntime==1.23.2 tqdm
POSE_FALLBACK=rtmpose_always python run_analysis.py test_batting.mov      # force RTMPose
POSE_FALLBACK=rtmpose python run_analysis.py <clip>                       # real fallback rule
python -m pytest tests/test_pose_fallback.py -q                           # trigger logic (no rtmlib needed)
BATTINGIQ_E2E=1 python -m pytest tests/test_reference_video_e2e.py -q     # default path unchanged
BATTINGIQ_E2E_RTMPOSE=1 python -m pytest tests/test_rtmpose_fallback_e2e.py -q   # forced RTMPose end to end
```

Code: `pose_extractor.py` (`_apply_pose_fallback`, `_extract_with_rtmpose`, `pose_fallback_status`). Settings: `config.py` (`POSE_FALLBACK*`). Tests: `tests/test_pose_fallback.py` (mocked extractors: detection below the threshold, the jump rule, rtmlib missing, RTMPose failing or worse, forced mode) and `tests/test_rtmpose_fallback_e2e.py` (opt-in).
