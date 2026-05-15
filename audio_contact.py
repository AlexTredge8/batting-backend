"""
Audio-first contact detection helpers.

Onset selection strategy (N-aware):
  N=0  → no audio onset found; caller falls back to pose detection.
  N=1  → single event in window; return it directly (audio_single).
  N=2  → exactly two distinct events (AUDIO_STRATEGY_N2="last"):
         On bowling machine videos the machine fires first, bat-on-ball
         is the second event. "last" reliably picks bat-on-ball here.
         Confirmed on 3/4 Elite bowling-machine videos (F3c, 2026-04-28).
  N≥3  → three or more distinct events (AUDIO_STRATEGY_N3_PLUS="loudest"):
         A third event after bat-on-ball is typically a follow-through
         thud or bounce. "last" would pick that thud. The loudest event
         is a more reliable proxy for bat-on-ball impact.
         Regression case: Cam T_LH Offdrive Elite — 3 events (machine
         fire frame 26, bat-on-ball frame 52, follow-through frame 83);
         "last" gave err=30; "loudest" gives err=1 (strength order:
         bat=0.74 > machine=0.66 > follow-through=0.64).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from config import (
    AUDIO_ONSET_DELTA,
    AUDIO_TEMPORAL_FILTER_START,
    AUDIO_TEMPORAL_FILTER_END,
    AUDIO_MIN_ONSET_GAP_SECONDS,
    AUDIO_STRATEGY_N2,
    AUDIO_STRATEGY_N3_PLUS,
)

try:
    import librosa
except Exception:  # pragma: no cover - graceful fallback if dependency is absent
    librosa = None


_TARGET_SR = 44100


def extract_audio_from_video(video_path: str) -> tuple[np.ndarray | None, int | None]:
    audio, sr, _ = extract_audio_from_video_with_diagnostics(video_path)
    return audio, sr


def extract_audio_from_video_with_diagnostics(
    video_path: str,
) -> tuple[np.ndarray | None, int | None, dict]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None, None, {"status": "ffmpeg_not_found"}

    if librosa is None:
        return None, None, {"status": "librosa_unavailable"}

    tmp_wav_fd = None
    tmp_wav_path = None
    command: list[str] | None = None
    try:
        tmp_wav_fd, tmp_wav_path = tempfile.mkstemp(suffix=".wav", prefix="battingiq_contact_")
        os.close(tmp_wav_fd)
        tmp_wav_fd = None

        command = [
            ffmpeg_path,
            "-i",
            str(video_path),
            "-vn",
            "-ar",
            str(_TARGET_SR),
            "-ac",
            "1",
            "-f",
            "wav",
            str(tmp_wav_path),
            "-y",
            "-loglevel",
            "error",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None, None, {
                "status": "ffmpeg_failed",
                "ffmpeg_command": command,
                "error": (completed.stderr or completed.stdout or "").strip() or None,
            }

        audio_array, sample_rate = librosa.load(str(tmp_wav_path), sr=_TARGET_SR, mono=True)
        if audio_array is None or sample_rate is None or len(audio_array) == 0:
            return None, None, {
                "status": "empty_audio",
                "ffmpeg_command": command,
            }

        return audio_array, int(sample_rate), {
            "status": "ok",
            "ffmpeg_command": command,
            "sample_rate": int(sample_rate),
            "sample_count": int(len(audio_array)),
        }
    except Exception as exc:  # pragma: no cover - defensive fallback
        return None, None, {
            "status": "audio_extract_exception",
            "ffmpeg_command": command,
            "error": str(exc),
        }
    finally:
        if tmp_wav_fd is not None:
            try:
                os.close(tmp_wav_fd)
            except OSError:
                pass
        if tmp_wav_path:
            try:
                Path(tmp_wav_path).unlink(missing_ok=True)
            except OSError:
                pass


def detect_contact_from_audio(
    audio_array: np.ndarray | None,
    sr: int | None,
    video_fps: float,
    video_n_frames: int,
) -> tuple[int | None, float]:
    frame, confidence, _ = detect_contact_from_audio_with_diagnostics(
        audio_array,
        sr,
        video_fps,
        video_n_frames,
    )
    return frame, confidence


def detect_contact_from_audio_with_diagnostics(
    audio_array: np.ndarray | None,
    sr: int | None,
    video_fps: float,
    video_n_frames: int,
) -> tuple[int | None, float, dict]:
    if librosa is None:
        return None, 0.0, {"status": "librosa_unavailable"}
    if audio_array is None or sr is None:
        return None, 0.0, {"status": "audio_unavailable"}

    try:
        onset_env = librosa.onset.onset_strength(y=audio_array, sr=sr)
        onset_times = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            units="time",
            backtrack=True,
            delta=AUDIO_ONSET_DELTA,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return None, 0.0, {"status": "onset_exception", "error": str(exc)}

    if onset_times is None or len(onset_times) == 0:
        return None, 0.0, {"status": "no_onsets"}

    filtered: list[dict] = []
    earliest_contact_frame = int(AUDIO_TEMPORAL_FILTER_START * video_n_frames)
    latest_contact_frame = int(AUDIO_TEMPORAL_FILTER_END * video_n_frames)
    for onset_time in onset_times:
        frame = int(round(float(onset_time) * float(video_fps)))
        if frame < earliest_contact_frame or frame > latest_contact_frame:
            continue
        env_idx = int(librosa.time_to_frames(float(onset_time), sr=sr))
        env_idx = max(0, min(env_idx, len(onset_env) - 1))
        filtered.append(
            {
                "time_s": round(float(onset_time), 6),
                "frame": frame,
                "strength": float(onset_env[env_idx]),
                "env_idx": env_idx,
            }
        )

    if not filtered:
        return None, 0.0, {
            "status": "no_onsets_in_video_window",
            "total_onsets": int(len(onset_times)),
            "temporal_window": {
                "start_frame": int(earliest_contact_frame),
                "end_frame": int(latest_contact_frame),
            },
        }

    # Group onsets into distinct events by merging those within
    # AUDIO_MIN_ONSET_GAP_SECONDS of each other.  Within each group the
    # representative is the loudest member.
    sorted_chron = sorted(filtered, key=lambda item: item["time_s"])
    distinct_events: list[dict] = []
    for onset in sorted_chron:
        if not distinct_events:
            distinct_events.append(onset)
        else:
            gap = onset["time_s"] - distinct_events[-1]["time_s"]
            if gap < AUDIO_MIN_ONSET_GAP_SECONDS:
                # Merge into current group — keep louder representative
                if onset["strength"] > distinct_events[-1]["strength"]:
                    distinct_events[-1] = onset
            else:
                distinct_events.append(onset)

    n_distinct = len(distinct_events)
    temporal_window_info = {
        "start_frame": int(earliest_contact_frame),
        "end_frame": int(latest_contact_frame),
    }

    if n_distinct == 1:
        chosen = distinct_events[0]
        return int(chosen["frame"]), 1.0, {
            "status": "single_onset",
            "selected_frame": int(chosen["frame"]),
            "selected_strength": round(float(chosen["strength"]), 6),
            "candidate_count": int(len(filtered)),
            "audio_onset_count": n_distinct,
            "audio_strategy_used": "audio_single",
            "temporal_window": temporal_window_info,
            "candidates": filtered,
        }

    # N-aware selection strategy — see module docstring for rationale.
    if n_distinct == 2:
        if AUDIO_STRATEGY_N2 == "last":
            chosen = distinct_events[-1]
        else:
            chosen = max(distinct_events, key=lambda e: e["strength"])
        strategy_used = "audio_last_of_2"
        confidence = 0.8
    else:
        # N≥3: loudest distinct event is the most reliable proxy for bat impact.
        if AUDIO_STRATEGY_N3_PLUS == "loudest":
            chosen = max(distinct_events, key=lambda e: e["strength"])
        else:
            chosen = distinct_events[-1]
        strategy_used = f"audio_loudest_of_{n_distinct}"
        confidence = 0.8

    # Reduce confidence when two events have very similar strength.
    strengths_desc = sorted((e["strength"] for e in distinct_events), reverse=True)
    if strengths_desc[0] < (2.0 * strengths_desc[1]):
        confidence = 0.5

    return int(chosen["frame"]), confidence, {
        "status": "multiple_onsets",
        "selected_frame": int(chosen["frame"]),
        "selected_strength": round(float(chosen["strength"]), 6),
        "candidate_count": int(len(filtered)),
        "audio_onset_count": n_distinct,
        "audio_strategy_used": strategy_used,
        "temporal_window": temporal_window_info,
        "candidates": filtered,
        "distinct_events": [
            {k: round(v, 6) if isinstance(v, float) else v for k, v in e.items()}
            for e in distinct_events
        ],
    }


def detect_contact_from_video(
    video_path: str | None,
    video_fps: float,
    video_n_frames: int,
) -> tuple[int | None, float, dict]:
    if not video_path:
        return None, 0.0, {"status": "video_path_missing"}

    audio_array, sample_rate, extract_diag = extract_audio_from_video_with_diagnostics(video_path)
    if audio_array is None or sample_rate is None:
        return None, 0.0, {
            "status": "audio_unavailable",
            "extract": extract_diag,
        }

    frame, confidence, detect_diag = detect_contact_from_audio_with_diagnostics(
        audio_array,
        sample_rate,
        video_fps,
        video_n_frames,
    )
    return frame, confidence, {
        "status": detect_diag.get("status", "ok"),
        "extract": extract_diag,
        "detection": detect_diag,
    }
