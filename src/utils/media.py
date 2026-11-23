"""媒体探测与解码工具。

本机无 decord / av（Windows 轮子易失败），改用：
  - ffprobe 探时长（已确认可用）
  - cv2 抽帧（cv2 4.13 已装）
  - soundfile / librosa 读音频
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np

# PATH first; an optional directory override supports portable FFmpeg installs.
_FFMPEG_DIR = os.environ.get("MERC_FFMPEG_DIR", "")


def _tool(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    if not _FFMPEG_DIR:
        return None
    cand = os.path.join(_FFMPEG_DIR, name + (".exe" if os.name == "nt" else ""))
    return cand if os.path.exists(cand) else None


def probe_duration(path: str) -> float | None:
    """用 ffprobe 取媒体时长（秒）。失败返回 None。"""
    exe = _tool("ffprobe")
    if exe is None or not os.path.exists(path):
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30, check=True,
        )
        d = json.loads(out.stdout)["format"]["duration"]
        return float(d)
    except Exception:
        return None


def read_video_frames(
    path: str,
    num_frames: int = 16,
    start: float | None = None,
    end: float | None = None,
    size: int = 224,
) -> np.ndarray:
    """在 [start, end] 内均匀抽 num_frames 帧，返回 (T, H, W, 3) uint8 RGB。

    帧数不足时重复末帧补齐，保证输出形状固定。
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        lo = 0 if start is None else max(0, int(start * fps))
        hi = (total - 1) if end is None else min(total - 1, int(end * fps))
        if hi < lo:
            lo, hi = 0, max(0, total - 1)
        idxs = np.linspace(lo, hi, num=num_frames).round().astype(int)

        frames: list[np.ndarray] = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                if frames:
                    frames.append(frames[-1])
                    continue
                raise RuntimeError(f"读帧失败：{path} @ {i}")
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames)
    finally:
        cap.release()


def read_audio(
    path: str,
    sr: int = 16000,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """读单声道音频，重采样到 sr，返回 float32 (N,)。"""
    import librosa

    offset = 0.0 if start is None else float(start)
    duration = None if (start is None or end is None) else max(0.0, float(end) - float(start))
    wav, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
    return wav.astype(np.float32)


def extract_wav_from_video(video_path: str, out_path: str, sr: int = 16000) -> str | None:
    """从 mp4 抽音轨为 wav（MOSI/MOSEI 原始数据可能只有 mp4）。"""
    exe = _tool("ffmpeg")
    if exe is None:
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        subprocess.run(
            [exe, "-y", "-v", "error", "-i", video_path, "-vn",
             "-ac", "1", "-ar", str(sr), out_path],
            capture_output=True, timeout=300, check=True,
        )
        return out_path if os.path.exists(out_path) else None
    except Exception:
        return None
