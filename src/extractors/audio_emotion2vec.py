"""H_a = E_a(A)：emotion2vec 提取语音情感表征（论文 §3.2, 原文 L31, E2）。

本地权重：`iic--emotion2vec_base/snapshots/master/emotion2vec_base.pt`
走 funasr.AutoModel（本机 funasr 1.4.0 已装），granularity="frame" 取帧级序列，
符合 E4 对 H_a ∈ R^{n_a × d_a} 的序列形状要求（d_a = 768）。

参考 `the emotion2vec upstream examples`（用户已有的 emotion2vec 调用范例）：
    granularity="utterance" -> {'feats': [768]}
    granularity="frame"     -> {'feats': [T, 768]}

funasr 只接受**文件路径**不接受 numpy 数组，而本项目需要按片段
（RAVDESS 的 K 个 segment）提特征，故 unit 级提取时先把切片写入临时 wav。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL_DIR = os.environ.get("MERC_EMOTION2VEC_DIR", "weights/emotion2vec-base")
SAMPLE_RATE = 16000


@dataclass
class AudioConfig:
    model_dir: str = DEFAULT_MODEL_DIR
    model_id: str = "iic/emotion2vec_base"
    granularity: str = "frame"  # frame -> (T,768) 序列；utterance -> (1,768)
    device: str = "cuda"
    # 帧级序列过长时按该上限做等间隔时间下采样（None 不限）
    max_frames: int | None = 256


class Emotion2VecExtractor:
    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self._model = None

    @property
    def out_dim(self) -> int:
        return 768

    def load(self) -> None:
        if self._model is not None:
            return
        from funasr import AutoModel

        # 优先用本地 snapshot 目录，避免联网；失败再退回 model id
        src = self.cfg.model_dir if os.path.isdir(self.cfg.model_dir) else self.cfg.model_id
        try:
            self._model = AutoModel(model=src, device=self.cfg.device, disable_update=True)
        except TypeError:  # 老版 funasr 无 disable_update 参数
            self._model = AutoModel(model=src, device=self.cfg.device)
        print(f"[audio] emotion2vec 加载完成 src={src} granularity={self.cfg.granularity}")

    def _postprocess(self, feats: np.ndarray) -> np.ndarray:
        arr = np.asarray(feats, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]  # utterance 粒度 -> (1, 768)
        mf = self.cfg.max_frames
        if mf and arr.shape[0] > mf:
            idx = np.linspace(0, arr.shape[0] - 1, mf).round().astype(int)
            arr = arr[idx]
        return arr

    def extract_path(self, wav_path: str) -> np.ndarray:
        """对整个 wav 文件提特征，返回 (n_a, 768)。"""
        self.load()
        # funasr 每次调用都打 tqdm 进度条与 rtf 统计，9808 次会淹没日志，
        # 故把它的 stdout/stderr 吞掉（真正的异常仍会抛出）。
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            res = self._model.generate(wav_path, granularity=self.cfg.granularity,
                                       extract_embedding=True)
        if not res:
            raise RuntimeError(f"emotion2vec 未返回结果：{wav_path}")
        return self._postprocess(res[0]["feats"])

    def extract_wave(self, wave: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """对内存中的波形提特征（用于 segment 级）。

        funasr 只吃文件路径，故写临时 wav 再删。
        """
        import soundfile as sf

        self.load()
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp, wave, sr, subtype="PCM_16")
            return self.extract_path(tmp)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def unload(self) -> None:
        self._model = None
        try:
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
