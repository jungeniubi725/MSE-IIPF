"""H_v = E_v(V)：VideoMAE 提取视觉动态表征（论文 §3.2, 原文 L31, E2）。

本地 `VideoMAE/` 只有源码无权重，按用户确认走 transformers 拉
`MCG-NJU/videomae-base`（首次约 350MB，之后走 HF 缓存）。

输出粒度
-------
论文 E4 要求 H_v ∈ R^{n_v × d_v} 是**帧级/token 级序列**，不是单向量。
VideoMAE-base 输入 16 帧 224²，patch 16²，tube size 2
  → token 数 = (16/2) × (224/16)² = 8 × 196 = 1568，维度 768。
1568 token/unit × 9808 unit 落盘约 1568×768×2B×9808 ≈ 23 GB，过大。
故做**空间池化保留时间轴**：把 1568 token reshape 为 (8, 196, 768)，
对 196 个空间 patch 取均值 → (8, 768)，即 8 个时间步的 token 序列。
这既满足 E4 的序列形状要求（n_v=8），又把缓存压到 ~120 MB。
`--pool none` 可保留全部 1568 token（调试用）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL_ID = os.environ.get("MERC_VIDEOMAE_DIR", "weights/videomae-base")

# VideoMAE-base 结构常量（用于 token 网格 reshape）
TUBELET_SIZE = 2
PATCH_SIZE = 16


@dataclass
class VisualConfig:
    model_id: str = DEFAULT_MODEL_ID
    num_frames: int = 16
    image_size: int = 224
    dtype: str = "float16"
    device: str = "cuda"
    # temporal: 空间池化保留时间轴 (8,768) | mean: 全池化 (1,768) | none: 全 token
    pool: str = "temporal"
    batch_size: int = 4


class VideoMAEExtractor:
    def __init__(self, cfg: VisualConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._proc = None

    @property
    def out_dim(self) -> int:
        return 768  # videomae-base hidden_size

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, VideoMAEModel

        dtype = getattr(torch, self.cfg.dtype)
        self._proc = AutoImageProcessor.from_pretrained(self.cfg.model_id)
        self._model = (
            VideoMAEModel.from_pretrained(self.cfg.model_id, torch_dtype=dtype)
            .to(self.cfg.device)
            .eval()
        )
        print(f"[visual] {self.cfg.model_id} 加载完成 dtype={self.cfg.dtype} "
              f"pool={self.cfg.pool}")

    def _grid(self) -> tuple[int, int]:
        t = self.cfg.num_frames // TUBELET_SIZE
        s = (self.cfg.image_size // PATCH_SIZE) ** 2
        return t, s

    def extract_batch(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        """clips: 每项 (T,H,W,3) uint8 RGB，T 须等于 cfg.num_frames。

        返回每项 (n_v, 768) float32。
        """
        self.load()
        import torch

        # processor 接受 list[list[frame]]，内部做 resize + normalize
        inputs = self._proc([list(c) for c in clips], return_tensors="pt")
        px = inputs["pixel_values"].to(self.cfg.device, dtype=getattr(torch, self.cfg.dtype))
        with torch.inference_mode():
            hidden = self._model(pixel_values=px).last_hidden_state  # (B, 1568, 768)

        t, s = self._grid()
        if self.cfg.pool == "temporal":
            b, n, d = hidden.shape
            if n == t * s:
                hidden = hidden.reshape(b, t, s, d).mean(dim=2)  # (B, 8, 768)
            else:  # token 数与预期网格不符时退化为全局均值，避免静默错位
                hidden = hidden.mean(dim=1, keepdim=True)
        elif self.cfg.pool == "mean":
            hidden = hidden.mean(dim=1, keepdim=True)
        # pool == "none": 保留 (B, 1568, 768)

        return [h.float().cpu().numpy() for h in hidden]

    def unload(self) -> None:
        self._model = None
        self._proc = None
        try:
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
