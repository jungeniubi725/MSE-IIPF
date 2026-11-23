"""H_sv, H_sa = E_t(D_v), E_t(D_a)：RoBERTa 编码模态转化文本（论文 E3）。

本地权重完整：`weights/roberta-base`（roberta-base，L=12, d=768, vocab=50265）。

输出约定
-------
返回 token 级序列 (n_s, 768)，**第 0 位是 <s>（[CLS]）**，因为论文 E5/E13 的
Pool_CLS 就取该位置：
    Z_sv = Norm(P_s(Pool_CLS(X_sv)))
padding 位一并返回 mask，供下游 MHA 的 key_padding_mask 与注意力池化屏蔽用
（E17 的 α_i softmax 前要把 padding 置 -inf）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL_DIR = os.environ.get("MERC_ROBERTA_DIR", "weights/roberta-base")


@dataclass
class TextConfig:
    model_dir: str = DEFAULT_MODEL_DIR
    max_length: int = 192  # 五要素模板实测约 90-130 token，192 足够且省显存
    dtype: str = "float16"
    device: str = "cuda"
    batch_size: int = 32


class RobertaEncoder:
    def __init__(self, cfg: TextConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tok = None

    @property
    def out_dim(self) -> int:
        return 768

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = getattr(torch, self.cfg.dtype)
        # 本地 config.json 的 architectures 是 RobertaForMaskedLM，
        # 用 AutoModel 只取 encoder 主干（丢弃 lm_head），符合 E3 的 E_t(·)
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model_dir)
        self._model = (
            AutoModel.from_pretrained(self.cfg.model_dir, torch_dtype=dtype)
            .to(self.cfg.device)
            .eval()
        )
        print(f"[text] roberta 加载完成 dir={self.cfg.model_dir} "
              f"max_length={self.cfg.max_length}")

    def encode(self, texts: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """返回 (每项 (n_s,768) float32, 每项 (n_s,) uint8 有效位掩码)。

        按各样本真实长度裁掉尾部 padding，避免落盘冗余。
        """
        self.load()
        import torch

        # 空串会退化为只有 <s></s>，下游注意力池化仍可工作
        safe = [t if (t and t.strip()) else "" for t in texts]
        enc = self._tok(
            safe,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.cfg.device)
        attn = enc["attention_mask"].to(self.cfg.device)
        with torch.inference_mode():
            out = self._model(input_ids=ids, attention_mask=attn).last_hidden_state

        feats, masks = [], []
        lens = attn.sum(dim=1).tolist()
        for i, n in enumerate(lens):
            n = max(1, int(n))
            feats.append(out[i, :n].float().cpu().numpy())
            masks.append(np.ones(n, dtype=np.uint8))
        return feats, masks

    def unload(self) -> None:
        self._model = None
        self._tok = None
        try:
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
