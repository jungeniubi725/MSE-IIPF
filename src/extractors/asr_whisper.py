"""D_a = G_a(A)：Whisper-large-v3 语音转录（论文 §3.2, 原文 L31）。

本地权重完整（`openai-whisper-large-v3/`，7 个文件），走 HF 原生
`WhisperForConditionalGeneration`，无额外依赖。fp16 约 3.1GB 显存。

关于 RAVDESS 的一点说明
---------------------
RAVDESS 全库只有两句固定台词（"Kids are talking by the door" /
"Dogs are sitting by the door"），故 D_a 的**判别信息量接近零** ——
文本模态的有效信息几乎全部来自 LLaVA 视觉描述 D_v。
这不影响代码正确性，但会直接影响：
  - §5.1 Krippendorff α（文本头与其他模态的一致度）
  - §5.4 消融 A1（去掉语义流）的解读
manifest 里的 meta_statement 列已记录台词编号，可用于核对转录质量。
MOSI/MOSEI 有真实口语内容，D_a 才承载实质语义。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL_DIR = os.environ.get("MERC_WHISPER_DIR", "weights/whisper-large-v3")

# Whisper 固定 16kHz 输入
SAMPLE_RATE = 16000


@dataclass
class AsrConfig:
    model_dir: str = DEFAULT_MODEL_DIR
    language: str = "en"
    dtype: str = "float16"
    device: str = "cuda"
    num_beams: int = 1  # 贪心，保证转录可复现
    max_new_tokens: int = 128
    batch_size: int = 8
    # 音频过短时 Whisper 易输出幻觉文本；短于该秒数直接给空串
    min_duration: float = 0.3


class WhisperTranscriber:
    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        dtype = getattr(torch, self.cfg.dtype)
        self._processor = WhisperProcessor.from_pretrained(self.cfg.model_dir)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.cfg.model_dir, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(self.cfg.device).eval()
        # 关掉强制 token，改由 generate 的 language 参数控制
        self._model.generation_config.forced_decoder_ids = None
        self._repair_generation_config()
        print(f"[asr] whisper-large-v3 加载完成 dtype={self.cfg.dtype} "
              f"device={self.cfg.device}")

    def _repair_generation_config(self) -> None:
        """本地权重目录缺 generation_config.json，需自建语言/任务 token 映射。

        实测：`openai-whisper-large-v3/` 只有 7 个文件，无 generation_config.json，
        且 config.json 里没有 lang_to_id，导致 generate(language=...) 抛
        "generation config is outdated"。这里从 tokenizer 的特殊 token 反查 id
        补齐，全程离线，无需联网下载官方 generation_config。
        """
        gc_ = self._model.generation_config
        tok = self._processor.tokenizer

        def tid(token: str) -> int | None:
            i = tok.convert_tokens_to_ids(token)
            return None if i is None or i == tok.unk_token_id else int(i)

        if getattr(gc_, "lang_to_id", None) in (None, {}):
            lang_to_id = {}
            for code in ("en", "zh", "de", "es", "fr", "ja", "ko", "ru"):
                i = tid(f"<|{code}|>")
                if i is not None:
                    lang_to_id[f"<|{code}|>"] = i
            if lang_to_id:
                gc_.lang_to_id = lang_to_id
        # 注意键格式不对称：transformers 用 lang_to_id["<|en|>"]（带尖括号）
        # 但 task_to_id["transcribe"]（裸名）。写错会 KeyError。
        if getattr(gc_, "task_to_id", None) in (None, {}):
            t2i = {}
            for task in ("transcribe", "translate"):
                i = tid(f"<|{task}|>")
                if i is not None:
                    t2i[task] = i
            if t2i:
                gc_.task_to_id = t2i
        for attr, token in (("no_timestamps_token_id", "<|notimestamps|>"),
                            ("decoder_start_token_id", "<|startoftranscript|>")):
            if getattr(gc_, attr, None) is None:
                i = tid(token)
                if i is not None:
                    setattr(gc_, attr, i)
        if getattr(gc_, "max_length", None) is None:
            gc_.max_length = 448

    def transcribe_batch(self, waves: list[np.ndarray]) -> list[str]:
        """批量转录。waves 为 16kHz 单声道 float32 数组列表。"""
        self.load()
        import torch

        texts: list[str] = [""] * len(waves)
        # 过短音频跳过，避免幻觉
        keep = [
            i for i, w in enumerate(waves)
            if w is not None and w.size >= int(self.cfg.min_duration * SAMPLE_RATE)
        ]
        if not keep:
            return texts

        feats = self._processor(
            [waves[i] for i in keep],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        dtype = getattr(torch, self.cfg.dtype)
        inputs = feats.input_features.to(self.cfg.device, dtype=dtype)
        attn = feats.get("attention_mask")
        kw = {"attention_mask": attn.to(self.cfg.device)} if attn is not None else {}

        with torch.inference_mode():
            ids = self._model.generate(
                inputs,
                language=self.cfg.language,
                task="transcribe",
                num_beams=self.cfg.num_beams,
                do_sample=False,
                max_new_tokens=self.cfg.max_new_tokens,
                **kw,
            )
        decoded = self._processor.batch_decode(ids, skip_special_tokens=True)
        for i, t in zip(keep, decoded):
            texts[i] = t.strip()
        return texts

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
