"""D_v = G_v(V)：LLaVA-Video-7B-Qwen2 生成结构化情感描述（论文 §3.2, 原文 L31）。

加载现状（实测，2026-08-05）
--------------------------
本地 `LLaVA-Video-7B-Qwen2/` 的 `architectures = ["LlavaQwenForCausalLM"]`，
这是 **LLaVA-NeXT 仓库的自定义架构**，不是 HF 原生 `LlavaForConditionalGeneration`：
  - 权重键为 `model.vision_tower.vision_tower.*`（SigLIP，421 键，已内置无需另下）
    与 `model.mm_projector.*`，HF 原生 llava 类无法直接吃。
  - 本机未装 `llava` 包（LLaVA-NeXT），故默认后端不可用。
  - 权重 bf16 共 **16.06 GB**，超过本机 15.9 GB 显存 → 必须 offload 或量化。

三种后端（--backend）
-------------------
llava_next : 需 `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT`（官方路径）。
             配 offload 后可在 16GB 卡上跑。**首选**。
hf_auto    : `AutoModel.from_pretrained(trust_remote_code=True)`。仅当权重目录
             自带 modeling 代码时可用；本地目录无 .py 文件，故预期失败，仅作探测。
dummy      : 不加载模型，全部写入 FALLBACK_CAPTION。用于打通 S2/S3 全链路、
             验证张量形状与训练循环，**不可用于出论文数字**。

显存策略：`device_map="auto"` + `max_memory` 把部分层 offload 到 CPU，
不依赖 bitsandbytes（本机未装）。慢但不炸显存。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .prompts import (
    FALLBACK_CAPTION,
    STRUCTURED_CAPTION_PROMPT,
    flatten,
    parse_structured,
)

DEFAULT_MODEL_DIR = os.environ.get("MERC_LLAVA_DIR", "weights/llava-video")


@dataclass
class CaptionConfig:
    model_dir: str = DEFAULT_MODEL_DIR
    backend: str = "llava_next"
    num_frames: int = 16  # 送入 LLaVA 的帧数
    max_new_tokens: int = 220  # 五句英文约 130 token，留余量
    temperature: float = 0.0  # 贪心解码，保证描述可复现
    gpu_mem_gb: float = 13.5  # 余 ~2.4GB 给激活值与 KV cache
    cpu_mem_gb: float = 32.0
    dtype: str = "bfloat16"
    # 量化：none=bf16 原权重(~15GiB，16GB 卡需 CPU offload)；
    # int8=~8.5GiB 全部驻留 GPU；int4=~5GiB，nf4 双重量化。
    quant: str = "none"


class LlavaCaptioner:
    """按需懒加载。generate(frames) -> (flat_text, fields_dict)。"""

    def __init__(self, cfg: CaptionConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._loaded = False

    # ------------------------------------------------------------------ 加载
    def load(self) -> None:
        if self._loaded:
            return
        if self.cfg.backend == "dummy":
            self._loaded = True
            print("[caption] backend=dummy，不加载模型，全部输出兜底描述"
                  "（仅用于打通链路，不可用于论文数字）")
            return
        if self.cfg.backend == "llava_next":
            self._load_llava_next()
        elif self.cfg.backend == "hf_auto":
            self._load_hf_auto()
        else:
            raise ValueError(f"未知 backend: {self.cfg.backend}")
        self._loaded = True

    def _max_memory(self) -> dict[Any, str]:
        return {0: f"{self.cfg.gpu_mem_gb:.1f}GiB", "cpu": f"{self.cfg.cpu_mem_gb:.0f}GiB"}

    def _quant_kwargs(self) -> dict[str, Any]:
        """返回传给 from_pretrained 的量化参数。quant=none 时为空 dict。"""
        if self.cfg.quant == "none":
            return {}
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"quant={self.cfg.quant} 需要 bitsandbytes，本机未装。\n"
                "  安装：pip install -U bitsandbytes>=0.45.0\n"
                "  RTX 50 系（Blackwell / sm_120）必须 >=0.45.0，旧版无对应 kernel。"
            ) from e
        import torch
        from transformers import BitsAndBytesConfig

        if self.cfg.quant == "int8":
            return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        if self.cfg.quant == "int4":
            return {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, self.cfg.dtype),
            )}
        raise ValueError(f"未知 quant: {self.cfg.quant}（可选 none/int8/int4）")

    def _load_llava_next(self) -> None:
        try:
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
        except ImportError as e:
            raise ImportError(
                "backend=llava_next 需要 LLaVA-NeXT 包，本机未装。\n"
                "  安装：pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git\n"
                "  或改用 --backend dummy 先打通后续流程（不可用于论文数字）。"
            ) from e
        import torch

        # Keep architecture selection independent of the user's local folder name.
        name = "llava_qwen"
        # LLaVA-NeXT 的 builder 用自己的 load_8bit/load_4bit 开关，
        # 不接受 transformers 的 quantization_config，故此处分支处理。
        kw: dict[str, Any] = {"attn_implementation": "sdpa"}  # 本机无 flash_attn
        if self.cfg.quant == "none":
            kw.update(torch_dtype=self.cfg.dtype, device_map="auto",
                      max_memory=self._max_memory())
            desc = f"dtype={self.cfg.dtype} max_memory={self._max_memory()}"
        else:
            self._quant_kwargs()  # 仅用于校验 bitsandbytes 可用与 quant 合法
            kw.update(load_8bit=self.cfg.quant == "int8",
                      load_4bit=self.cfg.quant == "int4",
                      device_map={"": 0})  # 量化后全部驻留 GPU，不做 offload
            desc = f"quant={self.cfg.quant} 全部驻留 GPU:0"

        self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
            self.cfg.model_dir, None, name, **kw
        )
        self._model.eval()
        print(f"[caption] llava_next 加载完成 {desc}")
        del torch

    def _load_hf_auto(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        dtype = getattr(torch, self.cfg.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_dir, trust_remote_code=True
        )
        quant_kw = self._quant_kwargs()
        # 量化后权重全部驻留 GPU，不需要 max_memory offload 上限
        mem_kw: dict[str, Any] = ({"device_map": {"": 0}} if quant_kw
                                  else {"device_map": "auto",
                                        "max_memory": self._max_memory()})
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
            **mem_kw,
            **quant_kw,
        )
        self._model.eval()
        try:
            self._image_processor = AutoProcessor.from_pretrained(
                self.cfg.model_dir, trust_remote_code=True
            )
        except Exception:
            self._image_processor = None
        print(f"[caption] hf_auto 加载完成 quant={self.cfg.quant}")

    def _runtime_device_dtype(self) -> tuple[Any, Any]:
        """取 vision tower 的实际 device/dtype（int8 权重不含浮点 dtype 信息，
        故跳过 int8/uint8 参数，回退到 model.dtype）。"""
        import torch

        vt = None
        if hasattr(self._model, "get_vision_tower"):
            try:
                vt = self._model.get_vision_tower()
            except Exception:
                vt = None
        src = vt if vt is not None else self._model
        for p in src.parameters():
            if p.dtype in (torch.float16, torch.bfloat16, torch.float32):
                return p.device, p.dtype
        dev = next(self._model.parameters()).device
        return dev, getattr(self._model, "dtype", torch.float16)

    # ------------------------------------------------------------------ 生成
    def generate(self, frames: np.ndarray) -> tuple[str, dict[str, str]]:
        """frames: (T, H, W, 3) uint8 RGB。返回 (拼接文本, 五字段)。"""
        if not self._loaded:
            self.load()
        if self.cfg.backend == "dummy":
            f = parse_structured(FALLBACK_CAPTION)
            return flatten(f), f

        raw = self._generate_llava(frames)
        fields = parse_structured(raw)
        if not any(v.strip() for v in fields.values()):
            # 模型没按模板输出：整段塞进 visual_affect，其余走兜底
            fields = parse_structured(FALLBACK_CAPTION)
            fields["visual_affect"] = raw.strip()[:600] or fields["visual_affect"]
        return flatten(fields), fields

    def _generate_llava(self, frames: np.ndarray) -> str:
        import copy

        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        proc = self._image_processor
        pixel = proc.preprocess(list(frames), return_tensors="pt")["pixel_values"]
        # dtype 必须取模型实际值而非 cfg.dtype：量化加载时 LLaVA-NeXT 的 builder
        # 会把计算 dtype 定成 fp16，与 cfg 的 bfloat16 不一致；preprocess 又固定
        # 输出 fp32。以 vision tower 的参数 dtype 为准，三种 quant 模式都对。
        device, dtype = self._runtime_device_dtype()
        pixel = pixel.to(device=device, dtype=dtype)

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{STRUCTURED_CAPTION_PROMPT}")
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(
            conv.get_prompt(), self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            out = self._model.generate(
                ids,
                images=[pixel],
                modalities=["video"],
                do_sample=self.cfg.temperature > 0,
                temperature=self.cfg.temperature if self.cfg.temperature > 0 else None,
                max_new_tokens=self.cfg.max_new_tokens,
                use_cache=True,
            )
        return self._tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    def unload(self) -> None:
        """显式释放显存，供 S1 跑完 caption 后接着跑 ASR。"""
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._loaded = False
        try:
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def probe_backend(model_dir: str = DEFAULT_MODEL_DIR) -> str:
    """探测哪个后端可用，返回首个可用者；都不可用返回 'dummy'。"""
    try:
        import llava  # noqa: F401
        return "llava_next"
    except ImportError:
        pass
    if any(f.endswith(".py") for f in os.listdir(model_dir)):
        return "hf_auto"  # 目录自带 modeling 代码
    return "dummy"
