"""LLaVA-Video 结构化情感描述的 prompt 模板。

论文 §3.2 (原文 L31) 要求 D_v = G_v(V) 覆盖五要素：
    场景、人物状态、表情线索、动作线索、视觉情绪判断

设计要点
--------
1. **定长模板**：五个字段各限一句，输出总长可控。RoBERTa 上限 512 token，
   定长模板保证 D_v 不被截断（实测五句英文约 90-130 token）。
2. **英文输出**：下游 E_t 是 roberta-base（英文词表 50265），描述必须英文。
3. **禁止直接给情感标签词**：若让模型输出 "happy/sad" 等 8 个 RAVDESS 类别词，
   D_v 会退化为一个弱分类器的预测，L_DSA 与 ITAC 学到的是标签泄漏而非语义对齐。
   故 visual affect 字段要求描述**可观测证据 + 效价/唤醒度**，不给类别名。
4. **不提示台词内容**：RAVDESS 只有两句固定台词，提及会引入无判别力的常量文本。
"""

from __future__ import annotations

import re

# 五要素字段名，顺序固定 —— 解析与拼接都依赖它
FIELDS = ("scene", "person_state", "facial_cues", "action_cues", "visual_affect")

STRUCTURED_CAPTION_PROMPT = """You are analyzing a short video clip of a single person speaking.
Describe what is visually observable, using exactly the five labeled lines below.
Write one concise English sentence per line. Do not add any other text.

Scene: the setting, background, framing and lighting.
PersonState: the person's identity-neutral appearance, posture, head pose and gaze.
FacialCues: concrete facial muscle activity you can see (brow, eyes, eyelids, nose, mouth corners, lips, jaw).
ActionCues: head, shoulder and body motion, and how it changes over the clip.
VisualAffect: the affective impression conveyed, described through valence (negative/neutral/positive) and arousal (low/medium/high) together with the visual evidence supporting it.

Rules:
- Report only what is visible. Do not guess the spoken words.
- Do not use emotion category names such as happy, sad, angry, fearful, calm, neutral, disgust or surprised.
- Never mention that this is a dataset, an actor, or a performance.
"""

# 兜底描述：视频解码失败或生成为空时写入，保证下游张量形状一致
FALLBACK_CAPTION = (
    "Scene: an indoor setting with plain background and even frontal lighting. "
    "PersonState: a single person is centered in frame facing the camera. "
    "FacialCues: facial activity is not clearly discernible. "
    "ActionCues: only slight head motion is present. "
    "VisualAffect: affective impression is unclear with neutral valence and low arousal."
)


_ALIASES = {
    "scene": "scene",
    "personstate": "person_state",
    "person state": "person_state",
    "person_state": "person_state",
    "facialcues": "facial_cues",
    "facial cues": "facial_cues",
    "facial_cues": "facial_cues",
    "actioncues": "action_cues",
    "action cues": "action_cues",
    "action_cues": "action_cues",
    "visualaffect": "visual_affect",
    "visual affect": "visual_affect",
    "visual_affect": "visual_affect",
}

# 标签可能出现在行首，也可能与前一字段挤在同一行（模型常把五段写成一行），
# 故不按行切分，而是在全文里按「标签:」的位置切段。
_LABEL_RE = re.compile(
    r"(?:^|(?<=[\s.;,!?)\]]))\**\s*"
    r"(scene|person[\s_]?state|facial[\s_]?cues|action[\s_]?cues|visual[\s_]?affect)"
    r"\**\s*:",
    re.IGNORECASE,
)


def parse_structured(text: str) -> dict[str, str]:
    """把模型输出解析为五字段。缺失字段留空串，由 flatten 补齐。

    容忍三种常见变形：五段挤在同一行；标签带 ** 或 - 前缀；
    结尾复读一遍空标签（此时不覆盖已有内容）。
    """
    out: dict[str, str] = {f: "" for f in FIELDS}
    marks = list(_LABEL_RE.finditer(text))
    if not marks:
        return out

    for i, m in enumerate(marks):
        # 标签归一化：下划线/多空格都折成单空格，再查别名表
        norm = re.sub(r"[\s_]+", " ", m.group(1).strip().lower())
        key = _ALIASES[norm]
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        val = text[m.end():end].strip().strip("-*# ").strip()
        if not out[key]:
            out[key] = val
        elif val:  # 同一标签重复出现且有实际内容：追加
            out[key] = (out[key] + " " + val).strip()
    return out


def flatten(fields: dict[str, str]) -> str:
    """五字段拼成单段文本，供 RoBERTa 编码。字段缺失时用兜底句补。"""
    fb = parse_structured(FALLBACK_CAPTION)
    parts = []
    label = {
        "scene": "Scene",
        "person_state": "PersonState",
        "facial_cues": "FacialCues",
        "action_cues": "ActionCues",
        "visual_affect": "VisualAffect",
    }
    for f in FIELDS:
        val = (fields.get(f) or "").strip() or fb[f]
        parts.append(f"{label[f]}: {val}")
    return " ".join(parts)
