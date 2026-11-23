"""RAVDESS 适配器。

目录事实（实测）
--------------
ravdess/Actor_01 .. Actor_24/
    01-*.mp4   音视频（h264 + aac），2452 个 —— 视觉流与 LLaVA 描述的输入
    02-*.mp4   纯视频（无音轨），2452 个 —— 不使用
    03-*.wav   纯音频，2452 个 —— emotion2vec 与 Whisper 的输入
三种前缀去掉首段后后缀完全一一配对。

文件名 7 段编码：modality-vocal-emotion-intensity-statement-repetition-actor
    emotion: 01 neutral 02 calm 03 happy 04 sad 05 angry 06 fearful 07 disgust 08 surprised
    intensity: 01 normal 02 strong（neutral 只有 01）
    statement: 01 "Kids are talking by the door" / 02 "Dogs are sitting by the door"
    actor 奇数=男，偶数=女

完整音视频样本数：neutral 188，disgust/surprised 各 192，其余各 376。
"""

from __future__ import annotations

import os
import re
from typing import Any

from .schema import (
    TASK_CLS,
    BaseDatasetAdapter,
    Sequence,
    Unit,
    register,
)

# 情感码 -> 标签 id（0-based，顺序即 class_names 顺序）
EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
CLASS_NAMES = list(EMOTION_CODES.values())
CODE_TO_ID = {code: i for i, code in enumerate(EMOTION_CODES)}

_FNAME = re.compile(
    r"^(?P<modality>\d{2})-(?P<vocal>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-"
    r"(?P<actor>\d{2})$"
)

# 说话人独立固定划分：train 01-20 / val 21,22 / test 23,24（各含 1 男 1 女）
DEFAULT_VAL_ACTORS = (21, 22)
DEFAULT_TEST_ACTORS = (23, 24)

# 5 折说话人独立 CV：每折 4 位演员作 test，性别平衡（奇男偶女）
CV_FOLDS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


def parse_filename(stem: str) -> dict[str, Any] | None:
    """解析 RAVDESS 文件名，返回各字段；不匹配则 None。"""
    m = _FNAME.match(stem)
    if m is None:
        return None
    d = m.groupdict()
    d["actor_id"] = int(d["actor"])
    d["gender"] = "male" if d["actor_id"] % 2 == 1 else "female"
    d["emotion_name"] = EMOTION_CODES.get(d["emotion"], "unknown")
    d["label"] = CODE_TO_ID.get(d["emotion"])
    return d


def split_into_segments(
    duration: float, num_segments: int, min_seg: float = 0.2
) -> list[tuple[float, float]]:
    """把 [0, duration] 等分为 num_segments 个连续片段（论文 E33 的 t）。

    片段过短时自动降低段数，避免 VideoMAE/emotion2vec 拿到近乎空的输入。
    """
    if duration is None or duration <= 0:
        return [(0.0, 0.0)]
    k = max(1, min(num_segments, int(duration // min_seg) or 1))
    step = duration / k
    return [(i * step, (i + 1) * step) for i in range(k)]


@register
class RavdessAdapter(BaseDatasetAdapter):
    name = "ravdess"
    task = TASK_CLS
    num_classes = 8
    class_names = CLASS_NAMES

    def __init__(
        self,
        root: str,
        num_segments: int = 4,
        vocal_channel: str = "both",  # both | speech | song
        durations: dict[str, float] | None = None,
        val_actors: tuple[int, ...] = DEFAULT_VAL_ACTORS,
        test_actors: tuple[int, ...] = DEFAULT_TEST_ACTORS,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, **kwargs)
        self.num_segments = num_segments
        self.vocal_channel = vocal_channel
        # 时长表由 s0_scan.py 用 ffprobe 预先探测后注入；缺失则 unit 退化为整段
        self.durations = durations or {}
        self.val_actors = set(val_actors)
        self.test_actors = set(test_actors)

    def build_sequences(self) -> list[Sequence]:
        seqs: list[Sequence] = []
        for actor_dir in sorted(os.listdir(self.root)):
            adir = os.path.join(self.root, actor_dir)
            if not (actor_dir.startswith("Actor_") and os.path.isdir(adir)):
                continue
            for fname in sorted(os.listdir(adir)):
                if not fname.endswith(".mp4") or not fname.startswith("01-"):
                    continue  # 只取 01- 音视频；02- 纯视频不用
                stem = os.path.splitext(fname)[0]
                info = parse_filename(stem)
                if info is None or info["label"] is None:
                    continue
                if self.vocal_channel == "speech" and info["vocal"] != "01":
                    continue
                if self.vocal_channel == "song" and info["vocal"] != "02":
                    continue

                video_path = os.path.join(adir, fname)
                # 配对 wav：把首段 modality 码 01 换成 03
                wav_name = "03-" + stem[3:] + ".wav"
                audio_path = os.path.join(adir, wav_name)
                if not os.path.exists(audio_path):
                    audio_path = None  # 缺失时下游音频分支置零并记 warning

                seq_id = f"ravdess/{actor_dir}/{stem}"
                dur = self.durations.get(seq_id)
                spans = split_into_segments(dur, self.num_segments) if dur else [(None, None)]

                units = [
                    Unit(
                        unit_id=f"{seq_id}#{i}",
                        video_path=video_path,
                        audio_path=audio_path,
                        start=s,
                        end=e,
                        transcript=None,  # RAVDESS 无自带转录，S1 用 Whisper 生成
                        label=None,  # 标签在 Sequence 级
                    )
                    for i, (s, e) in enumerate(spans)
                ]
                seqs.append(
                    Sequence(
                        seq_id=seq_id,
                        dataset=self.name,
                        units=units,
                        label=info["label"],
                        speaker=actor_dir,
                        meta={
                            "emotion_name": info["emotion_name"],
                            "intensity": info["intensity"],
                            "statement": info["statement"],
                            "repetition": info["repetition"],
                            "vocal_channel": info["vocal"],
                            "gender": info["gender"],
                            "duration": dur,
                        },
                    )
                )
        return seqs

    def assign_splits(self, seqs: list[Sequence]) -> list[Sequence]:
        """说话人独立划分 + 5 折 CV 折号标注。"""
        fold_of_actor = {a: i for i, actors in enumerate(CV_FOLDS) for a in actors}
        for s in seqs:
            actor_id = int(str(s.speaker).split("_")[-1])
            if actor_id in self.test_actors:
                s.split = "test"
            elif actor_id in self.val_actors:
                s.split = "val"
            else:
                s.split = "train"
            s.fold = fold_of_actor.get(actor_id)  # 21-24 不参与 CV，恒为 val/test
        return seqs
