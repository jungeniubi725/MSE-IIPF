"""数据层统一抽象。

三个数据集（RAVDESS / CMU-MOSI / CMU-MOSEI）在这里被统一为同一套概念，
使 S1/S2 提取脚本与主模型训练代码无需感知数据集差异。

核心概念
--------
Unit（时间单元）
    论文 §3.4 (E33) 中的 t。对 MOSI/MOSEI 一个 unit = 一句 utterance；
    对 RAVDESS 一个 unit = 把单段视频等分后的一个片段（segment）。
Sequence（序列样本）
    模型一次前向处理的对象，由 1..T 个按时间有序的 Unit 组成。
    IIPF 的时序演化建模（L_TPC）在 Sequence 内部展开。
    MOSI/MOSEI 一个 Sequence = 一个 video（含多条 utterance）；
    RAVDESS 一个 Sequence = 一个 mp4（含 K 个 segment）。

标签粒度
--------
RAVDESS      : Sequence 级离散标签（8 类），所有 unit 共享该标签。
MOSI / MOSEI : Unit 级标签（utterance 各自有情感值），可分类亦可回归。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# 任务类型：单标签多分类 / 连续情感值回归
TASK_CLS = "classification"
TASK_REG = "regression"


@dataclass
class Unit:
    """一个时间单元。路径 + 时间区间，实际解码交给 extractors。"""

    unit_id: str
    # 视频与音频来源。RAVDESS 两者指向同一 mp4 的配对文件；
    # MOSI/MOSEI 指向该 video 的原始 mp4 与抽出的 wav。
    video_path: str | None = None
    audio_path: str | None = None
    # 该 unit 在源文件中的时间区间（秒）。None 表示取整个文件。
    start: float | None = None
    end: float | None = None
    # 数据集自带转录文本。有则 S1 的 ASR 可跳过（MOSI/MOSEI 提供人工转录）。
    transcript: str | None = None
    # unit 级标签，仅 MOSI/MOSEI 使用；RAVDESS 为 None（用 Sequence 级）。
    label: Any | None = None

    @property
    def duration(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start


@dataclass
class Sequence:
    """一个序列样本，模型的一次前向单位。"""

    seq_id: str
    dataset: str
    units: list[Unit] = field(default_factory=list)
    # Sequence 级标签，仅 RAVDESS 使用。
    label: Any | None = None
    # 说话人标识。RAVDESS 为 Actor_XX，用于说话人独立划分；
    # MOSI/MOSEI 为 video id（同一 video 不跨 split）。
    speaker: str | None = None
    split: str | None = None  # train / val / test
    fold: int | None = None  # k-fold CV 中该样本作为 test 的折号
    # 数据集专属元信息（RAVDESS 的 intensity/statement/vocal_channel 等）
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_units(self) -> int:
        return len(self.units)

    def unit_labels(self) -> list[Any]:
        """统一取标签：unit 级缺失时回落到 Sequence 级。"""
        return [u.label if u.label is not None else self.label for u in self.units]

    def to_rows(self) -> list[dict[str, Any]]:
        """展平为 manifest.csv 的行，一行一个 unit。"""
        rows = []
        for idx, u in enumerate(self.units):
            row = {
                "seq_id": self.seq_id,
                "unit_id": u.unit_id,
                "unit_index": idx,
                "num_units": self.num_units,
                "dataset": self.dataset,
                "speaker": self.speaker,
                "split": self.split,
                "fold": self.fold,
                "video_path": u.video_path,
                "audio_path": u.audio_path,
                "start": u.start,
                "end": u.end,
                "transcript": u.transcript,
                "label": u.label if u.label is not None else self.label,
                "seq_label": self.label,
            }
            row.update({f"meta_{k}": v for k, v in self.meta.items()})
            rows.append(row)
        return rows


class BaseDatasetAdapter:
    """数据集适配器接口。

    每个数据集实现 build_sequences()，返回未划分的 Sequence 列表；
    划分逻辑由 assign_splits() 提供，允许各数据集覆写协议。
    """

    name: str = "base"
    task: str = TASK_CLS
    num_classes: int | None = None
    # 类别名，顺序即标签 id 顺序
    class_names: list[str] = []

    def __init__(self, root: str, **kwargs: Any) -> None:
        self.root = root
        self.kwargs = kwargs

    def build_sequences(self) -> list[Sequence]:
        raise NotImplementedError

    def assign_splits(self, seqs: list[Sequence]) -> list[Sequence]:
        raise NotImplementedError

    def build(self) -> list[Sequence]:
        return self.assign_splits(self.build_sequences())


# 适配器注册表，s0_scan.py 按 --dataset 取用
_REGISTRY: dict[str, type[BaseDatasetAdapter]] = {}


def register(cls: type[BaseDatasetAdapter]) -> type[BaseDatasetAdapter]:
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> type[BaseDatasetAdapter]:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的数据集 {name!r}，可选：{sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
