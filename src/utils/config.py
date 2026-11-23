"""YAML 配置加载，支持 `_base_` 继承与命令行覆盖。"""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并，override 优先。列表整体替换不合并。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> dict:
    """加载 YAML，递归处理 `_base_` 继承。"""
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = base_ref if os.path.isabs(base_ref) else os.path.join(
            os.path.dirname(path), base_ref
        )
        cfg = deep_merge(load_config(base_path), cfg)
    return cfg


def _cast(value: str) -> Any:
    """把命令行字符串转成合适类型。"""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    # 列表：支持两种写法 —— 裸逗号 "0.4,0.4,0.2" 与 YAML 风格 "[0.4,0.4,0.2]"。
    # 后者必须先剥括号：否则首尾元素会带着括号被当成字符串（'[0.4' / '0.2]'），
    # 模型解包时不报错但数值全错，属于静默错误。
    inner = value.strip()
    if len(inner) >= 2 and inner[0] in "[(" and inner[-1] in ")]":
        inner = inner[1:-1].strip()
        if not inner:
            return []
        return [_cast(v) for v in inner.split(",")]
    if "," in value:
        return [_cast(v) for v in value.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """应用 `a.b.c=value` 形式的覆盖，用于超参数扫描。"""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项须为 key=value 形式，得到 {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _cast(raw)
    return out


def flatten(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """展平为 {"model.d": 256, ...}，便于写入结果表。"""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
