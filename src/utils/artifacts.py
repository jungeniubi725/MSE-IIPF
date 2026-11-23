"""Portable run metadata and lossless, sample-addressable prediction exports."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if k != "_raw"}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def environment_info():
    packages = {}
    for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "transformers", "funasr", "matplotlib"):
        try:
            module_name = {"scikit-learn": "sklearn"}.get(name, name)
            imported = sys.modules.get(module_name)
            packages[name] = getattr(imported, "__version__", None) or importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda}


def export_predictions(raw, output_dir, prefix="test"):
    """Save NPZ plus unit/sequence CSVs. Uncertainty is never inferred as evidence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in raw.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(output_dir / f"{prefix}_predictions.npz", **arrays)
    for level, key, id_key in (("unit", "", "unit_ids"), ("sequence", "seq_", "seq_ids")):
        if key + "p_hat" not in arrays:
            continue
        p = arrays[key + "p_hat"]
        y = arrays[key + "y_true"]
        yp = arrays.get(key + "y_pred", p.argmax(-1))
        ids = arrays.get(id_key)
        if ids is None or len(ids) != len(y) or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"Missing, duplicate or misaligned {level} sample IDs")
        u = arrays.get(key + "u")
        fieldnames = ["seq_id" if level == "sequence" else "unit_id", "y_true", "y_pred", "uncertainty"]
        fieldnames += [f"p_{i}" for i in range(p.shape[1])]
        with (output_dir / f"{prefix}_{level}_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fieldnames)
            for i, identifier in enumerate(ids):
                writer.writerow([identifier, int(y[i]), int(yp[i]), "" if u is None else float(u[i]), *p[i].tolist()])

