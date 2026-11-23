"""Stream numerical CMU CSD (HDF5) data to the CSV schema used by the experiments."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def convert_csd(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".csv.part")
    rows = 0
    with h5py.File(source, "r") as store, partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        roots = [group["data"] for group in store.values() if isinstance(group, h5py.Group) and "data" in group]
        if not roots:
            raise ValueError(f"No computational-sequence data group found in {source}")
        if len(roots) != 1:
            raise ValueError(f"Expected one computational sequence in {source}")
        dimension = None
        for video in sorted(roots[0]):
            group = roots[0][video]
            features = np.asarray(group["features"])
            intervals = np.asarray(group["intervals"])
            if features.ndim == 1:
                features = features[:, None]
            if features.ndim != 2 or intervals.shape != (len(features), 2) or features.dtype.kind not in "fiub":
                raise ValueError(f"Invalid numerical CSD arrays: {video}")
            if dimension is None:
                dimension = features.shape[1]
                writer.writerow(["video_id", "start_time", "end_time"] + [f"feature_{i}" for i in range(dimension)])
            if features.shape[1] != dimension:
                raise ValueError(f"Feature dimension changed at {video}")
            # Keep within-video order and duplicate intervals, matching the original CSV conversion.
            for span, values in zip(intervals, features):
                writer.writerow([video, *span.tolist(), *values.tolist()])
                rows += 1
    if not rows:
        raise ValueError(f"Empty CSD input: {source}")
    partial.replace(destination)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csd-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    files = sorted(args.csd_dir.glob("*.csd"))
    if not files:
        parser.error("No .csd files found")
    for source in files:
        target = args.out_dir / (source.stem + ".csv")
        if target.exists():
            print(f"Already exists: {target}")
            continue
        print(f"{source.name}: {convert_csd(source, target)} rows -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
