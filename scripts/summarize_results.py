"""Collect newly generated metrics into CSV; do not populate missing experiments."""
import argparse
import json
from pathlib import Path

import pandas as pd


def scalars(data, prefix=""):
    out = {}
    for key, value in data.items():
        name = prefix + key
        if isinstance(value, dict):
            out.update(scalars(value, name + "."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = value
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    rows = [{"run": str(path.parent.relative_to(args.runs)), **scalars(json.loads(path.read_text(encoding="utf-8")))}
            for path in sorted(args.runs.rglob(f"{args.split}_metrics.json"))]
    if not rows:
        p.error("No completed metric files found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} completed runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
