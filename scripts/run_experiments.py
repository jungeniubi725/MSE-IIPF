"""Run a reproducible RAVDESS seed/ablation/grid plan; grids use validation only."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml

VARIANTS = {
    "main": [], "no_semantics": ["model.use_semantic_stream=false"], "no_gate": ["model.use_gate=false"],
    "no_mam": ["model.use_mam=false"], "no_itac": ["model.use_itac=false"], "no_itam": ["model.use_itam=false"],
    "no_iipf": ["model.use_iipf_module=false"], "no_cmc": ["model.use_cmc=false"],
    "no_tpc": ["model.use_tpc=false"], "no_prr": ["model.use_prr=false"],
    "no_proxy_weight": ["model.use_proxy_weight=false"], "softmax": ["model.use_evidential=false"],
    "no_cal": ["model.use_cal=false"],
}


def build_commands(args):
    grid = yaml.safe_load(args.grid.read_text(encoding="utf-8")) if args.grid else {}
    if grid and (not isinstance(grid, dict) or any(not isinstance(v, list) or not v for v in grid.values())):
        raise ValueError("Grid must map dotted config keys to nonempty lists")
    keys = list(grid)
    combinations = list(itertools.product(*(grid[k] for k in keys))) if keys else [()]
    commands = []
    for variant in args.variants:
        for seed in args.seeds:
            for i, values in enumerate(combinations):
                name = f"{variant}_seed{seed}" + (f"_grid{i:03d}" if grid else "")
                overrides = VARIANTS[variant] + [f"{k}={json.dumps(v)}" for k, v in zip(keys, values)]
                cmd = [sys.executable, "scripts/s3_train.py", "--config", str(args.config), "--exp-name", name,
                       "--out-root", str(args.out_root), "--seed", str(seed), "--device", args.device]
                if args.cv:
                    cmd.append("--cv")
                if grid or args.validation_only:
                    cmd.append("--validation-only")
                if overrides:
                    cmd.extend(["--set", *overrides])
                commands.append(cmd)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ravdess.yaml"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["main"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out-root", type=Path, default=Path("runs/experiments"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--grid", type=Path, help="YAML hyperparameter grid; always validation-only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commands = build_commands(args)
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return 0
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "experiment_plan.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    for command in commands:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

