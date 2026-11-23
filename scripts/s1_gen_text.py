"""S1：模态转化 —— 生成 D_v（LLaVA 视觉描述）与 D_a（Whisper 转录）。

用法
----
    # 先探测 LLaVA 后端可用性（不加载权重）
    python scripts/s1_gen_text.py --manifest cache/ravdess/manifest.csv --task probe

    # 视觉描述（最贵的一步：2452 次生成）
    python scripts/s1_gen_text.py --manifest cache/ravdess/manifest.csv \
        --task caption --out cache/ravdess/text/captions.jsonl --limit 20

    # 语音转录
    python scripts/s1_gen_text.py --manifest cache/ravdess/manifest.csv \
        --task asr --out cache/ravdess/text/asr.jsonl

粒度说明
-------
论文 E18 的 X_t = Σ αᵢuᵢ 是**单个向量**，文本模态天然是 sequence 级，
故默认 --granularity sequence：每个 sequence 生成一次，IIPF 中 h_l,t 对各
unit 广播。若按 unit 生成，RAVDESS 需 9808 次 LLaVA 调用（成本 ×4）。
MOSI/MOSEI 的 unit=utterance 本就是语义完整单元，用 --granularity unit。

断点续跑：jsonl 追加写，启动时读已有 id 集合跳过。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractors.asr_whisper import SAMPLE_RATE, AsrConfig, WhisperTranscriber
from src.extractors.caption_llava import CaptionConfig, LlavaCaptioner, probe_backend
from src.utils.console import setup_console
from src.utils.media import read_audio, read_video_frames

setup_console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S1 模态转化：视觉描述 + 语音转录")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task", required=True, choices=["caption", "asr", "probe"])
    p.add_argument("--out", help="输出 jsonl（task=probe 时不需要）")
    p.add_argument("--granularity", default="auto", choices=["auto", "sequence", "unit"])
    p.add_argument("--backend", default="auto",
                   help="caption 后端：auto|llava_next|hf_auto|dummy")
    p.add_argument("--llava-dir", default=None)
    p.add_argument("--whisper-dir", default=None)
    p.add_argument("--device", default="cuda", help="ASR device; LLaVA uses its device map")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8, help="ASR 批大小")
    p.add_argument("--gpu-mem-gb", type=float, default=13.5)
    p.add_argument("--quant", default="int8", choices=["none", "int8", "int4"],
                   help="caption 模型量化。16GB 卡默认 int8（~8.5GiB 全驻留 GPU）；"
                        "none=bf16 原权重 ~15GiB，会 offload 到 CPU 显著变慢")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="忽略已有结果重跑")
    return p.parse_args()


def resolve_granularity(args: argparse.Namespace, df: pd.DataFrame) -> str:
    if args.granularity != "auto":
        return args.granularity
    ds = str(df["dataset"].iloc[0])
    return "sequence" if ds == "ravdess" else "unit"


def load_done(path: str) -> set[str]:
    """读已完成 id，支持断点续跑。"""
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not record.get("error") and record.get("backend") != "dummy":
                    done.add(record["id"])
            except Exception:
                continue  # 上次中断产生的半行，忽略
    return done


def build_targets(df: pd.DataFrame, gran: str) -> list[dict]:
    """按粒度归并出待处理项。"""
    if gran == "sequence":
        # 一个 sequence 一项：视频取整段，音频取整段
        g = df.sort_values("unit_index").groupby("seq_id", sort=True)
        return [
            {
                "id": seq_id,
                "seq_id": seq_id,
                "video_path": grp["video_path"].iloc[0],
                "audio_path": grp["audio_path"].iloc[0],
                "start": None,
                "end": None,
            }
            for seq_id, grp in g
        ]
    return [
        {
            "id": r.unit_id,
            "seq_id": r.seq_id,
            "video_path": r.video_path,
            "audio_path": r.audio_path,
            "start": None if pd.isna(r.start) else float(r.start),
            "end": None if pd.isna(r.end) else float(r.end),
        }
        for r in df.itertuples()
    ]


def _vram() -> str:
    """返回 '  VRAM 峰值 8.4/16.0GiB' 形式的后缀；无 CUDA 时返回空串。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        peak = torch.cuda.max_memory_reserved() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        return f"  VRAM 峰值 {peak:.2f}/{total:.1f}GiB"
    except Exception:
        return ""


def run_caption(args: argparse.Namespace, targets: list[dict]) -> int:
    backend = args.backend
    if backend == "auto":
        backend = probe_backend(args.llava_dir or CaptionConfig.model_dir)
        print(f"[S1] 自动探测 caption 后端 -> {backend}")
        if backend == "dummy":
            raise RuntimeError("Install LLaVA-NeXT and provide real weights before caption generation.")
            print("[S1] ⚠ LLaVA-NeXT 未安装，将输出兜底描述。这些结果"
                  "只能用于打通流程，不能作为论文数字。\n"
                  "     安装：pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git")
    cfg = CaptionConfig(
        model_dir=args.llava_dir or CaptionConfig.model_dir,
        backend=backend,
        num_frames=args.num_frames,
        gpu_mem_gb=args.gpu_mem_gb,
        quant=args.quant,
    )
    cap = LlavaCaptioner(cfg)
    t_load = time.time()
    cap.load()  # 显式预加载，便于区分「权重占用」与「推理峰值」
    print(f"[S1] 模型加载 {time.time()-t_load:.1f}s{_vram()}", flush=True)
    # 小批量（试跑）时逐条打印，全量时每 20 条打印一次
    every = 1 if len(targets) <= 20 else 20
    fails, t0 = 0, time.time()
    with open(args.out, "a", encoding="utf-8") as fo:
        for i, t in enumerate(targets, 1):
            t_it = time.time()
            try:
                frames = read_video_frames(
                    t["video_path"], num_frames=cfg.num_frames,
                    start=t["start"], end=t["end"],
                )
                text, fields = cap.generate(frames)
                err = None
            except Exception as e:  # 单样本失败不中断全局
                from src.extractors.prompts import FALLBACK_CAPTION, flatten, parse_structured
                fields = parse_structured(FALLBACK_CAPTION)
                text, err, fails = flatten(fields), f"{type(e).__name__}: {e}", fails + 1
            rec = {"id": t["id"], "seq_id": t["seq_id"], "text": text,
                   "fields": fields, "backend": backend}
            if err:
                rec["error"] = err
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            if i % every == 0 or i == len(targets):
                el = time.time() - t0
                eta = el / i * (len(targets) - i)
                print(f"      {i}/{len(targets)}  本条 {time.time()-t_it:.2f}s  "
                      f"均 {el/i:.2f}s/it  ETA {eta/60:.1f}min  "
                      f"失败 {fails}{_vram()}", flush=True)
    cap.unload()
    print(f"[S1] caption 完成，失败 {fails}/{len(targets)} -> {args.out}{_vram()}")
    return int(fails > 0)


def run_asr(args: argparse.Namespace, targets: list[dict]) -> int:
    cfg = AsrConfig(
        model_dir=args.whisper_dir or AsrConfig.model_dir,
        batch_size=args.batch_size,
        device=args.device,
        dtype="float32" if args.device == "cpu" else "float16",
    )
    tr = WhisperTranscriber(cfg)
    fails, t0 = 0, time.time()
    with open(args.out, "a", encoding="utf-8") as fo:
        for s in range(0, len(targets), cfg.batch_size):
            chunk = targets[s : s + cfg.batch_size]
            waves, errs = [], []
            for t in chunk:
                try:
                    if not t["audio_path"] or not isinstance(t["audio_path"], str):
                        raise FileNotFoundError("manifest 中无 audio_path")
                    waves.append(read_audio(t["audio_path"], sr=SAMPLE_RATE,
                                            start=t["start"], end=t["end"]))
                    errs.append(None)
                except Exception as e:
                    waves.append(None)
                    errs.append(f"{type(e).__name__}: {e}")
                    fails += 1
            texts = tr.transcribe_batch(waves)
            for t, txt, err in zip(chunk, texts, errs):
                rec = {"id": t["id"], "seq_id": t["seq_id"], "text": txt}
                if err:
                    rec["error"] = err
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            done = min(s + cfg.batch_size, len(targets))
            if done % (cfg.batch_size * 10) == 0 or done == len(targets):
                el = time.time() - t0
                print(f"      {done}/{len(targets)}  {el/done:.2f}s/it  "
                      f"ETA {el/done*(len(targets)-done)/60:.1f}min  失败 {fails}",
                      flush=True)
    tr.unload()
    print(f"[S1] asr 完成，失败 {fails}/{len(targets)} -> {args.out}")
    return int(fails > 0)


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.manifest):
        print(f"[S1] manifest 不存在：{args.manifest}，先跑 s0_scan.py", file=sys.stderr)
        return 1
    df = pd.read_csv(args.manifest)

    if args.task == "probe":
        d = args.llava_dir or CaptionConfig.model_dir
        print(f"[S1] LLaVA 目录: {d}")
        print(f"[S1] 可用后端: {probe_backend(d)}")
        for mod, name in [("llava", "LLaVA-NeXT"), ("transformers", "transformers"),
                          ("torch", "torch")]:
            try:
                m = __import__(mod)
                print(f"       OK  {name} {getattr(m, '__version__', '?')}")
            except ImportError:
                print(f"       --  {name} 未安装")
        return 0

    if not args.out:
        print("[S1] 需要 --out", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    gran = resolve_granularity(args, df)
    targets = build_targets(df, gran)
    if args.limit:
        targets = targets[: args.limit]

    if args.overwrite and os.path.exists(args.out):
        os.remove(args.out)
    done = load_done(args.out)
    todo = [t for t in targets if t["id"] not in done]
    print(f"[S1] task={args.task} granularity={gran} "
          f"目标 {len(targets)}，已完成 {len(done)}，待处理 {len(todo)}")
    if not todo:
        print("[S1] 无待处理项")
        return 0

    return run_caption(args, todo) if args.task == "caption" else run_asr(args, todo)


if __name__ == "__main__":
    raise SystemExit(main())
