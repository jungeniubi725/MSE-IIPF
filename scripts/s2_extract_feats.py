"""S2：提取三路特征并落盘（论文 E2–E3）。

用法
----
    python scripts/s2_extract_feats.py --manifest cache/ravdess/manifest.csv \
        --mod visual --out-dir cache/ravdess/feats --limit 8
    python scripts/s2_extract_feats.py --manifest cache/ravdess/manifest.csv \
        --mod audio  --out-dir cache/ravdess/feats
    python scripts/s2_extract_feats.py --manifest cache/ravdess/manifest.csv \
        --mod text --out-dir cache/ravdess/feats \
        --caption-jsonl cache/ravdess/text/captions.jsonl \
        --asr-jsonl cache/ravdess/text/asr.jsonl

落盘布局
-------
    <out-dir>/visual/<safe_unit_id>.npy      (n_v, 768)   unit 级
    <out-dir>/audio/<safe_unit_id>.npy       (n_a, 768)   unit 级
    <out-dir>/text_sv/<safe_id>.npy          (n_sv, 768)  sequence 级（D_v）
    <out-dir>/text_sa/<safe_id>.npy          (n_sa, 768)  sequence 级（D_a）
    <out-dir>/<mod>_meta.json                形状与配置记录

视觉/音频按 unit（时间片段）提，文本按 sequence 提 —— 与 S1 的粒度一致，
理由见 E18：X_t 是单向量，文本模态天然 sequence 级。
断点续跑：逐文件落盘，已存在则跳过。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractors.audio_emotion2vec import AudioConfig, Emotion2VecExtractor
from src.extractors.text_roberta import RobertaEncoder, TextConfig
from src.extractors.visual_videomae import VideoMAEExtractor, VisualConfig
from src.utils.console import setup_console
from src.utils.media import read_audio, read_video_frames

setup_console()


def safe_name(uid: str) -> str:
    """unit_id 含 / 与 #，转成合法文件名。"""
    return uid.replace("/", "__").replace("#", "-at-").replace("\\", "__")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2 三路特征提取")
    p.add_argument("--manifest", required=True)
    p.add_argument("--mod", required=True, choices=["visual", "audio", "text"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--caption-jsonl", help="text 模态必需：S1 的 D_v")
    p.add_argument("--asr-jsonl", help="text 模态必需：S1 的 D_a")
    p.add_argument("--visual-model", default=VisualConfig.model_id)
    p.add_argument("--audio-model", default=AudioConfig.model_dir)
    p.add_argument("--text-model", default=TextConfig.model_dir)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--visual-pool", default="temporal", choices=["temporal", "mean", "none"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=256, help="音频帧级序列上限")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _todo(items: list[tuple[str, dict]], out_dir: str, overwrite: bool):
    os.makedirs(out_dir, exist_ok=True)
    if overwrite:
        return items
    return [(k, v) for k, v in items
            if not os.path.exists(os.path.join(out_dir, safe_name(k) + ".npy"))]


def _progress(i: int, n: int, t0: float, fails: int, every: int = 50) -> None:
    if i % every == 0 or i == n:
        el = time.time() - t0
        print(f"      {i}/{n}  {el/i:.2f}s/it  ETA {el/i*(n-i)/60:.1f}min  失败 {fails}",
              flush=True)


def run_visual(args: argparse.Namespace, df: pd.DataFrame) -> dict:
    out_dir = os.path.join(args.out_dir, "visual")
    items = [
        (r.unit_id, {"path": r.video_path,
                     "start": None if pd.isna(r.start) else float(r.start),
                     "end": None if pd.isna(r.end) else float(r.end)})
        for r in df.itertuples()
    ]
    if args.limit:
        items = items[: args.limit]
    todo = _todo(items, out_dir, args.overwrite)
    print(f"[S2] visual 目标 {len(items)}，待处理 {len(todo)}")
    if not todo:
        return {}

    ex = VideoMAEExtractor(VisualConfig(model_id=args.visual_model, device=args.device,
                                       dtype="float32" if args.device == "cpu" else "float16",
                                       num_frames=args.num_frames,
                                       pool=args.visual_pool,
                                       batch_size=args.batch_size))
    fails, t0, shapes = 0, time.time(), []
    for s in range(0, len(todo), args.batch_size):
        chunk = todo[s : s + args.batch_size]
        clips, keys = [], []
        for uid, info in chunk:
            try:
                clips.append(read_video_frames(info["path"], num_frames=args.num_frames,
                                               start=info["start"], end=info["end"]))
                keys.append(uid)
            except Exception as e:
                fails += 1
                print(f"      ⚠ 解码失败 {uid}: {type(e).__name__}: {e}")
        if not clips:
            continue
        try:
            feats = ex.extract_batch(clips)
        except Exception as e:
            fails += len(clips)
            print(f"      ⚠ 前向失败 batch@{s}: {type(e).__name__}: {e}")
            continue
        for uid, f in zip(keys, feats):
            np.save(os.path.join(out_dir, safe_name(uid) + ".npy"), f.astype(np.float32))
            shapes.append(f.shape)
        _progress(min(s + args.batch_size, len(todo)), len(todo), t0, fails)
    ex.unload()
    return {"modality": "visual", "n_done": len(shapes), "n_fail": fails,
            "shape_example": list(shapes[0]) if shapes else None,
            "pool": args.visual_pool, "num_frames": args.num_frames,
            "model": args.visual_model, "dim": 768}


def run_audio(args: argparse.Namespace, df: pd.DataFrame) -> dict:
    out_dir = os.path.join(args.out_dir, "audio")
    items = [
        (r.unit_id, {"path": r.audio_path,
                     "start": None if pd.isna(r.start) else float(r.start),
                     "end": None if pd.isna(r.end) else float(r.end)})
        for r in df.itertuples()
    ]
    if args.limit:
        items = items[: args.limit]
    todo = _todo(items, out_dir, args.overwrite)
    print(f"[S2] audio 目标 {len(items)}，待处理 {len(todo)}")
    if not todo:
        return {}

    ex = Emotion2VecExtractor(AudioConfig(model_dir=args.audio_model, device=args.device, max_frames=args.max_frames))
    fails, t0, shapes = 0, time.time(), []
    for i, (uid, info) in enumerate(todo, 1):
        try:
            if not info["path"] or not isinstance(info["path"], str):
                raise FileNotFoundError("manifest 无 audio_path")
            if info["start"] is None and info["end"] is None:
                feat = ex.extract_path(info["path"])
            else:  # 片段级：切波形后写临时 wav（funasr 只吃路径）
                wave = read_audio(info["path"], start=info["start"], end=info["end"])
                feat = ex.extract_wave(wave)
            np.save(os.path.join(out_dir, safe_name(uid) + ".npy"), feat.astype(np.float32))
            shapes.append(feat.shape)
        except Exception as e:
            fails += 1
            if fails <= 5:
                print(f"      ⚠ 失败 {uid}: {type(e).__name__}: {e}")
        _progress(i, len(todo), t0, fails)
    ex.unload()
    return {"modality": "audio", "n_done": len(shapes), "n_fail": fails,
            "shape_example": list(shapes[0]) if shapes else None,
            "granularity": AudioConfig.granularity, "max_frames": args.max_frames,
            "model": "emotion2vec_base", "dim": 768}


def _read_jsonl(path: str | None) -> dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records[r["id"]] = r
    out: dict[str, str] = {}
    for identifier, r in records.items():
        if r.get("error") or r.get("backend") == "dummy":
            raise ValueError(f"Failed/dummy text record {identifier}; regenerate it before training")
        out[identifier] = r.get("text", "") or ""
    return out


def run_text(args: argparse.Namespace, df: pd.DataFrame) -> dict:
    caps = _read_jsonl(args.caption_jsonl)
    asrs = _read_jsonl(args.asr_jsonl)
    if not caps or not asrs:
        raise ValueError("Both caption and ASR JSONL are required; run s1_gen_text.py first")
    print(f"[S2] 读入 D_v {len(caps)} 条，D_a {len(asrs)} 条")

    seq_ids = list(dict.fromkeys(df["seq_id"].tolist()))
    if args.limit:
        seq_ids = seq_ids[: args.limit]
    # manifest 的 transcript 列若非空（MOSI/MOSEI 自带转录），优先于 Whisper 结果
    own = {r.seq_id: r.transcript for r in df.itertuples()
           if isinstance(getattr(r, "transcript", None), str) and r.transcript.strip()}

    # ASR 是 unit 级（id 形如 "<seq_id>#3"），而本步按 seq_id 出特征，
    # 直接用 seq_id 查会全部落空。按 unit_index 升序拼回整句。
    asr_by_seq: dict[str, str] = {}
    if asrs:
        buckets: dict[str, list[tuple[int, str]]] = {}
        for uid, txt in asrs.items():
            sid, _, idx = uid.partition("#")
            if not txt or not txt.strip():
                continue
            try:
                order = int(idx) if idx else 0
            except ValueError:
                order = 0
            buckets.setdefault(sid, []).append((order, txt.strip()))
        for sid, parts in buckets.items():
            parts.sort(key=lambda x: x[0])
            # 相邻单元常被 Whisper 转出同一句，去重避免整句复读
            seen, merged = set(), []
            for _, t in parts:
                if t not in seen:
                    seen.add(t)
                    merged.append(t)
            asr_by_seq[sid] = " ".join(merged)
        print(f"[S2] ASR 按 seq 聚合：{len(asrs)} unit -> {len(asr_by_seq)} seq")

    enc = RobertaEncoder(TextConfig(model_dir=args.text_model, device=args.device,
                                    dtype="float32" if args.device == "cpu" else "float16",
                                    batch_size=args.batch_size or 32))
    meta: dict = {"modality": "text", "dim": 768, "n_fail": 0}
    for branch, table in (("text_sv", caps), ("text_sa", {**asr_by_seq, **own})):
        out_dir = os.path.join(args.out_dir, branch)
        items = [(sid, {}) for sid in seq_ids]
        todo = [k for k, _ in _todo(items, out_dir, args.overwrite)]
        print(f"[S2] {branch} 目标 {len(items)}，待处理 {len(todo)}")
        if not todo:
            continue
        bs, t0, shapes, miss = enc.cfg.batch_size, time.time(), [], 0
        for s in range(0, len(todo), bs):
            chunk = todo[s : s + bs]
            texts = []
            for sid in chunk:
                t = table.get(sid, "")
                if sid not in table:
                    raise ValueError(f"Missing {branch} text for {sid}; rerun text generation")
                texts.append(t)
            feats, _ = enc.encode(texts)
            for sid, f in zip(chunk, feats):
                np.save(os.path.join(out_dir, safe_name(sid) + ".npy"), f.astype(np.float32))
                shapes.append(f.shape)
            _progress(min(s + bs, len(todo)), len(todo), t0, miss, every=bs * 5)
        meta[branch] = {"n_done": len(shapes),
                        "shape_example": list(shapes[0]) if shapes else None,
                        "n_missing_text": miss}
        if miss:
            print(f"      ⚠ {branch} 有 {miss} 条文本缺失，已按空串编码")
    enc.unload()
    meta["model"] = args.text_model
    return meta


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.manifest):
        print(f"[S2] manifest 不存在：{args.manifest}", file=sys.stderr)
        return 1
    df = pd.read_csv(args.manifest)
    os.makedirs(args.out_dir, exist_ok=True)

    fn = {"visual": run_visual, "audio": run_audio, "text": run_text}[args.mod]
    meta = fn(args, df)
    if meta:
        mp = os.path.join(args.out_dir, f"{args.mod}_meta.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[S2] {args.mod} 完成 -> {mp}")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    return int(meta.get("n_fail", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
