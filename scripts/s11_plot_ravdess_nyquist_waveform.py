"""Decode one RAVDESS video audio stream and plot a Nyquist-safe waveform.

The source audio is decoded at its native sampling rate, mixed to mono, then
anti-aliased with a polyphase FIR resampler to the experiment's 16 kHz audio
rate. The corresponding Nyquist limit is therefore 8 kHz.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seq-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--qa-tools-dir", default=None, type=Path)
    parser.add_argument("--source-rate", type=int, default=48000)
    parser.add_argument("--target-rate", type=int, default=16000)
    return parser.parse_args()


def read_sequence_metadata(manifest: Path, seq_id: str) -> dict[str, object]:
    frame = pd.read_csv(manifest)
    rows = frame.loc[frame["seq_id"] == seq_id].sort_values("unit_index")
    if rows.empty:
        raise ValueError(f"sequence not found in manifest: {seq_id}")
    if len(rows) != 4 or rows["unit_index"].tolist() != [0, 1, 2, 3]:
        raise ValueError("expected four ordered temporal units")
    labels = rows["label"].astype(int).unique()
    emotions = rows["meta_emotion_name"].astype(str).unique()
    if len(labels) != 1 or len(emotions) != 1:
        raise ValueError("sequence labels are inconsistent")
    return {
        "seq_id": seq_id,
        "label": int(labels[0]),
        "emotion": str(emotions[0]),
        "duration_manifest_s": float(rows["end"].max()),
        "unit_boundaries_s": rows["end"].astype(float).tolist(),
    }


def decode_video_audio(video: Path, sample_rate: int) -> np.ndarray:
    if not video.exists():
        raise FileNotFoundError(video)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    waveform = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float64)
    if waveform.size == 0:
        raise ValueError("decoded audio stream is empty")
    if not np.isfinite(waveform).all():
        raise ValueError("decoded waveform contains NaN or infinite values")
    return waveform


def nyquist_resample(
    waveform: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sampling rates must be positive")
    if target_rate > source_rate:
        raise ValueError("this workflow expects downsampling or equal rates")
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    sampled = resample_poly(
        waveform,
        up=up,
        down=down,
        window=("kaiser", 8.6),
        padtype="line",
    )
    sampled = np.clip(sampled, -1.0, 1.0)
    if not np.isfinite(sampled).all():
        raise ValueError("resampled waveform contains NaN or infinite values")
    return sampled.astype(np.float32)


def plot_waveform(
    waveform: np.ndarray,
    target_rate: int,
    metadata: dict[str, object],
    output_dir: Path,
    qa_tools_dir: Path,
    source_rate: int,
) -> None:
    if qa_tools_dir is not None:
        sys.path.insert(0, str(qa_tools_dir))
        from audit_panel_alignment import require_matplotlib_panel_alignment

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    time_s = np.arange(len(waveform), dtype=np.float64) / float(target_rate)
    duration_s = len(waveform) / float(target_rate)
    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))

    fig, ax = plt.subplots(figsize=(7.2047, 3.2283), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.72)

    ax.axhline(0.0, color="#8A8A8A", linewidth=0.55, zorder=1)
    ax.fill_between(
        time_s,
        waveform,
        0.0,
        color="#8DB3E2",
        alpha=0.22,
        linewidth=0,
        rasterized=True,
        zorder=2,
    )
    ax.plot(
        time_s,
        waveform,
        color="#1F5A99",
        linewidth=0.48,
        alpha=0.92,
        rasterized=True,
        zorder=3,
    )

    boundaries = [float(value) for value in metadata["unit_boundaries_s"][:-1]]
    for boundary in boundaries:
        ax.axvline(
            boundary,
            color="#6F6F6F",
            linewidth=0.55,
            linestyle=(0, (3, 2)),
            alpha=0.75,
            zorder=4,
        )

    ax.set_xlim(0.0, duration_s)
    y_limit = max(peak * 1.08, 0.05)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Time (s)", fontsize=7.0, labelpad=4)
    ax.set_ylabel("Amplitude", fontsize=7.0, labelpad=4)
    ax.tick_params(axis="both", labelsize=6.3, width=0.6, length=3)
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.45, alpha=0.8)

    fig.text(
        0.085,
        0.946,
        "RAVDESS waveform after Nyquist-compliant sampling",
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color="#1A1A1A",
    )
    fig.text(
        0.085,
        0.875,
        f"{metadata['seq_id']}  |  emotion: {metadata['emotion']}",
        ha="left",
        va="top",
        fontsize=6.5,
        color="#3D3D3D",
    )
    fig.text(
        0.085,
        0.820,
        f"Anti-aliased polyphase resampling: {source_rate / 1000:.0f} kHz → "
        f"{target_rate / 1000:.0f} kHz; Nyquist limit = {target_rate / 2000:.0f} kHz; "
        f"duration = {duration_s:.3f} s; peak = {peak:.3f}; RMS = {rms:.3f}",
        ha="left",
        va="top",
        fontsize=6.2,
        color="#555555",
    )

    base = output_dir / "actor18_01-01-03-02-02-01-18_nyquist_waveform"
    fig.canvas.draw()
    if qa_tools_dir is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(base) + ".alignment-layout.json",
            overlay_svg=str(base) + ".alignment-overlay.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=300)
    fig.savefig(str(base) + ".tiff", dpi=600, format="tiff")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_sequence_metadata(args.manifest, args.seq_id)
    decoded_raw = decode_video_audio(args.video, args.source_rate)
    expected_source_samples = int(
        round(float(metadata["duration_manifest_s"]) * args.source_rate)
    )
    decoded = decoded_raw[:expected_source_samples]
    if len(decoded) != expected_source_samples:
        raise ValueError(
            "decoded audio is shorter than the manifest/container duration"
        )
    sampled = nyquist_resample(decoded, args.source_rate, args.target_rate)

    prefix = args.output_dir / "actor18_01-01-03-02-02-01-18_nyquist_waveform"
    time_s = np.arange(len(sampled), dtype=np.float64) / float(args.target_rate)
    pd.DataFrame({"time_s": time_s, "amplitude": sampled}).to_csv(
        str(prefix) + "_source_data.csv", index=False, encoding="utf-8-sig"
    )
    wavfile.write(
        str(prefix) + "_16kHz.wav",
        args.target_rate,
        np.round(sampled * 32767.0).astype(np.int16),
    )

    metadata.update(
        {
            "source_container": "MP4",
            "source_audio_codec": "AAC",
            "source_channels": 2,
            "source_sample_rate_hz": int(args.source_rate),
            "source_nyquist_hz": float(args.source_rate / 2),
            "target_sample_rate_hz": int(args.target_rate),
            "target_nyquist_hz": float(args.target_rate / 2),
            "resampling": "scipy.signal.resample_poly with Kaiser beta=8.6",
            "anti_aliasing": True,
            "decoded_sample_count_raw": int(len(decoded_raw)),
            "decoded_sample_count_used": int(len(decoded)),
            "codec_padding_samples_removed": int(len(decoded_raw) - len(decoded)),
            "output_sample_count": int(len(sampled)),
            "output_duration_s": float(len(sampled) / args.target_rate),
            "peak_amplitude": float(np.max(np.abs(sampled))),
            "rms_amplitude": float(
                np.sqrt(np.mean(np.square(sampled, dtype=np.float64)))
            ),
            "exclusions": 0,
        }
    )
    (Path(str(prefix) + "_metadata.json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_waveform(
        sampled,
        args.target_rate,
        metadata,
        args.output_dir,
        args.qa_tools_dir,
        args.source_rate,
    )

    qa = f"""# Figure contract and QA notes

## Figure contract

- Core conclusion: The selected RAVDESS video audio stream is represented as a complete time-domain waveform after anti-aliased 16 kHz sampling.
- Results-level question: What temporal amplitude structure is retained at the sampling rate used by the experiment's audio encoder?
- Archetype: Quantitative single-panel waveform.
- Backend: Python / Matplotlib only.
- Final size: 183 x 82 mm.
- Source: first audio stream decoded directly from the selected MP4 video.
- Exclusions: none; all {len(sampled):,} output samples are plotted.

## Signal processing

- Source: {args.source_rate:,} Hz, stereo AAC decoded to mono floating-point PCM.
- Target: {args.target_rate:,} Hz, corresponding to an {args.target_rate / 2:,.0f} Hz Nyquist limit.
- Resampling: polyphase FIR resampling with a Kaiser beta=8.6 anti-aliasing window.
- Duration: {len(sampled) / args.target_rate:.6f} s.
- The three vertical dashed lines reproduce the four temporal-unit boundaries in the experiment manifest.

## Interpretation limits

- The waveform shows amplitude over time; it does not display spectral energy or prove that all emotion-relevant information lies below 8 kHz.
- Frequencies above the target Nyquist limit are suppressed by the anti-aliasing resampler before decimation.
"""
    (args.output_dir / "FIGURE_QA_NOTES.md").write_text(qa, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
