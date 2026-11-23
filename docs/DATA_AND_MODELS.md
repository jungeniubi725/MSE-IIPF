# Data, models and environment

## Environment

Core training and metrics use Python 3.11–3.12 and PyTorch 2.8.0. CPU tests do not need pretrained encoders. Run all commands from the project root.

For an NVIDIA system compatible with CUDA 12.8, install the matching torch packages before the extraction requirements. Other platform builds are listed in the [PyTorch version guide](https://pytorch.org/get-started/previous-versions/#v280).

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -r requirements-extract.txt
```

Install LLaVA-NeXT separately. The inspected environment recorded commit `bce12e479bc4dfee2b9c50c88137b01ff51bd483`:

```bash
python -m pip install --no-deps "git+https://github.com/LLaVA-VL/LLaVA-NeXT.git@bce12e479bc4dfee2b9c50c88137b01ff51bd483"
```

`--no-deps` avoids replacing the torch/transformers versions above with older upstream pins. LLaVA may require additional optional packages for a particular platform; its full environment was not rebuilt from scratch for this source release. Follow the [LLaVA model's backend instructions](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2) if an optional import is missing. The feature cache and fusion model can be run independently of that backend.

Install FFmpeg so that both `ffmpeg` and `ffprobe` are in `PATH`. Alternatively set `MERC_FFMPEG_DIR` to its executable directory. Video decoding uses OpenCV; audio decoding uses soundfile/librosa. LLaVA defaults to GPU inference; `--quant int8`, `--quant int4` and `--quant none` change its memory requirements. Encoder/fusion workloads have different memory requirements, so a successful CPU smoke test does not establish that a machine can load the full caption model.

## Model registry

| Key | Upstream source | Default local folder | Feature dimension |
| --- | --- | --- | --- |
| `llava` | [lmms-lab/LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2) | `weights/llava-video` | Text generation |
| `whisper` | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | `weights/whisper-large-v3` | Speech transcription |
| `roberta` | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | `weights/roberta-base` | 768 |
| `videomae` | [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) | `weights/videomae-base` | 768 |
| `emotion2vec` | [iic/emotion2vec_base](https://modelscope.cn/models/iic/emotion2vec_base) | `weights/emotion2vec-base` | 768 |

The [emotion2vec model card](https://huggingface.co/emotion2vec/emotion2vec_base) documents the ModelScope/FunASR path used here. Models and datasets retain their own terms. They are not redistributed by this repository.

The downloader records the resolved Hugging Face commit for every download. To request a particular revision, select one model:

```bash
python scripts/download_models.py --models llava --revision 013210b3aff822f1558b166d39c1046dd109520f
python scripts/download_models.py --models roberta videomae
```

ModelScope downloads retain their requested revision and local snapshot metadata. Do not treat a mutable branch name as a verified immutable revision.

## Reuse existing local weights

| Script | Path option | Environment fallback |
| --- | --- | --- |
| `s1_gen_text.py --task caption` | `--llava-dir PATH` | `MERC_LLAVA_DIR` |
| `s1_gen_text.py --task asr` | `--whisper-dir PATH` | `MERC_WHISPER_DIR` |
| `s2_extract_feats.py --mod visual` | `--visual-model PATH` | `MERC_VIDEOMAE_DIR` |
| `s2_extract_feats.py --mod audio` | `--audio-model PATH` | `MERC_EMOTION2VEC_DIR` |
| `s2_extract_feats.py --mod text` | `--text-model PATH` | `MERC_ROBERTA_DIR` |

Quote paths containing spaces. The emotion2vec path must be the snapshot containing its configuration and checkpoint, rather than the parent cache directory. For VideoMAE, a source-code checkout alone is not a model checkpoint. Existing historical feature caches may use nested branch folders; the dataset loader accepts both that layout and the flat release layout.

## RAVDESS layout and protocol

```text
data/ravdess/
├── Actor_01/
│   ├── 01-01-...-01.mp4    # Audio-video
│   └── 03-01-...-01.wav    # Matching audio
├── ...
└── Actor_24/
```

The downloader reads the [Zenodo record](https://zenodo.org/records/1188976), selects speech/song audio and video archives, checks the publisher's checksum, and extracts modality `01` video and `03` audio. Modality `02` duplicates are not used. Actor 18 has no song video archive; availability is checked against the record metadata. `--channel speech` selects only speech and creates a different experiment from the default combined channels. `--actors` can restrict video downloads for debugging; the audio archives are distributed as all-actor bundles.

Scan with `--probe-duration` to create real temporal segments. Missing durations are errors. The manifest records `seq_id`, `unit_id`, `unit_index`, media paths, start/end times, labels, speaker, split and fold. The fixed split is Actors 01–20 / 21–22 / 23–24 for train/validation/test. The five-fold definition is documented in [EXPERIMENTS.md](EXPERIMENTS.md).

## CMU files and official splits

The download URLs follow the published SDK recipes for [MOSI](https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSI/cmu_mosi.py) and [MOSEI](https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py).

| Dataset | Required CSV files after conversion |
| --- | --- |
| MOSI | `CMU_MOSI_Opinion_Labels.csv`, `CMU_MOSI_OpenSmile_EB10.csv`, `CMU_MOSI_TimestampedWordVectors.csv`, `CMU_MOSI_Visual_Facet_42.csv` |
| MOSEI | `CMU_MOSEI_Labels.csv`, `CMU_MOSEI_COVAREP.csv`, `CMU_MOSEI_TimestampedWordVectors.csv`, `CMU_MOSEI_VisualFacet42.csv` |

Both directories also need `official_splits.json` with `train`, `val`, `test` video-ID lists. The split downloader parses literal lists without executing downloaded Python. It records the split source URL and hash; use `--sdk-revision COMMIT` to pin the source.

```bash
python scripts/download_data.py --dataset mosi --splits-only --out-dir data/cmu_mosi
python scripts/run_cmu.py --dataset mosi --data-dir data/cmu_mosi --stage prepare
```

Each CSV contains `video_id,start_time,end_time,feature_0,...`. The converter preserves duplicate intervals and keeps rows for each video contiguous. MOSI EB10 rows must align exactly to label intervals; text and visual tokens are selected by interval overlap. MOSEI streams larger feature CSVs by video. Scaling statistics are fitted on training features only.

The unvalidated raw-media CMU adapter from the old workspace is not exposed as a supported experiment path. IEMOCAP and M3ED have no completed pipeline in this release.

