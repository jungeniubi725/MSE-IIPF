# MSE-IIPF

**Modality-Transformed Affective Semantic Enhancement and Implicit Internal-State Proxy Fusion for Multimodal Emotion Recognition**

[Repository](https://github.com/jungeniubi725/MSE-IIPF) · [Releases](https://github.com/jungeniubi725/MSE-IIPF/releases) · [Method ↔ code](docs/METHOD_CODE_MAP.md) · [Data and models](docs/DATA_AND_MODELS.md) · [Experiments and figures](docs/EXPERIMENTS.md)

MSE-IIPF predicts emotion by combining clues from video, sound and text. It also gives each prediction an uncertainty score: a larger score means the model is less sure about its answer.

This project provides the code to download data and models, train the model, test it, save predictions and draw figures.

## Implemented workflows

| Dataset | What the model starts with | What gets a prediction |
| --- | --- | --- |
| RAVDESS | Original video and audio clips | A full clip, with results for shorter segments too |
| CMU-MOSI | Prepared sound, word and facial features from openSMILE, GloVe and FACET | An utterance: a short stretch of speech |
| CMU-MOSEI | Prepared sound, word and facial features from COVAREP, GloVe and FACET | An utterance: a short stretch of speech |

A **feature** is a set of numbers that describes part of the input, such as a sound or a facial movement. RAVDESS starts with media files, so the project first extracts these features. The CMU datasets already provide them.

For RAVDESS, SG-DSFE uses text descriptions to help the model interpret video and sound. The alignment step brings the different kinds of features together, and IIPF combines them to make a prediction. The CMU experiments use ALBEF for alignment, followed by IIPF. See [Method and code](docs/METHOD_CODE_MAP.md) for the formulas and their implementations.

## Installation

Use **Python 3.11 or 3.12**. Clone the repository with a GitHub account that has access, or extract the source archive:

```bash
git clone https://github.com/jungeniubi725/MSE-IIPF.git
cd MSE-IIPF
```

Open a terminal in the project folder and create a separate Python environment for its packages:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS, `source .venv/Scripts/activate` in Windows Git Bash, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the packages and run a small check on your CPU:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/smoke_test.py --device cpu
```

The smoke test is a quick practice run. It creates a small set of random example inputs, trains a model, saves and reloads it, tests it on separate examples, and draws figures. You can use it to check the installation before downloading the datasets and pretrained models.

To extract features from real media, follow the [environment setup](docs/DATA_AND_MODELS.md#environment). It covers GPU support, the LLaVA model and FFmpeg, which reads audio and video files. The [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/#v280) lists the matching GPU packages.

## RAVDESS: download and prepare

Download the clips from the [official RAVDESS release](https://zenodo.org/records/1188976) and the pretrained models used to read them. The first two commands show the download plan; the next two download the files. The last command lists the clips and divides each one into four segments:

```bash
python scripts/download_data.py --dataset ravdess --dry-run
python scripts/download_models.py --models all --dry-run
python scripts/download_data.py --dataset ravdess
python scripts/download_models.py --models all
python scripts/s0_scan.py --dataset ravdess --root data/ravdess --out cache/ravdess/manifest.csv --num-segments 4 --probe-duration
```

Next, describe what appears in each video and turn its speech into text. Then convert the video, audio and two kinds of text into features the model can learn from:

```bash
python scripts/s1_gen_text.py --manifest cache/ravdess/manifest.csv --task caption --backend llava_next --quant int8 --out cache/ravdess/text/captions.jsonl
python scripts/s1_gen_text.py --manifest cache/ravdess/manifest.csv --task asr --out cache/ravdess/text/asr.jsonl
python scripts/s2_extract_feats.py --manifest cache/ravdess/manifest.csv --mod visual --out-dir cache/ravdess/feats
python scripts/s2_extract_feats.py --manifest cache/ravdess/manifest.csv --mod audio --out-dir cache/ravdess/feats
python scripts/s2_extract_feats.py --manifest cache/ravdess/manifest.csv --mod text --out-dir cache/ravdess/feats --caption-jsonl cache/ravdess/text/captions.jsonl --asr-jsonl cache/ravdess/text/asr.jsonl
```

Downloaded models go in `weights/`, and the commands above save text and features in `cache/ravdess/`. If you already have the models, point the scripts to their folders using the [path options](docs/DATA_AND_MODELS.md#reuse-existing-local-weights). Rerunning text generation or feature extraction skips completed items.

## RAVDESS: train, test and save results

The data has three jobs: training examples teach the model, validation examples help choose the best saved model, and test examples check its final predictions. A saved model is called a **checkpoint**.

Start training and use the validation set to choose the checkpoint:

```bash
python scripts/s3_train.py --config configs/ravdess.yaml --exp-name ravdess_main --validation-only
```

This saves the selected model as `runs/ravdess_main/best.pt`. Load it to predict the test examples, then draw figures from the saved predictions:

```bash
python scripts/evaluate.py --checkpoint runs/ravdess_main/best.pt --split test --out-dir runs/ravdess_main/test
python scripts/plot_results.py --predictions runs/ravdess_main/test/test_predictions.npz --level sequence --out-dir figures/ravdess_main --history runs/ravdess_main/history.json --class-names neutral calm happy sad angry fearful disgust surprised
```

`evaluate.py` uses the saved model, so you can run it without training again. Give each new training run a different `--exp-name` to keep its files separate. To train and test in one command, omit `--validation-only` from the training command.

To check how stable the results are, repeat the experiment with different random starting points. You can also turn off model parts to see what they contribute. The first two commands run these experiments; the last previews a settings grid with `--dry-run`:

```bash
python scripts/run_experiments.py --cv --seeds 42 43 44 --variants main --validation-only
python scripts/run_experiments.py --cv --seeds 42 --variants no_mam no_itac no_itam no_iipf no_cmc no_tpc no_prr softmax no_cal --validation-only
python scripts/run_experiments.py --grid configs/hparam_grid.example.yaml --seeds 42 --dry-run
```

`--seeds` sets the random starting points. `--cv` divides Actors 01 to 20 into five groups and takes turns using each group for testing; Actors 21 and 22 stay in the validation set. Actors 23 and 24 are used by the separate fixed-split test. With `--validation-only`, each run stops after validation. A settings grid also uses validation results to choose settings. See [experiment instructions](docs/EXPERIMENTS.md) for the final test commands.

## CMU-MOSI and CMU-MOSEI

These datasets provide features, so you can start without processing raw video and audio. Download the features and the lists of training, validation and test examples. Then convert the downloaded CSD files into CSV tables, align the features, train and test:

```bash
python scripts/download_data.py --dataset mosi
python scripts/prepare_cmu.py --csd-dir data/cmu_mosi/csd --out-dir data/cmu_mosi
python scripts/run_cmu.py --dataset mosi --stage align --seed 42
python scripts/run_cmu.py --dataset mosi --stage train --seed 42
python scripts/run_cmu.py --dataset mosi --stage evaluate --seed 42
```

For MOSEI, replace `mosi` with `mosei` in those commands. The `align` stage teaches the model how the sound, word and facial features relate to each other. The `train` stage learns to predict labels and checks progress on validation examples. The `evaluate` stage loads the saved model, tests it and saves its answers. Use `--stage all` to run all stages together.

The CMU experiments predict three labels from the dataset's sentiment scores:

| Score | Label |
| --- | --- |
| Score ≤ −0.5 | Negative |
| −0.5 < score < 0.5 | Neutral |
| Score ≥ 0.5 | Positive |

Draw figures from the test predictions and collect the test scores from your runs into one table:

```bash
python scripts/plot_results.py --predictions runs/cmu_mosi/seed42/main/test_predictions.npz --level unit --class-names Negative Neutral Positive --out-dir figures/cmu_mosi
python scripts/summarize_results.py --runs runs --split test --out tables/test_metrics.csv
```

The downloads come from the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK). If the download server is unavailable, use existing CSD/CSV files or set `--cmu-base-url` to a mirror you have access to.

## Files you get

| File | Content |
| --- | --- |
| `best.pt` | The saved model chosen using validation results |
| `config.json`, `environment.json` | Settings, package versions and device details for repeating a run; CMU stores environment details in the folder for each seed |
| `history.json` | How training and validation results changed during learning |
| `validation_metrics.json`, `test_metrics.json` | Scores that summarize the model's predictions |
| `*_predictions.npz` | Each example's ID, true label, predicted probabilities and uncertainty; CMU also saves video IDs and times |
| `*_unit_predictions.csv`, `*_sequence_predictions.csv` | Prediction tables you can open in a spreadsheet; RAVDESS includes both segment and full-clip tables |
| Figure directory | PNG, SVG and PDF figures, plus the CSV data used to draw them |

NPZ files store arrays of numbers for Python. CSV files are easier to inspect in a spreadsheet. A `null` value in a JSON score file means that score could not be calculated for those examples.

## Repository layout

```text
MSE-IIPF/
├── configs/                # Training settings and an example settings grid
├── src/
│   ├── data/                # Read examples and group them for training
│   ├── extractors/          # Turn audio, video and text into features
│   ├── models/              # The model's building blocks
│   ├── losses/              # Rules used to guide the model during training
│   ├── engine/              # Training and evaluation
│   └── metrics/             # Calculate scores from predictions
├── experiments/cmu/        # MOSI and MOSEI experiments
├── scripts/                # Download, prepare, train, test, export, plot and package
├── tests/                  # Check that the code works
├── docs/                   # Method details and running instructions
├── licenses/               # Notices for reused code
└── .github/workflows/       # Automatic tests on GitHub
```

Downloads and generated results go in `data/`, `weights/`, `cache/`, `runs/`, `figures/` and `tables/`. Git and the packaging script leave these folders out of source uploads.

## Figures and checks

`plot_results.py` reads saved predictions and draws figures showing which labels the model gets right, which ones it mixes up, and how its confidence relates to its accuracy. It can also show training progress. Each figure comes with its source data so you can inspect or redraw it.

For feature and waveform plots, see the [figure instructions](docs/EXPERIMENTS.md#figures). The [validation record](docs/VALIDATION.md) lists the software checks and their results.

## Help and code notices

If a command fails, open an issue with the command, settings, package versions and full error message. These details help locate the problem.

Sources and license notices for reused code are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Create and upload a source archive

Create a ZIP containing the code, settings and documentation:

```bash
python scripts/package_release.py --output dist/MERC-GitHub-source.zip
```

The command also creates a SHA-256 checksum file, which lets you check that a downloaded ZIP matches the original. See [GitHub upload instructions](docs/RELEASING.md) to attach the ZIP to a Release.

For everyday edits and uploads with Git Bash, see [GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md).

