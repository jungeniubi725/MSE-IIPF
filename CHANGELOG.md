# Changes in the source release

## 0.1.0 — 2026-11-23

- Extract the model, loss, data, training, evaluation, metric and plotting source files from the experiment environment.
- Preserve the distinct RAVDESS and CMU architectures and document their relationship to the manuscript.
- Replace workstation paths with configurable paths and provide dataset/model download and CSD conversion commands.
- Add independent checkpoint evaluation, raw predictions with stable IDs, CSV exports, reusable plots and source-data tables.
- Preserve the corrected actor split; use complete ordered loaders for evaluation and feature export.
- Apply remainder batches in gradient accumulation and mask padded labels in the Softmax ablation loss.
- Require real text-generation outputs for feature extraction and report media-generation failures.
- Train CMU alignment independently per seed and keep fusion training separate from final test evaluation.
- Add CPU integration tests, an offline smoke workflow, source-only packaging and GitHub Actions configuration.
- Exclude datasets, model checkpoints, historical results, simulated result generators, document editing tools and unrelated upstream repositories.

Scientific defaults are inherited from the source implementation. Packaging does not convert the error-target calibration objective into the manuscript's conflict-target objective, or claim new benchmark results.

