# Third-party sources

The repository preserves the experiment's custom tri-modal ALBEF implementation in `src/models/tri_albef_align.py`. It adapts ideas and helper logic from [Salesforce ALBEF](https://github.com/salesforce/ALBEF); the license file found with that source is retained as [licenses/ALBEF.txt](licenses/ALBEF.txt). The entire upstream repository and its unrelated tasks are not vendored.

Pretrained encoders are obtained from their original repositories. Their weights and license files are not bundled here. See [model sources](docs/DATA_AND_MODELS.md#model-registry) and the corresponding model cards for terms and citations.

The dataset download recipes and split sources are attributed to the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) and the [RAVDESS release](https://zenodo.org/records/1188976). No dataset samples or pretrained checkpoints are included in this source archive.

This file does not assign a license to the authors' original code. A project-wide code license and final author/citation metadata should be provided by the rights holders.

