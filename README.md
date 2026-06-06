# DINOv3 JPEG Robustness Retrieval

This repository contains the source code used to evaluate whether frozen
DINOv3 image representations remain reliable for image retrieval under JPEG
compression artifacts.

Clean Oxford-IIIT Pets images are used as the retrieval gallery, and
JPEG-compressed versions of the same images are used as queries. The main
experiment compares DINOv3 ViT-B/16 and DINOv3 ViT-S/16 with ImageNet-pretrained
ResNet-50 and supervised ViT-B/16 baselines under the same cosine-retrieval
protocol.

## Repository Contents

- `configs/default.yaml`: experiment configuration.
- `src/ads_dinov3/`: data preparation, compression, model loading, feature
  extraction, retrieval metrics, visualization, and run logging.
- `scripts/`: command-line entry points for data preparation, feature
  extraction, retrieval evaluation, figure generation, and sanity/full runs.
- `tests/`: unit tests for metric computation, bootstrap intervals, failure-case
  mining, and run-log behavior.
- `environment.yml`: conda environment specification.

Generated artifacts are intentionally not committed, including raw datasets,
processed JPEG images, model checkpoints, external source checkouts, extracted
features, metric tables, figures, retrieval grids, and run logs.

## Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate dinov3-jpeg-retrieval
```

If your local PyTorch/MKL combination raises an `iJIT_NotifyEvent` runtime
symbol error, install the optional conda activation hook:

```bash
bash scripts/install_runtime_hook.sh
```

## Required Assets

Before running DINOv3 experiments, prepare these local assets:

- DINOv3 repository checkout at `external/dinov3/`.
- DINOv3 ViT-B/16 checkpoint at
  `data/models/dinov3/dinov3_vitb16_lvd1689m.pth`.
- DINOv3 ViT-S/16 checkpoint at
  `data/models/dinov3/dinov3_vits16_lvd1689m.pth`.
- Oxford-IIIT Pets images and annotations under `data/raw/oxford_iiit_pet/`.

The Oxford-IIIT Pets dataset can be downloaded by the pipeline with
`DOWNLOAD_FLAG=--download`. DINOv3 source code and weights are external assets
and should be obtained according to their own license and access instructions.

## Reproducing The Experiments

All public run scripts assume a single GPU by default. Set `GPU` to choose the
visible GPU and `DEVICE` if a different device string is needed.

Run the sanity pipeline:

```bash
DOWNLOAD_FLAG=--download GPU=0 bash scripts/run_sanity.sh
```

Run the full experiment:

```bash
GPU=0 bash scripts/run_full.sh
```

The pipeline writes generated files under `data/` and `outputs/`:

- prepared manifests and JPEG queries under `data/processed/`
- extracted features under `outputs/features/`
- metric tables and top-k exports under `outputs/tables/`
- figures and retrieval grids under `outputs/figures/` and
  `outputs/retrieval_grids/`
- execution logs at `outputs/run_log.csv`

## Individual Commands

Prepare data:

```bash
python scripts/prepare_data.py --mode full --download --compress
```

Extract features for one model:

```bash
python scripts/extract_features.py --mode full --model dinov3 --gpu 0 --device cuda:0
```

Evaluate retrieval and generate figures:

```bash
python scripts/evaluate_retrieval.py --mode full
python scripts/make_figures.py --mode full
```

## Tests

Run unit tests with:

```bash
PYTHONPATH=src pytest tests/
```

## Code Availability

The source code used in this project is publicly available at:

```text
https://github.com/jihyounchoi/dinov3-jpeg-retrieval
```

The public repository includes the implementation, preprocessing scripts,
configuration files, tests, and instructions required to reproduce the main
results. It does not include datasets, model weights, generated features,
figures, tables, or report documents.
