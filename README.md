# Data Augmentation Has Modest In-Domain Benefits but Improves Cross-Dataset Generalization in Thyroid Nodule Ultrasound Segmentation: A Multi-Dataset, Three-Architecture Benchmark

Code and fixed inputs for the manuscript *Data Augmentation Has Modest In-Domain Benefits but Improves Cross-Dataset Generalization in Thyroid Nodule Ultrasound Segmentation* (Prochazka, Zeman, Smutek; under review). Covers six datasets and three architectures under one fixed protocol with a per-image evaluation metric. Builds on Tupper & Gagné, *Revisiting Data Augmentation for Ultrasound Images*, TMLR 2025 (<https://openreview.net/forum?id=iGcxlTLIL5>).

## Contents

- `src/`: Python package `thyroid_augmentation` (training, per-image evaluation, selection, cross-dataset inference)
- `scripts/`: data preparation (`01_*`), training queue (`04_*`–`07_*`), cross-dataset inference (`08_*`), evaluation of released checkpoints (`09_*`)
- `configs/`: dataset, model and augmentation settings; `configs/published/`: model definitions and protocol of the preceding benchmark; `external/`: ultrasound-specific transforms from <https://github.com/adamtupper/ultrasound-augmentation> (commit `87123ee`, MIT)
- `manifests/`: fixed train/validation/test splits of all six datasets (`metric_v2/`) and the preparation record of PKTN, TNUI-2021 and ThyroidXL (`extension/`); `panel_maps/`: DDTI panels
- `runs/`: `summary.json` and `train_log.csv` for all 2160 training runs (`m2_single`, `m2_ta`, `m2_ra`)
- `metadata/`: validation rankings, TrivialAugment pools, RandAugment magnitude mapping, the 216 validation-selected checkpoints with SHA-256
- `per_image/`: per-image test Dice of the selected models averaged over seeds (in-domain and all 30 cross-dataset transfers)
- `tests/`: unit tests
- Statistics, tables and figures of the manuscript are not part of this repository

## Setup

- Python 3.12; the experiments were run on Windows 11 with one NVIDIA GPU (CUDA 12.8 wheels below); the training queue uses Windows file locks
- Activate the virtual environment before running the commands below

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e ".[models,dev]"
```

## Data

- Obtain the datasets from their providers: TUCC/Stanford Thyroid (<https://aimi.stanford.edu/datasets/thyroid-ultrasound-cine-clip>), TN3K (<https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation>), DDTI (<http://cimalab.unal.edu.co/?lang=en&mod=project&id=31>), PKTN (<https://github.com/jayxjsun/CLIP-TNseg>), TNUI-2021 (<https://github.com/zxg3017/TNUI-2021->), ThyroidXL (<https://huggingface.co/datasets/hunglc007/ThyroidXL>)
- Layout expected by the manifests: `data/{tucc,tn3k,ddti}/prepared/<sample_id>_image.png, _mask.png` (exports as in the preceding repository); `data/pktn/{images,masks}/`, `data/tnui2021/{train,val,test}/{images,masks}/` and `data/thyroidxl/{train,test}/{images,masks}/` as distributed by the providers; filenames must match `manifests/metric_v2/splits.csv`
- Optional: copy `configs/local_paths.example.yaml` to `configs/local_paths.yaml` and set `data_root`

## Reproduce

Prepare masks and scan masks, then check inputs and the GPU:

```bash
python scripts/01_prepare_extension_datasets.py --action prepare
python -m thyroid_augmentation prepare-scan-masks
python -m thyroid_augmentation prepare-arxiv-stanford-scan-masks
python scripts/04_preflight.py --gpu-smoke
python scripts/05_benchmark.py prepare
python -m pytest
```

Train (810 single, 702 TrivialAugment and 648 RandAugment runs; TrivialAugment pools are built from the validation ranking after the single stage):

```bash
python scripts/05_benchmark.py run --stage all --confirm-training
```

- `run-parallel --confirm-training` trains two runs concurrently where GPU memory permits (after `python scripts/06_parallel_gpu_smoke.py`); `status` prints progress; every run resumes from its last checkpoint
- Outputs: `runs/<phase>/<dataset>/<model>/<policy>/seed_<seed>/`, `results/metric_v2/per_image/<run_id>/{val,test}.csv`, rankings in `results/metric_v2/`

Cross-dataset inference of the 216 validation-selected checkpoints on the five other datasets (1080 evaluations):

```bash
python scripts/08_cross_dataset_inference.py prepare
python scripts/08_cross_dataset_inference.py run
python scripts/08_cross_dataset_inference.py analyze
```

## Checkpoints

- Released: baseline and validation-selected single, TrivialAugment and RandAugment configurations for every dataset, architecture and seed (216 files, 10.6 GiB)
- Download: <https://drive.google.com/drive/folders/1uAKmTXx6dZdKC9uUavr5ki82bzKhTXDm> (folder `checkpoints_release/`, 10.6 GiB; download the whole folder or individual models)
- Format: model weights only, stored in float16 (`{"model": state_dict, "dtype": "float16", "epoch", "best_dice", "config"}`); optimizer state is not included. Cast to float32 before inference. Float32 inference from the released weights reproduces the stored per-image test Dice within 0.004 per image and 0.0002 on average (checked on 12 DDTI models)
- Place files at the paths listed in `metadata/selected_checkpoints.csv` (`checkpoints_release/<phase>/<dataset>/<model>/<policy>/seed_<seed>/best.pt`; `release_sha256` gives the SHA-256 of every released file, `checkpoint_sha256` that of the original training checkpoint)
- Evaluate a released checkpoint on any split of any dataset, e.g. `python scripts/09_evaluate_release_checkpoint.py --checkpoint checkpoints_release/m2_single/TUCC/res34_unet/none/seed_0/best.pt --model res34_unet --dataset TN3K --output results/tucc_res34_baseline_on_tn3k.csv`
- `scripts/08_cross_dataset_inference.py` expects the original training checkpoints and their per-image exports and cannot be run from the released weights

## Citation

- Prochazka A, Zeman J, Smutek D. Data augmentation has modest in-domain benefits but improves cross-dataset generalization in thyroid nodule ultrasound segmentation: a multi-dataset, three-architecture benchmark. Under review.

## License

- MIT (see `LICENSE`); `external/ultrasound-augmentation/` remains under its own MIT License (Copyright (c) 2025 Adam Tupper)
