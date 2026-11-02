# Data Augmentation Has Modest In-Domain Benefits but Improves Cross-Dataset Generalization in Thyroid Nodule Ultrasound Segmentation: A Multi-Dataset, Three-Architecture Benchmark

This repository accompanies the article *Data Augmentation Has Modest In-Domain Benefits but Improves Cross-Dataset Generalization in Thyroid Nodule Ultrasound Segmentation: A Multi-Dataset, Three-Architecture Benchmark* (Prochazka, Zeman, Smutek; under review).

Fourteen single augmentations, TrivialAugment Top-N pools and a RandAugment grid were trained on **six public thyroid nodule ultrasound datasets** with **three architectures** (ResNet34-U-Net, DeepLabV3+, MiT-B2-U-Net) under one fixed protocol, 2,160 training runs in total. Every validation-selected model was evaluated on its own test split and, without retraining, on the test splits of the five other datasets. The protocol extends the ultrasound augmentation benchmark of Tupper & Gagné, *Revisiting Data Augmentation for Ultrasound Images*, TMLR 2025 (<https://openreview.net/forum?id=iGcxlTLIL5>).

## Datasets

| Dataset | Source | Train / val / test images | Unit of analysis | Split |
|---|---|---|---|---|
| TUCC (Stanford Thyroid) | <https://aimi.stanford.edu/datasets/thyroid-ultrasound-cine-clip> | 7,441 / 807 / 1,970 | clip (75 / 9 / 21 clips) | clip-level, Stanford clip selection of Tupper & Gagné |
| TN3K | <https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation> | 2,303 / 576 / 614 | image | released test kept; trainval 80/20 by image |
| DDTI | <http://cimalab.unal.edu.co/?lang=en&mod=project&id=31> | 278 / 102 / 86 | case (228 / 76 / 76 cases) | case-level 60/20/20 |
| PKTN | <https://github.com/jayxjsun/CLIP-TNseg> | 603 / 201 / 201 | image | image-level 60/20/20 |
| TNUI-2021 | <https://github.com/zxg3017/TNUI-2021-> | 966 / 276 / 139 | image | released partition kept |
| ThyroidXL | <https://huggingface.co/datasets/hunglc007/ThyroidXL> | 7,633 / 1,908 / 2,094 | patient (2,683 / 671 / 739 patients) | released test kept; 20 % of development patients to validation |

One description file per dataset is in `configs/datasets/` (source, expected file layout, preprocessing, scan mask, grouping unit, split). The split procedures are recorded in `configs/experiments/tucc_tn3k_ddti_splits.json` and `configs/experiments/pktn_tnui2021_thyroidxl_splits.json`; the fixed splits themselves are the manifests listed below. The images are not redistributed; obtain them from the providers under their terms.

## Repository layout

- `configs/experiments/benchmark.yaml`: the training protocol used for all six datasets and three architectures (Adam, 50 epochs, cosine schedule with a 200-epoch horizon, batch 128, 224 px, seeds 0–2, TrivialAugment Top-2…Top-14, RandAugment N ∈ {2,3,4} × M ∈ {5,10,15,30}); `configs/models/`: the three architecture definitions (segmentation-models-pytorch 0.5.0); `configs/augmentations/upstream_order.yaml`: the augmentation order of Tupper & Gagné; `configs/experiments/export_contract.json`: the per-image export format; `configs/upstream_snapshot/`: the upstream augmentation configurations (MIT, Adam Tupper) from which the parameter ranges were taken
- `manifests/all/`: the fixed train/validation/test split (`splits.csv`) and scan-mask manifest (`scan_masks.csv`) of all six datasets that every training run used; `manifests/tucc_tn3k_ddti/`: the same rows for TUCC, TN3K and DDTI with the image inventory (`pairs.csv`) and the DDTI panel map; `manifests/pktn_tnui2021_thyroidxl/`: the raw and prepared split rows, scan-mask manifest and preparation audit (SHA-256 of every source and prepared file) of PKTN, TNUI-2021 and ThyroidXL
- `src/thyroid_augmentation/`: the Python package (data loading and augmentation, training, per-image evaluation, selection, cross-dataset inference); `external/ultrasound-augmentation/`: the ultrasound-specific transforms of <https://github.com/adamtupper/ultrasound-augmentation> (commit `87123ee`, MIT) and the Stanford clip list
- `scripts/`: numbered entry points (01 preparation of PKTN, TNUI-2021 and ThyroidXL; 02 preflight; 03 benchmark training queue; 04 parallel GPU check; 05 RandAugment to TrivialAugment handover; 06 cross-dataset inference; 07 evaluation of one checkpoint on any split)
- `runs/`: `summary.json` and `train_log.csv` of all 2,160 training runs (`m2_single`: 810 single-augmentation runs, `m2_ta`: 702 TrivialAugment runs, `m2_ra`: 648 RandAugment runs)
- `selection/`: validation rankings of the single augmentations, TrivialAugment pools and RandAugment configurations, the Top-N pools, the RandAugment magnitude mapping and the 216 validation-selected configurations
- `per_image/`: per-image test Dice of the selected models averaged over seeds, in-domain and for all 30 cross-dataset transfers, with the cross-dataset summary
- `tests/`: unit tests of the metric and of the TrivialAugment transform

Folder names `prepared_data/extension/`, `scan_masks/extension/` and the run phases `m2_single`, `m2_ta`, `m2_ra` are recorded in the released manifests and logs and are therefore kept as they were when the experiments ran.

## Citation

- Prochazka A, Zeman J, Smutek D. Data augmentation has modest in-domain benefits but improves cross-dataset generalization in thyroid nodule ultrasound segmentation: a multi-dataset, three-architecture benchmark. Under review.

## License

- MIT (see `LICENSE`); `external/ultrasound-augmentation/` and `configs/upstream_snapshot/` remain under their own MIT License (Copyright (c) 2025 Adam Tupper)
