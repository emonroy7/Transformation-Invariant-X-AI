# Datasets

Datasets are **not** stored in this repository because of their size and licensing
terms. Download them from the official sources and place them here using the layout
below. Everything inside `raw/` and `processed/` is git-ignored.

## MS COCO 2017

Natural-image benchmark used for stability evaluation (L2 consistency under
transformation). Official site: https://cocodataset.org/#download

Files used by the thesis:
- `train2017` (118,287 images)
- `val2017` (5,000 images)
- `annotations/instances_{train,val}2017.json`

## CAMELYON16

Histopathology whole-slide images with pixel-level tumor annotations, used for
faithfulness evaluation (AUC-ROC against ground-truth masks). Official site:
https://camelyon16.grand-challenge.org/

The thesis extracts 224×224 patches at 10× magnification, balanced between tumor and
normal regions (~80,000 patches total, split 80/10/10).

## Expected layout

```
data/
├── raw/
│   ├── coco/
│   │   ├── train2017/
│   │   ├── val2017/
│   │   └── annotations/
│   └── camelyon16/
│       ├── slides/
│       └── masks/
└── processed/
    ├── coco/                # resized / canonicalized tensors or manifests
    └── camelyon16/          # extracted 224x224 patches + patch-level masks
```

Update the paths in `src/config.py` if you store the data elsewhere.
