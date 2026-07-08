# Transformation-Invariance Properties of Grad-CAM in Convolutional Neural Networks

M.Sc. thesis (Computer Science) — Ontario Tech University (University of Ontario
Institute of Technology), Oshawa, ON, Canada, December 2025.

This repository accompanies the thesis and stores the written document, the defense
presentation, and the reconstructed research code. Convolutional Neural Networks
(CNNs) are widely used for image classification, and **Grad-CAM** is a popular
post-hoc technique for explaining their decisions as visual heatmaps. This work
studies how reliable those explanations remain when the input image is
geometrically transformed (rotation, zoom, horizontal/vertical shift), and proposes
a **realignment-and-averaging framework** that produces robust,
transformation-invariant heatmaps.

> **Note on the code.** The original experiment code was lost. The modules under
> `src/`, `experiments/`, and `scripts/` are **reconstructed skeletons** rebuilt
> from the methodology described in the thesis. They define the intended structure,
> interfaces, and pipeline stages, with `TODO` markers where the original
> implementation details should be filled back in. They are a starting point, not a
> verified reproduction of the published results.

## What the thesis does

- **Models (ImageNet-pretrained, no fine-tuning):** ResNet152, DenseNet201, Xception.
- **Grad-CAM target layers:** `conv5_block3_out` (ResNet152), `conv5_block32_concat`
  (DenseNet201), `block14_sepconv2_act` (Xception).
- **Input sizes:** 224×224 (ResNet152, DenseNet201), 299×299 (Xception).
- **Datasets:** MS COCO 2017 (natural images, stability evaluation) and CAMELYON16
  histopathology patches (pixel-level tumor masks, faithfulness evaluation).
- **Transformations swept:** rotation (−180° to +180°), zoom (0.67× to 1.5×),
  horizontal shift and vertical shift (up to ±180 px).
- **Metrics:** Euclidean (L2) difference for heatmap consistency, and AUC-ROC
  against ground-truth masks for faithfulness.
- **Contribution:** transform-specific *invariant* Grad-CAM variants
  (Rotation / Zoom / Shift-X / Shift-Y) that transform the input, generate a
  Grad-CAM heatmap, warp the heatmap back to the canonical frame, and average across
  the sweep to suppress transformation-induced variance.

## Repository structure

```
transformation-invariant-gradcam/
├── README.md                  # this file
├── LICENSE                    # All rights reserved (see below)
├── CITATION.cff               # how to cite this work
├── GITHUB_SETUP.md            # how to push this repo to GitHub
├── .gitignore
├── requirements.txt           # Python dependencies (TensorFlow/Keras stack)
│
├── docs/                      # the thesis and defense materials
│   ├── Thesis_Paper.pdf       # full thesis (130 pages)
│   ├── Thesis_Presentation.pdf# defense slides (47 slides)
│   └── figures/               # exported figures (optional)
│
├── src/                       # reconstructed library code
│   ├── config.py              # paths, model configs, transform ranges
│   ├── data/
│   │   ├── preprocessing.py   # load / resize / canonicalize per backbone
│   │   └── datasets.py        # MS COCO & CAMELYON16 loaders + mask handling
│   ├── models/
│   │   └── backbones.py       # ResNet152 / DenseNet201 / Xception + target layers
│   ├── transforms/
│   │   └── geometric.py       # rotation, zoom, shift (+ inverse for realignment)
│   ├── gradcam/
│   │   ├── gradcam.py         # core Grad-CAM
│   │   └── invariant.py       # transformation-invariant Grad-CAM + averaging
│   ├── metrics/
│   │   └── metrics.py         # L2 difference, AUC-ROC
│   └── visualization/
│       └── heatmaps.py        # overlays and sensitivity plots
│
├── experiments/               # one script per thesis experiment
│   ├── exp1_single_image_consistency.py    # Exp 1: single-image consistency
│   ├── exp2_class_specific.py              # Exp 2: class-specific evaluation
│   ├── exp3_dataset_level.py               # Exp 3: dataset-level (MS COCO)
│   ├── exp4_histopathology_evaluation.py   # Exp 4: histopathology (CAMELYON16)
│   └── configs/               # experiment config files
│
├── scripts/
│   └── run_experiment.py      # CLI entry point
│
├── notebooks/                 # exploratory notebooks (optional)
│
├── data/                      # datasets are NOT committed — see data/README.md
│   ├── raw/
│   └── processed/
│
└── results/                   # generated figures and tables (git-ignored)
    ├── figures/
    └── tables/
```

## Setup

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU with CUDA is strongly recommended for running the CNN backbones over full
datasets.

## Datasets

Datasets are **not** included in this repository (size and licensing). See
[`data/README.md`](data/README.md) for how to obtain MS COCO 2017 and CAMELYON16
and the folder layout the code expects.

## Experiments

The thesis runs **four** experiments, one script each:

1. **Single-Image Transformation Consistency** (`exp1`) — sweep one transform over a
   single image and measure heatmap drift (L2) vs. transformation magnitude.
2. **Class-Specific Evaluation** (`exp2`) — aggregate transformation sensitivity per
   class to see which categories are most fragile.
3. **Dataset-Level Evaluation** (`exp3`) — MS COCO (80 classes, instance masks);
   baseline vs. all four invariant variants scored by AUC-ROC (faithfulness) and L2.
4. **Histopathology Image Evaluation** (`exp4`) — CAMELYON16 patches; **zoom-invariant**
   Grad-CAM, reporting L2 stability and per-patch AUC-ROC (mean ± std) against tumor
   masks (reconstructs Table 4.5).

## Usage

Once the skeletons are filled in, experiments are launched through the CLI:

```bash
# Exp 1 - single-image consistency: sweep one transform and plot L2 vs. magnitude
python scripts/run_experiment.py exp1 --model resnet152 --image path/to/img.jpg --transform rotation

# Exp 2 - class-specific effects across a set of images
python scripts/run_experiment.py exp2 --model densenet201 --dataset coco

# Exp 3 - dataset-level on MS COCO: baseline vs. invariant variants (AUC-ROC + L2)
python scripts/run_experiment.py exp3 --model xception

# Exp 4 - histopathology on CAMELYON16: zoom-invariant Grad-CAM (L2 + AUC-ROC)
python scripts/run_experiment.py exp4 --model resnet152
```

## Results

Figures and tables produced by the experiments are written to `results/` (ignored by
git so large artifacts stay out of history). The definitive results are the ones
reported in `docs/Thesis_Paper.pdf`.

## Citation

If you reference this work, please cite the thesis (see [`CITATION.cff`](CITATION.cff)):

> Roy, E. (2025). *Transformation-Invariance Properties of Grad-CAM in Convolutional
> Neural Networks* [Master's thesis, Ontario Tech University].

## License

**All rights reserved.** See [`LICENSE`](LICENSE). The thesis, presentation, and code
may not be reused or redistributed without the author's written permission.

---

*Reconstructed repository scaffold. Update the author name, GitHub handle, year, and
any `TODO` markers to match your own details before publishing.*
