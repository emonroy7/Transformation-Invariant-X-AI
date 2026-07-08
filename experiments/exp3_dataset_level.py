"""Experiment 3 - Dataset-Level Evaluation (MS COCO).

Tests whether the single-image benefits generalize to the large, diverse MS COCO
dataset (80 classes with pixel-level instance masks). For each image and its target
COCO class we build a binary ground-truth mask (pixel-wise union of that class's
instance masks). Then, for each backbone and each setting - Baseline, Rotation-,
Zoom-, Shift-X-, and Shift-Y-invariant - we produce a class-specific heatmap and
score it:
    - AUC-ROC against the COCO mask (localization faithfulness),
    - L2 difference vs. the baseline heatmap (how far the invariant map moves).

Reconstructs the MS COCO "baseline vs. transformation-invariant" table.
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.data.datasets import iter_coco_masks
from src.data.preprocessing import prepare, resize_to
from src.gradcam import invariant as inv
from src.gradcam.gradcam import GradCAM
from src.metrics.metrics import auc_roc, l2_difference
from src.models.backbones import load_backbone

# Setting name -> invariant function (None means baseline Grad-CAM)
SETTINGS = {
    "baseline": None,
    "rotation": inv.rotation_invariant,
    "zoom": inv.zoom_invariant,
    "shift_x": inv.shift_x_invariant,
    "shift_y": inv.shift_y_invariant,
}


def run(model_name, limit=1000):
    model, preprocess_fn, target_layer, input_size = load_backbone(model_name)
    gradcam = GradCAM(model, target_layer)

    records = {name: {"auc": [], "l2": []} for name in SETTINGS}
    for image_path, class_index, mask in iter_coco_masks(limit=limit):
        image = prepare(image_path, input_size)
        mask_r = resize_to(mask, input_size) if mask is not None else None

        base, cls = gradcam.heatmap(preprocess_fn(image.astype("float32").copy()),
                                    class_index=class_index)
        for name, fn in SETTINGS.items():
            heat = base if fn is None else fn(gradcam, preprocess_fn, image, class_index=cls)
            if mask_r is not None:
                records[name]["auc"].append(auc_roc(heat, mask_r))
            records[name]["l2"].append(l2_difference(base, heat))

    rows = []
    for name, rec in records.items():
        aucs = [v for v in rec["auc"] if v == v]     # drop NaN
        rows.append({
            "setting": name,
            "mean_AUC": float(np.mean(aucs)) if aucs else float("nan"),
            "mean_L2_vs_baseline": float(np.mean(rec["l2"])) if rec["l2"] else float("nan"),
            "n": len(rec["l2"]),
        })
    df = pd.DataFrame(rows)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.TABLES_DIR / f"exp3_coco_{model_name}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved dataset-level (MS COCO) results to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet152", choices=list(config.MODEL_CONFIGS))
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    run(args.model, args.limit)


if __name__ == "__main__":
    main()
