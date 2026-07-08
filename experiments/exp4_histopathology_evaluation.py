"""Experiment 4 - Histopathology Image Evaluation (CAMELYON16 / PatchCamelyon).

Evaluates ZOOM-invariant Grad-CAM on histopathology patches. Motivation: during
clinical review pathologists change magnification, so an explanation should not
"flicker" as scale changes. For each patch:
    1. compute a baseline Grad-CAM at the native scale,
    2. compute Grad-CAM at several zoom levels,
    3. inverse-zoom each map back to the base pixel grid and average them into a
       single zoom-invariant explanation.

Two measures (reconstructs Table 4.5, "Zoom-invariance evaluation"):
    - L2 difference between the baseline heatmap and each inverse-aligned zoomed
      heatmap, averaged over zoom factors (stability; lower = less scale-sensitive),
    - AUC-ROC of the baseline map and of the zoom-invariant map against the binary
      tumor mask, reported as mean +/- std over patches (localization faithfulness).
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.data.datasets import iter_camelyon16
from src.data.preprocessing import load_image, prepare, resize_to
from src.gradcam.gradcam import GradCAM
from src.gradcam.invariant import zoom_invariant
from src.metrics.metrics import auc_roc, l2_difference
from src.models.backbones import load_backbone
from src.transforms.geometric import Transform


def _mean_std(values):
    clean = [v for v in values if v == v]     # drop NaN
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean))


def run(model_name, limit=500):
    model, preprocess_fn, target_layer, input_size = load_backbone(model_name)
    gradcam = GradCAM(model, target_layer)

    baseline_aucs, invariant_aucs, mean_l2s = [], [], []
    for patch_path, mask_path, label in iter_camelyon16(limit=limit):
        image = prepare(patch_path, input_size)
        base, cls = gradcam.heatmap(preprocess_fn(image.astype("float32").copy()))

        # Zoom-invariant map: inverse-align each zoomed heatmap, then average
        robust = zoom_invariant(gradcam, preprocess_fn, image, class_index=cls)

        # Stability: mean L2 between baseline and each inverse-aligned zoomed map
        l2s = []
        for z in config.ZOOM_FACTORS:
            t = Transform(kind="zoom", value=float(z))
            zoomed, _ = gradcam.heatmap(
                preprocess_fn(t.apply(image).astype("float32").copy()), class_index=cls)
            l2s.append(l2_difference(base, t.invert(zoomed)))
        mean_l2s.append(float(np.mean(l2s)))

        # Faithfulness: AUC-ROC vs. the tumor mask (per patch)
        if mask_path is not None:
            mask = resize_to(load_image(mask_path)[..., 0], input_size)
            baseline_aucs.append(auc_roc(base, mask))
            invariant_aucs.append(auc_roc(robust, mask))

    base_mu, base_sd = _mean_std(baseline_aucs)
    inv_mu, inv_sd = _mean_std(invariant_aucs)
    df = pd.DataFrame([
        {"model": model_name, "setting": "Baseline (1.0x)",
         "mean_L2": float("nan"), "AUC_mean": base_mu, "AUC_std": base_sd,
         "n": len(baseline_aucs)},
        {"model": model_name, "setting": "Zoom-averaged",
         "mean_L2": float(np.mean(mean_l2s)) if mean_l2s else float("nan"),
         "AUC_mean": inv_mu, "AUC_std": inv_sd, "n": len(invariant_aucs)},
    ])
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.TABLES_DIR / f"exp4_histopathology_{model_name}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved histopathology zoom-invariance results to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet152", choices=list(config.MODEL_CONFIGS))
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    run(args.model, args.limit)


if __name__ == "__main__":
    main()
