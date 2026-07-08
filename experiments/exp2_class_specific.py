"""Experiment 2 - Class-specific effects.

Aggregate transformation sensitivity per class to test whether some categories are
more fragile than others. Reconstructed stub: wire in the dataset iterator and
accumulate mean L2 drift grouped by (predicted or ground-truth) class.
"""
import argparse
from collections import defaultdict

import numpy as np

from src import config
from src.data.datasets import iter_coco
from src.data.preprocessing import prepare
from src.gradcam.gradcam import GradCAM
from src.metrics.metrics import l2_difference
from src.models.backbones import load_backbone
from src.transforms.geometric import Transform

SWEEPS = {
    "rotation": config.ROTATION_ANGLES,
    "zoom": config.ZOOM_FACTORS,
    "shift_x": config.SHIFT_PIXELS,
    "shift_y": config.SHIFT_PIXELS,
}


def run(model_name, dataset="coco", transform="rotation", limit=200):
    model, preprocess_fn, target_layer, input_size = load_backbone(model_name)
    gradcam = GradCAM(model, target_layer)
    values = SWEEPS[transform]

    per_class = defaultdict(list)
    source = iter_coco(limit=limit) if dataset == "coco" else []
    for image_path, label in source:
        image = prepare(image_path, input_size)
        base, cls = gradcam.heatmap(preprocess_fn(image.astype("float32").copy()))
        drift = []
        for v in values:
            t = Transform(kind=transform, value=float(v))
            cam, _ = gradcam.heatmap(
                preprocess_fn(t.apply(image).astype("float32").copy()), class_index=cls)
            drift.append(l2_difference(base, t.invert(cam)))
        per_class[label if label is not None else cls].append(float(np.mean(drift)))

    # TODO: persist a per-class summary table under results/tables/
    ranked = sorted(per_class.items(), key=lambda kv: -np.mean(kv[1]))
    for k, v in ranked[:20]:
        print(f"class={k}: mean L2={np.mean(v):.4f} (n={len(v)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet152", choices=list(config.MODEL_CONFIGS))
    ap.add_argument("--dataset", default="coco", choices=["coco", "camelyon16"])
    ap.add_argument("--transform", default="rotation", choices=list(SWEEPS))
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    run(args.model, args.dataset, args.transform, args.limit)


if __name__ == "__main__":
    main()
