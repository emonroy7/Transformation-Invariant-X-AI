"""Experiment 1 - Single-image consistency.

Sweep one transform over one image, measure how far each transformed (then
realigned) heatmap drifts from the canonical heatmap using the L2 difference, and
plot the sensitivity curve (thesis Figures 4.x). Also produces the invariant
(averaged) heatmap for the same image for visual comparison.
"""
import argparse

from src import config
from src.data.preprocessing import prepare
from src.gradcam import invariant as inv
from src.gradcam.gradcam import GradCAM
from src.metrics.metrics import l2_difference
from src.models.backbones import load_backbone
from src.transforms.geometric import Transform
from src.visualization.heatmaps import plot_sensitivity, save_overlay

TRANSFORM_SWEEPS = {
    "rotation": (config.ROTATION_ANGLES, "Rotation angle (deg)"),
    "zoom": (config.ZOOM_FACTORS, "Zoom factor"),
    "shift_x": (config.SHIFT_PIXELS, "Horizontal shift (px)"),
    "shift_y": (config.SHIFT_PIXELS, "Vertical shift (px)"),
}

INVARIANT_FN = {
    "rotation": inv.rotation_invariant,
    "zoom": inv.zoom_invariant,
    "shift_x": inv.shift_x_invariant,
    "shift_y": inv.shift_y_invariant,
}


def run(model_name, image_path, transform):
    model, preprocess_fn, target_layer, input_size = load_backbone(model_name)
    gradcam = GradCAM(model, target_layer)

    image = prepare(image_path, input_size)                 # canonical uint8 RGB
    base_cam, cls = gradcam.heatmap(preprocess_fn(image.astype("float32").copy()))

    values, xlabel = TRANSFORM_SWEEPS[transform]
    l2s = []
    for v in values:
        t = Transform(kind=transform, value=float(v))
        moved = t.apply(image)
        cam, _ = gradcam.heatmap(preprocess_fn(moved.astype("float32").copy()),
                                 class_index=cls)
        l2s.append(l2_difference(base_cam, t.invert(cam)))

    curve = config.FIGURES_DIR / f"exp1_{model_name}_{transform}.png"
    plot_sensitivity(values, l2s, xlabel, curve,
                     title=f"{model_name} - {transform} sensitivity")

    robust = INVARIANT_FN[transform](gradcam, preprocess_fn, image, class_index=cls)
    save_overlay(image, robust,
                 config.FIGURES_DIR / f"exp1_{model_name}_{transform}_invariant.png",
                 title="Transformation-invariant Grad-CAM")
    print(f"Saved sensitivity curve and invariant overlay under {config.FIGURES_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet152", choices=list(config.MODEL_CONFIGS))
    ap.add_argument("--image", required=True, help="Path to a single input image")
    ap.add_argument("--transform", default="rotation", choices=list(TRANSFORM_SWEEPS))
    args = ap.parse_args()
    run(args.model, args.image, args.transform)


if __name__ == "__main__":
    main()
