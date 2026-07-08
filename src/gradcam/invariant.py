"""Transformation-invariant Grad-CAM: the realignment-and-averaging framework.

For each magnitude in a transform sweep:
    1. apply the forward transform to the input image,
    2. generate a Grad-CAM heatmap on the transformed image,
    3. warp the heatmap back to the canonical frame with the inverse transform,
    4. accumulate.

Averaging the realigned heatmaps yields a robust, transformation-invariant
explanation. Thin wrappers specialize the sweep for rotation, zoom, and
horizontal/vertical shift.
"""
from src import config
from src.transforms.geometric import Transform


def invariant_heatmap(gradcam, preprocess_fn, image, kind, values, class_index=None):
    """Average realigned Grad-CAM heatmaps over a transform sweep.

    Parameters
    ----------
    gradcam : GradCAM
    preprocess_fn : callable
        Backbone ``preprocess_input``, applied after the geometric transform.
    image : np.ndarray
        Raw RGB image resized to the backbone input size, shape (H, W, 3), uint8.
    kind : str
        "rotation" | "zoom" | "shift_x" | "shift_y".
    values : iterable of float
        Magnitudes to sweep (angles, zoom factors, or pixel shifts).
    class_index : int, optional
        Fix the target class across the sweep for a fair comparison.
    """
    accum = None
    n = 0
    for v in values:
        t = Transform(kind=kind, value=float(v))
        transformed = t.apply(image)
        pre = preprocess_fn(transformed.astype("float32").copy())
        cam, class_index = gradcam.heatmap(pre, class_index=class_index)
        cam_aligned = t.invert(cam)          # warp heatmap back to canonical frame
        accum = cam_aligned if accum is None else accum + cam_aligned
        n += 1
    return accum / max(n, 1)


def rotation_invariant(gradcam, preprocess_fn, image, class_index=None):
    return invariant_heatmap(gradcam, preprocess_fn, image, "rotation",
                             config.ROTATION_ANGLES, class_index)


def zoom_invariant(gradcam, preprocess_fn, image, class_index=None):
    return invariant_heatmap(gradcam, preprocess_fn, image, "zoom",
                             config.ZOOM_FACTORS, class_index)


def shift_x_invariant(gradcam, preprocess_fn, image, class_index=None):
    return invariant_heatmap(gradcam, preprocess_fn, image, "shift_x",
                             config.SHIFT_PIXELS, class_index)


def shift_y_invariant(gradcam, preprocess_fn, image, class_index=None):
    return invariant_heatmap(gradcam, preprocess_fn, image, "shift_y",
                             config.SHIFT_PIXELS, class_index)
