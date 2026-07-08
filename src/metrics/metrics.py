"""Quantitative metrics used in the thesis.

- L2 (Euclidean) difference: heatmap-to-heatmap consistency (lower = more stable).
- AUC-ROC: faithfulness of a heatmap against a binary ground-truth mask
  (higher = better localization of the true object/tumor region).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def l2_difference(h1, h2):
    """Euclidean distance between two heatmaps (flattened)."""
    h1 = np.asarray(h1, dtype="float64").ravel()
    h2 = np.asarray(h2, dtype="float64").ravel()
    return float(np.linalg.norm(h1 - h2))


def auc_roc(heatmap, mask):
    """AUC-ROC treating heatmap intensities as scores and ``mask`` as ground truth.

    Parameters
    ----------
    heatmap : np.ndarray
        Heatmap in [0, 1], shape (H, W).
    mask : np.ndarray
        Binary ground-truth mask, shape (H, W); non-zero = object/tumor.
    """
    scores = np.asarray(heatmap, dtype="float64").ravel()
    labels = (np.asarray(mask) > 0).astype(int).ravel()
    if labels.min() == labels.max():
        return float("nan")   # AUC undefined when only one class is present
    return float(roc_auc_score(labels, scores))


def roc_points(heatmap, mask):
    """Return ``(fpr, tpr, thresholds)`` for plotting an ROC curve."""
    scores = np.asarray(heatmap, dtype="float64").ravel()
    labels = (np.asarray(mask) > 0).astype(int).ravel()
    return roc_curve(labels, scores)
