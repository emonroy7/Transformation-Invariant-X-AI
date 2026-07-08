"""Visualization helpers: heatmap overlays and sensitivity curves."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay(image_rgb, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Superimpose a [0, 1] heatmap on an RGB uint8 image."""
    hm = np.uint8(255 * np.clip(heatmap, 0, 1))
    hm = cv2.applyColorMap(hm, colormap)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, hm, alpha, 0)


def save_overlay(image_rgb, heatmap, path, title=None):
    out = overlay(image_rgb, heatmap)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(out)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_sensitivity(x_values, l2_values, xlabel, path, title=None):
    """Plot L2 difference vs. transformation magnitude (thesis Figures 4.x)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3.5))
    plt.plot(x_values, l2_values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("L2 difference vs. canonical")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
