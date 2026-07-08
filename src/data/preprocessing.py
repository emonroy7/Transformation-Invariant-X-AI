"""Image loading and per-backbone canonicalization.

Note: the backbone-specific ``preprocess_input`` (mean subtraction / scaling) is
applied *after* the geometric transform, inside the Grad-CAM pipeline, so that the
transform operates on pixel-space images. These helpers only load and resize to the
canonical input frame.
"""
import cv2
import numpy as np


def load_image(path):
    """Load an image as RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_to(img, input_size):
    """Resize to a backbone input size given as ``(height, width)``."""
    h, w = input_size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def prepare(path, input_size):
    """Load and resize an image to a backbone's canonical input frame (uint8 RGB)."""
    return resize_to(load_image(path), input_size)
