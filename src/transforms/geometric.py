"""Geometric transforms and their inverses.

The invariant Grad-CAM framework needs both the forward transform (applied to the
input image) and its inverse (applied to the resulting heatmap to warp it back to
the canonical frame before averaging). Rotation and zoom are self-inverting under a
sign/reciprocal change; shift inverts by negating the offset.
"""
from dataclasses import dataclass

import cv2
import numpy as np


def _center(h, w):
    return (w / 2.0, h / 2.0)


def rotate(img, angle, inverse=False):
    """Rotate ``img`` by ``angle`` degrees about its centre."""
    h, w = img.shape[:2]
    a = -angle if inverse else angle
    M = cv2.getRotationMatrix2D(_center(h, w), a, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def zoom(img, factor, inverse=False):
    """Zoom ``img`` by ``factor`` (>1 zoom in, <1 zoom out) about its centre."""
    f = (1.0 / factor) if inverse else factor
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(_center(h, w), 0.0, f)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def shift(img, dx=0, dy=0, inverse=False):
    """Translate ``img`` by ``(dx, dy)`` pixels."""
    s = -1 if inverse else 1
    h, w = img.shape[:2]
    M = np.float32([[1, 0, s * dx], [0, 1, s * dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


@dataclass
class Transform:
    """A parameterized geometric transform with a forward and inverse application.

    kind : "rotation" | "zoom" | "shift_x" | "shift_y"
    value : angle in degrees, zoom factor, or pixel offset
    """
    kind: str
    value: float

    def apply(self, img):
        return self._dispatch(img, inverse=False)

    def invert(self, img):
        return self._dispatch(img, inverse=True)

    def _dispatch(self, img, inverse):
        if self.kind == "rotation":
            return rotate(img, self.value, inverse)
        if self.kind == "zoom":
            return zoom(img, self.value, inverse)
        if self.kind == "shift_x":
            return shift(img, dx=self.value, inverse=inverse)
        if self.kind == "shift_y":
            return shift(img, dy=self.value, inverse=inverse)
        raise ValueError(f"Unknown transform kind: {self.kind}")
