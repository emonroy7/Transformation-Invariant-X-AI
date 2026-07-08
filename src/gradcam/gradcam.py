"""Core Grad-CAM implementation (Selvaraju et al., 2017).

Produces a class-discriminative heatmap by weighting the target-layer activations
with the global-average-pooled gradients of the class score w.r.t. those
activations.
"""
import cv2
import numpy as np
import tensorflow as tf

from src.models.backbones import build_grad_model


class GradCAM:
    """Generate Grad-CAM heatmaps for a Keras classification model."""

    def __init__(self, model: "tf.keras.Model", target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.grad_model = build_grad_model(model, target_layer)

    def heatmap(self, preprocessed_input, class_index=None, eps=1e-8):
        """Return ``(cam, class_index)`` with ``cam`` a normalized [0, 1] heatmap.

        Parameters
        ----------
        preprocessed_input : np.ndarray
            A single image already preprocessed for the backbone, shape (H, W, 3).
        class_index : int, optional
            Target class. If ``None``, the top predicted class is used.
        """
        x = tf.convert_to_tensor(preprocessed_input[None, ...], dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(x)
            if class_index is None:
                class_index = int(tf.argmax(preds[0]))
            loss = preds[:, class_index]

        grads = tape.gradient(loss, conv_out)                  # dY_c / dA
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))        # global average pool
        cam = tf.reduce_sum(weights * conv_out[0], axis=-1)    # weighted combination
        cam = tf.nn.relu(cam).numpy()

        # Upsample to the input resolution and normalize to [0, 1]
        cam = cv2.resize(cam, (preprocessed_input.shape[1], preprocessed_input.shape[0]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam, class_index
