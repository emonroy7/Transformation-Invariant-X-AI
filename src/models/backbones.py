"""Load the CNN backbones used in the thesis and expose their Grad-CAM target layers.

Models: ResNet152, DenseNet201, Xception (ImageNet-pretrained, used without
fine-tuning). Grad-CAM is computed at the last convolutional layer before global
pooling in each architecture.
"""
from importlib import import_module

import tensorflow as tf

from src.config import MODEL_CONFIGS


def load_backbone(name: str):
    """Return ``(model, preprocess_fn, target_layer_name, input_size)``.

    Parameters
    ----------
    name : str
        One of the keys in ``MODEL_CONFIGS`` (e.g. ``"resnet152"``).
    """
    name = name.lower()
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_CONFIGS)}")

    cfg = MODEL_CONFIGS[name]
    module = import_module(cfg["module"])
    model_cls = getattr(module, cfg["keras_app"])
    preprocess_fn = getattr(module, "preprocess_input")

    model = model_cls(weights="imagenet", include_top=True)
    model.trainable = False
    return model, preprocess_fn, cfg["target_layer"], cfg["input_size"]


def build_grad_model(model: "tf.keras.Model", target_layer: str) -> "tf.keras.Model":
    """A model mapping the input to (target-layer activations, class predictions)."""
    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer).output, model.output],
    )
