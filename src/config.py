"""Central configuration for the transformation-invariance Grad-CAM experiments.

Reconstructed from the thesis methodology (Chapter 4). Adjust paths and sweep
ranges to match your environment and the exact protocol you used.
"""
from pathlib import Path
import numpy as np

# ---- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

DATASET_PATHS = {
    "coco": RAW_DIR / "coco",
    "camelyon16": RAW_DIR / "camelyon16",
}

# ---- Model configurations ---------------------------------------------------
# Each backbone: the Keras application, its Grad-CAM target layer (the last conv
# layer before global pooling), the input size, and parameter count (Table 4.1).
MODEL_CONFIGS = {
    "resnet152": {
        "keras_app": "ResNet152",
        "module": "tensorflow.keras.applications.resnet",
        "target_layer": "conv5_block3_out",
        "input_size": (224, 224),
        "params_millions": 60.2,
    },
    "densenet201": {
        "keras_app": "DenseNet201",
        "module": "tensorflow.keras.applications.densenet",
        "target_layer": "conv5_block32_concat",
        "input_size": (224, 224),
        "params_millions": 20.2,
    },
    "xception": {
        "keras_app": "Xception",
        "module": "tensorflow.keras.applications.xception",
        "target_layer": "block14_sepconv2_act",
        "input_size": (299, 299),
        "params_millions": 22.9,
    },
}

# ---- Transformation sweeps (from the thesis) --------------------------------
# Rotation: -180 to +180 degrees
ROTATION_ANGLES = np.arange(-180, 181, 15)              # degrees
# Zoom: 0.67x (zoom out) to 1.5x (zoom in)
ZOOM_FACTORS = np.round(np.linspace(0.67, 1.5, 13), 3)
# Horizontal / vertical shift: up to +/- 180 pixels
SHIFT_PIXELS = np.arange(-180, 181, 20)                 # pixels

# Canonical (identity) reference used when realigning and averaging
IDENTITY_ANGLE = 0.0
IDENTITY_ZOOM = 1.0
IDENTITY_SHIFT = 0
