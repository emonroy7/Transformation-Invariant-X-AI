"""Dataset iterators for MS COCO 2017 and CAMELYON16 / PatchCamelyon.

Reconstructed stubs. Fill in the loading logic to match how the data is stored
locally (see ``data/README.md``).

- MS COCO 2017: used by Experiment 2 (class-specific) and Experiment 3
  (dataset-level, with instance-segmentation masks).
- CAMELYON16 / PatchCamelyon: used by Experiment 4 (histopathology), which needs
  pixel-level tumor masks for AUC-ROC.
"""
from pathlib import Path

from src import config


def iter_coco(split="val2017", limit=None):
    """Yield ``(image_path, class_label)`` for MS COCO images (Experiment 2).

    TODO: parse ``annotations/instances_{split}.json`` (pycocotools) and attach the
    dominant category per image. Masks are not required for the class-specific
    stability sweep.
    """
    root = Path(config.DATASET_PATHS["coco"]) / split
    paths = sorted(root.glob("*.jpg"))
    if limit:
        paths = paths[:limit]
    for p in paths:
        yield p, None                      # TODO: attach category label


def iter_coco_masks(split="val2017", limit=None):
    """Yield ``(image_path, class_index, mask)`` for MS COCO (Experiment 3).

    TODO: using pycocotools, for each image and each target class present, decode
    every instance (COCO stores them as polygons or RLE bitmaps) and take the
    pixel-wise union to build a binary ground-truth mask (1 = any instance of the
    class, 0 = background). ``mask`` is a 2-D uint8 array the size of the image;
    ``class_index`` is the class the backbone should be explained for.
    """
    root = Path(config.DATASET_PATHS["coco"]) / split
    paths = sorted(root.glob("*.jpg"))
    if limit:
        paths = paths[:limit]
    for p in paths:
        yield p, None, None                # TODO: (image_path, class_index, mask)


def iter_camelyon16(split="test", limit=None):
    """Yield ``(patch_path, mask_path, label)`` for CAMELYON16 / PCam patches (Experiment 4).

    PatchCamelyon tiles are extracted from 40x whole-slide images; each tile's label
    is defined by its central region (1 = tumor, 0 = normal), while the surrounding
    context is kept for the model. ``mask_path`` points to the pixel-level tumor mask
    used for AUC-ROC.
    TODO: return the patch, its tumor mask, and label from your local layout.
    """
    root = Path(config.DATASET_PATHS["camelyon16"])
    patch_dir = root / "patches"
    patches = sorted(patch_dir.glob("*.png")) if patch_dir.exists() else []
    if limit:
        patches = patches[:limit]
    for p in patches:
        mask = root / "masks" / p.name
        yield p, (mask if mask.exists() else None), None   # TODO: attach label
