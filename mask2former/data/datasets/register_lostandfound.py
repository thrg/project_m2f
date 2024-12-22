# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

LOSTANDFOUND_SEM_SEG_CATEGORIES = [
    {
        "color": [0, 0, 0],
        "instances": True,
        "readable": "Inliers",
        "name": "inliers",
        "evaluate": True,
    },
    {
        "color": [255, 255, 255],
        "instances": True,
        "readable": "Outlier",
        "name": "outlier",
        "evaluate": True,
    }
]


def _get_lostandfound_meta():
    stuff_classes = [k["readable"] for k in LOSTANDFOUND_SEM_SEG_CATEGORIES if k["evaluate"]]
    stuff_colors = [k["color"] for k in LOSTANDFOUND_SEM_SEG_CATEGORIES if k["evaluate"]]

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_all_lostandfound(root):
    root = os.path.join(root, "lostandfound")
    meta = _get_lostandfound_meta()

    image_dir = os.path.join(root, "leftimg8bit")
    gt_dir = os.path.join(root, "gtFine")
    name = "lostandfound"
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="labels.png", image_ext="leftImg8bit.png")
    )
    MetadataCatalog.get("lostandfound").set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="ood_detection",
        ignore_label=255,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
        **meta,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_lostandfound(_root)
