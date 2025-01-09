import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from PIL import Image

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from detectron2.evaluation import DatasetEvaluator
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score
from detectron2.data import detection_utils as utils

class EAMEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }
        self._label = []
        self._s_eam = []

    def reset(self):
        self._label = []
        self._s_eam = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            mask_pred = output['mask_pred']
            mask_cls = output['mask_cls']
            mask_pred = mask_pred.sigmoid()

            p = mask_cls.softmax(-1)[..., :-1]
            s_ahm = -p.max(1)[0]
            s_eam = (mask_pred * s_ahm.view(-1, 1, 1)).sum(0)
            s_eam = s_eam.to(self._cpu_device)

            label = utils.read_image(self.input_file_to_gt_file[input["file_name"]]).view(-1)

            self._s_eam += [s_eam.view(-1)[label != self._metadata.ignore_label]]
            self._label += [label[label != self._metadata.ignore_label]]

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        label = torch.cat(self._label, 0)
        pred = torch.cat(self._s_eam, 0)

        AP = 100 * average_precision_score(label, pred)
        AUROC = 100 * roc_auc_score(label, pred)

        fpr, tpr, _ = roc_curve(label, pred)
        fpr = np.array(fpr)
        tpr = np.array(tpr)
        FPR = 100 * fpr[tpr >= 0.95][0]

        results = OrderedDict({"ood_detection": {"AP": AP, "AUROC": AUROC, "FPR": FPR}})
        self._logger.info(results)
        return results