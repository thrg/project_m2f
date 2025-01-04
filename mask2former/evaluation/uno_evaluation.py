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
from sklearn.metrics import average_precision_score, roc_curve, auc
from detectron2.data import detection_utils as utils

class UNOEvaluator(DatasetEvaluator):
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
        self._s_m2f_uno = []

    def reset(self):
        self._label = []
        self._s_m2f_uno = []

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
            mask_cls_ = output['mask_cls']
            mask_pred = mask_pred.sigmoid()

            s_no = mask_cls_.softmax(-1)[..., -2]
            s_unc = -mask_cls_.softmax(-1)[..., :-2].max(1)[0]

            s_uno = s_no + s_unc

            s_m2f_uno = (mask_pred * s_uno.view(-1, 1, 1)).sum(0)
            s_m2f_uno = s_m2f_uno.to(self._cpu_device)

            label = utils.read_image(self.input_file_to_gt_file[input["file_name"]]).view(-1)

            self._s_m2f_uno += [s_m2f_uno.view(-1)[label != self._metadata.ignore_label]]
            self._label += [label[label != self._metadata.ignore_label]]

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        label = torch.cat(self._label, 0)
        pred = torch.cat(self._s_m2f_uno, 0)

        fpr, tpr, threshold = roc_curve(label, pred)
        fpr = np.array(fpr)
        tpr = np.array(tpr)

        AUROC = auc(fpr, tpr)
        FPR = fpr[tpr >= 0.95][0]

        AP = average_precision_score(label, pred)

        res = {}
        res["AP"] = 100 * AP
        res["AUROC"] = 100 * AUROC
        res["FPR@TPR95"] = 100 * FPR


        results = OrderedDict({"ood_detection": res})
        self._logger.info(results)
        return results