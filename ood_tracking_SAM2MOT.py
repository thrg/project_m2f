import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor
import pickle
from collections import Counter, defaultdict
import random
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from scipy.ndimage import measurements
from munkres import Munkres


class TrajectoryManager:
    def __init__(self,
                 image_shape=(1080, 1920),
                 untracked_threshold: float = 0.7,
                 lost_tolerance: int = 5,
                 tau_r: float = 0.8,
                 tau_p: float = 0.6,
                 tau_s: float = 0.4):

        self.untracked_threshold = untracked_threshold
        self.lost_tolerance = lost_tolerance
        self.tau_r, self.tau_p, self.tau_s = tau_r, tau_p, tau_s

        self.image_shape = image_shape

        self.tracked_objects = {}
        self.next_object_id = 0

    def add_object(self, box, mask, logits):
        obj_id = self.next_object_id
        self.tracked_objects[obj_id] = {
            'box': box,
            'mask': mask,
            'logits': logits,
            'state': self._get_state(logits),
            'lost_count': 0
        }
        self.next_object_id += 1
        return obj_id

    def remove_object(self, obj_id):
        if obj_id in self.tracked_objects:
            del self.tracked_objects[obj_id]

    def update_object(self, obj_id, box, mask, logits):
        if obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['box'] = box
            self.tracked_objects[obj_id]['mask'] = mask
            self.tracked_objects[obj_id]['logits'] = logits
            self.tracked_objects[obj_id]['state'] = self._get_state(logits)

            if self._get_state(logits) == "reliable":
                self.tracked_objects[obj_id]['lost_count'] = 0

            if self.tracked_objects[obj_id]['state'] == "lost":
                self.tracked_objects[obj_id]['lost_count'] += 1

    def _get_state(self, logits):
        if logits > self.tau_r:
            return "reliable"
        elif logits > self.tau_p:
            return "pending"
        elif logits > self.tau_s:
            return "suspicious"
        else:
            return "lost"

    def _build_iou_matrix(self, detections):
        iou_matrix = []
        for det in detections:
            iou_matrix.append([])
            for obj_id, obj in self.tracked_objects.items():
                iou = self._calculate_iou(det, obj['box'])
                iou_matrix[-1].append(1 - iou)

        iou_matrix = self._pad_matrix_to_square(iou_matrix)
        return iou_matrix

    @staticmethod
    def _pad_matrix_to_square(matrix, pad_value=1):
        rows = len(matrix)
        cols = len(matrix[0])

        if rows == cols:
            return matrix

        size = max(rows, cols)

        while len(matrix) < size:
            matrix.append([pad_value] * cols)

        for row in matrix:
            while len(row) < size:
                row.append(pad_value)

        return matrix

    @staticmethod
    def _get_unfit_from_indexes(indexes, iou_matrix, num_existing, type):
        lost = []

        for index in indexes:
            if index[type] < num_existing and iou_matrix[index[0]][index[1]] == 1:
                lost.append(index[type])
        return lost

    def _hungarian_matching(self, detections):
        if not self.tracked_objects:
            return [], [], detections

        iou_matrix = self._build_iou_matrix(detections)
        m = Munkres()
        indexes = m.compute(iou_matrix)

        lost_indexes = self._get_unfit_from_indexes(indexes, iou_matrix, len(self.tracked_objects), 1)
        new_indexes = self._get_unfit_from_indexes(indexes, iou_matrix, len(detections), 0)

        matched_pairs = []
        for index in indexes:
            if index[1] not in lost_indexes and index[0] not in new_indexes:
                matched_pairs.append((detections[index[0]], list(self.tracked_objects.keys())[index[1]]))

        lost = [list(self.tracked_objects.keys())[i] for i in lost_indexes]
        new = [detections[i] for i in new_indexes]

        return matched_pairs, lost, new

    def _calculate_iou(self, box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        inter_area = (x_right - x_left) * (y_bottom - y_top)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def _get_untracked_region_mask(self):
        mask_none = np.ones(self.image_shape, dtype=np.uint8)

        for obj in self.tracked_objects.values():
            obj_mask = obj['mask']
            obj_mask = (obj_mask > 0).cpu().numpy()
            mask_none &= ~obj_mask

        return mask_none

    def _filter_new_objects(self, detections, untracked_mask):
        filtered_detections = []

        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]

            box_mask = untracked_mask[y1:y2, x1:x2]

            box_area = (x2 - x1) * (y2 - y1)
            overlap_area = np.sum(box_mask)
            overlap_ratio = overlap_area / box_area

            if overlap_ratio >= self.untracked_threshold:
                filtered_detections.append(det)

        return filtered_detections

    def _reconstruct_prompt(self, obj_id, detection, frame_id):
        print(f"reconstructing object: {obj_id}")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_id,
            obj_id=obj_id,
            box=detection,
        )

        self.update_object(
            obj_id=obj_id,
            box=detection,
            mask=(out_mask_logits[obj_id][0] > 0.0),
            logits=out_mask_logits[obj_id][0][out_mask_logits[obj_id][0] > 0.0].mean(dim=0)
        )

    def process_frame(self, detections, frame_id):
        matched_pairs, tracked_lost, new_detected = self._hungarian_matching(detections)

        if self.tracked_objects:
            prediction = predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=frame_id,
                max_frame_num_to_track=0
            )

            # print(prediction)
            for _, out_obj_id, out_mask_logit in prediction:
                _, out_obj_ids, out_mask_logits = _, out_obj_id, out_mask_logit

            # print(out_mask_logits.shape)
            # print(out_mask_logits)

        for det, obj_id in matched_pairs:
            # print(obj_id)
            self.update_object(
                obj_id=obj_id,
                box=det,
                mask=(out_mask_logits[obj_id][0] > 0.0),
                logits=out_mask_logits[obj_id][0][out_mask_logits[obj_id][0] > 0.0].mean(dim=0)
            )

            if self.tracked_objects[obj_id]['state'] == "pending":
                self._reconstruct_prompt(obj_id, det, frame_id)

        for obj_id in tracked_lost:
            prev_box = self.tracked_objects[obj_id]['box']
            # print(obj_id)

            self.update_object(
                obj_id=obj_id,
                box=prev_box,
                mask=(out_mask_logits[obj_id][0] > 0.0),
                logits=out_mask_logits[obj_id][0][out_mask_logits[obj_id][0] > 0.0].mean(dim=0)
            )

            if self.tracked_objects[obj_id]['lost_count'] > self.lost_tolerance:
                self.remove_object(obj_id)

        if new_detected:
            untracked_mask = self._get_untracked_region_mask()

            new_object_candidates = self._filter_new_objects(new_detected, untracked_mask)

            for det in new_object_candidates:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_id,
                    obj_id=self.next_object_id,
                    box=det,
                )
                # print(out_mask_logits.shape)

                self.add_object(
                    box=det,
                    mask=(out_mask_logits[self.next_object_id][0] > 0.0),
                    logits=out_mask_logits[self.next_object_id][0][out_mask_logits[self.next_object_id][0] > 0.0].mean(
                        dim=0)
                )

        return self.tracked_objects


def propagate_in_video(start_tracking_database, inference_state, start_frame, reverse=False):
    tracking_database = start_tracking_database
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                    start_frame_idx=start_frame,
                                                                                    reverse=reverse):
        if len(out_obj_ids) == 0:
            continue
        segment_masks = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            segment_mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
            if not segment_mask.any():
                continue

            cnt = np.sum(segment_mask)
            if cnt > 30000:
                # print('removed big', cnt)
                continue

            segment_masks[out_obj_id] = segment_mask

        tracking_database[out_frame_idx] = segment_masks
    return tracking_database


def save_results(refined_tracking_database, point_database, frame_names, sequence, good_ids):
    for frame_idx in tqdm(range(len(frame_names))):
        frame_name = frame_names[frame_idx]
        result = -1 * np.ones((1080, 1920))

        if refined_tracking_database:
            for id, segment_mask in refined_tracking_database[frame_idx].items():
                if id not in good_ids:
                    continue
                result[segment_mask] = id

        np.save(os.path.join(res_dir, sequence, frame_name.replace('_raw_data', '').replace('jpg', 'npy')), result)

        # plot the result
        prediction = colorize_labels(result.astype(np.int64))

        image = Image.open(os.path.join(video_dir, frame_name))

        image = np.asarray(image)
        # Convert RGB to BGR
        image = image[:, :, ::-1].copy()

        alpha = 0.5
        result = alpha * image + (1 - alpha) * prediction

        # print(point_database)
        # print((frame_idx, points))
        if frame_idx in point_database:
            points = point_database[frame_idx]
            for i in range(len(points)):
                point = points[i]
                cv2.drawMarker(result, (int(point[0]), int(point[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                               markerSize=10, thickness=2)

        os.makedirs(os.path.join(result_video_dir, sequence), exist_ok=True)
        cv2.imwrite(os.path.join(result_video_dir, sequence, frame_name), result)


def colorize_labels(labels):
    # Create an empty RGB image
    colorized_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    unique_labels = np.unique(labels)
    # print(unique_labels)

    colorized_image[labels == -1] = (0, 0, 0)
    for label in unique_labels:
        if label == -1: continue
        colorized_image[labels == label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return colorized_image


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

sam2_checkpoint = "./checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

videos_dir = "../datasets/street_obstacle_sequences/raw_data_tmp"
res_dir = '../datasets/street_obstacle_sequences/ood_prediction_tracked_n_sam/'
result_video_dir = '../datasets/street_obstacle_sequences/result_videos_n_sam/'
threshold = 0.70

sequences = os.listdir(videos_dir)
sequences.sort()

for sequence in sequences:
    print(sequence)
    video_dir = os.path.join(videos_dir, sequence)
    seq_res_dir = os.path.join(res_dir, sequence)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    point_database = {}

    trajectory_manager = TrajectoryManager()

    start_frame = len(frame_names) - len(frame_names) // 2
    for frame_id in tqdm(range(len(frame_names))):
        image_path = os.path.join(video_dir, frame_names[frame_id])
        image = cv2.imread(image_path)
        masks = mask_generator.generate(image)

        score_path = image_path.replace('raw_data_tmp', 'ood_score').replace('_ood_score', '').replace('jpg', 'npy')
        ood_score = np.load(score_path)
        binary_mask = (ood_score >= threshold).astype(np.uint8)

        road_path = image_path.replace('raw_data_tmp', 'street_masks').replace('.jpg', '_street_masks.npy')
        road = np.load(road_path)

        kernel = np.ones((50, 50), np.uint8)  # Adjust the kernel size as needed

        # Apply morphological closing to fill holes
        closed_image = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # fill in big objects
        closed_image[-1, :] = 1  # street starts from the bottom of the image

        # Removes background, remains 0 if it is surrounded by other classes and calls that the street
        image_copy = closed_image.copy().astype(np.uint8)
        h, w = closed_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for i in range(h):
            if image_copy[i, 0] == 0:
                cv2.floodFill(image_copy, mask, (0, i), 2)
            if image_copy[i, w - 1] == 0:
                cv2.floodFill(image_copy, mask, (w - 1, i), 2)
        for j in range(w):
            if image_copy[0, j] == 0:
                cv2.floodFill(image_copy, mask, (j, 0), 2)
            if image_copy[h - 1, j] == 0:
                cv2.floodFill(image_copy, mask, (j, h - 1), 2)
        closed_image[image_copy == 0] = 1

        # Returns centers and masks of all anomalies seperatly
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        masks_filtered, stats_filtered, centroids_filtered, boxes_filtered = [], [], [], []
        for i in range(num_labels):
            if i == 0:
                continue  # background
            if stats[i, 4] < 500:
                continue
            mask = labels_im == i
            # If most of the anomaly off road
            if np.abs(np.sum(np.logical_and(mask, closed_image)) - np.sum(mask)) > 100:
                continue
            masks_filtered.append(mask)
            stats_filtered.append(stats[i])
            centroids_filtered.append(np.expand_dims(centroids[i], 0))

            mask_locations = np.where(labels_im == i)
            x_min = int(np.min(mask_locations[1]))
            x_max = int(np.max(mask_locations[1]))
            y_min = int(np.min(mask_locations[0]))
            y_max = int(np.max(mask_locations[0]))

            box = [x_min, y_min, x_max, y_max]
            boxes_filtered.append(box)
            # print(box)
            # print(centroids[i])

        num_labels, component_masks, stats, centroids, boxes = (
            len(masks_filtered), masks_filtered, stats_filtered, centroids_filtered, boxes_filtered)

        if num_labels == 0:
            continue

        all_points = []
        for i in range(num_labels):
            points = centroids[i].astype(np.float32)
            all_points.append(points)

        if len(all_points) > 0:
            point_database[frame_id] = np.concatenate(all_points).astype(np.int32)

        trajectory_manager.process_frame(boxes, frame_id)

    os.makedirs(seq_res_dir, exist_ok=True)

    if len(point_database) == 0:
        print('no points found')

        save_results({}, point_database, frame_names, sequence, [])
        continue

    tracked_ids = []
    tracking_database = propagate_in_video({}, inference_state, start_frame)

    if 0 not in tracking_database:
        tracking_database = propagate_in_video(tracking_database, inference_state, start_frame, True)

    tracking_database = dict(sorted(tracking_database.items()))
    refined_tracking_database = list(tracking_database.values())

    good_ids = set()
    consecutive_occurrences = defaultdict(int)

    for frame_dict in refined_tracking_database:

        for object_id in frame_dict.keys():
            consecutive_occurrences[object_id] += 1
            if consecutive_occurrences[object_id] >= 5:
                good_ids.add(object_id)

        # Reset counts for object IDs not in the current frame
        for object_id in list(consecutive_occurrences.keys()):
            if object_id not in frame_dict:
                consecutive_occurrences[object_id] = 0

    save_results(refined_tracking_database, point_database, frame_names, sequence, good_ids)
