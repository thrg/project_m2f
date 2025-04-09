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

        #print(point_database)
        #print((frame_idx, points))
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


sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2) #TODO: Mask generator

videos_dir = "../datasets/street_obstacle_sequences/raw_data_tmp"
res_dir = '../datasets/street_obstacle_sequences/ood_prediction_tracked_n_sam/'
result_video_dir = '../datasets/street_obstacle_sequences/result_videos_n_sam/'
threshold = 0.52

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

    start_frame = len(frame_names) - len(frame_names) // 2 # TODO: Why start frame this?
    for frame_id in tqdm(range(0, len(frame_names), len(frame_names) // 10)):
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
        closed_image[-1, :] = 1 # street starts from the bottom of the image

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
        masks_filtered, stats_filtered, centroids_filtered = [], [], []
        for i in range(num_labels):
            if i == 0:
                continue # background
            if stats[i, 4] < 500:
                continue
            mask = labels_im==i
            # If most of the anomaly off road
            if np.abs(np.sum(np.logical_and(mask, closed_image)) - np.sum(mask)) > 100:
                continue
            masks_filtered.append(mask)
            stats_filtered.append(stats[i])
            centroids_filtered.append(np.expand_dims(centroids[i], 0))
        num_labels, component_masks, stats, centroids = len(masks_filtered), masks_filtered, stats_filtered, centroids_filtered

        if num_labels == 0:
            continue

        # Adds those points to samv2 to track the object
        all_points = []
        ann_frame_idx = frame_id
        for i in range(num_labels):
            points = centroids[i].astype(np.float32)
            all_points.append(points)

            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1] * points.shape[0], np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=i,
                points=points,
                labels=labels,
            )
        if len(all_points) > 0:
            point_database[frame_id] = np.concatenate(all_points).astype(np.int32)

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
    tracking_database = list(tracking_database.values())

    refined_tracking_database = []

    tracked_ids = []
    histoy_mask = {}
    history_time = {}

    for t, frame_dict in enumerate(tracking_database):
        if tracked_ids == []:
            new_frame_dict = {}
            for i, (id, mask) in enumerate(frame_dict.items()):
                new_frame_dict[id] = mask
                histoy_mask[id] = mask
                history_time[id] = t
                tracked_ids.append(id)
            refined_tracking_database.append(new_frame_dict)
            continue

        new_frame_dict = {}
        for id, mask in frame_dict.items():

            c1 = measurements.center_of_mass(mask)

            closest = 99999
            closest_id = -1
            for idh, maskh in histoy_mask.items():
                c2 = measurements.center_of_mass(maskh)
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
                if dist < closest:
                    closest = dist
                    closest_id = idh

            if closest < 200:
                new_frame_dict[closest_id] = mask
                histoy_mask[closest_id] = mask
                history_time[closest_id] = t
            else:
                new_frame_dict[len(tracked_ids)] = mask
                tracked_ids.append(len(tracked_ids))
                histoy_mask[len(tracked_ids)] = mask
                history_time[len(tracked_ids)] = t

        refined_tracking_database.append(new_frame_dict)

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
