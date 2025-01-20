import os
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import random
from sort.sort import Sort

def save_results(track_bbs_ids_all, point_database, frame_names, sequence):
    for frame_idx in tqdm(range(len(frame_names))):
        frame_name = frame_names[frame_idx]
        result = -1 * np.ones((1080, 1920))

        if frame_idx in track_bbs_ids_all:
            for bbx in track_bbs_ids_all[frame_idx]:
                result[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])] = bbx[4]

        np.save(os.path.join(res_dir, sequence, frame_name.replace('_raw_data', '').replace('jpg', 'npy')), result)

        if sequence == "sequence_001":
            # plot the result
            prediction = colorize_labels(result.astype(np.int64))

            image = Image.open(os.path.join(video_dir, frame_name))

            image = np.asarray(image)
            # Convert RGB to BGR
            image = image[:, :, ::-1].copy()

            alpha = 0.5
            result = alpha * image + (1 - alpha) * prediction

            # if len(point_database) > 0 and (frame_idx, points) in point_database.items():
            #     for i in range(len(points)):
            #         point = points[i]
            #         cv2.drawMarker(result, (int(point[0]), int(point[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS,
            #                        markerSize=10, thickness=2)

            os.makedirs(os.path.join(result_video_dir, sequence), exist_ok=True)
            cv2.imwrite(os.path.join(result_video_dir, sequence, frame_name), result)


def colorize_labels(labels):
    # Create an empty RGB image
    colorized_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    unique_labels = np.unique(labels)

    colorized_image[labels == -1] = (0, 0, 0)
    for label in unique_labels:
        if label == -1: continue
        colorized_image[labels == label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return colorized_image


videos_dir = "./datasets/street_obstacle_sequences/raw_data_tmp"
res_dir = './datasets/street_obstacle_sequences/ood_prediction_tracked_n/'
result_video_dir = './datasets/street_obstacle_sequences/result_videos_n/'
threshold = 0.52

sequences = os.listdir(videos_dir)
# sequences = ["sequence_001"]
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

    predictor = Sort()

    point_database = {}

    track_bbs_ids_all = {}

    start_frame = len(frame_names) - len(frame_names) // 2 # TODO: Why start frame this?
    for frame_id in tqdm(range(len(frame_names))):
        image_path = os.path.join(video_dir, frame_names[frame_id])
        image = cv2.imread(image_path)

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
        bboxes = []
        for i in range(num_labels):
            points = centroids[i].astype(np.float32)
            stat = stats[i].astype(np.float32)
            bbox = [stat[0], stat[1], stat[0] + stat[2], stat[1] + stat[3], 1]
            all_points.append(points)
            bboxes.append(bbox)

        if len(bboxes) > 0:
            bboxes = np.array(bboxes).astype(np.int32)

            # print(bboxes)
            track_bbs_ids = predictor.update(bboxes)
            # print(track_bbs_ids)
            track_bbs_ids_all[frame_id] = track_bbs_ids

        if len(all_points) > 0:
            point_database[frame_id] = np.concatenate(all_points).astype(np.int32)

    os.makedirs(seq_res_dir, exist_ok=True)

    if len(point_database) == 0:
        print('no points found')

        save_results([], point_database, frame_names, sequence)
        continue

    save_results(track_bbs_ids_all, point_database, frame_names, sequence)


