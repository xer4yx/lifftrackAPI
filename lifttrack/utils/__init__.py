import cv2
import numpy as np

KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255, 0, 255), (0, 2): (0, 255, 255), (1, 3): (255, 0, 255),
    (2, 4): (0, 255, 255), (0, 5): (255, 0, 255), (0, 6): (0, 255, 255),
    (5, 7): (255, 0, 255), (7, 9): (255, 0, 255), (6, 8): (0, 255, 255),
    (8, 10): (0, 255, 255), (5, 6): (255, 255, 0), (5, 11): (255, 0, 255),
    (6, 12): (0, 255, 255), (11, 12): (255, 255, 0), (11, 13): (255, 0, 255),
    (13, 15): (255, 0, 255), (12, 14): (0, 255, 255), (14, 16): (0, 255, 255)
}


def _keypoints_and_edges_for_display(keypoints_with_scores, height, width, keypoint_threshold=0.11):
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1).astype(int)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start, y_start = kpts_absolute_xy[edge_pair[0]]
                x_end, y_end = kpts_absolute_xy[edge_pair[1]]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)

    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2), dtype=int)

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2), dtype=int)
    return keypoints_xy, edges_xy, edge_colors


def draw_prediction(image, keypoints_with_scores, crop_region=None, output_image_height=None):
    height, width, _ = image.shape
    keypoints_xy, edges_xy, edge_colors = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    # Draw edges
    for edge, color in zip(edges_xy, edge_colors):
        cv2.line(image, tuple(edge[0]), tuple(edge[1]), color, 2)

    # Draw keypoints
    for keypoint in keypoints_xy:
        cv2.circle(image, tuple(keypoint), 3, (255, 0, 255), -1)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        cv2.rectangle(image, (int(xmin), int(ymin)),
                      (int(xmin + rec_width), int(ymin + rec_height)),
                      (0, 255, 0), 2)

    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image = cv2.resize(image, (output_image_width, output_image_height),
                           interpolation=cv2.INTER_CUBIC)

    return image
