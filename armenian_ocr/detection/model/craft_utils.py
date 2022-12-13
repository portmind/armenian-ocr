import math
from collections import deque, OrderedDict
from typing import List

import cv2
import numpy as np
from numpy import ndarray
from scipy.signal import find_peaks

cv2.setNumThreads(0)


def copy_state_dict(state_dict):
    """
    Copy the state_dict - the weights of each module
    Args:
        state_dict: Model state_dict

    Returns:
        Copy of  a state dict as an OrderedDict()
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def find_cuts(arr: np.ndarray, div_coef_max: float = 1.5, div_coef_min: float = 1.2) -> np.ndarray:
    hist = np.sum(arr, axis=1).astype(np.int)
    mask = np.zeros_like(hist)
    maxes = find_peaks(hist)[0]
    mines = find_peaks(-hist)[0]
    peaks = [0] + sorted(np.concatenate([mines, maxes])) + [len(hist) - 1]
    if len(peaks) < 4:
        return mask.astype("bool")

    for i in range(1, len(peaks) - 1):
        idx = peaks[i]
        val_max = hist[idx] * div_coef_max
        val_min = hist[idx] * div_coef_min

        if (val_max < hist[peaks[i - 1]] + 2) | (val_max < hist[peaks[i + 1]] + 2):
            if (val_min < hist[peaks[i - 1]] + 2) & (val_min < hist[peaks[i + 1]] + 2):
                mask[idx] = 1

    return mask.astype("bool")


def get_breakpoints(zero_counts: np.ndarray, height_thresh: int = 3) -> list:
    """
    Get peaks of zero counts and return them as breakpoints
    Args:
        zero_counts: Counts of zeros
        height_thresh: Difference threshold

    Returns:
        Indices of peaks
    """
    peaks, _ = find_peaks(zero_counts)
    neg_zero_count = -zero_counts
    peaks_neg, _ = find_peaks(neg_zero_count)
    real_peaks = []
    if len(peaks) > 0:
        if len(peaks_neg) == 0:
            real_peaks = list(peaks)
        else:
            if len(peaks) + 1 != len(peaks_neg):
                peaks_neg = np.concatenate((peaks_neg, peaks_neg[-1:]))
            for i in range(len(peaks)):
                if zero_counts[peaks][i] > zero_counts[peaks_neg][i] + height_thresh:
                    real_peaks.append(peaks[i])

    return real_peaks


def get_labels(text_scores: np.ndarray, offset: int = 1) -> tuple:
    """
    Run connected component analysis with additional pre-processing
    Args:
        text_scores: Predicted text scores
        offset: Offset for boundaries

    Returns:
        Number of labels, labels, stats, centroids
    """
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(text_scores, connectivity=4)
    # refinement
    text_scores_refined = text_scores.copy()
    for i in range(1, n_labels + 1):
        label_image = np.where(labels == i, 255, 0)

        x_where = np.where((label_image != 0).sum(axis=0) > 0)[0]
        y_where = np.where((label_image != 0).sum(axis=1) > 0)[0]

        if (len(x_where) > 1) and (len(y_where) > 1):
            label_bounded = label_image[y_where.min() : y_where.max(), x_where.min() : x_where.max()]
        else:
            continue

        zero_counts = (label_bounded == 0).sum(axis=1)

        if len(zero_counts) < 5:
            continue

        to_be_zeroed = get_breakpoints(zero_counts)

        x, y = np.where(label_bounded > 0)
        for x_, y_ in zip(x, y):
            if x_ in to_be_zeroed:
                if (abs(label_bounded.shape[0] - x_) <= offset) or (x_ <= offset):
                    # discard on boundaries
                    continue
                if label_bounded[max(x_ - offset, 0) : x_ + offset + 1, y_].mean() < 255:
                    # check if it has values both upper and below
                    continue
                text_scores_refined[y_where.min() + x_, x_where.min() + y_] = 0
    return cv2.connectedComponentsWithStats(text_scores_refined, connectivity=4)


def get_detection_boxes(
    text_map: np.ndarray,
    link_map: np.ndarray,
    link_threshold: float = 0.4,
    low_text: float = 0.3,
    link_threshold2: float = 0.6,
    low_text2: float = 0.6,
    improve_steps: int = 0,
) -> list:
    """
    Create text boxes from predicted text_map and link_map
    Args:
        text_map: Text map from CRAFT prediction
        link_map: Link map from CRAFT prediction
        link_threshold: Soft threshold for links
        low_text: Soft threshold for text
        low_text2: Hard threshold for text
        link_threshold2: Hard threshold for links
        improve_steps: How many of the new changes apply (temporary should be removed)

    Returns:
        object:
        Found text boxes, labels of soft connected components, map of label_ids
        Only found text boxes are used in future
    """
    # prepare data
    link_map = link_map.copy()
    text_map = text_map.copy()

    image_height, image_width = text_map.shape

    # threshold by soft thresholds
    _, text_score = cv2.threshold(text_map, low_text, 1, 0)
    _, link_score = cv2.threshold(link_map, link_threshold, 1, 0)

    # add the maps and run connected components
    text_score_combined = np.clip(text_score + link_score, 0, 1).astype(np.uint8)
    if improve_steps > 0:
        n_labels, labels, stats, _ = get_labels(text_score_combined)
    else:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_score_combined, connectivity=4)
    labels = labels.astype(np.uint64)

    # threshold by hard thresholds
    _, text_score_hard = cv2.threshold(text_map, low_text2, 1, 0)
    _, link_score_hard = cv2.threshold(link_map, link_threshold2, 1, 0)

    # add the maps and run connected components
    text_score_hard_combined = np.clip(text_score_hard + link_score_hard, 0, 1).astype(np.uint8)
    if improve_steps > 0:
        _, labels_hard, stats_hard, _ = get_labels(text_score_hard_combined)
    else:
        _, labels_hard, stats_hard, _ = cv2.connectedComponentsWithStats(text_score_hard_combined, connectivity=4)

    numer = np.arange(0, len(stats_hard)).reshape((len(stats_hard), 1))
    stats_hard = np.concatenate((stats_hard, numer), axis=1)
    stats_hard = stats_hard[1:]

    detections, mapper = [], []
    labels_deq = deque(list(range(1, n_labels)))
    stats_deq = deque(stats[1:].tolist())
    id_multiplier = 1000
    long_box_coef = 15
    max_id = n_labels + 1

    while labels_deq:
        # size filtering
        label_id = labels_deq.popleft()
        old_stats = stats_deq.popleft()
        x, y, width, height, size = old_stats

        if size < 2:
            continue

        if (size > 30) and (label_id < n_labels) and (improve_steps >= 0):
            segmentation_map = np.zeros(text_map.shape, dtype=np.uint8)
            segmentation_map[(labels == label_id) & (text_score == 1)] = 1
            cuts = find_cuts(segmentation_map[y : y + height, x : x + width])
            segmentation_map[y : y + height, x : x + width][cuts] = 0
            n_labels_current, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(
                segmentation_map[y : y + height, x : x + width], connectivity=4
            )
            if n_labels_current > 2:
                sub_stats[:, 0] += x
                sub_stats[:, 1] += y
                for i in range(1, n_labels_current):
                    new_label_id = max_id
                    max_id += 1
                    labels[y : y + height, x : x + width][sub_labels == i] = new_label_id
                    labels_deq.append(new_label_id)
                    stats_deq.append(sub_stats[i].tolist())
                continue

        if (label_id < id_multiplier) and ((width / height) > long_box_coef):
            # is small box's top left smaller then targets
            top_left_in = (stats_hard[:, 0] >= x) & (stats_hard[:, 1] >= y)
            # is small box's bot right bigger then targets
            bottom_right_in = (stats_hard[:, 0] + stats_hard[:, 2] <= x + width) & (
                stats_hard[:, 1] + stats_hard[:, 3] <= y + height
            )
            small_boxes = stats_hard[top_left_in & bottom_right_in]
            for box in small_boxes:
                label_i = box[5]
                new_label_id = label_id * id_multiplier + label_i
                labels[labels_hard == label_i] = new_label_id
                labels_deq.append(new_label_id)
                t_box = [box[0], y, box[2], height, box[4]]
                stats_deq.append(t_box)
            if len(small_boxes) > 0:
                continue

        # make segmentation map
        segmentation_map = np.zeros(text_map.shape, dtype=np.uint8)
        segmentation_map[labels == label_id] = 255
        # segmentation_map[np.logical_and(link_score == 1, text_score == 0)] = 0   # remove link area

        if height > 20:
            num_iter_coef = 1.1
            num_iter_offset = 0
        elif height > 5:
            num_iter_coef = 1
            num_iter_offset = 1
        else:
            num_iter_coef = 1
            num_iter_offset = 2

        num_iter = int(math.sqrt(size * min(width, height) / (width * height)) * num_iter_coef)
        x_left, x_right, y_top, y_bottom = (
            x - num_iter,
            x + width + num_iter + 1,
            y - num_iter,
            y + height + num_iter + num_iter_offset,
        )
        # boundary check
        if x_left < 0:
            x_left = 0
        if y_top < 0:
            y_top = 0
        if x_right >= image_width:
            x_right = image_width
        if y_bottom >= image_height:
            y_bottom = image_height

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + num_iter, 1 + num_iter + num_iter_offset))
        segmentation_map[y_top:y_bottom, x_left:x_right] = cv2.dilate(
            segmentation_map[y_top:y_bottom, x_left:x_right], kernel
        )

        # make box
        np_contours = np.roll(np.array(np.where(segmentation_map != 0)), 1, axis=0).transpose().reshape(-1, 2)
        # rectangle = cv2.minAreaRect(np_contours)
        # box = cv2.boxPoints(rectangle)

        # align diamond-shape
        try:
            left, right = min(np_contours[:, 0]), max(np_contours[:, 0])
            top, bottom = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
        except:
            print("dumped", label_id)
            # print(Counter(labels.flatten()))
            print("__", (labels == label_id).sum())
            print("sum0", (segmentation_map != 0).sum())
            print("sum", np.array(np.where(segmentation_map != 0)).sum())
            print("stats", old_stats)
            print("s", x_left, y_top, x_right, y_bottom)
            continue

        # make clock-wise order
        start_index = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_index, 0)
        detections.append(box.tolist())
        mapper.append(label_id)

    return detections  # , labels, mapper


def adjust_result_coordinates(
    polygons: list, ratio_width: float, ratio_height: float, ratio_net: int = 2
) -> np.ndarray:
    """
    During inference, original image size is changed by (ratio_w, ratio_h) times.
    As predicted boxes are for changed image, this method will map the model's predicted boxes to
    original image  scale

    Args:
        polygons: Polygons to adjust
        ratio_width: Ratio by width
        ratio_height: Ratio by height
        ratio_net: Network outputs ratio_net times smaller maps

    Returns:
        Adjusted polygons
    """
    if len(polygons) > 0:
        polygons = np.array(polygons, dtype=object)
        for index in range(len(polygons)):
            if polygons[index] is not None:
                polygons[index] = polygons[index] * (ratio_width * ratio_net, ratio_height * ratio_net)
                polygons[index] = polygons[index].astype(int)
    return polygons


def remove_padding(box: tuple, upper_padding: int, left_padding: int) -> tuple:
    """
    Remove padding created by centered padding
    Args:
        box: Predicted box
        upper_padding: Padding from above
        left_padding: Padding from left

    Returns:
        Corrected box
    """
    return box[0] - left_padding, box[1] - upper_padding, box[2] - left_padding, box[3] - upper_padding
