import math
from collections import OrderedDict, deque
from typing import List, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks

cv2.setNumThreads(0)


box_type = Tuple[int, int, int, int]


def copy_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """Copy the state_dict - the weights of each module

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


def find_cuts(
    segmentation_map: np.ndarray,
    div_coef_max: float = 1.5,
    div_coef_min: float = 1.2,
) -> np.ndarray:
    """Find horizontal cuts in segmentation map"""
    non_zero_counts = np.sum(a=segmentation_map, axis=1).astype(int)
    mask = np.zeros_like(non_zero_counts)
    max_indices = find_peaks(non_zero_counts)[0]
    min_indices = find_peaks(-non_zero_counts)[0]
    peak_indices = (
        [0]
        + sorted(np.concatenate([min_indices, max_indices]))
        + [len(non_zero_counts) - 1]
    )

    if len(peak_indices) >= 4:
        for index in range(1, len(peak_indices) - 1):
            peak_index = peak_indices[index]
            val_max = non_zero_counts[peak_index] * div_coef_max
            val_min = non_zero_counts[peak_index] * div_coef_min

            if (val_max < non_zero_counts[peak_indices[index - 1]] + 2) | (
                val_max < non_zero_counts[peak_indices[index + 1]] + 2
            ) and (val_min < non_zero_counts[peak_indices[index - 1]] + 2) & (
                val_min < non_zero_counts[peak_indices[index + 1]] + 2
            ):
                mask[peak_index] = 1

    return mask.astype("bool")


def get_breakpoints(zero_counts: np.ndarray, height_thresh: int = 3) -> list:
    """Get peaks of zero counts and return them as breakpoints

    Args:
        zero_counts: Counts of zeros
        height_thresh: Difference threshold

    Returns:
        Indices of peaks
    """
    peaks, _ = find_peaks(zero_counts)
    neg_zero_count = -zero_counts
    peaks_neg, _ = find_peaks(neg_zero_count)
    peak_indices = []
    if len(peaks) > 0:
        if len(peaks_neg) == 0:
            peak_indices = list(peaks)
        else:
            if len(peaks) + 1 != len(peaks_neg):
                peaks_neg = np.concatenate((peaks_neg, peaks_neg[-1:]))
            for index in range(len(peaks)):
                if (
                    zero_counts[peaks][index]
                    > zero_counts[peaks_neg][index] + height_thresh
                ):
                    peak_indices.append(peaks[index])

    return peak_indices


def get_detection_boxes(
    text_map: np.ndarray,
    link_map: np.ndarray,
    soft_link_threshold: float = 0.4,
    soft_text_threshold: float = 0.3,
    hard_link_threshold: float = 0.6,
    hard_text_threshold: float = 0.6,
) -> List[box_type]:
    """Create text boxes from predicted text_map and link_map

    Args:
        text_map: Text map from CRAFT prediction
        link_map: Link map from CRAFT prediction
        soft_link_threshold: Soft threshold for links
        soft_text_threshold: Soft threshold for text
        hard_text_threshold: Hard threshold for text
        hard_link_threshold: Hard threshold for links

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
    _, text_score = cv2.threshold(
        src=text_map, thresh=soft_text_threshold, maxval=1, type=0
    )
    _, link_score = cv2.threshold(
        src=link_map, thresh=soft_link_threshold, maxval=1, type=0
    )

    # add the maps and run connected components
    text_score_combined = np.clip(
        a=text_score + link_score, a_min=0, a_max=1
    ).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image=text_score_combined, connectivity=4
    )
    labels = labels.astype(np.uint64)

    # threshold by hard thresholds
    _, text_score_hard = cv2.threshold(
        src=text_map, thresh=hard_text_threshold, maxval=1, type=0
    )
    _, link_score_hard = cv2.threshold(
        src=link_map, thresh=hard_link_threshold, maxval=1, type=0
    )

    # add the maps and run connected components
    text_score_hard_combined = np.clip(
        a=text_score_hard + link_score_hard, a_min=0, a_max=1
    ).astype(np.uint8)
    _, labels_hard, stats_hard, _ = cv2.connectedComponentsWithStats(
        image=text_score_hard_combined, connectivity=4
    )

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

        if (size > 30) and (label_id < n_labels):
            segmentation_map = np.zeros(shape=text_map.shape, dtype=np.uint8)
            segmentation_map[(labels == label_id) & (text_score == 1)] = 1
            cuts = find_cuts(segmentation_map[y : y + height, x : x + width])
            segmentation_map[y : y + height, x : x + width][cuts] = 0
            (
                n_labels_current,
                sub_labels,
                sub_stats,
                _,
            ) = cv2.connectedComponentsWithStats(
                image=segmentation_map[y : y + height, x : x + width],
                connectivity=4,
            )
            if n_labels_current > 2:
                sub_stats[:, 0] += x
                sub_stats[:, 1] += y
                for i in range(1, n_labels_current):
                    new_label_id = max_id
                    max_id += 1
                    labels[y : y + height, x : x + width][
                        sub_labels == i
                    ] = new_label_id
                    labels_deq.append(new_label_id)
                    stats_deq.append(sub_stats[i].tolist())
                continue

        if (label_id < id_multiplier) and ((width / height) > long_box_coef):
            # is small box's top left smaller then targets
            top_left_in = (stats_hard[:, 0] >= x) & (stats_hard[:, 1] >= y)
            # is small box's bot right bigger then targets
            bottom_right_in = (
                stats_hard[:, 0] + stats_hard[:, 2] <= x + width
            ) & (stats_hard[:, 1] + stats_hard[:, 3] <= y + height)
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

        num_iter = int(
            math.sqrt(size * min(width, height) / (width * height))
            * num_iter_coef
        )
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

        kernel = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT,
            ksize=(1 + num_iter, 1 + num_iter + num_iter_offset),
        )
        segmentation_map[y_top:y_bottom, x_left:x_right] = cv2.dilate(
            src=segmentation_map[y_top:y_bottom, x_left:x_right], kernel=kernel
        )

        # make box
        np_contours = (
            np.roll(
                a=np.array(np.where(segmentation_map != 0)), shift=1, axis=0
            )
            .transpose()
            .reshape(-1, 2)
        )

        # align diamond-shape
        try:
            left, right = min(np_contours[:, 0]), max(np_contours[:, 0])
            top, bottom = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array(
                [[left, top], [right, top], [right, bottom], [left, bottom]],
                dtype=np.float32,
            )
        except BaseException as e:
            print(f"Exception {e} with {label_id=}")
            continue

        # make clock-wise order
        start_index = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_index, 0)
        detections.append(box.tolist())
        mapper.append(label_id)

    return detections


def adjust_result_coordinates(
    polygons: list, ratio_width: float, ratio_height: float, ratio_net: int = 2
) -> np.ndarray:
    """During inference, original image size is changed by (ratio_width, ratio_height) times.
    As predicted boxes are for changed image, this method will map the model's predicted boxes to
    original image scale

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
                polygons[index] = polygons[index] * (
                    ratio_width * ratio_net,
                    ratio_height * ratio_net,
                )
                polygons[index] = polygons[index].astype(int)
    return polygons


def remove_padding(
    box: box_type, upper_padding: int, left_padding: int
) -> box_type:
    """Remove padding created by centered padding

    Args:
        box: Predicted box
        upper_padding: Padding from above
        left_padding: Padding from left

    Returns:
        Corrected box
    """
    return (
        box[0] - left_padding,
        box[1] - upper_padding,
        box[2] - left_padding,
        box[3] - upper_padding,
    )
