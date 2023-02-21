"""Helper functions for using boxes predicted by detection model to form a layout. The formed layout may not always
correct. The process is includes heuristics found during experimentation stages on a fixed test set and may not cover
all possible layouts and forms.
"""

from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate moving average values

    Args:
        values: Input values
        window_size: Window size

    Returns:
        Moving average values
    """
    return np.convolve(values, np.ones(window_size), "same") / window_size


def get_horizontal_breakpoints(
    image_1d: np.ndarray, window_size: int = 10
) -> List[int]:
    """Find horizontal breakpoints using detected boxes.

    Args:
        image_1d: Image with colored detection boxes
        window_size: Moving average window size (used for smoothing the amounts of pixels covered by each horizontal
            line)

    Returns:
        Horizontal breakpoints
    """
    horizontal_whites = np.where(
        moving_average((image_1d != 255).sum(axis=1), window_size).astype(int)
        == 0
    )[0]
    y_breakpoints, window = [], []

    for index in horizontal_whites:
        if len(window) == 0:
            window.append(index)
        else:
            if window[-1] != index - 1:
                y_breakpoints.append(window[len(window) // 2])
                window = [index]
            else:
                window.append(index)
    if len(window) == 0:
        return []
    else:
        y_breakpoints.append(window[len(window) // 2])
    return y_breakpoints


def get_vertical_breakpoints(
    image_1d: np.ndarray, divisor: int = 4, window_size: int = 100
) -> np.ndarray:
    """Find vertical breakpoints using detected boxes

    Args:
        image_1d: Image with colored detection boxes
        divisor: What fraction change compared to image height should be considered as breakpoint
            (for example, if divisor is 4 if a prominence of image height / 4 occurs a break line will be added)
        window_size: Moving average window size (used for smoothing the amounts of pixels covered by each vertical
            line)

    Returns:
        Vertical breakpoints
    """
    values = (image_1d != 255).sum(axis=0)
    moving_averages = moving_average(
        values=values, window_size=window_size
    )  # to make smoother
    breakpoints = find_peaks(
        -moving_averages, prominence=image_1d.shape[0] // divisor
    )[0]
    return breakpoints


def unify_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Assign "row_id"s to predicted words

    Args:
        df: Dataframe from make_df

    Returns:
        Same dataframe, where close y1 coordinates are unified as one coordinate
    """

    df["y_mean"] = (df["y1"] + df["y2"]) / 2
    df = df.sort_values(by="y_mean")

    row_id = 0
    row_ids = [row_id]
    prev = df.iloc[0]
    for i in range(1, len(df)):
        next_ = df.iloc[i]
        if (prev["y1"] <= next_["y_mean"] <= prev["y2"]) and (
            next_["y1"] <= prev["y_mean"] <= next_["y2"]
        ):
            # if the y coordinate is close to prev, change its coordinate to prev
            pass
        else:
            # if the y coordinate differs too much, start using it as prev
            row_id += 1
        row_ids.append(row_id)
        prev = next_
    df["row_id"] = row_ids

    return df


def unify_paragraphs(
    detections: pd.DataFrame, group_by: str = "paragraph_id"
) -> pd.DataFrame:
    """Joins overlapping paragraphs and assigns "row_id"s to each paragraph

    Args:
        detections: Detection boxes
        group_by: Grouping column

    Returns:
        Updated DataFrame with 'row_id' column
    """
    min_values = detections.groupby(group_by)[["x1", "y1"]].min()
    max_values = detections.groupby(group_by)[["x2", "y2"]].max()

    group_coordinates_items = list(
        pd.concat((min_values, max_values), axis=1).iterrows()
    )
    group_mapping = {
        group_id: group_id for group_id in detections[group_by].unique()
    }

    median_box_width = np.median(detections["x2"] - detections["x1"])
    median_box_height = np.median(detections["y2"] - detections["y1"])

    for group_id1, coordinates1 in group_coordinates_items:
        for group_id2, coordinates2 in group_coordinates_items:
            if (
                group_id1 >= group_id2
            ):  # for joining the lower index is considered as parent
                continue

            x_intersection = min(coordinates1["x2"], coordinates2["x2"]) - max(
                coordinates1["x1"], coordinates2["x1"]
            )
            y_intersection = min(coordinates1["y2"], coordinates2["y2"]) - max(
                coordinates1["y1"], coordinates2["y1"]
            )

            if (
                x_intersection > median_box_width
                and y_intersection > median_box_height
            ):
                group_mapping[group_id2] = group_id1

    for (
        child,
        parent,
    ) in group_mapping.items():  # find parent index and join only to it
        while parent != group_mapping[parent]:
            parent = group_mapping[parent]
            group_mapping[child] = parent

    detections[group_by] = detections[group_by].map(group_mapping)

    detections["row_id"] = 0
    for group_id in detections[group_by].unique():
        group_mask = detections[group_by] == group_id
        group_df = detections[group_mask].copy()
        group_df = unify_rows(group_df)
        detections.loc[group_mask] = group_df
    detections["row_id"] = (
        detections[group_by].astype(str)
        + "_"
        + detections["row_id"].astype(str)
    )
    detections = detections[["x1", "y1", "x2", "y2", "row_id", group_by]]
    return detections


def improve_row_grouping(detections: pd.DataFrame) -> pd.DataFrame:
    """Split row groups that are likely to contain more than one row (heuristic based)

    Args:
        detections: Detections dataframe that should contain "row_id" column

    Returns:
        Detections with updated "row_id" column
    """
    median_height = (detections["y2"] - detections["y1"]).median()
    threshold_width = np.percentile(detections["x2"] - detections["x1"], 5)
    group_heights = detections.groupby("row_id").apply(
        lambda x: x["y2"].max() - x["y1"].min()
    )
    big_groups = set(group_heights[group_heights > 1.4 * median_height].index)

    for group in big_groups:
        group_mask = detections["row_id"] == group
        group_df = detections[group_mask].copy()
        if len(group_df) <= 1:
            continue

        num_rows = []
        for x_coord in range(group_df["x1"].min(), group_df["x2"].max()):
            num_rows.append(
                (
                    (x_coord >= group_df["x1"]) & (x_coord <= group_df["x2"])
                ).sum()
            )

        thresholded_counts = [
            i
            for i in sorted(Counter(num_rows).items(), reverse=True)
            if i[1] > threshold_width
        ]
        if len(thresholded_counts) == 0:
            print(
                "Issue in improve_row_grouping, len(thresholded_counts) == 0, to be fixed"
            )
            continue
        max_row_count = thresholded_counts[0][0]

        if max_row_count == 1:
            continue
        else:
            group_df["y_mean"] = (group_df["y1"] + group_df["y2"]) / 2
            threshold_height = (
                group_df["y2"].max() - group_df["y1"].min()
            ) / max_row_count
            threshold_min = group_df["y1"].min()
            group_suffix = 0
            while threshold_min < group_df["y2"].max():
                threshold_max = threshold_min + threshold_height
                threshold_mask = (group_df["y_mean"] > threshold_min) & (
                    group_df["y_mean"] <= threshold_max
                )
                group_df.loc[threshold_mask, "row_id"] = (
                    group_df["row_id"][threshold_mask] + f"_{group_suffix}"
                )
                group_suffix += 1
                threshold_min = threshold_max
        detections.loc[group_mask] = group_df
    return detections


def detect_layout(
    detections: pd.DataFrame,
    image_shape: Tuple[int, ...],
    median_multiplier: int = 5,
    window_size: int = 15,
    divisor: int = 4,
) -> pd.DataFrame:
    """For each predicted box, assign a "paragraph_id" and "row_id". The "paragraph_id" indicates the visual grouping
    into paragraphs, while the "row_id" specifies the row number for each paragraph.

    Args:
        detections: Detected boxes (should have x1, y1, x2, y2 columns; x1, y1 is the top left corner and
         x2, y2 is the bottom right corner of the box)
        image_shape: Image shape
        median_multiplier: An argument that is used to calculate the threshold for vertically splitting a page part.
            The product of median_multiplier and median row height is compared to the page part height.
            If the page part height exceeds this product, it will be divided vertically.
        window_size: Moving average window size (used for smoothing the amounts of pixels covered by each vertical
            line)
        divisor: An argument that is used to calculate the threshold for vertically splitting a page part.
            The height of the image is divided by divisor. If the difference between two vertical lines covering
            detection box pixels is exceeding this threshold, then a break line will be inserted

    Returns:
        Detections dataframe with two new columns: "paragraph_id" and "row_id"
    """
    detected_boxes = np.full(
        shape=image_shape[:2], fill_value=255, dtype=np.uint8
    )  # Create an empty image for
    # putting predicted detection boxes atop. The image will be used for layout detection.
    for x1, y1, x2, y2 in detections[["x1", "y1", "x2", "y2"]].values:
        detected_boxes[
            y1:y2, x1:x2
        ] = 0  # fill all the pixels of detected boxes with 0s

    detections["y_mean"] = (detections["y1"] + detections["y2"]) / 2
    detections["x_mean"] = (detections["x1"] + detections["x2"]) / 2

    group_by = "y_mean"
    paragraph_name = "paragraph_id"
    detections[paragraph_name] = -1  # initialize paragraph ID with -1
    paragraph_id = 0

    height_median = (detections["y2"] - detections["y1"]).median()
    horizontal_breakpoints = (
        [0]
        + get_horizontal_breakpoints(
            image_1d=detected_boxes, window_size=int(height_median / 2)
        )
        + [image_shape[0]]
    )

    for y_min, y_max in zip(
        horizontal_breakpoints[:-1], horizontal_breakpoints[1:]
    ):
        paragraph_ids = []
        if y_max - y_min < median_multiplier * height_median:
            # This region is not big enough to be divided vertically
            # So all detections in this region are assigned to the same paragraph
            group_mask = (detections["y_mean"] > y_min) & (
                detections["y_mean"] <= y_max
            )
            detections.loc[group_mask, paragraph_name] = paragraph_id
            paragraph_id += 1
        else:
            vertical_breakpoints = (
                [0]
                + list(
                    get_vertical_breakpoints(
                        image_1d=detected_boxes[y_min:y_max],
                        divisor=divisor,
                        window_size=window_size,
                    )
                )
                + [image_shape[1] + 1]
            )
            for x_min, x_max in zip(
                vertical_breakpoints[:-1], vertical_breakpoints[1:]
            ):
                x_mask = (detections["x_mean"] > x_min) & (
                    detections["x_mean"] <= x_max
                )
                y_mask = (detections["y_mean"] > y_min) & (
                    detections["y_mean"] <= y_max
                )
                group_mask = x_mask & y_mask
                group_df = detections[group_mask].copy()
                group_by_values = sorted(group_df[group_by].unique())
                if len(group_by_values) == 0:
                    continue
                prev_value = group_by_values[0]
                for value in group_by_values:
                    if abs(value - prev_value) > 1.5 * height_median:
                        # if there is a larger space that 1.5 row then assign the word a new paragraph ID

                        group_df.loc[
                            group_df[group_by].isin(paragraph_ids),
                            paragraph_name,
                        ] = paragraph_id
                        paragraph_ids = [value]
                        paragraph_id += 1
                    else:
                        # otherwise assign it to the previous paragraph ID
                        paragraph_ids.append(value)
                    prev_value = value
                group_df.loc[
                    group_df[group_by].isin(paragraph_ids), paragraph_name
                ] = paragraph_id
                paragraph_id += 1
                detections.loc[group_mask, :] = group_df

    # join overlapping paragraphs
    detections = unify_paragraphs(
        detections=detections, group_by=paragraph_name
    )

    # split row groups that are likely to contain more than one row
    detections = improve_row_grouping(detections)
    return detections.sort_index()
