import string
import numpy as np
import cv2


def filter_boxes(boxes, texts):
    """
    Args:
        boxes (list): A list of boxes of the format [x_min, y_min, x_max, y_max].
        texts (list): A list of strings.

    Returns:
        boxes_with_text (list): A list of tuples of the format ([x_min, y_min, x_max, y_max], text).
        num_boxes (list): A list of boxes of the format [x_min, y_min, x_max, y_max].
    """
    boxes_with_text = [(boxes[i], texts[i]) for i in range(len(boxes)) if hull_area(boxes[i]) > 50]
    boxes_with_nums = [i[0] for i in boxes_with_text if is_digit_or_sign(i[1])]
    boxes_with_text = [i for i in boxes_with_text if not is_digit_or_sign(i[1])]
    num_boxes = merge_intersecting(np.array(boxes_with_nums), overlap_vertical=0.5, x_margins=30)
    return boxes_with_text, num_boxes


def hull_area(hull):
    """
    Args:
        hull (tuple): A tuple of the format (x_min, y_min, x_max, y_max).

    Returns:
        area (int): The area of the hull.
    """
    tl_x, tl_y = hull[0], hull[1]
    br_x, br_y = hull[2], hull[3]
    area = (br_x - tl_x) * (br_y - tl_y)
    return area


def is_digit_or_sign(text, max_len=10):
    """

    Args:
        text (str): A text.
        max_len (int): Max allowed length.

    Returns:
        digit_or_sign (bool): True if text is digit or sign.
    """
    if len(text) > max_len:
        return False
    allowed_symbols = set(string.digits + "$€ " + string.punctuation)
    set_text = set(text)
    digit_or_sign = set_text.issubset(allowed_symbols)
    if digit_or_sign:
        return True
    no_num_text = "".join([*filter(str.isalpha, text)]).lower()
    currencies = {
        "eu",
        "eur",
        "euro",
        "euros",
        "us",
        "usd",
        "dollars",
        "dollar",
        "cfa",
        "fcfa",
        "francs",
        "franc",
        "naira",
    }
    return (no_num_text in currencies) & (text.lower() not in currencies)


def merge_intersecting(boxes_, overlap_vertical=0.0, overlap_horizontal=0.0, x_margins=15, y_margins=-5):
    """
    Args:
        boxes_  (np.array): A array of boxes of the format [x_min, y_min, x_max, y_max].
        overlap_vertical (float): Allowed vertical overlap.
        overlap_horizontal (float): Allowed horizontal overlap
        x_margins (int): Margins ot be added by x-axis.
        y_margins (int): Margins to be added by y-axis.

    Returns:
        new_boxes (np.array): A array of boxes of the format [x_min, y_min, x_max, y_max].
    """
    if len(boxes_) == 0:
        return []

    boxes = boxes_.copy()
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # sort by top left coordinate
    boxes = np.array(sorted(boxes, key=lambda x: x[0]))

    pick = boxes[:1]

    # for merging, we should iterate over all boxes,
    # and check whether they should merge with all the other current iteration boxes,
    # current iteration boxes include yet unchecked as well as already merged boxes
    while len(boxes) > 0:
        # join already iterated and merged boxes with yet unchecked boxes
        candidate_boxes = np.concatenate((pick, boxes))

        # grab their coordinates
        x1 = candidate_boxes[:, 0]
        y1 = candidate_boxes[:, 1]
        x2 = candidate_boxes[:, 2]
        y2 = candidate_boxes[:, 3]

        # always take the first box yet remaining in boxes and
        # find intersection with the rest (itself included)
        xx1 = np.maximum(boxes[0, 0], x1)
        yy1 = np.maximum(boxes[0, 1], y1)
        xx2 = np.minimum(boxes[0, 2], x2)
        yy2 = np.minimum(boxes[0, 3], y2)

        # compute the width and height of the bounding box,
        # taking into account the merging margins

        w = np.maximum(0, xx2 - xx1 + x_margins)
        h = np.maximum(0, yy2 - yy1 + y_margins)

        # compute the ratio of overlap to the area of the smaller box
        # from the merging candidates
        overlap_h = w / np.minimum(
            abs(candidate_boxes[:, 2] - candidate_boxes[:, 0]),
            abs(boxes[0][2] - boxes[0][0]),
        )
        overlap_v = h / np.minimum(
            abs(candidate_boxes[:, 3] - candidate_boxes[:, 1]),
            abs(boxes[0][3] - boxes[0][1]),
        )
        overlap = (overlap_v > overlap_vertical) & (overlap_h > overlap_horizontal)

        # keep for merging only those
        # which have higher overlap area than threshold
        overlap_ixs = np.concatenate((np.where(overlap)[0], [len(pick)]), axis=0)

        # all the boxes that overlap with current box in iteration
        overlap_boxes = candidate_boxes[overlap_ixs]
        # calculate the bounding box around all these intersecting boxes
        xx1 = min(overlap_boxes[:, 0])
        yy1 = min(overlap_boxes[:, 1])
        xx2 = max(overlap_boxes[:, 2])
        yy2 = max(overlap_boxes[:, 3])

        # draw a new box around intersecting boxes
        # new_box = np.array([[xx1, yy1-3, xx2, yy2+5]])
        new_box = np.array([[xx1, yy1, xx2, yy2]])

        # drop the small boxes, to replace them with the new bigger box
        pick_ixs_to_drop = overlap_ixs[np.where(overlap_ixs < len(pick))[0]]
        boxes_ixs_to_drop = overlap_ixs[np.where(overlap_ixs >= len(pick))[0]]
        boxes_ixs_to_drop -= len(pick)
        pick = np.delete(pick, pick_ixs_to_drop, axis=0)
        boxes = np.delete(boxes, boxes_ixs_to_drop, axis=0)

        # add new box to the list
        pick = np.concatenate((pick, new_box)) if len(pick) > 0 else new_box
    new_boxes = np.array(pick).astype("int").tolist()
    return new_boxes


def grayscale(image):
    """
    Args:
        image (np.array): An input image.
    Returns:
        img_gs (np.array): The grayscale version of the image.
    """
    if len(image.shape) == 3:
        img_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gs = image.copy()
    return img_gs
