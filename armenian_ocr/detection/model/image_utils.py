from typing import Tuple

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)


def normalize_mean_variance(
    input_image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    variance: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize input image for model training

    Args:
        input_image: Input image, should be in RGB
        mean: Mean for normalization
        variance: Variance for normalization

    Returns:
        Normalized image
    """
    image = input_image.copy().astype(np.float32)

    image -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    image /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return image


def denormalize_mean_variance(
    input_image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    variance: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Denormalize input image

    Args:
        input_image: Input image, should be in RGB
        mean: Mean for normalization
        variance: Variance for normalization

    Returns:
        Denormalized image
    """
    image = input_image.copy()
    image = ((image * variance) + mean) * 255
    image = np.clip(a=image, a_min=0, a_max=255).astype(np.uint8)
    return image


def resize_aspect_ratio(
    image: np.ndarray,
    square_size: int,
    interpolation,
    fill_value: int = 0,
    center: bool = True,
) -> Tuple[np.ndarray, float]:
    """Resize image to square size by using interpolation.
    Big side of the image will be resized to square size, then small side will be resized
    by the same ratio as big side was.

    Args:
        image: Image to resize
        square_size: Size of resized image's big side
        interpolation (cv2 interpolation) : Interpolation type
        in our case always image > square_size
        fill_value: Fill value for padding
        center: Pad in a way that original image is in the center

    Returns:

    """
    if len(image.shape) == 3:
        height, width, channel = image.shape
    else:
        height, width = image.shape
        channel = 1

    target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    processed = cv2.resize(
        image, (target_w, target_h), interpolation=interpolation
    )

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    max_side = max(target_h32, target_w32)

    resized = (
        np.ones((max_side, max_side, channel), dtype=np.float32) * fill_value
    )

    if center:
        height_start, height_end = int((max_side - target_h) / 2), int(
            (max_side + target_h) / 2
        )
        width_start, width_end = int((max_side - target_w) / 2), int(
            (max_side + target_w) / 2
        )
        if channel == 1:
            resized[
                height_start:height_end, width_start:width_end, 0
            ] = processed
        else:
            resized[
                height_start:height_end, width_start:width_end, :
            ] = processed
    else:
        if channel == 1:
            resized[0:target_h, 0:target_w, 0] = processed
        else:
            resized[0:target_h, 0:target_w, :] = processed

    return resized, ratio


def cvt2heatmap_image(image: np.ndarray) -> np.ndarray:
    """Make heatmap from model predicted maps

    Args:
        image: Vertical stack of region and affinity boxes

    Returns:
        Input heatmap
    """
    image = (np.clip(a=image, a_min=0, a_max=1) * 255).astype(np.uint8)
    image = cv2.applyColorMap(src=image, colormap=cv2.COLORMAP_JET)
    return image
