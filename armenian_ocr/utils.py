import cv2
import numpy as np


def grayscale(image: np.ndarray) -> np.ndarray:
    """Make image grayscale

    Args:
        image: Input image.
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image.copy()
    return grayscale_image
