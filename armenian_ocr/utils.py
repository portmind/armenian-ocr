import cv2


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
