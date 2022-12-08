from time import time

from armenian_ocr import utils
from armenian_ocr.detection.model.det_wrapper import DetWrapper
from armenian_ocr.recognition.model.rec_wrapper import RecWrapper


class OcrWrapper:
    """
    The wrapper class of OCR Pipeline.

    Attributes:
        det_wrapper (DetWrapper): An detection wrapper.
        rec_wrapper (RecWrapper): An recognition wrapper
    """

    def __init__(self):
        self.det_wrapper = DetWrapper()
        self.rec_wrapper = RecWrapper()

    def load(self, det_model_dir, rec_model_dir, device="cpu"):
        """
        Args:
            det_model_dir (str): A path to the directory of the detection model.
            rec_model_dir (str): A path to the directory of the recognition model.
            device (str): The device to run the models.
        """
        self.det_wrapper.load(det_model_dir, device=device)
        self.rec_wrapper.load(rec_model_dir, device=device)

    def predict(
        self,
        image,
        link_threshold=0.3,
        low_text=0.4,
        link_threshold2=0.6,
        low_text2=0.6,
        timer=False,
    ):
        """
        Args:
            image (np.array): An input image (RGB).
            link_threshold: same as DetWrapper.predict.link_threshold
            low_text: same as DetWrapper.predict.low_text
            link_threshold2: same as DetWrapper.predict.link_threshold2
            low_text2: same as DetWrapper.predict.low_text2
            timer: whether to print the execution times
        Returns:
            predictions (list): A list of tuples of the format ([x_min, y_min, x_max, y_max], text).
            image (np.array): The rotated image.
        """

        det_start = time()
        boxes = self.det_wrapper.predict(
            image,
            link_threshold=link_threshold,
            low_text=low_text,
            link_threshold2=link_threshold2,
            low_text2=low_text2,
        )

        if timer:
            print(f"detection took {time() - det_start} seconds")
        image_grayscale = utils.grayscale(image)
        images = [image_grayscale[box[1] : box[3], box[0] : box[2]] for box in boxes]
        rec_start = time()
        texts = self.rec_wrapper.predict(images)
        if timer:
            print(f"recognition took {time() - rec_start} seconds")
        predictions = list(zip(boxes, texts))
        return predictions
