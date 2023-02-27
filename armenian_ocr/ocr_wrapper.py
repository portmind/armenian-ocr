from time import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from armenian_ocr import utils
from armenian_ocr.detection.layout_detection import detect_layout
from armenian_ocr.detection.model.det_wrapper import DetWrapper
from armenian_ocr.recognition.model.rec_wrapper import RecWrapper


class OcrWrapper:
    """The wrapper class of OCR Pipeline.

    Attributes:
        det_wrapper (DetWrapper): An detection wrapper.
        rec_wrapper (RecWrapper): An recognition wrapper
    """

    def __init__(self):
        self.det_wrapper = DetWrapper()
        self.rec_wrapper = RecWrapper()

    def load(
        self, det_model_dir: str, rec_model_dir: str, device: str = "cpu"
    ):
        """Load model

        Args:
            det_model_dir: A path to the directory of the detection model.
            rec_model_dir: A path to the directory of the recognition model.
            device: The device to run the models.
        """
        self.det_wrapper.load(det_model_dir, device=device)
        self.rec_wrapper.load(rec_model_dir, device=device)

    def predict(
        self,
        image: np.ndarray,
        predict_layout: bool = False,
        timer: bool = False,
    ) -> List[Dict[str, Any]]:
        """

        Args:
            image: Input image (RGB).
            predict_layout: Whether to predict layout groups.
            timer: Whether to print the execution times.

        Returns:
            predictions: A list of tuples of the format ([x_min, y_min, x_max, y_max], text).
        """

        # Detection
        det_start = time()
        boxes = self.det_wrapper.predict(image=image)
        paragraphs, rows = None, None

        if timer:
            print(f"Detection took {time() - det_start:.2f} seconds")

        if predict_layout:
            layout_start = time()
            detections_df = pd.DataFrame(
                boxes, columns=["x1", "y1", "x2", "y2"]
            )
            detections_df = detect_layout(
                detections=detections_df, image_shape=image.shape
            )
            paragraphs = detections_df["paragraph_id"]
            rows = detections_df["row_id"]

            if timer:
                print(
                    f"Layout detection took {time() - layout_start:.2f} seconds"
                )

        # Recognition
        rec_start = time()
        image_grayscale = utils.grayscale(image)
        images = [
            image_grayscale[box[1] : box[3], box[0] : box[2]] for box in boxes
        ]

        texts = self.rec_wrapper.predict(images)
        if timer:
            print(f"Recognition took {time() - rec_start:.2f} seconds")

        # Creating output
        if predict_layout:
            predictions = [
                {
                    "box": box,
                    "text": text,
                    "paragraph": paragraph_id,
                    "row": row_id,
                }
                for box, paragraph_id, row_id, text in zip(
                    boxes, paragraphs, rows, texts
                )
            ]
        else:
            predictions = [
                {"box": box, "text": text} for box, text in zip(boxes, texts)
            ]

        return predictions
