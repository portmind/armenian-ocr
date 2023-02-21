import json
from argparse import ArgumentParser

import numpy as np
import torch.cuda
from PIL import Image

from armenian_ocr import OcrWrapper

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-dd",
        "--detection_dir",
        type=str,
        help="Path to the detection model directory.",
    )
    parser.add_argument(
        "-rd",
        "--recognition_dir",
        type=str,
        help="Path to the recognition model directory.",
    )
    parser.add_argument(
        "-i", "--image_path", type=str, help="Path to the image."
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="Path to the output file."
    )
    parser.add_argument(
        "-l", "--layout", action="store_true", help="Detect layout."
    )
    parser.add_argument(
        "-t", "--timer", action="store_true", help="Show processing times."
    )
    parser.add_argument(
        "-cuda", "--cuda", action="store_true", help="Use cuda."
    )
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    ocr = OcrWrapper()
    ocr.load(
        det_model_dir=args.detection_dir,
        rec_model_dir=args.recognition_dir,
        device=device,
    )

    image = np.array(Image.open(args.image_path))
    predictions = ocr.predict(
        image=image, predict_layout=args.layout, timer=args.timer
    )

    with open(args.output_path, "w", encoding="utf-8") as fp:
        json.dump(obj=predictions, fp=fp, ensure_ascii=False)
