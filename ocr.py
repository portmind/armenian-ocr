import json
import time
from argparse import ArgumentParser

import numpy as np
import torch.cuda
from pdf2image import convert_from_path
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
        "-d",
        "--document_path",
        type=str,
        help="Path to the document (image or PDF).",
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

    start = time.time()

    document_extension = args.document_path.split(".")[-1].lower()

    if document_extension == "pdf":
        pil_images = convert_from_path(args.document_path)
        np_images = [np.array(pil_image) for pil_image in pil_images]
    elif document_extension in ["png", "jpg", "jpeg"]:
        pil_image = Image.open(args.document_path)
        np_image = np.array(pil_image)

        if np_image.shape[-1] == 4:  # handle images with alpha channel
            pil_image.load()
            blended_image = Image.new(
                mode="RGB", size=pil_image.size, color=(255, 255, 255)
            )
            blended_image.paste(
                im=pil_image, mask=pil_image.split()[3]
            )  # blend alpha channel
            np_image = np.array(blended_image)
        np_images = [np_image]
    else:
        raise NotImplementedError(
            f"Documents with {document_extension} extension are not supported"
        )

    predictions = [
        ocr.predict(
            image=np_image, predict_layout=args.layout, timer=args.timer
        )
        for np_image in np_images
    ]

    print(f"OCR process took {time.time() - start:.2f} seconds")

    with open(args.output_path, "w", encoding="utf-8") as fp:
        json.dump(obj=predictions, fp=fp, ensure_ascii=False)
