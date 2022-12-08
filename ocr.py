import json
import numpy as np
from PIL import Image
from armenian_ocr import OcrWrapper
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--detection_dir", type=str, help="Path to the detection model directory.")
    parser.add_argument("--recognition_dir", type=str, help="Path to the recognition model directory.")
    parser.add_argument("--image_path", type=str, help="Path to the image.")
    parser.add_argument("--output_path", type=str, help="Path to the output file.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda.")
    args = parser.parse_args()
    
    ocr = OcrWrapper()
    ocr.load(args.detection_dir, args.recognition_dir, device='cuda' if args.cuda else 'cpu')

    img = np.array(Image.open(args.image_path))
    predictions = ocr.predict(img)

    with open(args.output_path, "w", encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False)