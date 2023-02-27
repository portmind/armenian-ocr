import json
import os
from collections import defaultdict
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from armenian_ocr.detection.model import craft_utils, image_utils
from armenian_ocr.detection.model.craft import CRAFT
from armenian_ocr.detection.model.craft_utils import copy_state_dict

box_type = Tuple[int, int, int, int]


class DetWrapper:
    def __init__(self):
        self.cuda = False
        self.args = dict()
        self.model = CRAFT()
        self.device = None

    def load(
        self,
        path: str,
        device: str = "cpu",
        model_file_name: str = "detection.pth",
        args_file_name: str = "args.json",
    ):
        """Loads model weights and args from specified paths to object

        Args:
            path: Path to where model.pth and opt.txt are saved
            device: "cpu" or "cuda"
            model_file_name: File name of the pth file
            args_file_name: File name of the arguments file
        """

        self.device = torch.device(device)
        self.model.load_state_dict(
            copy_state_dict(
                torch.load(
                    os.path.join(path, model_file_name),
                    map_location=self.device,
                )
            )
        )

        if device == "cuda":
            self.model.to(device)
            cudnn.benchmark = False

        self.model.eval()

        with open(os.path.join(path, args_file_name)) as fp:
            self.args = json.load(fp)

    @staticmethod
    def postprocess_output(predictions: List[box_type]) -> List[box_type]:
        """Postprocess predictions. The process includes:
        Find boxes with small heights and widths. If a box has both small height and width, it is considered very small,
        otherwise it is considered small. If a box is very small, then it most likely a punctuation and will be joined
        to a box to its left or bottom. If a box is small, then it will be joined to a box to its left or right. After
        calculating those distances, if the minimum distance is less than half the medium box height, the other box will
        be added to parent box candidates. Then, for all candidate boxes a weighted distance is calculated and the one
        with minimal is chosen as parent box.

        Args:
            predictions: Predicted detection boxes

        Returns:
            Postprocessed output
        """
        if len(predictions) == 0:
            return predictions

        box_heights = np.array([box[3] - box[1] for box in predictions])
        box_heights_norm = box_heights[
            (box_heights >= np.percentile(box_heights, 10))
            & (box_heights <= np.percentile(box_heights, 90))
        ]
        median_box_height = np.median(box_heights_norm)
        small_height_mask = box_heights < median_box_height - 3 * np.std(
            box_heights_norm
        )

        box_widths = np.array([box[2] - box[0] for box in predictions])
        box_widths_norm = box_widths[
            (box_widths >= np.percentile(box_widths, 10))
            & (box_widths <= np.percentile(box_widths, 90))
        ]
        median_box_width = np.median(box_widths)
        small_width_mask = box_widths < median_box_width - 3 * np.std(
            box_widths_norm
        )

        delete_indices = set()

        join_mapping = {index: index for index in range(len(predictions))}
        for index_small, small_box in enumerate(predictions):
            if (
                small_height_mask[index_small]
                and small_width_mask[index_small]
            ):
                very_small = True
            elif (
                small_height_mask[index_small] or small_width_mask[index_small]
            ):
                very_small = False
            else:  # not a small box
                continue
            small_centre_height, small_centre_width = (
                small_box[3] + small_box[1]
            ) / 2, (small_box[0] + small_box[2]) / 2
            half_height, half_width = (small_box[3] - small_box[1]) / 2, (
                small_box[0] - small_box[2]
            ) / 2

            parent_candidates = []
            for index_other, other_box in enumerate(predictions):
                if (index_small == index_other) or (
                    small_height_mask[index_other]
                    and small_width_mask[index_other]
                ):
                    continue
                other_centre_height, other_centre_width = (
                    other_box[3] + other_box[1]
                ) / 2, (other_box[0] + other_box[2]) / 2
                other_half_height, other_half_width = (
                    other_box[3] - other_box[1]
                ) / 2, (other_box[0] - other_box[2]) / 2

                distance_left = np.sqrt(
                    (
                        (small_centre_width + half_width)
                        - (other_centre_width - other_half_width)
                    )
                    ** 2
                    + (small_centre_height - other_centre_height) ** 2
                )

                if very_small:
                    distance_lower = np.sqrt(
                        (
                            (small_centre_height + half_height)
                            - (other_centre_height - other_half_height)
                        )
                        ** 2
                        + (small_centre_width - other_centre_width) ** 2
                    )
                    distances = [distance_left, distance_lower]
                else:
                    distance_right = np.sqrt(
                        (
                            (small_centre_width - half_width)
                            - (other_centre_width + other_half_width)
                        )
                        ** 2
                        + (small_centre_height - other_centre_height) ** 2
                    )

                    distances = [distance_left, distance_right]

                if min(distances) > median_box_height * 0.5:
                    continue
                else:
                    parent_candidates.append((distances, index_other))

            if len(parent_candidates) > 0:
                _, candidate_index = min(
                    parent_candidates,
                    key=lambda box_distances: 2 * box_distances[0][0]
                    + box_distances[0][1],
                )
                # logical preference is given to the left box, that's why its distance is multiplied by 2

                indices = [index_small, candidate_index]

                parent, child = min(indices), max(indices)
                join_mapping[parent] = child
            else:
                if very_small:
                    delete_indices.add(index_small)

        for (
            child,
            parent,
        ) in join_mapping.items():  # find parent box and join only to it
            while parent != join_mapping[parent]:
                parent = join_mapping[parent]
                join_mapping[child] = parent

        join_mapping_final, join_set = defaultdict(list), set()
        for child, parent in join_mapping.items():
            if child != parent:
                assert parent == join_mapping[parent]
                join_mapping_final[parent].append(child)
                join_set.add(child)
                join_set.add(parent)

        final_boxes = [
            box
            for index, box in enumerate(predictions)
            if index not in join_set.union(delete_indices)
        ]

        for index_small, indices in join_mapping_final.items():
            if index_small in delete_indices:
                continue
            bounding_boxes = [predictions[index_small]]
            for index_other in indices:
                bounding_boxes.append(predictions[index_other])
            bounding_boxes = np.array(bounding_boxes)
            joined_box = (
                bounding_boxes[:, :2].min(axis=0).tolist()
                + bounding_boxes[:, 2:].max(axis=0).tolist()
            )
            final_boxes.append(tuple(joined_box))
        return final_boxes

    def predict(
        self,
        image: np.ndarray,
        return_heatmap: bool = False,
        postprocess: bool = True,
    ) -> Union[List[box_type], Tuple[List[box_type], np.ndarray]]:
        """Inference on CRAFT model.
        During the inference, the output maps of the model will be thresholded twice: soft and hard
        Soft is thresholded by link_threshold and low_text which may result in big boxes, but also may find small text.
        Hard is thresholded by link_threshold2 and low_text2. As thresh values are higher, boxes will become small
        and small text will be dropped.
        Inference is a mix of the two stages in a way that if hard breaks a box founded by soft into several parts,
        it is applicable. But also small boxes found by soft are kept.

        Args:
            image: Input image (RGB)
            return_heatmap: Whether to return heatmap of predicted maps
            postprocess: Postprocess boxes to join small boxes

        Returns:
            Predicted bounded boxes of text, heatmap of region and affinity maps
        """
        center = self.args.get("center", False)
        # resize to model's input size
        image = image[..., ::-1]  # RGB -> BGR
        image_resized, target_ratio = image_utils.resize_aspect_ratio(
            image=image,
            square_size=self.args.get("canvas_size", 1280),
            interpolation=cv2.INTER_AREA,
            center=center,
        )
        ratio_height = ratio_width = 1 / target_ratio

        # preprocessing
        image = image_utils.normalize_mean_variance(image_resized)
        image = torch.from_numpy(image).permute(
            2, 0, 1
        )  # [h, w, c] to [c, h, w]
        image = image.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
        image = image.to(self.device)

        with torch.no_grad():
            prediction, _ = self.model(image)
            score_text = prediction[0, :, :, 0].cpu().numpy()
            score_link = prediction[0, :, :, 1].cpu().numpy()

        # Post-processing
        boxes = craft_utils.get_detection_boxes(
            text_map=score_text,
            link_map=score_link,
            soft_link_threshold=self.args.get("soft_link_threshold", 0.4),
            soft_text_threshold=self.args.get("soft_text_threshold", 0.3),
            hard_link_threshold=self.args.get("hard_link_threshold", 0.6),
            hard_text_threshold=self.args.get("hard_text_threshold", 0.6),
        )

        # coordinate adjustment to original image size
        boxes = craft_utils.adjust_result_coordinates(
            polygons=boxes, ratio_width=ratio_width, ratio_height=ratio_height
        )
        boxes = list(map(self.hull2box, boxes))
        boxes = [
            box
            for box in boxes
            if ((box[3] - box[1]) > 1) and ((box[2] - box[0]) > 1)
        ]

        if postprocess:
            boxes = self.postprocess_output(boxes)

        if center:
            target_height = image.shape[0] / ratio_height
            target_width = image.shape[1] / ratio_width
            left_padding = int(
                ratio_width * (self.args["canvas_size"] - target_width) / 2
            )
            upper_padding = int(
                ratio_height * (self.args["canvas_size"] - target_height) / 2
            )
            boxes = [
                craft_utils.remove_padding(
                    box=box,
                    upper_padding=upper_padding,
                    left_padding=left_padding,
                )
                for box in boxes
            ]

        # render results (optional)
        if return_heatmap:
            render_image = score_text.copy()
            render_image = np.hstack((render_image, score_link))
            heatmap = image_utils.cvt2heatmap_image(render_image)
            return boxes, heatmap

        return boxes

    @staticmethod
    def hull2box(hull: np.ndarray) -> box_type:
        """Make a rectangle from hull

        Args:
            hull: Predicted hull

        Returns:
            left, top, right, bottom coordinates of the hull
        """
        top = int(hull[:, 1].min())
        bottom = int(hull[:, 1].max())
        left = int(hull[:, 0].min())
        right = int(hull[:, 0].max())
        return left, top, right, bottom

    @staticmethod
    def draw_pred(image: np.ndarray, boxes: List[box_type]) -> np.ndarray:
        """Draw predicted text boxes rectangles on image

        Args:
            image: Image to draw boxes on
            boxes: Predicted boxes by self.predict

        Returns:
            Image with drawn predicted boxes
        """
        image = image.copy()
        for top, left, bottom, right in boxes:
            cv2.rectangle(
                img=image,
                pt1=(int(top), int(left)),
                pt2=(int(bottom), int(right)),
                color=(0, 255, 0),
                thickness=2,
            )
        return image
