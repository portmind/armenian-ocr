import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from armenian_ocr.recognition.model.model import Model
from armenian_ocr.recognition.model.utils import (
    AlignCollate,
    AttnLabelConverter,
    CTCLabelConverter,
)


class Opt(object):
    """Class for arguments required by recognition network"""

    def __init__(self, path: str):
        """Get and set arguments

        Args:
            path: path to opt.txt saved with the model
        """
        with open(path, "r", encoding="utf8") as fp:
            args = fp.readlines()

        for arg in args:
            if ":" in arg:
                split_point = arg.index(":")
                arg_name = arg[:split_point]
                arg_value = arg[(split_point + 2) : -1]
                setattr(self, arg_name, arg_value)

        # convert boolean and integer attributes to correct format from string
        boolean_fields = ["sensitive", "PAD", "rgb"]
        int_fields = [
            "imgW",
            "imgH",
            "input_channel",
            "output_channel",
            "hidden_size",
            "num_fiducial",
            "batch_max_length",
            "batch_size",
        ]

        for field in boolean_fields:
            if hasattr(self, field):
                setattr(
                    self,
                    field,
                    (True if getattr(self, field) == "True" else False),
                )

        for field in int_fields:
            if hasattr(self, field):
                setattr(self, field, int(getattr(self, field)))


class ImageDataset(Dataset):
    def __init__(self, images: List[np.ndarray]):
        """Initialize dataset

        Args:
            images: Input images
        """
        self.images = [
            Image.fromarray(image) for image in images
        ]  # convert to PIL Image, suitable format for the network

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item: int):
        return self.images[item], None


class RecWrapper(object):
    """
    Wrapper class for the recognition network
    """

    def __init__(self):
        self.model = None
        self.converter = None
        self.opt = None
        self.device = None

    def load(
        self,
        path: str,
        device: str = "cpu",
        model_file_name: str = "best_accuracy.pth",
        opt_file_name: str = "opt.txt",
    ):
        """
        Args:
            path: path to where model.pth and opt.txt are saved
            device: cpu or cuda
            model_file_name: file name of the pth file
            opt_file_name: file name of the arguments file

        Returns:

        """
        self.opt = Opt(os.path.join(path, opt_file_name))
        self.device = torch.device(device)
        self.opt.device = device
        """ model configuration """
        if "CTC" in self.opt.Prediction:
            self.converter = CTCLabelConverter(
                character=self.opt.character, device=self.opt.device
            )
        else:
            self.converter = AttnLabelConverter(
                character=self.opt.character, device=self.opt.device
            )
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        self.model = Model(self.opt)

        t_load = torch.load(
            f=os.path.join(path, model_file_name), map_location=self.device
        )
        self.model.load_state_dict(
            {k[k.find(".") + 1 :]: v for k, v in t_load.items()}
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images: List[np.ndarray]) -> List[str]:
        """Predict

        Args:
            images: images to be read by recognition network

        Returns:
            Texts (Recognition network outputs)
        """
        image_data = ImageDataset(images)
        collate = AlignCollate(
            imgH=self.opt.imgH,
            imgW=self.opt.imgW,
            keep_ratio_with_pad=self.opt.PAD,
        )
        image_loader = DataLoader(
            dataset=image_data,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=collate,
            pin_memory=(self.opt.device != "cpu"),
        )

        predictions = []

        # predict
        self.model.eval()
        with torch.no_grad():
            for image_tensors, _ in image_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(self.device)
                # For max length prediction
                length_for_pred = torch.IntTensor(
                    [self.opt.batch_max_length] * batch_size
                ).to(self.device)
                prediction_text = (
                    torch.LongTensor(batch_size, self.opt.batch_max_length + 1)
                    .fill_(0)
                    .to(self.device)
                )

                if "CTC" in self.opt.Prediction:
                    prediction = self.model(
                        input=image, text=prediction_text
                    ).log_softmax(2)

                    # Select max probability (greedy decoding) then decode index to character
                    prediction_size = torch.IntTensor(
                        [prediction.size(1)] * batch_size
                    )
                    _, prediction_indices = prediction.permute(1, 0, 2).max(2)
                    prediction_indices = (
                        prediction_indices.transpose(1, 0)
                        .contiguous()
                        .view(-1)
                    )
                    prediction_str = self.converter.decode(
                        prediction_indices.data, prediction_size.data
                    )
                else:
                    prediction = self.model(
                        input=image, text=prediction_text, is_train=False
                    )

                    # select max probability (greedy decoding) then decode index to character
                    _, prediction_indices = prediction.max(2)
                    prediction_str = self.converter.decode(
                        prediction_indices, length_for_pred
                    )

                soft_prob = torch.nn.functional.softmax(prediction, dim=2)
                soft_prob, index = soft_prob.max(2)

                predicted_texts = []
                for index, predicted_text in enumerate(prediction_str):
                    if "Attn" in self.opt.Prediction:
                        box_prob = torch.mean(
                            soft_prob[index][: predicted_text.find("[s]")]
                        ).item()
                        if box_prob < 0.7:
                            predicted_text = ""
                        else:
                            predicted_text = predicted_text[
                                : predicted_text.find("[s]")
                            ]  # prune after "end of sentence" token ([s])
                    predicted_texts.append(predicted_text)
                predictions += predicted_texts

        return predictions
