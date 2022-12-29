import os

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
    """
    Class for arguments required by recognition network
    """

    def __init__(self, path):
        """
        Args:
            path (str): path to opt.txt saved with the model
        """
        with open(path, "r") as fp:
            args = fp.readlines()

        for arg_i in args:
            if ":" in arg_i:
                split_point = arg_i.index(":")
                arg_name = arg_i[:split_point]
                arg_value = arg_i[(split_point + 2) : -1]
                setattr(self, arg_name, arg_value)

        # convert boolean and integer attributes to correct format from string
        bool_fields = ["sensitive", "PAD", "rgb"]
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

        for i in bool_fields:
            if hasattr(self, i):
                setattr(
                    self, i, (True if getattr(self, i) == "True" else False)
                )

        for i in int_fields:
            if hasattr(self, i):
                setattr(self, i, int(getattr(self, i)))


class ImageDataset(Dataset):
    def __init__(self, images):
        """
        Args:
            images (list of numpy arrays):
        """
        self.images = [
            Image.fromarray(i) for i in images
        ]  # convert to PIL Image, suitable format for the network

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
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
        path,
        device="cpu",
        model_file_name="best_accuracy.pth",
        opt_file_name="opt.txt",
    ):
        """
        Args:
            path (str): path to where model.pth and opt.txt are saved
            device (str): cpu or cuda
            model_file_name (str): file name of the pth file
            opt_file_name (str): file name of the arguments file

        Returns:

        """
        self.opt = Opt(os.path.join(path, opt_file_name))
        self.device = torch.device(device)
        self.opt.device = device
        """ model configuration """
        if "CTC" in self.opt.Prediction:
            self.converter = CTCLabelConverter(
                self.opt.character, device=self.opt.device
            )
        else:
            self.converter = AttnLabelConverter(
                self.opt.character, device=self.opt.device
            )
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        self.model = Model(self.opt)

        t_load = torch.load(
            os.path.join(path, model_file_name), map_location=self.device
        )
        self.model.load_state_dict(
            {k[k.find(".") + 1 :]: v for k, v in t_load.items()}
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images):
        """
        Args:
            images (list of numpy arrays): images to be read by recognition network

        Returns:
            preds_list (list of str): list of read texts from the images_
        """
        image_data = ImageDataset(images)
        collate = AlignCollate(
            imgH=self.opt.imgH,
            imgW=self.opt.imgW,
            keep_ratio_with_pad=self.opt.PAD,
        )
        image_loader = DataLoader(
            image_data,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=collate,
            pin_memory=(self.opt.device != "cpu"),
        )

        preds_list = []

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
                text_for_pred = (
                    torch.LongTensor(batch_size, self.opt.batch_max_length + 1)
                    .fill_(0)
                    .to(self.device)
                )

                if "CTC" in self.opt.Prediction:
                    preds = self.model(image, text_for_pred).log_softmax(2)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.permute(1, 0, 2).max(2)
                    preds_index = (
                        preds_index.transpose(1, 0).contiguous().view(-1)
                    )
                    preds_str = self.converter.decode(
                        preds_index.data, preds_size.data
                    )
                else:
                    preds = self.model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(
                        preds_index, length_for_pred
                    )

                soft_prob = torch.nn.functional.softmax(preds, dim=2)
                soft_prob, index = soft_prob.max(2)

                preds_i = []
                for i, pred in enumerate(preds_str):
                    if "Attn" in self.opt.Prediction:
                        box_prob = torch.mean(
                            soft_prob[i][: pred.find("[s]")]
                        ).item()
                        if box_prob < 0.7:
                            pred = ""
                        else:
                            pred = pred[
                                : pred.find("[s]")
                            ]  # prune after "end of sentence" token ([s])
                    preds_i.append(pred)
                preds_list += preds_i

        return preds_list
