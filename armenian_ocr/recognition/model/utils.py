import math

import torch
import torchvision.transforms as transforms
from PIL import Image


class CTCLabelConverter(object):
    """Convert between text-label and text-index"""

    def __init__(
        self,
        character,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = [
            "[blank]"
        ] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = "".join(text)
        text = [self.dict[char] for char in text]

        return torch.IntTensor(text).to(self.device), torch.IntTensor(
            length
        ).to(self.device)

    def decode(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        index = 0
        for l in length:  # noqa: E741
            t = text_index[index : index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (
                    not (i > 0 and t[i - 1] == t[i])
                ):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = "".join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """Convert between text-label and text-index"""

    def __init__(
        self,
        character,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ["[GO]", "[s]"]  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append("[s]")
            text = [self.dict[char] for char in text]
            batch_text[i][1 : 1 + len(text)] = torch.LongTensor(
                text
            )  # batch_text[:, 0] = [GO] token
        return batch_text.to(self.device), torch.IntTensor(length).to(
            self.device
        )

    def decode(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        for index, l in enumerate(length):
            text = "".join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


# Image preprocessing related util classes
class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type="right"):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = (
                img[:, :, w - 1]
                .unsqueeze(2)
                .expand(c, h, self.max_size[2] - w)
            )

        return Pad_img


class AlignCollate(object):
    def __init__(
        self, imgH=32, imgW=100, keep_ratio_with_pad=False, aug_func=None
    ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.aug_func = aug_func

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)
        if self.aug_func is not None:
            images = [self.aug_func(img) for img in images]

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            transform = NormalizePAD((1, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize(
                    (resized_w, self.imgH), Image.BICUBIC
                )
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat(
                [t.unsqueeze(0) for t in resized_images], 0
            )

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat(
                [t.unsqueeze(0) for t in image_tensors], 0
            )

        return image_tensors, labels
