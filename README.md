
# Armenian OCR

![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-brightgreen)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## About

armenian-ocr is a package that contains an **Armenian Document OCR solution** that also supports *Latin* and *Cyrillic* characters. The solution is designed to work with scanned documents. It supports documents with *different **layouts**, **densities** and **scan qualities***.
A modified version of the solution is currently used in the process of digitalization of the documents of the [National Library of Armenia](https://nla.am/en/).
The results are slightly better than the Google Cloud Vision OCR, and are shown in the table below.

## Test Dataset
We have annotated 4 documents of different layouts and densities to assess the quality of our OCR and also to be able to compare with other OCR solutions.
The dataset with corresponding annotations are located in *test_images* directory.
## Comparison with Google Cloud Vision OCR
We compared our trained model with the OCR provided by Google.
In the table below you can see the *character level error rates* of both models on the
annotated dataset.


| Method                    |1.png             |2.png|3.png|4.png|`mean`|
|---------------------------|------------------|-----|-----|-----|------|
| `Ours`                    |7.5%              |0.8% |3.5% |4.4% |4%    |
| `Google Cloud Vision OCR` |14%               |1.4% |4%   |2.5% |5.5%  |

## Solution

The solution is a pipeline of two models - *detection* and *recognition*.
### Detection
The purpose of this stage is to find bounding boxes around the words in the document.\
We chose the [CRAFT](https://github.com/clovaai/CRAFT-pytorch) architecture for our detection model.

### Recognition
The *recognition* model takes the output of the detection network (*image patches inside the bounding boxes*), and returns the text of that area.\
This part shares the architecture of [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

## Installation

We use poetry as a development manager.
If you do not have a poetry installed, follow the instructions [here](https://python-poetry.org/docs/#installation)

```shell
$ poetry install
```

## Usage

1. Download model files from public S3 bucket, and uncompress the archive

```shell
$ wget https://armenian-ocr-objects.s3.eu-west-3.amazonaws.com/objects.zip
$ unzip objects.zip -d objects
```
2. Run the provided `ocr.py` script.
If you have an Nvidia GPU you can use the `--cuda` flag to utilize it. \
If you want to also predict layout structure you should pass `--layout` flag. \
This code will save the output in a list format, where each element is a list of the form `{"box": box, "text": text}`.\
If `--layout` flag is passed the form will be `{"box": box, "text": text, "paragraph": paragraph_id, "row": row_id}`.

```shell
$ poetry run python ocr.py \
    --detection_dir /path/to/detection \
    --recognition_dir /path/to/recognition \
    --image_path ./test_images/1.png \
    --output_path ./output.json
```

## License

This project is licensed under the CC BY-NC 4.0 License - see the [LICENSE](LICENSE-CC-BY-NC-4.0.md) file for details
