
# Armenian OCR

## About

armenian-ocr is a package that contains an **Armenian Document OCR solution** that also supports *English* and *Russian* languages.

## Test Dataset
We have annotated 4 documents of different layouts and densities to assess the quality of our OCR and also to be able to compare with other OCR solutions.
The dataset with corresponding annotations are located in *test_images* directory.
## Comparison with Google Cloud Vision OCR
We compared our trained model with the OCR provided by Google.
In the table below you can see the *character level error rates* of both models on the
annotated dataset.


|Method                   |1.png             |2.png|3.png|4.png|`mean`|
|-------------------------|------------------|-----|-----|-----|------|
|`ours`                   |7.5%              |0.8% |3.5% |4.4% |4%    |
|`Google Cloud Vision OCR`|14%               |1.4% |4%   |2.5% |5.5%  |

## Solution

The solution is a pipeline of two models - *detection* and *recognition*.
### Detection
The purpose of this stage is to find bounding boxes around the words in the document.\
We chose the [CRAFT](https://github.com/clovaai/CRAFT-pytorch) architecture for our detection model.

### Recognition
The *recognition* model takes the output of the detection network (*image patches inside the bounding boxes*), and returns the text of that area.\
This part shares the architecture of [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

## Installation

First clone the repository then run the following commands
```bash
cd armenian-ocr
pip3 install .
```

## Usage

To use the solution first download the model files from this [link](dummy_link) and uncompress the archive.

Once it is done you can import the package and use it like so.

```python
from armenian_ocr import OcrWrapper

ocr = OcrWrapper()
ocr.load(
    "/path/to/objects/detection",
    "/path/to/objects/recognition",
    device='cpu'
    )

img = np.array('/path/to/image')
predictions = ocr.predict(img)
```


Or run the `ocr.py` script from the command line.
```bash
python ocr.py \
 --detection_dir /path/to/objects/detection \
 --recognition_dir /path/to/objects/recognition \
 --image_path ./test_images/3.png \
 --output_path ./output.json
```



If you have a Nvidia GPU you can use the `--cuda` flag to use it.

This code will save the output in a list format, where each element is a list of the form `[bounding_box, word]`.

## License

This project is licensed under the CC BY-NC 4.0 License - see the [LICENSE](LICENSE-CC-BY-NC-4.0.md) file for details