import setuptools

with open("./requirements.txt", "r") as f:
    packages_required = [line.rstrip("\n") for line in f]

setuptools.setup(
    name="armenian-ocr",
    version="1.0",
    author="Portmind AM",
    description="PyTorch implementation of OCR pipeline.",
    packages=[
        "armenian_ocr",
        "armenian_ocr.recognition.model",
        "armenian_ocr.recognition.model.modules",
        "armenian_ocr.detection.model",
        "armenian_ocr.detection.model.basenet",
    ],
    install_requires=packages_required,
    python_requires=">=3.6.8",
)
