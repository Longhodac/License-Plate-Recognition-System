  
A Python-based License Plate Recognition System using OpenCV and Tesseract OCR.

## Features

- Detects license plates in images and video streams
- Extracts and recognizes license plate numbers using OCR
- Supports multiple image formats
- Easy to use and extend

## Requirements

- Python 3.7+
- OpenCV
- pytesseract
- numpy

## Installation

```bash
git clone https://github.com/yourusername/License-Plate-Recognition-System.git
cd License-Plate-Recognition-System
pip install -r requirements.txt
```

Make sure Tesseract OCR is installed on your system.  
[Download Tesseract](https://github.com/tesseract-ocr/tesseract)

## Usage

```bash
python main.py --image path/to/image.jpg
```

Or for video stream:

```bash
python main.py --video path/to/video.mp4
```

## Project Structure

```
License-Plate-Recognition-System/
├── main.py
├── detector.py
├── recognizer.py
├── requirements.txt
└── README.md
```

## References

This project was inspired by and adapted from:
- [Developing a License Plate Recognition System with Machine Learning in Python](https://medium.com/devcenter/developing-a-license-plate-recognition-system-with-machine-learning-in-python-787833569ccd) by DevCenter on Medium


## License

This project is licensed under the MIT License.

