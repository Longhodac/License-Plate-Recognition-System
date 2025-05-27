# License Plate Recognition System
A Python-based License Plate Recognition System using OpenCV and Machine Learning.

## Features

- Detects license plates in images and video streams
- Extracts and recognizes license plate numbers using ML and OCR
- Web interface for real-time processing
- Supports multiple image formats
- Cross-platform compatibility

## Requirements

- Python 3.7+
- OpenCV
- scikit-image
- scikit-learn
- Flask
- numpy

## Installation

```bash
git clone https://github.com/yourusername/License-Plate-Recognition-System.git
cd License-Plate-Recognition-System
pip install -r requirements.txt
```

## Usage

For command line interface:
```bash
python prediction.py 
```

For web interface:
```bash
cd source
python app.py
```

Then open `http://localhost:5000` in your browser.

## Project Structure

```
License-Plate-Recognition-System/
├── source/
│   ├── static/
│   │   └── styles.css
│   ├── templates/
│   │   └── index.html
│   └── app.py
├── models/
│   └── svc/
│       └── svc.pkl
├── training-data/
│   └── train20x20/
├── test-img/
├── machine_train.py
├── prediction.py
├── segmentation.py
└── README.md
```

## Current Development

🚧 **Work in Progress** 🚧

Currently implementing:
- Web interface with Flask and real-time processing (currently doesn't work)
- Character segmentation improvements using scikit-image
- SVM model training pipeline for better accuracy
- Additional image preprocessing features
- API endpoints for external integration

Status:
- [x] Basic plate detection
- [x] Character segmentation
- [x] SVM model training
- [x] Web interface layout
- [ ] Real-time video processing
- [ ] API documentation
- [ ] Performance optimization
- [ ] Comprehensive testing suite

## References

This project was inspired by and adapted from:
- [Developing a License Plate Recognition System with Machine Learning in Python](https://medium.com/devcenter/developing-a-license-plate-recognition-system-with-machine-learning-in-python-787833569ccd) by DevCenter on Medium

## License

This project is licensed under the MIT License.