# LuminaRetouch

LuminaRetouch is a Python-based image processing library designed for automated photo enhancement and retouching.

## Features

- **Auto-Exposure**: Intelligently adjusts brightness and contrast.
- **Color Correction**: Balances white balance and saturation.
- **Noise Reduction**: Removes digital noise while preserving details.
- **Batch Processing**: Process entire directories of images efficiently.
- **CLI Support**: Easy-to-use command-line interface.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LuminaRetouch.git
    cd LuminaRetouch
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Basic Usage

```python
from luminaretouch import ImageProcessor

processor = ImageProcessor()
processor.enhance('input.jpg', save_path='output.jpg')
```

### CLI

```bash
python main.py --input ./images/photo.jpg --output ./processed/
```

## Technologies

- **Python 3.x**
- **OpenCV**: Computer vision tasks.
- **Pillow (PIL)**: Image manipulation.
- **NumPy**: Matrix calculations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License

This project is licensed under the MIT License.