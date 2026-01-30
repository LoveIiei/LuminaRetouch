# LuminaRetouch ✨

**Professional AI-Powered Portrait Retouching Desktop App**

LuminaRetouch is a sophisticated desktop application built with Python and PySide6, designed for professional-grade portrait photo enhancement. It combines traditional image processing techniques with powerful AI models to deliver stunning results, all through an intuitive and refined user interface.

![LuminaRetouch Screenshot](assets/screenshot.jpg)  
Image credit: [link](https://unsplash.com/photos/woman-in-white-crew-neck-shirt-smiling-IF9TK5Uy-KI)

---

## Key Features

LuminaRetouch provides a comprehensive suite of tools for detailed portrait enhancement, featuring a real-time, split-screen canvas to compare your edits instantly.

#### Retouching Toolkit
- **Skin Enhancement**: Achieve flawless skin with AI-powered smoothing that preserves natural texture and advanced blemish removal that intelligently targets spots and imperfections.
- **Eyes & Mouth**: Make eyes pop with brightness and enlargement tools, reduce dark circles, whiten teeth, and enhance lip color for a vibrant look.
- **Facial Sculpting**: Subtly refine facial features with tools for face, nose, and chin slimming, and jawline sharpening.

#### AI-Powered Engine
- **Two Processing Modes**:
    - **Fast Mode**: Utilizes high-speed, traditional algorithms for quick edits.
    - **Quality Mode**: Leverages the GFPGAN AI model for superior, professional-quality face restoration and enhancement.
- **AI Upscaling**: Increase image resolution by 2x or 4x, perfect for preparing images for high-resolution displays or print.

#### Streamlined Workflow
- **Intuitive UI**: A sleek, modern interface built with PySide6, designed for an efficient and enjoyable editing experience.
- **Before/After Split View**: Instantly compare your original image with the retouched version using a draggable split-screen slider.
- **Retouching Templates**: Save your favorite settings combinations as templates and apply them to other photos with a single click.

## Before & After
| Original | Retouched with LuminaRetouch |
| :---: | :---: |
| ![Original](assets/before.jpg) | ![Retouched](assets/after.png) |

---

## Installation & Setup

Follow these steps to get LuminaRetouch running on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/LoveIiei/LuminaRetouch.git
cd LuminaRetouch
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

LuminaRetouch has two modes. You can start with the standard installation and add the AI-powered features later.

#### Standard Installation (CPU-based)

This will install the core application with all traditional (non-AI) processing features.

```bash
pip install -r requirements.txt
```

#### Full Installation (AI-Enhanced - GPU Recommended)

For the best results and performance, install the AI dependencies. A CUDA-enabled GPU is highly recommended.

You can use the automated setup script:
```bash
python setup_ai.py
```
This script will detect if you have a compatible GPU and guide you through installing PyTorch and GFPGAN.

Alternatively, you can install them manually:
```bash
# 1. Install PyTorch (visit pytorch.org for the correct command for your system)
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install AI enhancement and upscaling libraries
pip install gfpgan realesrgan
```

### 4. AI Model Weights

The required AI model files (`.pth`) will be **downloaded automatically** the first time you use an AI-powered feature. Please be patient, as the initial download can be several hundred megabytes.

---

## How to Run

Once the installation is complete, launch the application:

```bash
python main.py
```

1.  Click **"Open"** to load a portrait photo.
2.  Use the sliders on the left control panel to adjust the retouching settings.
3.  For the highest quality, select the **"Quality"** processing mode.
4.  Click **"Process"** to see the results.
5.  Use the split-view slider on the image to compare the before and after.
6.  Click **"Save"** to export your masterpiece.

## Technologies Used

- **GUI**: PySide6 (Qt for Python)
- **Core Image Processing**: OpenCV, NumPy
- **Face Landmark Detection**: MediaPipe
- **AI Face Enhancement**: GFPGAN / PyTorch
- **AI Upscaling**: Real-ESRGAN

## Contributing

Contributions are welcome! If you have ideas for improvements or find a bug, please open an issue or submit a pull request.

