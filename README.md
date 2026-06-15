# Medical AI Segmentation Prototype

A production-ready **Brain Tumor Segmentation System** built with **Flask**, **PyTorch**, and a custom **U-Net** implementation.

This project is designed for:
- 3rd-year AIML / CSE students
- academic portfolio showcase
- GitHub open-source presentation
- resume-ready deep learning project demonstrations

---

## Project Overview

The system accepts an MRI brain scan image, predicts the tumor region using a binary segmentation model, generates a segmentation mask, overlays the mask on the original image, and presents confidence-related outputs in a professional web interface.

The implementation is **dataset-ready** for **BraTS-style workflows** and includes:
- U-Net architecture from scratch
- training pipeline
- validation pipeline
- prediction pipeline
- Dice score
- IoU score
- checkpoint saving
- model loading
- Flask-based inference UI

> Note: BraTS provides multi-parametric MRI scans and expert segmentation labels for tumor sub-regions such as enhancing tumor, tumor core, and whole tumor. This project uses a simplified **binary segmentation** setup suitable for student portfolio deployment and extension. The official BraTS segmentation benchmark evaluates segmentation with Dice-based metrics.

---

## Features

### Web Application
- Upload MRI brain scan image
- Run segmentation inference
- Display original image
- Display predicted mask
- Display overlay image
- Show confidence and mask coverage

### AI Pipeline
- U-Net model implemented from scratch
- Binary segmentation training support
- Dice score computation
- IoU score computation
- Checkpoint saving and loading
- CLI prediction support

### Engineering Quality
- Modular folder structure
- Input validation
- Error handling
- Resume-friendly documentation
- Dataset-ready design for future experimentation

---

## Tech Stack

### Backend
- Python
- Flask

### AI / Deep Learning
- PyTorch
- Custom U-Net

### Libraries
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Scikit-Learn

### Frontend
- HTML
- CSS
- JavaScript

---

## Architecture Diagram

```text
                +----------------------+
                |   MRI Brain Image    |
                |   (Upload / CLI)     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Preprocessing      |
                | Resize / Normalize   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    U-Net Model       |
                | Brain Segmentation   |
                +----------+-----------+
                           |
               +-----------+------------+
               |                        |
               v                        v
      +-------------------+    +-------------------+
      | Segmentation Mask |    | Probability Map   |
      +---------+---------+    +---------+---------+
                |                        |
                +-----------+------------+
                            |
                            v
                 +------------------------+
                 | Overlay + Metrics      |
                 | Confidence / Coverage  |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 | Flask Web Interface    |
                 +------------------------+
```

---

## Folder Structure

```text
medical-ai-segmentation/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
├── model/
│   └── unet.py
│
├── dataset/
│   ├── images/
│   ├── masks/
│   └── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── uploads/
├── outputs/
└── checkpoints/
```

---

## Dataset Setup

## BraTS-style Integration

BraTS data is distributed in NIfTI format and contains multi-parametric MRI modalities with expert labels. Since automatic download may not be available in your environment, this repository is implemented as **dataset-ready**, not dataset-bundled.

### Recommended student workflow

#### Option 1: Use preprocessed 2D PNG slices
Convert selected BraTS slices into paired 2D images and masks:

```text
dataset/
├── images/
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
├── masks/
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
```

Rules:
- filenames must match exactly between `images/` and `masks/`
- masks should be binary or convertible to binary
- MRI image can be grayscale or RGB PNG
- code resizes samples during training

#### Option 2: Extend to NIfTI
For a more research-oriented version:
- load `.nii` / `.nii.gz` MRI volumes
- extract slices
- merge tumor labels into binary masks
- train the same U-Net on generated 2D samples

This prototype is intentionally structured to support that extension.

---

## Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/your-username/medical-ai-segmentation.git
cd medical-ai-segmentation
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

#### Windows
```bash
venv\Scripts\activate
```

#### macOS / Linux
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide

## Train the model

Make sure your dataset is available under `dataset/images` and `dataset/masks`.

```bash
python train.py --epochs 20 --batch_size 4 --image_size 256
```

Optional arguments:
- `--epochs`
- `--batch_size`
- `--lr`
- `--image_size`
- `--val_size`
- `--seed`

### Output artifacts
Training saves:
- `checkpoints/latest_model.pth`
- `checkpoints/best_model.pth`
- `checkpoints/training_history.json`

---

## Run prediction from CLI

```bash
python predict.py --image uploads/sample.png --checkpoint checkpoints/best_model.pth
```

Output files are saved inside:
```text
outputs/
```

---

## Run the Flask web app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## Metrics

The training and validation pipelines support:

### Dice Score
Used to measure overlap between predicted segmentation and ground truth.

### IoU Score
Used to evaluate the intersection over union between prediction and mask.

### Confidence
Displayed in the web app as the maximum predicted probability in the tumor region probability map.

### Coverage Percentage
Represents the percentage of image pixels predicted as tumor after thresholding.

---

## Model Details

### U-Net
The U-Net is implemented from scratch in `model/unet.py` with:
- encoder blocks
- bottleneck
- decoder blocks
- skip connections
- 1-channel output head for binary segmentation

### Loss Function
Training uses a combined objective:
- BCEWithLogitsLoss
- Dice loss

This balances pixel-wise learning and overlap quality.

---

## Screenshots

Add your screenshots here after running the app:

```text
docs/screenshots/home-page.png
docs/screenshots/result-view.png
docs/screenshots/sample-mask.png
```

Suggested README image blocks:

```markdown



```

---

## Folder Explanation

### `app.py`
Runs the Flask application and handles file upload, validation, inference, and JSON responses.

### `train.py`
Contains the training loop, validation loop, metrics, checkpointing, and dataset loading.

### `predict.py`
Loads a trained model, performs inference, generates mask and overlay outputs.

### `model/unet.py`
Defines the U-Net architecture from scratch.

### `dataset/`
Stores training images and segmentation masks.

### `templates/` and `static/`
Contain the frontend files for the web interface.

### `uploads/`
Temporary storage for uploaded user files.

### `outputs/`
Stores prediction results such as original, mask, and overlay images.

### `checkpoints/`
Stores trained model checkpoints and training history.

---

## Resume Description

**Medical AI Segmentation Prototype**  
Developed a deep learning-based brain tumor segmentation system using PyTorch and U-Net with Dice/IoU evaluation, Flask deployment, and MRI mask overlay visualization in a portfolio-ready web application.

---

## Future Improvements

- Support full BraTS NIfTI volumetric pipeline
- Multi-class segmentation for ET / TC / WT
- Add MONAI-based medical preprocessing
- Add Grad-CAM or uncertainty visualization
- Add Docker deployment
- Add unit tests and CI/CD
- Add experiment tracking with TensorBoard or Weights & Biases
- Export prediction reports as PDF
- Deploy on cloud with GPU inference

---

## License

This project is released under the **MIT License**.

You can create a `LICENSE` file with the MIT license text before publishing publicly.

---

## Important Notes

- This project is an educational prototype for portfolio and learning use.
- It is **not** a certified medical device.
- Real clinical deployment requires extensive validation, regulatory compliance, and expert review.