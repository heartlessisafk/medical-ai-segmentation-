# Dataset Instructions

This project is built to be **BraTS-style dataset ready**.

## Expected 2D training format

Place your preprocessed image-mask pairs here:

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

## Rules
- Image and mask filenames must match exactly.
- Images can be PNG/JPG/BMP.
- Masks should be binary segmentation masks.
- Non-zero mask pixels are treated as tumor.

## Suggested BraTS preprocessing workflow
1. Obtain BraTS data through official access channels.
2. Load MRI modalities from NIfTI volumes.
3. Extract relevant 2D slices.
4. Convert selected tumor labels into binary masks.
5. Export slices to `dataset/images/` and `dataset/masks/`.

## Example
- `dataset/images/case_001_slice_45.png`
- `dataset/masks/case_001_slice_45.png`

## Why this format?
This repository is designed for easy student use, rapid experimentation, and simple Flask deployment. You can later extend it to full 3D or multimodal training.