# CIFAR-10 Cats and Dogs Image Denoiser

## Overview

This repository contains a PyTorch implementation of a Conditional Transformer-based image denoising model specifically tailored for CIFAR-10 images of cats and dogs. It demonstrates the use of advanced neural network techniques, including Transformer architectures, feature extraction, and customized loss functions to effectively remove noise from images.

## Dataset

- **Dataset Used**: CIFAR-10
- **Classes Targeted**: Cats and Dogs
- **Dataset Splits**:
  - Training Set: 90%
  - Validation Set: 10%
  - Separate Test Set from CIFAR-10 provided

## Requirements

- Python
- PyTorch
- torchvision
- numpy
- matplotlib

## Installation

Clone the repository and install the dependencies:

```bash
git clone <repository_url>
cd <repository_directory>
pip install torch torchvision numpy matplotlib
```

## Usage

Run the training script:

```bash
python train.py
```

Ensure your device supports CUDA or MPS for optimal training speed:
- CUDA (GPU) recommended
- MPS (Apple Silicon Macs)
- CPU fallback available

## Model Architecture

The implemented model (`CIFARPetDenoiser`) leverages:
- **Feature Extraction**: CNN layers reducing input to embeddings.
- **Transformer Encoder**: Conditional transformer encoder with positional, noise-level, and label embeddings.
- **Decoder**: CNN-based decoder reconstructing images back to 32x32 RGB format.

## Training Details

- **Loss Functions**:
  - MSE Loss
  - L1 Loss
  - Cosine Similarity Loss
  - Combined weighted loss
- **Optimization**:
  - Optimizer: AdamW
  - Learning Rate: 3e-4 with OneCycleLR scheduler
  - Epochs: 60 (with learning rate resets at epochs 20 and 40)
- **Gradient Clipping**: Enabled (max gradient norm = 1.0)
- **Batch Size**: 64

## Results

- Model checkpoints, loss curves, and sample images are saved under `./cifar_pet_denoiser_results`
- Best performing model based on validation loss is automatically saved as `best_model.pt`

## Visualization

- Visualization scripts included to demonstrate model denoising across diffusion steps.
- Sample generation provided to qualitatively evaluate model performance.

![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v1/output/myplot1.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v1/output/myplot2.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v1/output/myplot3.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v1/output/myplot4.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v1/output/myplot5.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v1/output/myplot6.png)

## Requirements

- PyTorch
- torchvision
- Matplotlib
- NumPy

Install requirements via:

```bash
pip install torch torchvision matplotlib numpy
```
