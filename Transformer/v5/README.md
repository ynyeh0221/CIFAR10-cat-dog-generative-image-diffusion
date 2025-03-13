# CIFAR-10 Cat vs. Dog Diffusion Denoiser

## Overview

This project implements a diffusion-based denoising model specifically designed to classify and generate images of cats and dogs from the CIFAR-10 dataset. Utilizing a U-Net architecture enhanced with patch-based convolutions, self-attention modules, and residual blocks, this model efficiently learns to denoise and generate high-quality RGB images.

## Features

- **Dataset Filtering:** Utilizes CIFAR-10 dataset, filtering specifically for cat and dog classes.
- **Patch-based Convolutions:** Custom convolutional modules (`PatchConv` and `PatchExpand`) for enhanced local context handling.
- **Linear-Attention:** Incorporates efficient linear self-attention mechanisms.
- **U-Net Architecture:** A modified U-Net architecture with time and class conditioning, suitable for diffusion-based image generation tasks.
- **Custom Training Loop:** Includes a diffusion-based denoising training loop with combined loss functions (MSE, L1, and Cosine Similarity).

## Model Components

- **ResBlock:** Residual blocks used in refinement stages to improve image quality.
- **PatchConv & PatchExpand:** Specialized convolutional layers for handling patches, enabling better spatial context understanding and image reconstruction.
- **DoubleConv, Down, and Up Blocks:** Key building blocks in the U-Net architecture, adapted to use patch convolutions and self-attention.
- **Self-Attention Module:** Efficient linear attention mechanism to capture global context with reduced computational complexity.
- **CIFARUNetDenoiser:** A complete U-Net architecture with embedded conditioning for noise levels and class labels.

## Training Setup

- **Optimizer:** AdamW with weight decay and adjusted learning rate.
- **Loss Functions:** Combination of Mean Squared Error (MSE), L1 loss, and Cosine Similarity Loss.
- **Learning Rate Scheduler:** Cosine annealing schedule with warmup period for stable convergence.
- **Diffusion Process:** Advanced diffusion schedule with 10 diffusion steps, enabling high-quality denoising results.

## Visualization

- **Epoch-wise Visualization:** Images generated at each epoch, visualizing the progression of noise reduction.
- **Denoising Visualization:** Step-by-step visualization of the denoising process.

## Requirements

- PyTorch
- Torchvision
- NumPy
- Matplotlib

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd <project-folder>
pip install -r requirements.txt
```

## Usage

Run the training script to train the model:

```bash
python train.py
```

## Visualization

Visualizations for each epoch and the generation process will be saved in the `epoch_visualizations` folder.
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_1_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_10_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_20_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_30_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_40_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_50_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_60_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_70_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_80_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_90_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v5/output/epoch_100_generation.png)
