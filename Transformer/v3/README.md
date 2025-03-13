# CIFAR-10 Pet Diffusion Model

## Overview

This repository contains an implementation of an improved diffusion model for generating and classifying cat and dog images from the CIFAR-10 dataset. The model uses a transformer-based architecture with linear attention mechanisms to efficiently denoise images at various noise levels, providing high-quality pet image generation.

## Features

- **Linear Transformer Architecture**: Uses a more efficient attention mechanism for faster training and inference
- **Classifier-Free Guidance**: Enhances the class-conditional generation quality
- **Advanced Loss Function**: Balanced adaptive loss combining MSE, L1, cosine similarity, and edge/high-frequency components
- **Improved Sampling Strategy**: Adaptive guidance scaling for better generation quality
- **Exponential Moving Average (EMA)**: For more stable model performance
- **2D Sinusoidal Positional Embeddings**: Provides better spatial awareness
- **Residual Connections**: In both encoder and decoder for improved information flow

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- matplotlib
- numpy

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cifar-pet-diffusion.git
cd cifar-pet-diffusion

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model from scratch:

```bash
python cifar_pet_diffusion.py --mode train
```

The training process:
1. Filters cat and dog images from CIFAR-10
2. Splits data into training and validation sets
3. Trains the diffusion model with an adaptive noise schedule
4. Saves checkpoints and visualizations in the `improved_cifar_pet_denoiser` directory

### Generating Images

To generate pet images using a trained model:

```bash
python cifar_pet_diffusion.py --mode generate --model_path /path/to/model.pt --num_samples 5 --steps 40
```

Parameters:
- `--model_path`: Path to a saved model checkpoint
- `--num_samples`: Number of images to generate per class (default: 5)
- `--steps`: Number of diffusion steps (higher = better quality but slower, default: 40)

## Model Architecture

The model consists of several key components:

1. **Patch Embedding**: Transforms input images into patch embeddings (similar to ViT)
2. **Label Embedding**: Encodes class information for conditional generation
3. **Noise Level Embedding**: Encodes the current noise level in the diffusion process
4. **Linear Transformer Encoder**: Processes embeddings with an efficient attention mechanism
5. **Residual Decoder**: Transforms transformer outputs back to image space
6. **Balanced Loss Functions**: Combines different loss types for improved training

## Advanced Features

### Improved Sampling with Classifier-Free Guidance

The model uses an adaptive guidance scale that changes throughout the sampling process:
- Early steps: Minimal guidance to establish basic structure
- Middle steps: Standard guidance for class conditioning
- Late steps: Increased guidance to enhance class-specific features

### Balanced Adaptive Loss

The loss function adaptively balances:
- MSE loss for overall reconstruction
- L1 loss for pixel-level accuracy
- Cosine similarity for directional consistency
- High-frequency loss for texture details
- Edge-preserving loss for structural integrity

## Results

The model generates high-quality 32x32 cat and dog images. Example visualizations are saved during training showing the progressive denoising process and final generated samples.

![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v3/output/myplot1.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v3/output/myplot2.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v3/output/myplot3.png)
