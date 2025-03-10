# CIFAR Pet Image Denoiser

## Overview

This project implements a sophisticated deep-learning-based diffusion model specifically designed to denoise and generate high-quality images of cats and dogs from the CIFAR-10 dataset. It leverages advanced techniques including sinusoidal positional embeddings, linear transformers with efficient linear attention mechanisms, and an enhanced decoder featuring residual connections and spatial attention modules.

## Features

- **Advanced Transformer Architecture**: Utilizes linear transformers for computationally efficient attention, improving speed without sacrificing performance.
- **Enhanced Decoder**: Incorporates residual blocks, spatial attention, and detail enhancement modules for superior image reconstruction.
- **Balanced Adaptive Loss Function**: Combines multiple loss criteria (MSE, L1, cosine similarity, high-frequency detail preservation, and edge consistency) dynamically to ensure visually appealing and detailed outputs.
- **Classifier-free Guidance**: Offers adaptive guidance during image generation, improving class-conditioned outputs.

## Dependencies

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- NumPy

Install dependencies using:
```bash
pip install torch torchvision matplotlib numpy
```

## Usage

### Training the Model

To train the model:
```bash
python your_script_name.py --mode train
```

### Generating Images
```bash
python your_script.py --mode generate --model_path path_to_checkpoint.pt --num_samples 5 --num_steps 40
```

## Results
- Checkpoints are automatically saved during training.
- Generated images and training visualizations are stored in designated directories for easy access and evaluation.
