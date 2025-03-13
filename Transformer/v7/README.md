# CIFAR-10 Cat & Dog Classification and Denoising with Advanced U-Net

This repository contains an implementation of an advanced U-Net model designed specifically for classifying and denoising images of cats and dogs from the CIFAR-10 dataset. The model incorporates unique architectural enhancements like patch-based convolutions, self-attention mechanisms, and conditional embeddings to improve performance and image clarity.

---

## Key Features

- **Patch-based Convolution:** Efficiently captures spatial context by processing image patches.
- **Self-Attention Modules:** Improve image representations with linear attention mechanisms for computational efficiency.
- **Dynamic Loss Functions:** Combines Mean Squared Error (MSE), L1 loss, and Cosine Similarity loss dynamically during training.
- **Visualization Tools:** Integrated functions to visualize training progress, denoising processes, and output samples.

---

## Dataset

Utilizes the CIFAR-10 dataset filtered to contain only:

- **Cats (Label: 3)**
- **Dogs (Label: 5)**

Images are resized and normalized for optimal training.

---

## Model Architecture

The U-Net model is enhanced with:
- **ResBlocks:** For refined image details.
- **PatchConv & PatchExpand:** For efficient spatial dimension handling.
- **Attention Layers:** Integrated at multiple stages for improved context awareness.

---

## Training Details

- **Optimizer:** AdamW
- **Learning Rate Schedule:** Warmup with cosine annealing.
- **Epochs:** 140
- **Batch Size:** 64
- **Device Compatibility:** CUDA, MPS, or CPU

---

## Training Visualization

Visualizations generated during training:
- **Epoch-wise Sample Generations:** Monitor progress.
- **Attention Maps:** Insight into the model's focus during denoising.
- **Final Generated Samples:** Evaluate the quality after training.

Check the `epoch_visualizations` folder post-training.

![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_1_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_10_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_20_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_30_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_40_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_50_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_60_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_70_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_80_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_90_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_100_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_110_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_120_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_130_generation.png)
![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-diffusion/blob/main/Transformer/v7/output/epoch_140_generation.png)

---

## Requirements

- Python
- PyTorch
- Torchvision
- Matplotlib
- NumPy

Install requirements:
```bash
pip install torch torchvision matplotlib numpy
```

---

## Usage

Run the training script:

```bash
python train.py
```

Ensure your environment supports GPU or MPS for the best performance.

---

## Outputs

- **Model Checkpoints:** Saved periodically.
- **Loss Graphs:** Visualize training convergence.
- **Generated Images & Animations:** Stored in `epoch_visualizations`.


Enjoy exploring advanced classification and denoising techniques with CIFAR-10!
