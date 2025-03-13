import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import math
import numpy as np
import torch.nn.functional as F

# Increase image size for CIFAR-10 (32x32 RGB images)
img_size = 32

# Data loading and preprocessing for CIFAR-10
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB
])

# Load CIFAR-10 dataset
batch_size = 64
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Filter to only include cats (class 3) and dogs (class 5)
cat_dog_indices = [i for i, (_, label) in enumerate(train_dataset) if label in [3, 5]]
cat_dog_dataset = Subset(train_dataset, cat_dog_indices)


# Create a new target mapping: 0 for cats, 1 for dogs
# We need to remap the original CIFAR-10 labels (3 for cat, 5 for dog) to 0 and 1
class CatDogSubset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # Remap labels: 3 (cat) -> 0, 5 (dog) -> 1
        new_label = 0 if label == 3 else 1
        return image, new_label

    def __len__(self):
        return len(self.subset)


train_dataset = CatDogSubset(cat_dog_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Define class names
class_names = ['Cat', 'Dog']


# Define a basic ResBlock for potential use in refinement
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection with projection if dimensions change
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Create a new class for handling patches in convolutional layers
class PatchConv(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2, stride=1, padding=0):
        super(PatchConv, self).__init__()
        self.patch_size = patch_size

        # Create a convolution that operates on patches
        self.conv = nn.Conv2d(
            in_channels * patch_size * patch_size,  # Input channels × patch area
            out_channels,
            kernel_size=1,  # 1x1 convolution on the flattened patches
            stride=1,
            padding=0
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Ensure dimensions are divisible by patch_size
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            pad_h = self.patch_size - (height % self.patch_size)
            pad_w = self.patch_size - (width % self.patch_size)
            x = F.pad(x, (0, pad_w, 0, pad_h))
            batch_size, channels, height, width = x.shape

        # Reshape to extract patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, channels, height // self.patch_size, width // self.patch_size,
                                self.patch_size * self.patch_size)
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, channels * self.patch_size * self.patch_size,
                                                       height // self.patch_size, width // self.patch_size)

        # Apply convolution to the flattened patches
        x = self.conv(x)

        return x


# Add a class for unpatchifying - the reverse operation of patching
class PatchExpand(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2):
        super(PatchExpand, self).__init__()
        self.patch_size = patch_size

        # Create a convolution to map patch features to pixel features
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * patch_size * patch_size,  # Output channels × patch area
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Apply 1x1 convolution to expand features
        x = self.conv(x)

        # Reshape to expand spatial dimensions
        channels_per_pixel = x.shape[1] // (self.patch_size * self.patch_size)
        x = x.view(batch_size, channels_per_pixel, self.patch_size, self.patch_size, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, channels_per_pixel, height * self.patch_size, width * self.patch_size)

        return x


# Modify the DoubleConv class to use patch convolutions
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, patch_size=1):
        super().__init__()
        self.residual = residual
        self.patch_size = patch_size

        if not mid_channels:
            mid_channels = out_channels

        # First convolution using patches if patch_size > 1
        if patch_size > 1:
            self.conv1 = PatchConv(in_channels, mid_channels, patch_size=patch_size)
        else:
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)

        self.norm1 = nn.GroupNorm(1, mid_channels)
        self.act1 = nn.GELU()

        # Second convolution using normal 3x3 conv to maintain spatial information
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        # Apply the first patch-based convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        # Apply the second standard convolution
        out = self.conv2(out)
        out = self.norm2(out)

        if self.residual and x.shape == out.shape:
            return F.gelu(x + out)
        else:
            return F.gelu(out)


# Modify the Down class to incorporate patch convolutions
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, patch_size=1):
        super().__init__()

        self.patch_size = patch_size

        # Replace maxpool with patch-based downsampling
        if patch_size > 1:
            self.patch_down = PatchConv(in_channels, in_channels, patch_size=patch_size)
            self.double_conv1 = DoubleConv(in_channels, in_channels, residual=True, patch_size=1)
            self.double_conv2 = DoubleConv(in_channels, out_channels, patch_size=1)
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, in_channels, residual=True),
                DoubleConv(in_channels, out_channels),
            )

        # Time and class embedding projection
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t_emb):
        if self.patch_size > 1:
            x = self.patch_down(x)
            x = self.double_conv1(x)
            x = self.double_conv2(x)
        else:
            x = self.maxpool_conv(x)

        emb = self.emb_layer(t_emb)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb


# Modify the Up class to use patch-based upsampling
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, patch_size=1):
        super().__init__()

        self.patch_size = patch_size

        # Use patch expansion if patch_size > 1, otherwise use bilinear upsampling
        if patch_size > 1:
            self.up = PatchExpand(in_channels // 2, in_channels // 2, patch_size=patch_size)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Convolutions using patch size
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # Time and class embedding projection
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t_emb):
        x = self.up(x)

        # Handle different input sizes (ensure feature maps have same dimensions)
        diffY = skip_x.size()[2] - x.size()[2]
        diffX = skip_x.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate skip connections
        x = torch.cat([skip_x, x], dim=1)

        # Apply convolutions
        x = self.conv(x)

        emb = self.emb_layer(t_emb)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb


# Self-attention block (can be included in the U-Net architecture)
class SelfAttention(nn.Module):
    """
    Linear Attention module with O(n) complexity instead of O(n²)
    """

    def __init__(self, channels, num_heads=4):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = (channels // num_heads) * 2
        self.inner_dim = self.head_dim * num_heads

        self.query = nn.Conv2d(channels, self.inner_dim, kernel_size=1)
        self.key = nn.Conv2d(channels, self.inner_dim, kernel_size=1)
        self.value = nn.Conv2d(channels, self.inner_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(self.inner_dim, channels, kernel_size=1)

        self.norm_q = nn.GroupNorm(1, channels)
        self.norm_k = nn.GroupNorm(1, channels)
        self.norm_v = nn.GroupNorm(1, channels)

        self.attention_maps = None

    def forward(self, x):
        """
        Forward pass using linear attention mechanism
        """
        batch_size, c, h, w = x.shape

        q = self.query(self.norm_q(x))
        k = self.key(self.norm_k(x))
        v = self.value(self.norm_v(x))

        q = q.view(batch_size, self.num_heads, self.head_dim, h * w)
        k = k.view(batch_size, self.num_heads, self.head_dim, h * w)
        v = v.view(batch_size, self.num_heads, self.head_dim, h * w)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        k_sum = k.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        k_normalized = k / k_sum

        # (B, H, D, N) x (B, H, D, N) -> (B, H, D, D)
        context = torch.matmul(v, k_normalized.transpose(-2, -1))

        # (B, H, D, D) x (B, H, D, N) -> (B, H, D, N)
        out = torch.matmul(context, q)

        out = out.view(batch_size, self.inner_dim, h, w)
        out = self.out_proj(out)

        return out

# U-Net Model with time and class conditioning
class CIFARUNetDenoiser(nn.Module):
    def __init__(self, img_size=32, in_channels=3, num_classes=2, time_dim=256,
                 attention_heads=8, patch_size=4):
        super(CIFARUNetDenoiser, self).__init__()

        # Store important parameters
        self.img_size = img_size
        self.in_channels = in_channels
        self.time_dim = time_dim
        self.patch_size = patch_size

        # Time embedding network: converts scalar timestep to time_dim vector
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Class embedding: learns a vector for each class label
        self.class_embedding = nn.Embedding(num_classes, time_dim)

        # Initial convolutional layer (standard conv, not patch-based)
        self.init_conv = DoubleConv(in_channels, 64, patch_size=1)

        # Encoder (downsampling path) with patch convolutions
        self.down1 = Down(64, 128, emb_dim=time_dim, patch_size=patch_size)
        self.sa1 = SelfAttention(128, num_heads=attention_heads)
        self.down2 = Down(128, 256, emb_dim=time_dim, patch_size=patch_size)
        self.sa2 = SelfAttention(256, num_heads=attention_heads)
        self.down3 = Down(256, 256, emb_dim=time_dim, patch_size=patch_size)
        self.sa3 = SelfAttention(256, num_heads=attention_heads)

        # Bottleneck layers at the lowest resolution (use standard convs)
        self.bottleneck1 = DoubleConv(256, 512, patch_size=1)
        self.bottleneck2 = DoubleConv(512, 512, patch_size=1)
        self.bottleneck3 = DoubleConv(512, 256, patch_size=1)

        # Decoder (upsampling path) with patch convolutions
        self.up1 = Up(512, 128, emb_dim=time_dim, patch_size=patch_size)
        self.sa4 = SelfAttention(128, num_heads=attention_heads)
        self.up2 = Up(256, 64, emb_dim=time_dim, patch_size=patch_size)
        self.sa5 = SelfAttention(64, num_heads=attention_heads)
        self.up3 = Up(128, 64, emb_dim=time_dim, patch_size=patch_size)
        self.sa6 = SelfAttention(64, num_heads=attention_heads)

        # Final output convolution (standard conv)
        self.final_conv = nn.Sequential(
            DoubleConv(64, 64, patch_size=1),
            nn.Conv2d(64, in_channels, kernel_size=1)  # 1x1 conv to map to RGB
        )

        # Refinement module using ResBlocks for enhanced details
        self.refinement = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, self.in_channels, kernel_size=3, padding=1)
        )

        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, noise_level, labels):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]
            noise_level (torch.Tensor): Noise level tensor (timestep in diffusion process)
            labels (torch.Tensor): Class labels for conditioning

        Returns:
            torch.Tensor: Denoised output with same shape as input
        """
        # Store original dimensions for ensuring output size matches input
        orig_height, orig_width = x.shape[2], x.shape[3]

        # Process noise level - handle different formats
        batch_size = x.shape[0]

        # Handle different noise_level formats
        if noise_level.dim() == 0 or (noise_level.dim() == 1 and noise_level.size(0) == 1):
            # It's a scalar or single-element tensor, expand to batch size
            noise_level = noise_level.view(1).expand(batch_size)
        elif noise_level.dim() == 1 and noise_level.size(0) == batch_size:
            # Already batch-sized but needs reshaping
            pass
        elif noise_level.dim() >= 2:
            # Flatten multi-dimensional tensor
            noise_level = noise_level.view(-1)
            if len(noise_level) == 1:
                noise_level = noise_level.expand(batch_size)
            elif len(noise_level) == batch_size:
                pass
            else:
                raise ValueError(f"Noise level shape {noise_level.shape} cannot be matched to batch size {batch_size}")

        # Ensure final shape is [batch_size, 1]
        noise_level = noise_level.view(batch_size, 1)
        t_emb = self.time_mlp(noise_level)

        # Process class label
        c_emb = self.class_embedding(labels)

        # Combine time and class embeddings
        combined_emb = t_emb + c_emb

        # Initial convolution
        x1 = self.init_conv(x)  # [B, 64, H, W]

        # Encoder path with self-attention
        x2 = self.down1(x1, combined_emb)  # [B, 128, H/2, W/2]
        x2 = self.sa1(x2)
        x3 = self.down2(x2, combined_emb)  # [B, 256, H/4, W/4]
        x3 = self.sa2(x3)
        x4 = self.down3(x3, combined_emb)  # [B, 256, H/8, W/8]
        x4 = self.sa3(x4)

        # Bottleneck
        x4 = self.bottleneck1(x4)  # [B, 512, H/8, W/8]
        x4 = self.bottleneck2(x4)  # [B, 512, H/8, W/8]
        x4 = self.bottleneck3(x4)  # [B, 256, H/8, W/8]

        # Decoder path with skip connections and self-attention
        x = self.up1(x4, x3, combined_emb)  # [B, 128, H/4, W/4]
        x = self.sa4(x)
        x = self.up2(x, x2, combined_emb)  # [B, 64, H/2, W/2]
        x = self.sa5(x)
        x = self.up3(x, x1, combined_emb)  # [B, 64, H, W]
        x = self.sa6(x)

        # Final convolution
        output = self.final_conv(x)  # [B, in_channels, H, W]

        # Ensure output size matches input size using interpolation if needed
        if output.shape[2] != orig_height or output.shape[3] != orig_width:
            output = F.interpolate(output, size=(orig_height, orig_width), mode='bilinear', align_corners=True)

        # Apply refinement with residual connection
        refined_output = self.refinement(output) + output

        return refined_output

    def get_attention_maps(self):
        """
        Collect attention maps from all self-attention layers for visualization.

        Returns:
            list: List of attention maps from each self-attention layer
        """
        attention_maps = []
        for module in [self.sa1, self.sa2, self.sa3, self.sa4, self.sa5, self.sa6]:
            if hasattr(module, 'attention_maps'):
                attention_maps.append(module.attention_maps)
        return attention_maps


# Enhanced training with improved schedules and more diffusion steps
if __name__ == '__main__':
    # Initialize model, optimizer, loss function, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Current using device: {device}")

    # Create a U-Net model for CIFAR-10
    model = CIFARUNetDenoiser(
        img_size=img_size,
        in_channels=3,
        num_classes=2,  # Cat and dog
        time_dim=256,
        patch_size=4
    ).to(device)

    # Use AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.02, betas=(0.9, 0.99))

    # Multiple loss functions combined
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()


    # Learning rate scheduler with warmup
    def warmup_cosine_schedule(step, warmup_steps, max_steps):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))


    # Improved diffusion process
    epochs = 140
    num_diffusion_steps = 10  # More steps for better quality
    max_steps = epochs * len(train_loader)
    warmup_steps = max_steps // 10  # 10% warmup

    # Track losses for plotting
    training_losses = []

    # Create output directory for visualizations
    import os

    os.makedirs('epoch_visualizations', exist_ok=True)


    def cosine_similarity_loss(outputs, targets):
        outputs_flat = outputs.view(outputs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        # Normalize
        outputs_norm = F.normalize(outputs_flat, p=2, dim=1)
        targets_norm = F.normalize(targets_flat, p=2, dim=1)
        # Cosine similarity
        cos_sim = (outputs_norm * targets_norm).sum(dim=1)
        # Convert to loss
        return (1 - cos_sim).mean()


    # Function to generate and save visualizations with attention maps
    def generate_visualizations(epoch_num):
        model.eval()
        with torch.no_grad():
            # Create a figure for the generation process
            fig, axes = plt.subplots(2, num_diffusion_steps + 1, figsize=(20, 5))

            # Generate cat and dog images
            for class_idx in range(2):
                # Fix random seed for consistent visualization across epochs
                torch.manual_seed(42 + class_idx)
                denoised_img = torch.randn((1, 3, img_size, img_size)).to(device)
                label = torch.tensor([class_idx], device=device)

                # Plot the starting noise
                noise_img = denoised_img.cpu().permute(0, 2, 3, 1).squeeze()
                noise_img = (noise_img * 0.5 + 0.5).clip(0, 1)  # Denormalize
                axes[class_idx, 0].imshow(noise_img)
                axes[class_idx, 0].set_title(f"Start (Noise)")
                axes[class_idx, 0].axis('off')

                # Perform denoising steps
                for step_idx, t in enumerate(reversed(range(1, num_diffusion_steps + 1))):
                    noise_level = torch.ones(1, 1, device=device) * (t / num_diffusion_steps)
                    denoised_img = model(denoised_img, noise_level, label)

                    # When adding noise during early steps, reduce noise to preserve structure
                    if t > num_diffusion_steps * 0.7:
                        denoised_img = denoised_img + 0.03 * torch.randn_like(denoised_img)  # From 0.05

                    # Show the denoised image
                    img_to_show = denoised_img.cpu().permute(0, 2, 3, 1).squeeze()
                    img_to_show = (img_to_show * 0.5 + 0.5).clip(0, 1)  # Denormalize

                    # Plot the denoised image
                    axes[class_idx, step_idx + 1].imshow(img_to_show)
                    axes[class_idx, step_idx + 1].set_title(f"Step {step_idx + 1}")
                    axes[class_idx, 0].set_ylabel(f"{class_names[class_idx]}", size='large',
                                                  rotation=0, labelpad=40, va='center', ha='right')
                    axes[class_idx, step_idx + 1].axis('off')

            plt.suptitle(f"Cat and Dog Generation Process - Epoch {epoch_num + 1}", fontsize=16)
            plt.subplots_adjust(left=0.1, wspace=0.1, hspace=0.2)
            plt.tight_layout()
            plt.savefig(f'epoch_visualizations/epoch_{epoch_num + 1}_generation.png', dpi=200)
            plt.close()

            # Create a grid of multiple samples
            num_samples = 5
            fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

            for class_idx in range(2):
                for sample_idx in range(num_samples):
                    # Use consistent seeds for each sample across epochs
                    torch.manual_seed(sample_idx * 10 + class_idx)
                    image = torch.randn((1, 3, img_size, img_size)).to(device)
                    label = torch.tensor([class_idx], device=device)

                    # Perform full denoising process
                    for t in reversed(range(1, num_diffusion_steps + 1)):
                        noise_level = torch.ones(1, 1, device=device) * (t / num_diffusion_steps)
                        image = model(image, noise_level, label)

                        # Add some noise early to prevent artifacts
                        if t > num_diffusion_steps * 0.7:
                            image = image + 0.05 * torch.randn_like(image)

                    # Show final image
                    img_to_show = image.cpu().permute(0, 2, 3, 1).squeeze()
                    img_to_show = (img_to_show * 0.5 + 0.5).clip(0, 1)

                    axes[class_idx, sample_idx].imshow(img_to_show)
                    axes[class_idx, sample_idx].axis('off')

                    if sample_idx == 0:
                        axes[class_idx, sample_idx].set_ylabel(f"{class_names[class_idx]}",
                                                               size='large', rotation=0,
                                                               labelpad=40, va='center', ha='right')

            plt.suptitle(f"Generated Cat and Dog Samples - Epoch {epoch_num + 1}", fontsize=16)
            plt.subplots_adjust(left=0.1, wspace=0.1, hspace=0.1)
            plt.tight_layout()
            plt.savefig(f'epoch_visualizations/epoch_{epoch_num + 1}_samples.png', dpi=200)
            plt.close()

        model.train()  # Set back to training mode

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        torch.backends.cudnn.benchmark = True  # Speed up training

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Randomly select diffusion step with importance sampling (focus more on early steps)
            t_importance = np.random.beta(0.5, 0.5)  # Beta distribution for importance sampling
            t = int(t_importance * num_diffusion_steps) + 1
            t = max(1, min(t, num_diffusion_steps))  # Clamp between 1 and num_diffusion_steps

            noise_level = torch.tensor([t / num_diffusion_steps], device=device)

            # Add noise based on noise level with cosine schedule for better image quality
            noise = torch.randn_like(images).to(device)
            alpha = math.cos((t / num_diffusion_steps) * 0.4 * math.pi / 2)
            noisy_images = alpha * images + (1 - alpha) * noise

            optimizer.zero_grad()
            outputs = model(noisy_images, noise_level, labels)

            # Calculate losses
            mse = mse_loss(outputs, images)
            l1 = l1_loss(outputs, images)
            cos_sim = cosine_similarity_loss(outputs, images)

            # Initialize scaling factors
            if step == 0 and epoch == 0:
                # First batch
                mse_scale = mse.item()
                l1_scale = l1.item()
                cos_sim_scale = cos_sim.item()
            else:
                # Smooth update (use a small factor to avoid rapid changes)
                mse_scale = 0.99 * mse_scale + 0.01 * mse.item()
                l1_scale = 0.99 * l1_scale + 0.01 * l1.item()
                cos_sim_scale = 0.99 * cos_sim_scale + 0.01 * cos_sim.item()

            # Scale factors to make losses comparable
            l1_factor = mse_scale / max(l1_scale, 1e-8)
            cos_sim_factor = mse_scale / max(cos_sim_scale, 1e-8)

            # Dynamic weighting with scaling
            # Adjust loss weights for better convergence
            mse_weight = 0.8 - 0.15 * (epoch / epochs)
            l1_weight = 0.2 + 0.15 * (epoch / epochs)

            # Combined loss
            loss = (mse_weight * mse) + (l1_weight * l1 * l1_factor)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update learning rate with warmup cosine schedule
            global_step = epoch * len(train_loader) + step
            lr_scale = warmup_cosine_schedule(global_step, warmup_steps, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scale * 8e-5  # Updated base lr to match earlier recommendation

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Generate and save visualizations after each epoch
        generate_visualizations(epoch)

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'cifar10_catdog_diffusion_epoch_{epoch + 1}.pt')

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()


    # Function to visualize attention through the denoising process
    def visualize_attention_through_denoising():
        print("Attention visualization has been disabled to avoid errors.")
        # You can re-enable this function later when the attention issues are fixed
        pass


    # Generate final visualizations with more samples
    with torch.no_grad():
        print("Skipping attention visualization due to dimension mismatch.")

        # Create a summary grid with samples from different epochs
        try:
            import imageio
            import glob

            # Create a figure comparing first, middle and final epoch results
            epochs_to_show = [0, epochs // 2, epochs - 1]
            fig, axes = plt.subplots(2, len(epochs_to_show), figsize=(15, 6))

            for i, epoch_num in enumerate(epochs_to_show):
                # Load the sample images if they exist
                epoch_file = f'epoch_visualizations/epoch_{epoch_num + 1}_samples.png'
                if os.path.exists(epoch_file):
                    img = imageio.imread(epoch_file)

                    # Display the image in the appropriate subplot
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f"Epoch {epoch_num + 1}")
                    axes[0, i].axis('off')

                    # For the first column, add class labels
                    if i == 0:
                        axes[0, i].set_ylabel("Cats", size='large', rotation=0,
                                              labelpad=40, va='center', ha='right')
                        axes[1, i].set_ylabel("Dogs", size='large', rotation=0,
                                              labelpad=40, va='center', ha='right')

            plt.suptitle("Training Progress Comparison", fontsize=16)
            plt.tight_layout()
            plt.savefig('training_progress_comparison.png', dpi=300)
            plt.close()

            # Create a GIF from all epoch visualizations
            sample_images = []
            for epoch_num in range(epochs):
                if os.path.exists(f'epoch_visualizations/epoch_{epoch_num + 1}_samples.png'):
                    img = imageio.imread(
                        f'epoch_visualizations/epoch_{epoch_num + 1}_samples.png')
                    sample_images.append(img)

            if sample_images:
                imageio.mimsave('training_progress.gif', sample_images, fps=3)
                print("Created animation GIF of training progress")

        except Exception as e:
            print(f"Could not create summary visualization: {e}")

        # Generate a large grid of final samples
        num_samples = 8
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))

        for class_idx in range(2):
            for sample_idx in range(num_samples):
                # Use different seeds for variety
                torch.manual_seed(sample_idx * 10 + class_idx * 100)
                image = torch.randn((1, 3, img_size, img_size)).to(device)
                label = torch.tensor([class_idx], device=device)

                # Perform full denoising process
                for t in reversed(range(1, num_diffusion_steps + 1)):
                    noise_level = torch.tensor([[t / num_diffusion_steps]], device=device)
                    image = model(image, noise_level, label)

                    # Add some noise early to prevent artifacts
                    if t > num_diffusion_steps * 0.7:
                        image = image + 0.05 * torch.randn_like(image)

                # Show final image
                img_to_show = image.cpu().permute(0, 2, 3, 1).squeeze()
                img_to_show = (img_to_show * 0.5 + 0.5).clip(0, 1)

                axes[class_idx, sample_idx].imshow(img_to_show)
                axes[class_idx, sample_idx].axis('off')

                if sample_idx == 0:
                    axes[class_idx, sample_idx].set_ylabel(f"{class_names[class_idx]}",
                                                           size='large', rotation=0,
                                                           labelpad=40, va='center', ha='right')

        plt.suptitle("Final Generated Cat and Dog Samples", fontsize=16)
        plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
        plt.tight_layout()
        plt.savefig('final_cifar10_catdog_samples.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Create a visualization that shows the step-by-step denoising process
        steps_to_show = [0, 1, 2, 3, 5, 7, 10, 15, 19]  # Selected steps to show
        fig, axes = plt.subplots(2, len(steps_to_show), figsize=(20, 6))

        for class_idx in range(2):
            # Use consistent seed for this visualization
            torch.manual_seed(42 + class_idx)
            denoised_img = torch.randn((1, 3, img_size, img_size)).to(device)
            label = torch.tensor([class_idx], device=device)

            # Store intermediate results
            step_results = []

            # Run through all diffusion steps
            for t in reversed(range(1, num_diffusion_steps + 1)):
                noise_level = torch.tensor([[t / num_diffusion_steps]], device=device)
                denoised_img = model(denoised_img, noise_level, label)

                # Add some residual noise for early steps
                if t > num_diffusion_steps * 0.7:
                    denoised_img = denoised_img + 0.1 * torch.randn_like(denoised_img)

                # Save step result
                if num_diffusion_steps - t in steps_to_show:
                    img_to_show = denoised_img.cpu().permute(0, 2, 3, 1).squeeze()
                    img_to_show = (img_to_show * 0.5 + 0.5).clip(0, 1)
                    step_results.append(img_to_show)

            # Add the initial noise as first step
            noise_img = torch.randn((1, 3, img_size, img_size)).to(device)
            noise_img = noise_img.cpu().permute(0, 2, 3, 1).squeeze()
            noise_img = (noise_img * 0.5 + 0.5).clip(0, 1)
            step_results.insert(0, noise_img)

            # Plot each selected step
            for i, img in enumerate(step_results):
                axes[class_idx, i].imshow(img)
                if i == 0:
                    step_label = "Noise"
                else:
                    step_label = f"Step {steps_to_show[i]}"
                axes[class_idx, i].set_title(step_label)
                axes[class_idx, i].axis('off')

                if i == 0:
                    axes[class_idx, i].set_ylabel(f"{class_names[class_idx]}",
                                                  size='large', rotation=0,
                                                  labelpad=40, va='center', ha='right')

        plt.suptitle("Detailed Denoising Process", fontsize=16)
        plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
        plt.tight_layout()
        plt.savefig('detailed_denoising_process.png', dpi=300, bbox_inches='tight')
