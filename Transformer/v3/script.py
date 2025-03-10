import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import copy


# Sinusoidal positional embedding
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # we have 2D grids of shape (2, grid_size, grid_size)

    # Flatten the grids
    grid = np.stack(grid, axis=0)  # (2, grid_size, grid_size)
    grid = grid.reshape([2, 1, grid_size, grid_size])  # (2, 1, grid_size, grid_size)

    # Get dimensions for sin/cos embedding
    # Each dimension uses (embed_dim // 4) dimensions
    dim_per_component = embed_dim // 4

    # Create position embedding for each dimension
    pos_embed = np.zeros((1, grid_size, grid_size, embed_dim), dtype=np.float32)

    # Create frequency bases
    omega = np.arange(dim_per_component, dtype=np.float32)
    omega /= dim_per_component
    omega = 1. / (10000 ** omega)

    # Apply sin and cos to positions
    out_h = grid[0] * omega.reshape([1, dim_per_component, 1, 1])  # (1, dim_per_component, grid_size, grid_size)
    out_w = grid[1] * omega.reshape([1, dim_per_component, 1, 1])  # (1, dim_per_component, grid_size, grid_size)

    # Apply sin and cos to different dimensions
    # First quarter: sin on height
    pos_embed[0, :, :, 0:dim_per_component] = np.sin(out_h).transpose([2, 3, 0, 1]).reshape(
        [grid_size, grid_size, dim_per_component])
    # Second quarter: cos on height
    pos_embed[0, :, :, dim_per_component:2 * dim_per_component] = np.cos(out_h).transpose([2, 3, 0, 1]).reshape(
        [grid_size, grid_size, dim_per_component])
    # Third quarter: sin on width
    pos_embed[0, :, :, 2 * dim_per_component:3 * dim_per_component] = np.sin(out_w).transpose([2, 3, 0, 1]).reshape(
        [grid_size, grid_size, dim_per_component])
    # Fourth quarter: cos on width
    pos_embed[0, :, :, 3 * dim_per_component:] = np.cos(out_w).transpose([2, 3, 0, 1]).reshape(
        [grid_size, grid_size, dim_per_component])

    # Reshape to final form
    pos_embed = pos_embed.reshape([1, grid_size * grid_size, embed_dim])

    return torch.from_numpy(pos_embed)


# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
np.random.seed(42)

# Data loading and preprocessing for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB images
])

# Load CIFAR-10 dataset
cifar_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Filter only cats (class 3) and dogs (class 5)
cat_indices = [i for i, (_, label) in enumerate(cifar_full) if label == 3]
dog_indices = [i for i, (_, label) in enumerate(cifar_full) if label == 5]
pet_indices = cat_indices + dog_indices

print(f"Found {len(cat_indices)} cats and {len(dog_indices)} dogs in the training set")

# Create a new dataset with only cats and dogs
pet_subset = Subset(cifar_full, pet_indices)


# Map original labels (3 and 5) to new labels (0 for cat, 1 for dog)
class PetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Map labels: 3 (cat) -> 0, 5 (dog) -> 1
        if label == 3:
            return img, 0  # Cat
        else:
            return img, 1  # Dog


# Create the pet dataset
pet_dataset = PetDataset(pet_subset)

# Split into train/validation sets
train_size = int(0.9 * len(pet_dataset))
val_size = len(pet_dataset) - train_size
train_dataset, val_dataset = random_split(pet_dataset, [train_size, val_size])

print(f"Training set: {len(train_dataset)} samples, Validation set: {len(val_dataset)} samples")

# Smaller batch size to reduce memory requirements
batch_size = 64  # Reduced from 128 to be more memory-friendly on Mac
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2 if torch.cuda.is_available() else 0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Test dataset (CIFAR-10 test set, filtered for cats and dogs)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_cat_indices = [i for i, (_, label) in enumerate(cifar_test) if label == 3]
test_dog_indices = [i for i, (_, label) in enumerate(cifar_test) if label == 5]
test_pet_indices = test_cat_indices + test_dog_indices
test_pet_subset = Subset(cifar_test, test_pet_indices)
test_pet_dataset = PetDataset(test_pet_subset)
test_loader = DataLoader(test_pet_dataset, batch_size=batch_size, shuffle=False)

print(f"Test set: {len(test_pet_dataset)} samples")


# IMPROVEMENT 1: Added Residual Block for the decoder
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


# IMPROVEMENT 2: Added ResUpBlock for better information flow in decoder
class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip = self.skip(x)
        skip = self.upsample(skip)

        x = self.act(self.norm1(self.conv1(x)))
        x = self.upsample(x)
        x = self.act(self.norm2(self.conv2(x)))

        return x + skip


# Linear Attention implementation - refined for better stability
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # Feature map function φ(x) = elu(x) + 1
        self.feature_map = lambda x: F.elu(x) + 1.0

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        # Get queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        # Scale queries
        q = q * self.scale

        # Apply feature map to queries and keys
        q = self.feature_map(q)
        k = self.feature_map(k)

        # Linear attention computation
        k_cumsum = k.sum(dim=-2)
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)

        # Add epsilon to avoid division by zero
        denominator = torch.einsum('bhnd,bhd->bhn', q, k_cumsum).unsqueeze(-1) + 1e-8
        out = out / denominator

        # Merge heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


# Linear Transformer Encoder Layer
class LinearTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu",
                 batch_first=True, norm_first=True):
        super().__init__()
        self.self_attn = LinearAttention(d_model, heads=nhead, dim_head=d_model // nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu
        self.norm_first = norm_first
        self.batch_first = batch_first

    def forward(self, src):
        if self.norm_first:
            src = src + self._sa_block(self.norm1(src))
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(src + self._sa_block(src))
            src = self.norm2(src + self._ff_block(src))

        return src

    def _sa_block(self, x):
        x = self.self_attn(x)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# Linear Transformer Encoder
class LinearTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


# Patch Embedding module for CIFAR-10 (32x32 images)
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=2, embed_dim=704):
        super().__init__()
        self.patch_size = patch_size
        # Number of patches for 32x32 image with patch_size=4: (32/4)*(32/4) = 64 patches
        num_patches = (32 // patch_size) ** 2

        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # LayerNorm after patch embedding
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Project and flatten patches: (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        # Rearrange to (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        # Apply LayerNorm
        x = self.norm(x)
        return x


# IMPROVEMENT 3: Refined CIFAR Pet Denoiser with improved architecture
class ImprovedCIFARPetDenoiser(nn.Module):
    def __init__(self, num_classes=2, embed_dim=1280, num_heads=10, num_layers=5, dropout=0.1, patch_size=2):
        super(ImprovedCIFARPetDenoiser, self).__init__()

        # Enhanced label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Noise level embedding with smoother progression
        self.noise_level_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Replace feature extractor with patch embedding
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size, embed_dim=embed_dim)

        # Number of patches for 32x32 image with patch_size=4: (32/4)*(32/4) = 64 patches
        self.num_patches = (32 // patch_size) ** 2

        # Grid size for positional embedding (8x8 for patch_size=4)
        self.grid_size = 32 // patch_size

        # Use sinusoidal positional embedding instead of learned
        self.register_buffer('positional_embedding', get_2d_sincos_pos_embed(embed_dim, self.grid_size))

        # Transformer encoder with linear attention
        encoder_layer = LinearTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = LinearTransformerEncoder(encoder_layer, num_layers=num_layers)

        # IMPROVEMENT 4: Refined decoder with residual connections for better feature preservation
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            ResidualBlock(256),  # Added residual block

            # Only need one upsampling block now (16x16 -> 32x32)
            ResUpBlock(256, 64, scale_factor=2),  # 16x16 -> 32x32
            ResidualBlock(64),  # Added residual block

            # Final adjustment layer
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output normalized RGB values
        )

    def forward(self, x, noise_level, labels):
        B, C, H, W = x.shape

        # Apply patch embedding instead of feature extraction
        features = self.patch_embedding(x)  # [B, num_patches, embed_dim]

        # Add positional embeddings
        features = features + self.positional_embedding

        # Noise level embedding
        noise_embed = self.noise_level_embedding(noise_level.view(B, 1))  # [B, embed_dim]
        noise_embed = noise_embed.unsqueeze(1).expand(-1, self.num_patches, -1)  # [B, num_patches, embed_dim]

        # Label embedding
        label_embed = self.label_embedding(labels)  # [B, embed_dim]
        label_embed = self.label_proj(label_embed)  # [B, embed_dim]
        label_embed = label_embed.unsqueeze(1).expand(-1, self.num_patches, -1)  # [B, num_patches, embed_dim]

        # Combine all inputs
        transformer_input = features + noise_embed + label_embed

        # Apply Transformer
        transformer_output = self.transformer_encoder(transformer_input)

        # Reshape back to spatial dimensions
        patch_grid_size = int(math.sqrt(self.num_patches))
        spatial_features = transformer_output.permute(0, 2, 1).reshape(B, -1, patch_grid_size, patch_grid_size)

        # Decode to original image size
        output = self.decoder(spatial_features)

        return output


# IMPROVEMENT 5: Balanced loss function
def balanced_adaptive_loss(outputs, targets, noise_level, total_steps=1.0):
    # Calculate basic losses
    mse = F.mse_loss(outputs, targets)
    l1 = F.l1_loss(outputs, targets)

    # Extract high-frequency components for loss calculation
    high_freq = high_frequency_loss(outputs, targets)
    edge = edge_preserving_loss(outputs, targets)

    # Calculate cosine similarity for directional consistency
    cosine = cosine_similarity_loss(outputs, targets)

    # More balanced step progression
    step_progress = 1.0 - noise_level.mean().item()

    sigmoid_progress = 1.0 / (1.0 + math.exp(-10 * (step_progress - 0.5)))

    mse_weight = 0.3 - 0.1 * sigmoid_progress

    l1_weight = 0.25 + 0.05 * sigmoid_progress

    cosine_weight = 0.2 - 0.05 * sigmoid_progress

    high_freq_weight = 0.12 + 0.08 * sigmoid_progress

    edge_weight = 0.13 + 0.07 * sigmoid_progress

    # Normalize weights
    total_weight = mse_weight + l1_weight + cosine_weight + high_freq_weight + edge_weight
    mse_weight /= total_weight
    l1_weight /= total_weight
    cosine_weight /= total_weight
    high_freq_weight /= total_weight
    edge_weight /= total_weight

    return (mse_weight * mse +
            l1_weight * l1 +
            cosine_weight * cosine +
            high_freq_weight * high_freq +
            edge_weight * edge)


# These helper loss functions remain the same
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


def high_frequency_loss(outputs, targets):
    channels = outputs.shape[1]

    base_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32).view(1, 1, 3, 3).to(outputs.device)

    laplacian_kernel = torch.zeros(channels, 1, 3, 3, device=outputs.device)
    for i in range(channels):
        laplacian_kernel[i] = base_kernel

    outputs_high = F.conv2d(outputs, laplacian_kernel, padding=1, groups=channels)
    targets_high = F.conv2d(targets, laplacian_kernel, padding=1, groups=channels)

    return F.l1_loss(outputs_high, targets_high)


def edge_preserving_loss(outputs, targets):
    # Sobel operator for edge detection
    channels = outputs.shape[1]

    # Create horizontal and vertical Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3).to(outputs.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32).view(1, 1, 3, 3).to(outputs.device)

    # Create filters for each channel
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)

    # Calculate image gradients
    outputs_grad_x = F.conv2d(outputs, sobel_x, padding=1, groups=channels)
    outputs_grad_y = F.conv2d(outputs, sobel_y, padding=1, groups=channels)
    targets_grad_x = F.conv2d(targets, sobel_x, padding=1, groups=channels)
    targets_grad_y = F.conv2d(targets, sobel_y, padding=1, groups=channels)

    # Calculate gradient magnitude
    outputs_grad = torch.sqrt(outputs_grad_x ** 2 + outputs_grad_y ** 2 + 1e-6)
    targets_grad = torch.sqrt(targets_grad_x ** 2 + targets_grad_y ** 2 + 1e-6)

    # Use L1 loss for gradient differences
    return F.l1_loss(outputs_grad, targets_grad)


# IMPROVEMENT 6: Denormalize image function remains the same
def denormalize_image(img_tensor):
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize and clip
    return img


# IMPROVEMENT 7: Improved sampling with classifier-free guidance
def improved_sampling(model, device, class_idx, num_steps=40, guidance_scale=3.0, seed=None):
    """Improved sampling with stronger classifier-free guidance and adaptive scaling

    Args:
        model: The diffusion model
        device: Device to run inference on
        class_idx: Class index (0 for cat, 1 for dog)
        num_steps: Number of diffusion steps (higher = better quality, slower)
        guidance_scale: Base guidance scale strength (higher = stronger class features but less diversity)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (final generated image, list of all intermediate images)
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    else:
        torch.manual_seed(42 + class_idx)

    # Start with random noise
    img = torch.randn((1, 3, 32, 32)).to(device)
    label = torch.tensor([class_idx], device=device)

    # Store intermediate results
    intermediates = [img.clone()]

    # Define adaptive guidance scaling function
    def adaptive_scale(t, total_steps=num_steps):
        # Early steps: use minimal guidance to establish basic structure
        if t > total_steps * 0.8:  # First 20% of steps
            return guidance_scale * 0.5
        # Middle steps: use standard guidance for class conditioning
        elif t > total_steps * 0.3:  # Next 50% of steps
            return guidance_scale
        # Late steps: increase guidance to enhance class-specific features
        elif t > total_steps * 0.1:  # Next 20% of steps
            return guidance_scale * 1.5
        # Final steps: strongest guidance for feature refinement
        else:  # Last 10% of steps
            return guidance_scale * 2.0

    # Use more steps for smoother progression
    for t in reversed(range(1, num_steps + 1)):
        # Normalized noise level in [0,1]
        noise_level = torch.tensor([[[[t / num_steps]]]], device=device)

        # Get model prediction
        with torch.no_grad():
            pred = model(img, noise_level, label)

        # Apply classifier-free guidance adaptively
        if t < num_steps * 0.95:  # Skip guidance in very early steps
            # Get unconditional prediction (use opposite class for stronger contrast)
            unconditional_label = torch.tensor([1 - class_idx], device=device)
            with torch.no_grad():
                unconditional_pred = model(img, noise_level, unconditional_label)

            # Calculate current guidance scale based on diffusion step
            current_scale = adaptive_scale(t)

            # Mix conditional and unconditional predictions with adaptive scaling
            pred = unconditional_pred + current_scale * (pred - unconditional_pred)

        # Small correction near the end to prevent over-sharpening
        if t < num_steps * 0.05:
            # Blend with a tiny bit of previous step to stabilize very end of generation
            blend_factor = 0.95
            img = blend_factor * pred + (1 - blend_factor) * img
        else:
            # Controlled step size based on current noise level
            alpha = 0.9 if t > num_steps * 0.7 else 1.0  # Smaller steps early on
            img = alpha * pred + (1 - alpha) * img

        # Save intermediate
        intermediates.append(img.clone())

    return img, intermediates


# IMPROVEMENT 8: Updated visualization function that uses improved sampling
def visualize_improved_samples(model, device, epoch, num_steps=10, save_path=None):
    model.eval()
    class_labels = ['Cat', 'Dog']

    num_samples_per_class = 3

    # Create larger figure with better spacing
    fig, axes = plt.subplots(num_samples_per_class * len(class_labels), num_steps + 1,
                             figsize=(20, 3 * num_samples_per_class * len(class_labels)))

    with torch.no_grad():
        sample_idx = 0
        for class_idx in range(len(class_labels)):
            for sample in range(num_samples_per_class):
                # Use different seeds for different samples
                seed = 42 + sample * 10 + class_idx

                # Generate sample using improved sampling
                _, intermediates = improved_sampling(
                    model, device, class_idx, num_steps=num_steps,
                    guidance_scale=1.2, seed=seed
                )

                # Display all intermediates
                for step, img in enumerate(intermediates):
                    axes[sample_idx, step].imshow(denormalize_image(img[0]))
                    if sample == 0 and step == 0:
                        axes[sample_idx, step].set_title("Initial Noise")
                    elif sample == 0:
                        axes[sample_idx, step].set_title(f"Step {step}")
                    axes[sample_idx, step].axis('off')

                # Add class label
                if sample == 0:
                    axes[sample_idx, 0].set_ylabel(class_labels[class_idx], fontsize=14)

                sample_idx += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/improved_samples_epoch_{epoch}.png")

    # Show the figure before closing it
    plt.show()
    plt.close()


# IMPROVEMENT 9: Evaluation function remains largely the same but uses new loss
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Use varying noise levels for evaluation
            noise_level = (0.5 * torch.rand(images.size(0), 1, 1, 1) + 0.1).to(device)
            noisy_images = images + torch.randn_like(images).to(device) * noise_level

            outputs = model(noisy_images, noise_level, labels)
            loss = balanced_adaptive_loss(outputs, images, noise_level)

            total_loss += loss.item()

    return total_loss / len(dataloader)


# IMPROVEMENT 10: Added visualization of comparison between steps
def visualize_step_comparison(model, device, save_path=None):
    model.eval()
    class_labels = ['Cat', 'Dog']
    step_options = [5, 10, 20, 40]  # Different numbers of diffusion steps

    fig, axes = plt.subplots(len(class_labels), len(step_options), figsize=(16, 6))

    with torch.no_grad():
        for i, class_idx in enumerate(range(len(class_labels))):
            # Use same random seed for all step counts (fair comparison)
            torch.manual_seed(42 + class_idx)

            for j, steps in enumerate(step_options):
                # Generate using improved sampling with the same seed
                denoised_img, _ = improved_sampling(
                    model, device, class_idx, num_steps=steps,
                    guidance_scale=1.2, seed=42 + class_idx
                )

                # Display final result
                axes[i, j].imshow(denormalize_image(denoised_img[0]))

                if i == 0:
                    axes[i, j].set_title(f"{steps} steps", fontsize=14)

                if j == 0:
                    axes[i, j].set_ylabel(class_labels[class_idx], fontsize=14)

                axes[i, j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/diffusion_steps_comparison.png")
    plt.close()

# Function to update the EMA model
def update_ema_model(model, ema_model, decay):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            if param.requires_grad:
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

# Main training function
def train_diffusion_model():
    # Device setup with MPS for Mac Pro
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_dir = './improved_cifar_pet_denoiser'
    os.makedirs(save_dir, exist_ok=True)

    # Create model instances (main and EMA)
    model = ImprovedCIFARPetDenoiser().to(device)
    ema_model = copy.deepcopy(model)
    ema_decay = 0.995

    # Print model size
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {param_count:,}")

    # Optimization settings
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

    # Training settings - reduced epochs for faster training
    epochs = 40  # Reduced from 60 to be more time-efficient on Mac
    warmup_epochs = 3

    # Use cosine annealing LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Enable gradient clipping
    max_grad_norm = 1.0

    # Training and validation history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Training loop
    print("Starting training...")
    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            # Progress tracking
            num_batches = len(train_loader)
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            for i, (images, labels) in enumerate(train_loader):
                # Move data to device
                images, labels = images.to(device), labels.to(device)

                # IMPROVEMENT: Better noise scheduling during training
                if epoch < warmup_epochs:
                    # Focus on low-noise steps during warmup
                    noise_level = (0.3 * torch.rand(images.size(0), 1, 1, 1) + 0.05).to(device)
                else:
                    # Use more balanced approach for noise levels
                    beta = min(0.5, epoch / epochs)  # Gradually increase emphasis on difficult steps
                    if torch.rand(1).item() < beta:
                        # Focus on mid-range noise (representing mid-steps)
                        noise_level = (0.3 * torch.rand(images.size(0), 1, 1, 1) + 0.35).to(device)
                    else:
                        # Use full range
                        noise_level = torch.rand(images.size(0), 1, 1, 1).to(device)

                # Add noise to images based on noise level
                noise = torch.randn_like(images).to(device) * noise_level
                noisy_images = images + noise

                # Forward pass
                optimizer.zero_grad()
                outputs = model(noisy_images, noise_level, labels)

                # Compute loss using balanced loss function
                loss = balanced_adaptive_loss(outputs, images, noise_level)

                # Backward pass and optimize
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                # Update EMA model
                update_ema_model(model, ema_model, ema_decay)

                # Track progress
                running_loss += loss.item()

                # Print batch progress
                if (i + 1) % 20 == 0 or (i + 1) == num_batches:
                    print(f"  Batch {i + 1}/{num_batches} - Loss: {loss.item():.4f}")

            # Step the learning rate scheduler
            scheduler.step()

            # Calculate average training loss
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validate the model using EMA weights
            print("Validating...")
            val_loss = evaluate(ema_model, val_loader, device)
            val_losses.append(val_loss)

            # Print epoch summary
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # Save best model (using EMA weights)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss
                }, f"{save_dir}/best_model.pt")
                print(f"  New best model saved with val loss: {val_loss:.4f}")

            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss
                }, f"{save_dir}/checkpoint_epoch_{epoch + 1}.pt")
                print(f"  Checkpoint saved at epoch {epoch + 1}")

            # Visualize samples periodically
            if (epoch + 1) % 1 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Generating visualization samples...")
                visualize_improved_samples(ema_model, device, epoch + 1, num_steps=10, save_path=save_dir)

        # Final evaluation and visualization
        print("\nTraining complete. Generating final visualizations...")

        # Evaluate on test set
        test_loss = evaluate(ema_model, test_loader, device)
        print(f"Final test loss: {test_loss:.4f}")

        # Generate samples with different numbers of steps
        visualize_step_comparison(ema_model, device, save_path=save_dir)

        # Generate final sample visualizations
        visualize_improved_samples(ema_model, device, epochs, num_steps=40, save_path=save_dir)

        print(f"All results saved to {save_dir}")
        return ema_model, save_dir

    except Exception as e:
        print(f"Error during training: {e}")
        # Save emergency checkpoint
        torch.save({
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict() if 'ema_model' in locals() else None,
            'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
        }, f"{save_dir}/emergency_checkpoint.pt")
        print(f"Emergency checkpoint saved to {save_dir}/emergency_checkpoint.pt")
        raise


# For inference only after loading a trained model
def generate_samples(model_path, device="auto", num_samples=5, num_steps=40):
    # Set device
    if device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else
                              'cpu')

    print(f"Using device: {device}")

    # Create model and load weights
    model = ImprovedCIFARPetDenoiser().to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # Try to load EMA weights first, fall back to regular weights
    if 'ema_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_model_state_dict'])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")

    model.eval()

    # Generate samples
    print("Generating samples...")

    class_labels = ['Cat', 'Dog']
    samples = []

    for class_idx in range(len(class_labels)):
        class_samples = []
        for i in range(num_samples):
            sample, _ = improved_sampling(
                model, device, class_idx, num_steps=num_steps,
                guidance_scale=1.5, seed=1000 + i
            )
            class_samples.append(sample)
        samples.append(class_samples)

    # Display samples
    fig, axes = plt.subplots(len(class_labels), num_samples, figsize=(num_samples * 3, 6))

    for i, class_samples in enumerate(samples):
        for j, sample in enumerate(class_samples):
            img = denormalize_image(sample[0])
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(class_labels[i], fontsize=14)

    plt.tight_layout()
    plt.savefig(f"generated_samples.png")
    plt.show()

    print("Done! Samples saved to generated_samples.png")

    return samples


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or use CIFAR Pet Diffusion model")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                        help='Mode: train or generate')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (for generate mode)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate per class')
    parser.add_argument('--steps', type=int, default=40,
                        help='Number of diffusion steps for generation')

    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting model training...")
        train_diffusion_model()
    elif args.mode == 'generate':
        if args.model_path is None:
            print("Error: --model_path is required for generate mode")
        else:
            generate_samples(args.model_path, num_samples=args.num_samples, num_steps=args.steps)
    else:
        print(f"Unknown mode: {args.mode}")
