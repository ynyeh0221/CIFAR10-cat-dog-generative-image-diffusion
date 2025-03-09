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

# Sinusoidal positional embedding
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Create 2D sine-cosine positional embedding.

    Args:
        embed_dim: Embedding dimension (must be divisible by 4 since we have 2D with sin/cos each)
        grid_size: Int - grid height and width

    Returns:
        Positional embedding of shape (grid_size*grid_size, embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # Here, we have 2D grids of shape (2, grid_size, grid_size)

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
batch_size = 128
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


# Linear Attention implementation
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
        out = out / (torch.einsum('bhnd,bhd->bhn', q, k_cumsum).unsqueeze(-1) + 1e-8)

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
    def __init__(self, in_channels=3, patch_size=4, embed_dim=704):
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


# Optimized Conditional Transformer Denoiser with Linear Attention for CIFAR-10 cats and dogs
class CIFARPetDenoiser(nn.Module):
    def __init__(self, num_classes=2, embed_dim=1280, num_heads=10, num_layers=5, dropout=0.12, patch_size=4):
        super(CIFARPetDenoiser, self).__init__()

        # Enhanced label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Noise level embedding
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

        # Positional embedding for patches
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

        # Decoder for CIFAR-10 (upsampling back to 32x32x3)
        # With patch_size=4, the spatial size after patch embedding is 8x8
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # Final adjustment at 32x32
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

        # Label embedding - fixed simpler approach
        label_embed = self.label_embedding(labels)  # [B, embed_dim]
        label_embed = self.label_proj(label_embed)  # [B, embed_dim]
        label_embed = label_embed.unsqueeze(1).expand(-1, self.num_patches, -1)  # [B, num_patches, embed_dim]

        # Combine all inputs
        transformer_input = features + noise_embed + label_embed

        # Apply Transformer
        transformer_output = self.transformer_encoder(transformer_input)

        # Reshape back to spatial dimensions
        # For 4x4 patches, we have 8x8=64 patches, and want to reshape to 8x8 spatial grid
        patch_grid_size = int(math.sqrt(self.num_patches))
        spatial_features = transformer_output.permute(0, 2, 1).reshape(B, -1, patch_grid_size, patch_grid_size)

        # Decode to original image size
        output = self.decoder(spatial_features)

        return output


# Helper function to convert normalized tensor to displayable image
def denormalize_image(img_tensor):
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize and clip
    return img


# Visualization function for RGB images
def visualize_samples(model, device, epoch, save_path=None):
    model.eval()
    class_labels = ['Cat', 'Dog']

    num_samples_per_class = 3
    num_diffusion_steps = 10

    fig, axes = plt.subplots(num_samples_per_class * len(class_labels), num_diffusion_steps + 1,
                             figsize=(15, 3 * num_samples_per_class * len(class_labels)))

    with torch.no_grad():
        sample_idx = 0
        for class_idx in range(len(class_labels)):
            for sample in range(num_samples_per_class):
                # Use different seeds for different samples
                torch.manual_seed(42 + sample * 10 + class_idx)

                # Start with random noise
                denoised_img = torch.randn((1, 3, 32, 32)).to(device)
                label = torch.tensor([class_idx], device=device)

                # Show initial noise
                axes[sample_idx, 0].imshow(denormalize_image(denoised_img[0]))
                if sample == 0:
                    axes[sample_idx, 0].set_title("Initial Noise")
                axes[sample_idx, 0].axis('off')

                # Sequential denoising
                for t in reversed(range(1, num_diffusion_steps + 1)):
                    noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)
                    denoised_img = model(denoised_img, noise_level, label)

                    axes[sample_idx, num_diffusion_steps - t + 1].imshow(denormalize_image(denoised_img[0]))
                    if sample == 0 and class_idx == 0:
                        axes[sample_idx, num_diffusion_steps - t + 1].set_title(f"Step {num_diffusion_steps - t + 1}")

                    axes[sample_idx, num_diffusion_steps - t + 1].axis('off')

                # Add class label to the right of the final image
                if num_diffusion_steps - 0 + 1 == num_diffusion_steps:
                    axes[sample_idx, -1].text(1.05, 0.5, f"{class_labels[class_idx]} {sample + 1}",
                                              transform=axes[sample_idx, -1].transAxes,
                                              fontsize=10, verticalalignment='center')

                sample_idx += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/samples_epoch_{epoch}.png")
    plt.show()


# Evaluation function
def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    total_loss = 0

    if criterion is None:
        criterion = nn.MSELoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Fixed mid-level noise for consistent evaluation
            noise_level = torch.ones(images.size(0), 1, 1, 1).to(device) * 0.3
            noisy_images = images + torch.randn_like(images).to(device) * noise_level

            outputs = model(noisy_images, noise_level, labels)
            loss = criterion(outputs, images)

            total_loss += loss.item()

    return total_loss / len(dataloader)


# Custom loss functions
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


def combined_loss(outputs, targets, mse_weight=0.65, l1_weight=0.15, cosine_weight=0.2):
    mse = F.mse_loss(outputs, targets)
    l1 = F.l1_loss(outputs, targets)
    cosine = cosine_similarity_loss(outputs, targets)

    return mse_weight * mse + l1_weight * l1 + cosine_weight * cosine


# Device setup with MPS for Mac Pro
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else
                      'cpu')
print(f"Using device: {device}")

# Create model instance (reduced size for Mac)
model = CIFARPetDenoiser().to(device)
print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Optimization settings
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Training settings
epochs = 60
warmup_epochs = 3
reset_optimizer_epochs = [20, 40]


# Learning rate scheduler with warm-up - FIXED PATH
def get_scheduler(optimizer, epochs, steps_per_epoch):
    return torch.optim.lr_scheduler.OneCycleLR(  # Fixed path here
        optimizer,
        max_lr=4e-4,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1000.0
    )


scheduler = get_scheduler(optimizer, epochs, len(train_loader))

# Gradient clipping
max_grad_norm = 1.0

# Training and validation history
train_losses = []
val_losses = []
best_val_loss = float('inf')

# Create directory for saving
save_dir = './cifar_pet_denoiser_results'
os.makedirs(save_dir, exist_ok=True)

# Training loop
print("Starting training...")
try:
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Reset optimizer if at specified epoch
        if epoch in reset_optimizer_epochs:
            print(f"Resetting optimizer state at epoch {epoch + 1}")
            optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            scheduler = get_scheduler(optimizer, epochs - epoch, len(train_loader))

        for i, (images, labels) in enumerate(train_loader):
            # Check for NaN or Inf values in input
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"Warning: NaN or Inf detected in input images at batch {i}")
                continue

            images, labels = images.to(device), labels.to(device)

            # Progressive noise strategy
            if epoch < warmup_epochs:
                noise_level = (0.2 * torch.rand(images.size(0), 1, 1, 1) + 0.05).to(device)
            else:
                noise_level = (0.8 * torch.rand(images.size(0), 1, 1, 1) + 0.1).to(device)

            noise = torch.randn_like(images).to(device) * noise_level
            noisy_images = images + noise

            optimizer.zero_grad()
            outputs = model(noisy_images, noise_level, labels)

            # Check for NaN or Inf values in output
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN or Inf detected in outputs at batch {i}, skipping...")
                continue

            loss = combined_loss(outputs, images)

            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf detected in loss at batch {i}, skipping...")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Print progress every 50 batches
            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate the model
        val_loss = evaluate(model, val_loader, device, criterion=lambda x, y: combined_loss(x, y))
        val_losses.append(val_loss)

        # Print progress
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

        # Save checkpoint every 10 epochs for safety
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, f"{save_dir}/checkpoint_epoch_{epoch + 1}.pt")

        # Periodically visualize results
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
            visualize_samples(model, device, epoch + 1, save_path=save_dir)

except Exception as e:
    print(f"Error during training: {e}")
    # Save checkpoint on error
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }, f"{save_dir}/emergency_checkpoint.pt")
    raise

# Plot training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(f"{save_dir}/loss_curves.png")
plt.show()

# Load best model for final evaluation and visualization
try:
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt"))
    test_loss = evaluate(model, test_loader, device, criterion=lambda x, y: combined_loss(x, y))
    print(f"Final test loss: {test_loss:.4f}")

    # Generate final samples
    print("Generating final samples...")
    visualize_samples(model, device, epochs, save_path=save_dir)


    # Function to generate a grid of samples with varying diffusion steps
    def visualize_diffusion_grid(model, device, save_path=None):
        model.eval()
        class_labels = ['Cat', 'Dog']
        step_options = [5, 10, 20, 40]

        fig, axes = plt.subplots(len(class_labels), len(step_options), figsize=(12, 6))

        with torch.no_grad():
            for i, class_idx in enumerate(range(len(class_labels))):
                # Use same random seed for all step counts (fair comparison)
                torch.manual_seed(42 + class_idx)

                for j, steps in enumerate(step_options):
                    # Start with consistent noise
                    denoised_img = torch.randn((1, 3, 32, 32)).to(device)
                    label = torch.tensor([class_idx], device=device)

                    # Sequential denoising
                    for t in reversed(range(1, steps + 1)):
                        noise_level = torch.tensor([[[[t / steps]]]], device=device)
                        denoised_img = model(denoised_img, noise_level, label)

                    # Display final result
                    axes[i, j].imshow(denormalize_image(denoised_img[0]))

                    if i == 0:
                        axes[i, j].set_title(f"{steps} steps")

                    if j == 0:
                        axes[i, j].set_ylabel(class_labels[class_idx])

                    axes[i, j].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/diffusion_steps_comparison.png")
        plt.show()


    # Compare different numbers of diffusion steps
    visualize_diffusion_grid(model, device, save_path=save_dir)
except Exception as e:
    print(f"Error during evaluation: {e}")

print("Training and evaluation complete!")
