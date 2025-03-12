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


# Define a basic ResNet block
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


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


# Self-attention with additional tricks for better convergence
class ImprovedSelfAttention(nn.Module):
    def __init__(self, dim, heads=10, dim_head=128, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Initialize projection with small values for stable training
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # Initialize parameters with small values
        with torch.no_grad():
            nn.init.xavier_uniform_(self.to_qkv.weight, gain=0.01)
            nn.init.xavier_uniform_(self.to_out[0].weight, gain=0.01)

        # For storing attention maps
        self.attention_maps = None

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        # Get queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        # Apply scaling to queries
        q = q * self.scale

        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2))

        # Apply softmax to get attention weights
        attn = dots.softmax(dim=-1)

        # Store attention maps for visualization
        self.attention_maps = attn.detach().clone()

        # Apply attention weights to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)

        return self.to_out(out)


# Improved transformer block with layer normalization positioning optimized
class ImprovedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(ImprovedTransformerBlock, self).__init__()
        self.attention = ImprovedSelfAttention(embed_dim, heads=num_heads, dropout=dropout)

        # Pre-norm architecture (more stable training)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Improved feed-forward network with SwiGLU activation
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim * 2),  # Wider for SwiGLU
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class ResNetPatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # This is correct

        # ResNet feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),  # Smaller kernel, no stride
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            ResBlock(64, 128, stride=1),  # No stride to maintain spatial dimensions
            ResBlock(128, 256, stride=1),
            ResBlock(256, embed_dim, stride=1)
        )

        # Critical change: Make sure this creates the right number of patches
        # For patch_size=2, this should create 16×16=256 patches from a 32×32 image
        self.final_conv = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=patch_size,  # This determines the patch size
            stride=patch_size  # This must match patch_size
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Process through ResNet components
        x = self.initial(x)
        x = self.res_blocks(x)

        # Final patch-based projection
        x = self.final_conv(x)  # Should give [batch_size, embed_dim, img_size/patch_size, img_size/patch_size]

        # Verify the shape matches expectations
        expected_grid_size = self.img_size // self.patch_size
        assert x.shape[2] == expected_grid_size and x.shape[3] == expected_grid_size, \
            f"Expected grid size {expected_grid_size}, got {x.shape[2]}x{x.shape[3]}"

        x = x.flatten(2)  # [batch_size, embed_dim, grid_size*grid_size]
        x = x.transpose(1, 2)  # [batch_size, grid_size*grid_size, embed_dim]

        return x

# Improved Transformer-based Conditional Denoiser for CIFAR-10
class CIFARTransformerDenoiser(nn.Module):
    def __init__(self, img_size=32, patch_size=2, num_classes=2, embed_dim=480,
                 num_heads=10, num_layers=6, ff_dim=384, dropout=0.1):
        super(CIFARTransformerDenoiser, self).__init__()

        # Save image and patch size as attributes
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = 3  # RGB for CIFAR-10
        self.patch_embed = ResNetPatchEmbedding(
            img_size,
            patch_size,
            in_channels=self.in_channels,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        # Rest of the transformer components remain the same
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.noise_embedding = nn.Linear(1, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Initialize positional embeddings
        with torch.no_grad():
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        with torch.no_grad():
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, patch_size * patch_size * self.in_channels)

        # NEW: Add ResNet refinement module for output enhancement
        self.refinement = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, self.in_channels, kernel_size=3, padding=1)
        )

        self.grid_size = img_size // patch_size

    def forward(self, x, noise_level, labels):
        batch_size = x.shape[0]

        # Get patch embeddings with ResNet features
        x = self.patch_embed(x)

        # The rest of the transformer processing remains the same
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        label_embed = self.label_embedding(labels).unsqueeze(1)
        noise_level_flat = noise_level.flatten().unsqueeze(1).expand(batch_size, 1)
        noise_embed = self.noise_embedding(noise_level_flat).unsqueeze(1)
        x[:, 0] = x[:, 0] + label_embed.squeeze(1) + noise_embed.squeeze(1)

        for block in self.transformer_blocks:
            x = block(x)

        x = x[:, 1:]
        x = self.norm(x)
        patch_pixels = self.projection(x)

        # Reshape into image
        num_patches_per_side = int(self.img_size // self.patch_size)
        output = patch_pixels.view(
            batch_size, num_patches_per_side, num_patches_per_side,
            self.patch_size, self.patch_size, self.in_channels
        )
        output = output.permute(0, 5, 1, 3, 2, 4).contiguous()
        output = output.view(batch_size, self.in_channels, self.img_size, self.img_size)

        # NEW: Apply ResNet refinement with skip connection
        refined_output = self.refinement(output) + output

        return refined_output


# Enhanced training with improved schedules and more diffusion steps
if __name__ == '__main__':
    # Initialize model, optimizer, loss function, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Current using device: {device}")

    # Create a more powerful model for CIFAR-10
    model = CIFARTransformerDenoiser(
        img_size=img_size,
        patch_size=4,
        num_classes=2,  # Cat and dog
        embed_dim=384,
        num_heads=8,
        num_layers=4,
        ff_dim=384,
        dropout=0.1
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
    epochs = 100
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


    def get_attention_maps(self):
        # Assuming your transformer has attention layers with attention weights
        attention_maps = []

        # This implementation depends on how your transformer is structured
        # For example, if you're using a standard transformer with multiple layers:
        for layer in self.transformer_layers:  # Adjust this based on your model structure
            # Extract attention weights from the layer
            # For example: attention_weights = layer.self_attention.attention_weights
            attention_weights = layer.get_attention_weights()  # Implement this method in your layer class
            attention_maps.append(attention_weights)

        return attention_maps

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
                    noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)
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
                        noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)
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

            # Create attention map visualizations
            # We'll generate one sample of each class and visualize attention
            os.makedirs('epoch_visualizations/attention_maps', exist_ok=True)

            for class_idx in range(2):
                # Use consistent seed
                torch.manual_seed(42 + class_idx)
                image = torch.randn((1, 3, img_size, img_size)).to(device)
                label = torch.tensor([class_idx], device=device)

                # Select specific timesteps for visualization (beginning, middle, end)
                timesteps = [num_diffusion_steps - 1, num_diffusion_steps // 2, 1]

                for timestep in timesteps:
                    # Clear previous attention maps
                    for block in model.transformer_blocks:
                        if hasattr(block.attention, 'attention_maps'):
                            block.attention.attention_maps = None

                    # Run the model for this specific timestep
                    noise_level = torch.tensor([[[[timestep / num_diffusion_steps]]]], device=device)
                    output = model(image, noise_level, label)

                    # Get attention maps from the model
                    attention_maps = model.get_attention_maps()

                    if attention_maps:
                        # Get the generated image
                        img_to_show = output.cpu().permute(0, 2, 3, 1).squeeze()
                        img_to_show = (img_to_show * 0.5 + 0.5).clip(0, 1)

                        # Select specific layers to visualize (first, middle, last)
                        layers_to_show = [0, len(attention_maps) // 2, len(attention_maps) - 1]

                        # Create visualization
                        fig, axs = plt.subplots(len(layers_to_show), 3, figsize=(12, 4 * len(layers_to_show)))

                        for i, layer_idx in enumerate(layers_to_show):
                            if layer_idx < len(attention_maps):
                                # Get attention map for this layer (first head)
                                # Skip the class token (index 0)
                                attn = attention_maps[layer_idx][0, 0, 1:, 1:].cpu()

                                # Reshape attention to match image grid
                                grid_size = model.grid_size
                                attn_map = attn.reshape(grid_size, grid_size, grid_size, grid_size)

                                # Compute average attention for each target position
                                attn_img = attn_map.mean(dim=(0, 1)).numpy()

                                # Upsample attention map to match image size
                                attn_img = np.kron(attn_img, np.ones((model.patch_size, model.patch_size)))

                                # Show the image
                                axs[i, 0].imshow(img_to_show)
                                axs[i, 0].set_title(f"Generated Image")
                                axs[i, 0].axis('off')

                                # Show attention map
                                im = axs[i, 1].imshow(attn_img, cmap='viridis')
                                axs[i, 1].set_title(f"Attention Map (Layer {layer_idx})")
                                axs[i, 1].axis('off')
                                plt.colorbar(im, ax=axs[i, 1])

                                # Overlay attention on image
                                axs[i, 2].imshow(img_to_show)
                                axs[i, 2].imshow(attn_img, alpha=0.5, cmap='viridis')
                                axs[i, 2].set_title(f"Attention Overlay")
                                axs[i, 2].axis('off')

                        plt.suptitle(f"{class_names[class_idx]} - Timestep {timestep} - Epoch {epoch_num + 1}",
                                     fontsize=16)
                        plt.tight_layout()
                        plt.savefig(f'epoch_visualizations/attention_maps/epoch_{epoch_num + 1}_'
                                    f'{class_names[class_idx]}_timestep_{timestep}.png', dpi=200)
                        plt.close()

            # Generate a GIF of the first sample from each class at this epoch
            if epoch_num + 1 == epochs:
                try:
                    import imageio
                    import glob

                    # Collect images for a generation process GIF
                    cat_images = []
                    dog_images = []

                    for epoch_num in range(epochs):
                        if os.path.exists(f'epoch_visualizations/epoch_{epoch_num + 1}_samples.png'):
                            img = imageio.imread(f'epoch_visualizations/epoch_{epoch_num + 1}_samples.png')
                            cat_images.append(img)
                            dog_images.append(img)

                    # Save GIF if we have images
                    if cat_images:
                        imageio.mimsave('cat_dog_generation_progress.gif', cat_images, fps=3)
                        print("Created animation GIF of training progress")
                except Exception as e:
                    print(f"Could not create GIF: {e}")

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

            noise_level = torch.tensor([t / num_diffusion_steps], device=device).view(-1, 1, 1, 1)

            # Add noise based on noise level with cosine schedule for better image quality
            noise = torch.randn_like(images).to(device)
            alpha = math.cos((t / num_diffusion_steps) * 0.4 * math.pi / 2)
            noisy_images = alpha * images + (1 - alpha) * noise

            optimizer.zero_grad()
            outputs = model(noisy_images, noise_level, labels)

            # Define these at the beginning of your script, before the training loop
            mse_scale = None
            l1_scale = None
            cos_sim_scale = None

            # Calculate losses
            mse = mse_loss(outputs, images)
            l1 = l1_loss(outputs, images)
            cos_sim = cosine_similarity_loss(outputs, images)

            # Initialize or update the scaling factors
            if mse_scale is None:
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
            l1_factor = mse_scale / l1_scale
            cos_sim_factor = mse_scale / cos_sim_scale

            # Dynamic weighting with scaling
            # Adjust loss weights for better convergence
            mse_weight = 0.55 - 0.2 * (epoch / epochs)  # Changed from 0.65 - 0.25
            l1_weight = 0.3 - 0.1 * (epoch / epochs)  # Changed from 0.25 - 0.15
            cos_sim_weight = 0.15 + 0.3 * (epoch / epochs)  # Changed from 0.1 + 0.4

            # Combined loss
            loss = (mse_weight * mse) + (l1_weight * l1 * l1_factor) + (cos_sim_weight * cos_sim * cos_sim_factor)

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


    # Function to visualize attention heatmaps over the full denoising process
    def visualize_attention_through_denoising():
        model.eval()
        with torch.no_grad():
            # Create a directory for the attention visualization
            os.makedirs('attention_visualization', exist_ok=True)

            # Run through each class
            for class_idx in range(2):
                # Use consistent seed
                torch.manual_seed(42 + class_idx)
                image = torch.randn((1, 3, img_size, img_size)).to(device)
                label = torch.tensor([class_idx], device=device)

                # We'll collect images at specific steps
                step_images = []
                step_attention_maps = []
                selected_steps = [1, 5, 10, 15, 19]  # Choose specific steps to visualize

                # Perform full denoising process
                for t in reversed(range(1, num_diffusion_steps + 1)):
                    noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)

                    # Clear previous attention maps
                    for block in model.transformer_blocks:
                        if hasattr(block.attention, 'attention_maps'):
                            block.attention.attention_maps = None

                    # Generate the next denoised image
                    image = model(image, noise_level, label)

                    # Add some residual noise for early steps
                    if t > num_diffusion_steps * 0.7:
                        image = image + 0.05 * torch.randn_like(image)

                    # If this is a step we want to visualize, save the image and attention maps
                    if num_diffusion_steps - t in selected_steps:
                        # Get the current image
                        img_to_show = image.cpu().permute(0, 2, 3, 1).squeeze()
                        img_to_show = (img_to_show * 0.5 + 0.5).clip(0, 1)
                        step_images.append(img_to_show)

                        # Get attention maps from all transformer blocks
                        attention_maps = model.get_attention_maps()

                        # Average across all layers and heads to get a single attention map
                        if attention_maps:
                            avg_attention = []
                            for attn in attention_maps:
                                # Skip the class token, focus on patches
                                patch_attn = attn[0, :, 1:, 1:].mean(dim=0).cpu()  # Average across heads
                                avg_attention.append(patch_attn)

                            if avg_attention:
                                # Average across layers
                                avg_attn = torch.stack(avg_attention).mean(dim=0)

                                # Reshape to grid
                                grid_size = model.grid_size
                                attn_map = avg_attn.reshape(grid_size, grid_size, grid_size, grid_size)

                                # Average across source positions (where attention comes from)
                                attn_img = attn_map.mean(dim=(0, 1)).numpy()

                                # Upsample to match image size
                                attn_img = np.kron(attn_img, np.ones((model.patch_size, model.patch_size)))
                                step_attention_maps.append(attn_img)

                                # Now create the visualization with images and attention maps
                            if step_images and step_attention_maps:
                                # Create a figure showing images and attention maps
                                fig, axes = plt.subplots(3, len(step_images), figsize=(20, 10))

                                for i in range(len(step_images)):
                                    # Original image
                                    axes[0, i].imshow(step_images[i])
                                    axes[0, i].set_title(f"Step {selected_steps[i]}")
                                    axes[0, i].axis('off')

                                    # Attention heatmap
                                    im = axes[1, i].imshow(step_attention_maps[i], cmap='hot')
                                    axes[1, i].set_title(f"Attention Map")
                                    axes[1, i].axis('off')

                                    # Overlay attention on image
                                    axes[2, i].imshow(step_images[i])
                                    axes[2, i].imshow(step_attention_maps[i], alpha=0.6, cmap='hot')
                                    axes[2, i].set_title(f"Attention Overlay")
                                    axes[2, i].axis('off')

                                plt.suptitle(f"Attention Focus During {class_names[class_idx]} Generation", fontsize=16)
                                plt.tight_layout()
                                plt.savefig(
                                    f'attention_visualization/{class_names[class_idx]}_attention_progression.png',
                                    dpi=300)
                                plt.close()

                                # Call the attention visualization function
                            visualize_attention_through_denoising()

                            # Generate final visualizations with more samples
                            with torch.no_grad():
                                # Create a summary grid with samples from different epochs
                                # This will show how the model improved over training
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
                                            noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)
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
                                # with higher resolution and more details
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
                                        noise_level = torch.tensor([[[[t / num_diffusion_steps]]]], device=device)
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
