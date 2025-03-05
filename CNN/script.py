import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Data loading and preprocessing for CIFAR-10
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Modified U-Net without enc4 (temporarily removed labels, only include noise_level)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.pool = nn.MaxPool2d(2)

        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, noise_level):
        x = torch.cat([x, noise_level], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d1 = self.dec1(torch.cat([self.upconv1(e3), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d1), e1], dim=1))

        return torch.tanh(self.final(d2))

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.MSELoss()

# Training loop
num_diffusion_steps = 30
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for images, _ in train_loader:
        images = images.to(device)
        t = torch.randint(1, num_diffusion_steps + 1, (images.size(0),), device=device)
        noise_level = (t / num_diffusion_steps).view(-1, 1, 1, 1).repeat(1, 1, images.size(2), images.size(3))
        noisy_images = images + torch.randn_like(images) * noise_level

        optimizer.zero_grad()
        outputs = model(noisy_images, noise_level)  # fixed here
        loss = criterion(outputs, images)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Save model
torch.save(model.state_dict(), 'unet_diffusion_cifar10.pth')

# Evaluation and visualization
model.eval()

with torch.no_grad():
    fig, axes = plt.subplots(2, num_diffusion_steps + 1, figsize=(20, 6))

    for row in range(2):
        img = torch.randn((1, 3, 32, 32)).to(device)
        axes[row, 0].imshow(img.cpu().squeeze().permute(1, 2, 0) * 0.5 + 0.5)
        axes[row, 0].set_title("Noise")
        axes[row, 0].axis('off')

        for step in reversed(range(1, num_diffusion_steps + 1)):
            noise_level = torch.tensor([step / num_diffusion_steps], device=device).view(1, 1, 1, 1).repeat(1, 1, 32, 32)
            img = model(img, noise_level)
            axes[row, num_diffusion_steps - step + 1].imshow(img.cpu().squeeze().permute(1, 2, 0) * 0.5 + 0.5)
            axes[row, num_diffusion_steps - step + 1].set_title(f"Step {num_diffusion_steps - step + 1}")
            axes[row, num_diffusion_steps - step + 1].axis('off')

plt.tight_layout()
plt.show()
