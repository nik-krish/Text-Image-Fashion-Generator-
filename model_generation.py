# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install torch torchvision datasets transformers

import torch
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class FashionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = item['text']
        sketch = transform(item['sketch'].convert("RGB"))  # Optional if used
        image = transform(item['image'].convert("RGB"))

        # Encode text with CLIP
        inputs = clip_processor(text=prompt, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_feat = clip_model.get_text_features(**inputs)

        return text_feat.squeeze(0), image, prompt  # 👈 Include prompt for CLIP loss

from datasets import load_dataset
ds = load_dataset("Abhi5ingh/Dresscodepromptsketch", split='train')
print(ds[0])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os

# ✅ Set output directory
save_dir = "/kaggle/working/gan_outputs2"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🟢 Adaptive Instance Normalization Layer
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.gamma = nn.Linear(embedding_dim, in_channels)
        self.beta = nn.Linear(embedding_dim, in_channels)

    def forward(self, x, embedding):
        gamma = self.gamma(embedding).unsqueeze(2).unsqueeze(3)
        beta = self.beta(embedding).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

# 🔵 Generator
class Generator(nn.Module):
    def __init__(self, embedding_dim):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256 * 8 * 8),
            nn.ReLU()
        )

        self.adain = AdaptiveInstanceNorm(256, embedding_dim)

        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # -> [128, 16, 16]
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # -> [64, 32, 32]
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  # -> [3, 64, 64]
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, text_embedding):
        x = self.fc(text_embedding).view(-1, 256, 8, 8)
        x = self.adain(x, text_embedding)
        img = self.deconv(x)
        return img

# 🔴 Discriminator with Spectral Normalization
class Discriminator(nn.Module):
    def __init__(self, embedding_dim):
        super(Discriminator, self).__init__()

        self.image_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),  # -> [64, 32, 32]
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),  # -> [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),  # -> [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten()  # [B, 256*8*8]
        )

        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(256 * 8 * 8 + embedding_dim, 1))
        )

    def forward(self, image, text_embedding):
        image_feat = self.image_conv(image)
        combined = torch.cat((image_feat, text_embedding), dim=1)
        out = self.fc(combined)
        return out

# 🟡 Gradient Penalty (WGAN-GP)
def compute_gradient_penalty(D, real_samples, fake_samples, text_embedding):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    interpolated_preds = D(interpolated, text_embedding)

    gradients = torch.autograd.grad(
        outputs=interpolated_preds,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_preds),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def clip_loss(generated_images, text_prompts):
    # 🧼 Rescale from [-1, 1] to [0, 1] for CLIP
    generated_images = (generated_images + 1) / 2.0

    # Preprocess for CLIP
    image_inputs = clip_processor(images=generated_images, return_tensors="pt").to(device)
    text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    # Get CLIP features
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
        text_features = clip_model.get_text_features(**text_inputs)

    # Normalize embeddings
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity loss
    similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
    return 1 - similarity.mean()

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def show_image_with_text(image_tensor, text_prompt, epoch, save_path):
    image = TF.to_pil_image(image_tensor.cpu().detach().clamp(-1, 1) * 0.5 + 0.5)  # unnormalize
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"Prompt: {text_prompt}", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 🟣 Training Setup
batch_size = 32
lr = 2e-4
epochs = 30
lambda_gp = 10  # Weight for gradient penalty
lambda_clip = 2.0  # CLIP-guided loss weight
embedding_dim = 512  # CLIP-based

dataset = FashionDataset(ds)  # Must return (text_embedding, real_image, text_prompt)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(embedding_dim).to(device)
discriminator = Discriminator(embedding_dim).to(device)

g_optimizer = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(epochs):
    for i, (text_embeddings, real_images, text_prompts) in enumerate(loader):
        text_embeddings = text_embeddings.to(device)
        real_images = real_images.to(device)

        # ✅ Train Discriminator
        fake_images = generator(text_embeddings).detach()
        real_preds = discriminator(real_images, text_embeddings)
        fake_preds = discriminator(fake_images, text_embeddings)

        d_loss_real = torch.mean(F.relu(1.0 - real_preds))
        d_loss_fake = torch.mean(F.relu(1.0 + fake_preds))
        gp = compute_gradient_penalty(discriminator, real_images, fake_images, text_embeddings)
        d_loss = d_loss_real + d_loss_fake + lambda_gp * gp

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ✅ Train Generator
        fake_images = generator(text_embeddings)
        fake_preds = discriminator(fake_images, text_embeddings)
        g_loss_adv = -torch.mean(fake_preds)

        # 🆕 CLIP-guided loss
        clip_guided_loss = clip_loss(fake_images, text_prompts)

        g_loss = g_loss_adv + lambda_clip * clip_guided_loss

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    # ✅ Save Checkpoint
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')

    # ✅ Save Prompt + Image — get from last batch
    sample_prompt = text_prompts[0] if isinstance(text_prompts[0], str) else text_prompts[0].decode("utf-8")
    show_image_with_text(fake_images[0], sample_prompt, epoch, f'{save_dir}/prompt_image_epoch_{epoch+1}.png')
    print(f"📝 Prompt: {sample_prompt}")

    print(f"[Epoch {epoch+1}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, CLIP Loss: {clip_guided_loss.item():.4f}")
    print(f"Checkpoint and samples saved for epoch {epoch+1}")


# ✅ Save Final Models
torch.save(generator.state_dict(), f'{save_dir}/generator_final.pth')
torch.save(discriminator.state_dict(), f'{save_dir}/discriminator_final.pth')
print("🎉 Final models saved successfully!")

# ✅ Save Final Models
torch.save(generator.state_dict(), f'{save_dir}/generator_final.pth')
torch.save(discriminator.state_dict(), f'{save_dir}/discriminator_final.pth')
print("🎉 Final models saved successfully!")