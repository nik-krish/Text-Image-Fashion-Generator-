# model.py
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Adaptive Instance Norm + Generator class
class AdaptiveInstanceNorm(torch.nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.gamma = torch.nn.Linear(embedding_dim, in_channels)
        self.beta = torch.nn.Linear(embedding_dim, in_channels)

    def forward(self, x, embedding):
        gamma = self.gamma(embedding).unsqueeze(2).unsqueeze(3)
        beta = self.beta(embedding).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

class Generator(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(Generator, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 256 * 8 * 8),
            torch.nn.ReLU()
        )
        self.adain = AdaptiveInstanceNorm(256, embedding_dim)
        self.deconv = torch.nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, text_embedding):
        x = self.fc(text_embedding).view(-1, 256, 8, 8)
        x = self.adain(x, text_embedding)
        img = self.deconv(x)
        return img

# Load generator
embedding_dim = 512
generator = Generator(embedding_dim).to(device)
checkpoint_path = r"D:\projects\Fashion_generator\server\generator_final (4).pth"
generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
generator.eval()

# Function to generate image from prompt
def generate_from_prompt(generator, prompt):
    inputs = clip_processor(text=prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**inputs)
        generated_image = generator(text_feat)
    return generated_image

# Exported symbols
__all__ = ['generator', 'clip_processor', 'clip_model', 'generate_from_prompt']
