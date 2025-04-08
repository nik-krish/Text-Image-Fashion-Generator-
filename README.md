# 👗 Text-to-Fashion Image Generation with CLIP-Guided GAN

This project trains a GAN model to generate realistic **fashion images** from **text prompts** using CLIP-guided supervision. It uses the [DressCodePromptSketch](https://huggingface.co/datasets/Abhi5ingh/Dresscodepromptsketch) dataset and improves generation quality using **CLIP loss**, **WGAN-GP**, and **Adaptive Instance Normalization (AdaIN)**.

## 🚀 Features

- ✅ Text-to-image generation using CLIP text embeddings  
- ✅ Residual Generator with AdaIN for text-conditioning  
- ✅ Spectral Normalized Discriminator  
- ✅ CLIP loss for text-image alignment  
- ✅ Gradient Penalty with WGAN-GP  
- ✅ Visualization and checkpoint saving per epoch  
- ✅ Optimized for GPU training (Kaggle/Colab)  

---

## 🧠 Model Architecture

- **Generator**:  
  - Fully connected + AdaIN + ConvTranspose layers  
  - Input: CLIP text embedding → Output: 64×64 RGB image  

- **Discriminator**:  
  - Spectral normalized CNN + FC  
  - Conditional on both image and text embedding  

- **CLIP Model**:  
  - `openai/clip-vit-base-patch32` (via 🤗 Transformers)  
  - Used for text embedding and CLIP loss  

---

## 📦 Dataset

**Name**: [`Abhi5ingh/Dresscodepromptsketch`](https://huggingface.co/datasets/Abhi5ingh/Dresscodepromptsketch)  
**Contents**:  
- `text`: Prompt for fashion description  
- `image`: Ground truth image  
- `sketch`: Optional sketch of the fashion item  

---

## 🛠️ Setup

```bash
pip install torch torchvision datasets transformers
```

---

## 📄 Training Script Overview

```python
# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load dataset
from datasets import load_dataset
ds = load_dataset("Abhi5ingh/Dresscodepromptsketch", split='train')

# Create DataLoader
dataset = FashionDataset(ds)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize Generator & Discriminator
generator = Generator(embedding_dim=512).to(device)
discriminator = Discriminator(embedding_dim=512).to(device)

# Train for 20 epochs
for epoch in range(epochs):
    ...
```

---

## 🧪 Loss Functions

- **Discriminator Loss**: WGAN-GP + Hinge loss  
- **Generator Loss**: Adversarial + `λ * CLIP loss`  
- **CLIP Loss**: Cosine similarity between image/text features

---

## 📊 Outputs

- Checkpoints saved per epoch: `gan_outputs2/checkpoint_epoch_*.pth`  
- Image previews saved: `gan_outputs2/prompt_image_epoch_*.png`  
- Final models:
  ```
  generator_final.pth
  discriminator_final.pth
  ```

---

## 📷 Sample Output

<p align="center">
  <img src="![prompt_image_epoch_21](https://github.com/user-attachments/assets/cf75dad2-1636-491f-8045-7dc615d1d8f7)
" width="200"/>
</p>

---

## 🧠 Future Improvements

- 🔼 Generate 128×128 or 256×256 images (Real-ESRGAN upscaling)  
- ⚙️ Add residual connections and attention in Generator  
- 💬 Text-to-sketch-to-image pipeline  

---

## 👨‍💻 Author

Built with ❤️ by **Nikhil Krishan**

---

Let me know if you'd like a version that includes inference/testing or a `requirements.txt` too
