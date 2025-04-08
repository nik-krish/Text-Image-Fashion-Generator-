# 👗 Text-to-Fashion Image Generation with CLIP-Guided GAN

This project is a Text-to-Fashion Image Generation App built with a custom GAN model to generate realistic **fashion images** from **text prompts** using CLIP-guided supervision. It uses the [DressCodePromptSketch](https://huggingface.co/datasets/Abhi5ingh/Dresscodepromptsketch) dataset and improves generation quality using **CLIP loss**, **WGAN-GP**, and **Adaptive Instance Normalization (AdaIN)**.

---

## 🌐 Demo

![Screenshot 2025-04-08 160732](https://github.com/user-attachments/assets/365ac863-c223-4627-89bb-f1d1c347f9f7)

---

## 🚀 Features

### 🧠 AI Model

- ✅ Text-to-image generation using CLIP text embeddings  
- ✅ Residual Generator with AdaIN for text conditioning  
- ✅ Spectral Normalized Discriminator  
- ✅ CLIP loss for text-image alignment  
- ✅ Gradient Penalty with WGAN-GP  
- ✅ Visualization and checkpoint saving per epoch  
- ✅ Optimized for GPU training (Kaggle/Colab)  
- ✅ Output: 64×64 images, upscaled to 256×256 using Real-ESRGAN in the browser

### 🖼️ UI & UX (React)

- Real-time image upscaling using canvas for enhanced display  
- Prompt suggestions and animated loading visuals  
- Toggle between original and enhanced images  
- Download option for generated images  
- Responsive and themed using custom CSS variables  

### 🖥️ Backend (Flask)

- REST API with `/generate` endpoint  
- Accepts JSON prompt from frontend and returns the generated image  
- Handles inference and image conversion to PNG  
- CORS enabled for local frontend communication  

---

## 🧠 Model Architecture

- **Generator**  
  - Fully connected + AdaIN + ConvTranspose layers  
  - Input: CLIP text embedding → Output: 64×64 RGB image  

- **Discriminator**  
  - Spectral normalized CNN + Fully Connected layers  
  - Conditional on both image and text embedding  

- **CLIP Model**  
  - `openai/clip-vit-base-patch32` (via 🤗 Transformers)  
  - Used for text embedding and CLIP loss  

---

## 📦 Dataset

**Name**: [`Abhi5ingh/Dresscodepromptsketch`](https://huggingface.co/datasets/Abhi5ingh/Dresscodepromptsketch)  
**Contents**:
- `text`: Prompt describing the fashion item  
- `image`: Ground truth image  
- `sketch`: Optional sketch of the fashion item  

---

## 🛠️ Model Setup

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


---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/fashion-gan.git
cd fashion-gan
```

### 2️⃣ Backend Setup (Flask)

```bash
cd backend
pip install -r requirements.txt
```

Ensure `app.py` imports your model:

```python
from one import generator, clip_processor, clip_model, generate_from_prompt
```

Run the Flask server:

```bash
python app.py
```

Server runs at: `http://localhost:5000`

### 3️⃣ Frontend Setup (React)

```bash
cd frontend
npm install
npm start
```

React app runs at: `http://localhost:3000`

---

## 🧪 Sample Prompts

- `dress lilac pink`  
- `blue surplice jersey maxi dress`  
- `sheath dress gray cut-out dress`  
- `pink haley tee`  

---

## 📂 Project Structure

```
.
├── backend/
│   ├── app.py
│   └── one/          # contains generator, CLIP models, etc.
├── frontend/
│   ├── src/App.js
│   ├── src/App.css
│   └── ...
└── README.md
```

---

## 🧠 Future Improvements

- 🔼 Generate 128×128 or 256×256 images (Real-ESRGAN upscaling)  
- ⚙️ Add residual connections and attention in the Generator  
- 💬 Implement a text-to-sketch-to-image pipeline  

---

## 👨‍💻 Author

Built by **Nikhil Krishan**
Contact:s.nikhilkrishnan@gmail.com

---
