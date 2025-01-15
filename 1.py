import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AdamW
from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Authenticate with Hugging Face
login(token=os.getenv("TOKEN"))

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, images_dir, captions_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        with open(captions_file, 'r') as file:
            self.captions = file.readlines()
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        
        # Debug prints
        print(f"Found {len(self.image_files)} images.")
        print(f"Found {len(self.captions)} captions.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        caption = self.captions[idx].strip()
        if self.transform:
            image = self.transform(image)
        return image, caption

# Image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# DataLoader
dataset = CustomDataset(images_dir='./data/images/pebble', captions_file='./data/captions/pebble.txt', transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load model
torch.cuda.empty_cache()
stable_diffusion = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
stable_diffusion = stable_diffusion.to("cuda")

# Optimizer
optimizer = AdamW(stable_diffusion.parameters(), lr=1e-6)

# Training loop
epochs = 3
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (images, captions) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to('cuda', dtype=torch.float16)
        loss = torch.rand(1).item()  # Placeholder for actual loss calculation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader)}")

# Save the trained model
model_save_path = './trained_pebble_diffusion_3.5_model'
torch.save(stable_diffusion.state_dict(), model_save_path)

# Load trained model for inference
trained_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
trained_model.load_state_dict(torch.load(model_save_path))
trained_model.eval().to("cuda")

# Generate an image based on a prompt
def generate_image(prompt):
    generated_image = trained_model(prompt).images[0]
    generated_image.show()

generate_image("Pebble the rabbit sitting in a field of flowers, smiling")
