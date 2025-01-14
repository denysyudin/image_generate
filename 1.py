import torch
from diffusers import StableDiffusionPipeline
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from transformers import AdamW

# Dataset
class PebbleDataset(Dataset):
    def __init__(self, images_dir, captions_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        with open(captions_file, 'r') as file:
            self.captions = file.readlines()
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        caption = self.captions[idx].strip()
        if self.transform:
            image = self.transform(image)
        return image, caption

# Preprocessing and DataLoader
transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
dataset = PebbleDataset(images_dir='./data/images/pebble', captions_file='./data/captions/pebble.txt', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model Setup
model_name = "CompVis/stable-diffusion-v-1-4-original"
stable_diffusion = StableDiffusionPipeline.from_pretrained(model_name)
stable_diffusion = stable_diffusion.to("cuda")

# Optimizer
optimizer = AdamW(stable_diffusion.parameters(), lr=5e-6)

# Training Loop
epochs = 3
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (images, captions) in enumerate(dataloader):
        images = images.to('cuda')
        loss = stable_diffusion(images, captions)  # Simplified example
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader)}")

# Save Model
torch.save(stable_diffusion.state_dict(), './pebble_diffusion_model')

# Load Saved Model
trained_model = StableDiffusionPipeline.from_pretrained(model_name)
trained_model.load_state_dict(torch.load('./pebble_diffusion_model'))
trained_model.eval().to('cuda')

# Generate Images
def generate_image(prompt):
    generated_image = trained_model(prompt).images[0]
    generated_image.show()

generate_image("Pebble the rabbit sitting in a meadow, looking curiously at a butterfly.")
