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
from torch.optim import AdamW
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.amp import autocast, GradScaler

load_dotenv()

# Authenticate with Hugging Face
login(token=os.getenv("TOKEN"))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, images_dir, captions_file, transform=None):
        print(torch.cuda.is_available())
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

# Custom transform to add an extra channel
class AddChannel:
    def __call__(self, image):
        # Add an extra channel (e.g., all zeros)
        extra_channel = torch.zeros((1, image.size(1), image.size(2)))
        return torch.cat((image, extra_channel), dim=0)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    AddChannel(),  # Add the extra channel here
    transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]),
])

# DataLoader
dataset = CustomDataset(images_dir='./data/images/pebble', captions_file='./data/captions/pebble.txt', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load model
torch.cuda.empty_cache()
stable_diffusion = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
stable_diffusion = stable_diffusion.to("cuda")

# Assuming 'stable_diffusion' is your pipeline object
unet_model = stable_diffusion.unet  # Access the UNet model within the pipeline

# Create an optimizer for the UNet model
optimizer = AdamW(unet_model.parameters(), lr=1e-6)

# Initialize GradScaler for mixed precision
scaler = GradScaler(device='cuda')

# Training loop
epochs = 3
criterion = nn.MSELoss()  # Example loss function, replace with appropriate one
accumulation_steps = 10  # Example accumulation steps, replace with appropriate value

for epoch in range(epochs):
    total_loss = 0
    optimizer.zero_grad()  # Move zero_grad outside the loop
    for batch_idx, (images, captions) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to('cuda', dtype=torch.float16)
        images.requires_grad_(True)

        # Clear cache before each batch
        torch.cuda.empty_cache()

        timesteps = torch.randint(0, 1000, (images.size(0),), device=images.device)
        encoder_hidden_states = torch.rand((images.size(0), 77, 768), device=images.device, dtype=torch.float16)

        with autocast(device_type='cuda'):
            def custom_forward(*inputs):
                return unet_model(*inputs)

            outputs = checkpoint.checkpoint(custom_forward, images, timesteps, encoder_hidden_states, use_reentrant=False)
            output_tensor = outputs.sample
            target = torch.rand_like(output_tensor)
            loss = criterion(output_tensor, target) / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader)}")

    # Clear cache after each epoch
    torch.cuda.empty_cache()

# Save the trained model
model_save_path = './trained_pebble_diffusion_3.5_model'
stable_diffusion.save_pretrained(model_save_path)

# Load trained model for inference
trained_model = DiffusionPipeline.from_pretrained(model_save_path)
trained_model.eval().to("cuda")

# Generate an image based on a prompt
def generate_image(prompt):
    with torch.no_grad():  # Disable gradient calculation
        generated_image = trained_model(prompt).images[0]
        generated_image.show()

generate_image("Pebble the rabbit sitting in a field of flowers, smiling")
