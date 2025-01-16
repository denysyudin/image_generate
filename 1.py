import os
# Set environment variables before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# Correct AdamW import
from torch.optim import AdamW
from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm
# Correct login import
from huggingface_hub import login, HfApi
from dotenv import load_dotenv
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.amp import autocast, GradScaler
from transformers import BertModel
from torchvision import models
import torch.nn.functional as F

load_dotenv()

# Authenticate with Hugging Face
api = HfApi(token=os.getenv("TOKEN"))
api.token = os.getenv("TOKEN")
login(token=api.token)

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
        if isinstance(image, torch.Tensor):
            extra_channel = torch.zeros((1, image.size(1), image.size(2)))
            return torch.cat((image, extra_channel), dim=0)
        else:
            raise TypeError("Input image must be a tensor")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # Convert to tensor first
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    AddChannel(),  # Add the extra channel after ToTensor
])

# DataLoader
dataset = CustomDataset(images_dir='./data/images/pebble', captions_file='./data/captions/pebble.txt', transform=transform)
dataloader = DataLoader(
    dataset, 
    batch_size=1,  # Reduce batch size
    shuffle=True,
    num_workers=2,  # Add parallel data loading
    pin_memory=True  # Speed up data transfer to GPU
)

# Load model
torch.cuda.empty_cache()
stable_diffusion = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
stable_diffusion = stable_diffusion.to("cuda")

# Assuming 'stable_diffusion' is your pipeline object
unet_model = stable_diffusion.unet  # Access the UNet model within the pipeline

# Create an optimizer for the UNet model
optimizer = AdamW(unet_model.parameters(), lr=1e-6)

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Before the training loop, add gradient checkpointing to save memory
unet_model.enable_gradient_checkpointing()

# Before the training loop, modify the model and optimizer setup:
unet_model.train()  # Set model to training mode
unet_model = unet_model.float()  # Ensure model parameters are in float32
optimizer = AdamW(unet_model.parameters(), lr=1e-6)

# Enable gradient checkpointing
unet_model.enable_gradient_checkpointing()

# Before training, enable memory efficient attention
unet_model.set_attention_slice(1)

# Training loop
epochs = 3
criterion = nn.MSELoss()  # Example loss function, replace with appropriate one
accumulation_steps = 32  # Increase this value

class MultiModalModel(nn.Module):
    def __init__(self, num_classes=1000):  # Add default num_classes parameter
        super(MultiModalModel, self).__init__()
        # Image Encoder
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final classification layer
        
        # Text Encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Fusion Layer
        self.fc = nn.Linear(self.cnn.fc.in_features + self.bert.config.hidden_size, 256)
        
        # Output Layer
        self.output = nn.Linear(256, num_classes)

    def forward(self, image, text_input_ids, text_attention_mask):
        # Image features
        image_features = self.cnn(image)
        
        # Text features
        text_outputs = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_features = text_outputs.pooler_output
        
        # Concatenate features
        combined_features = torch.cat((image_features, text_features), dim=1)
        
        # Pass through fusion and output layers
        x = self.fc(combined_features)
        x = F.relu(x)  # Add activation function
        x = self.output(x)
        return x

# Example usage
# model = MultiModalModel()
# image_tensor = ...  # Preprocessed image tensor
# text_input_ids, text_attention_mask = ...  # Tokenized text input
# output = model(image_tensor, text_input_ids, text_attention_mask)

# Add near the top of your script
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (images, captions) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            
        optimizer.zero_grad()
        
        try:
            # Convert images to float32 and enable gradients
            images = images.float().to('cuda')
            images.requires_grad_(True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # Generate timesteps
                timesteps = torch.randint(0, 1000, (images.size(0),), device=images.device)
                
                # Create encoder hidden states with gradients
                encoder_hidden_states = torch.rand(
                    (images.size(0), 77, 2048), 
                    device=images.device, 
                    dtype=torch.float16,
                    requires_grad=True
                )
                
                # Generate time_ids and text_embeds with gradients
                time_ids = torch.zeros(
                    (images.size(0), 6), 
                    device=images.device, 
                    dtype=torch.float16,
                    requires_grad=True
                )
                
                text_embeds = torch.rand(
                    (images.size(0), 1280), 
                    device=images.device, 
                    dtype=torch.float16,
                    requires_grad=True
                )
                
                # Direct forward pass without checkpoint
                added_cond_kwargs = {
                    'text_embeds': text_embeds,
                    'time_ids': time_ids
                }
                
                output = unet_model(
                    images, 
                    timesteps, 
                    encoder_hidden_states, 
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                
                # Calculate loss (ensure target has gradients too)
                target = torch.rand_like(output, requires_grad=True)
                loss = criterion(output, target) / accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Unscale gradients
                scaler.unscale_(optimizer)
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(unet_model.parameters(), max_norm=0.5)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                # Clear gradients and cache
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise e

        total_loss += loss.item() * accumulation_steps

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader)}")
    torch.cuda.empty_cache()

# Save the trained model
model_save_path = './trained_pebble_diffusion_3.5_model'
stable_diffusion.save_pretrained(model_save_path)

# Load trained model for inference
trained_model = DiffusionPipeline.from_pretrained(model_save_path)
trained_model.eval().to("cuda")

# Generate an image based on a prompt
def generate_image(prompt, num_inference_steps=50):
    try:
        with torch.no_grad():
            generated_image = trained_model(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5
            ).images[0]
            generated_image.save(f"generated_{prompt[:30]}.png")
            return generated_image
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

generate_image("Pebble the rabbit sitting in a field of flowers, smiling")
