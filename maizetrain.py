import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SegmentationDataset
from model import Unet

# Hyperparameters
NUM_EPOCHS = 25
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pth"

# Paths
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"
MODEL_PATH = "model.pth"

# Data transforms
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Dataset and dataloader
dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR, transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = Unet().to(DEVICE)

# Loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load model if specified
if LOAD_MODEL:
    model.load_state_dict(torch.load(LOAD_MODEL_FILE))

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}/{NUM_EPOCHS}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)