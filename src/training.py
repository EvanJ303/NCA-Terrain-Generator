import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
import matplotlib.pyplot as plt
from model import NCA

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
STATE_CHANNELS = 16
HIDDEN_CHANNELS = 32
UPDATE_PROB = 0.5
MIN_STEPS = 64
MAX_STEPS = 96
NUM_EPOCHS = 100
CLIP_GRAD_NORM = 1.0
PERCEPTUAL_LOSS_WEIGHT_1 = 0.3
PERCEPTUAL_LOSS_WEIGHT_2 = 0.7
MSE_LOSS_WEIGHT = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('./data/models', exist_ok=True)
os.makedirs('./data/plots', exist_ok=True)

transform = transforms.Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.EuroSAT(
    root='./data',
    download=True,
    transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

nca_model = NCA(state_channels=STATE_CHANNELS, hidden_channels=HIDDEN_CHANNELS, update_prob=UPDATE_PROB).to(device)

vgg_model = models.vgg16(pretrained=True).features.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

vgg_slice_1 = nn.Sequential(*list(vgg_model.children())[:9]).to(device)
vgg_slice_2 = nn.Sequential(*list(vgg_model.children())[:16]).to(device)

optimizer = optim.Adam(nca_model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

def save_checkpoint(model, epoch):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'nca_model_epoch_{epoch}_{timestamp}.pt'
    path = os.path.join('./data/models', filename)
    torch.save(model.state_dict(), path)

    latest_path = os.path.join('./data', 'latest_checkpoint.txt')
    with open(latest_path, 'w', encoding='utf-8') as f:
        f.write(filename)

    print(f'Saved model checkpoint: {path}')

for epoch in range(NUM_EPOCHS):
    for batch_idx, (targets, _) in enumerate(train_loader):
        targets = targets.to(device)

        steps = torch.randint(MIN_STEPS, MAX_STEPS + 1, (BATCH_SIZE,), device=device)
        max_steps = int(steps.max().item())

        states = torch.zeros((BATCH_SIZE, STATE_CHANNELS, targets.size(2), targets.size(3)), device=device)
        states[:, 4:, :, :] = 0.01 * torch.randn((BATCH_SIZE, STATE_CHANNELS - 4, targets.size(2), targets.size(3)), device=device)
        states[:, 3, targets.size(2) // 2, targets.size(3) // 2] = 1.0

        history = [states.clone()]

        for _ in range(max_steps):
            states = nca_model(states)
            history.append(states.clone())

        states = torch.stack((history[s.item()][b] for b, s in enumerate(steps)), dim=0)

        images = states[:, :3, :, :]

        vgg_features_images_1 = vgg_slice_1(images)
        vgg_features_images_2 = vgg_slice_2(images)

        with torch.no_grad():
            vgg_features_targets_1 = vgg_slice_1(targets)
            vgg_features_targets_2 = vgg_slice_2(targets)

        perceptual_loss_1 = criterion(vgg_features_images_1, vgg_features_targets_1)
        perceptual_loss_2 = criterion(vgg_features_images_2, vgg_features_targets_2)

        mse_loss = criterion(images, targets)

        total_loss = (
            PERCEPTUAL_LOSS_WEIGHT_1 * perceptual_loss_1 +
            PERCEPTUAL_LOSS_WEIGHT_2 * perceptual_loss_2 +
            MSE_LOSS_WEIGHT * mse_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(nca_model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

    for batch_idx, (targets, _) in enumerate(val_loader):
        # Validation loop can be implemented here if needed, similar to the training loop but without gradient updates.

    if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
        save_checkpoint(nca_model, epoch + 1)
        