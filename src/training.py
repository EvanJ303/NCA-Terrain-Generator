import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
import matplotlib.pyplot as plt
from model import NCA

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
STATE_CHANNELS = 16
HIDDEN_CHANNELS = 32
ALPHA_CHANNEL = 3
UPDATE_PROB = 0.5
MIN_STEPS = 64
MAX_STEPS = 96
NUM_EPOCHS = 100
CLIP_GRAD_NORM = 1.0
PERCEPTUAL_LOSS_WEIGHT_1 = 0.3
PERCEPTUAL_LOSS_WEIGHT_2 = 0.7
MSE_LOSS_WEIGHT = 0.1

def generate_images(targets, model):
    device = next(model.parameters()).device
    batch_size = targets.size(0)

    steps = torch.randint(MIN_STEPS, MAX_STEPS + 1, (batch_size,), device=device)
    max_steps = int(steps.max().item())

    states = torch.zeros((batch_size, STATE_CHANNELS, targets.size(2), targets.size(3)), device=device)
    states[:, 4:, :, :] = 0.01 * torch.randn((batch_size, STATE_CHANNELS - 4, targets.size(2), targets.size(3)), device=device)
    states[:, 3, targets.size(2) // 2, targets.size(3) // 2] = 1.0

    history = [states.clone()]

    for _ in range(max_steps):
        states = model(states)
        history.append(states.clone())

    states = torch.stack(tuple(history[s.item()][b] for b, s in enumerate(steps)), dim=0)

    alpha = torch.sigmoid(states[:, ALPHA_CHANNEL:ALPHA_CHANNEL + 1, :, :])
    dead_mask = alpha < 0.1

    rgb = states[:, :3, :, :]
    rgb[dead_mask.repeat(1, 3, 1, 1)] = 0.0
    rgb = torch.clamp(rgb, 0.0, 1.0)

    return rgb

def calculate_loss(images, targets, model, criterion):
    device = next(model.parameters()).device

    images = images.to(device)
    targets = targets.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = normalize(images)

    slice_1 = nn.Sequential(*list(model.children())[:9]).to(device)
    slice_2 = nn.Sequential(*list(model.children())[:16]).to(device)

    features_images_1 = slice_1(images)
    features_images_2 = slice_2(images)

    with torch.no_grad():
        features_targets_1 = slice_1(targets)
        features_targets_2 = slice_2(targets)

    perceptual_loss_1 = criterion(features_images_1, features_targets_1)
    perceptual_loss_2 = criterion(features_images_2, features_targets_2)

    mse_loss = criterion(images, targets)

    total_loss = (
        PERCEPTUAL_LOSS_WEIGHT_1 * perceptual_loss_1 +
        PERCEPTUAL_LOSS_WEIGHT_2 * perceptual_loss_2 +
        MSE_LOSS_WEIGHT * mse_loss
    )

    return total_loss

def save_checkpoint(model, epoch):
    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'nca_model_epoch_{epoch}_{current_timestamp}.pt'
    path = os.path.join('./data/models', filename)
    torch.save(model.state_dict(), path)

    latest_path = os.path.join('./data', 'latest_checkpoint.txt')
    with open(latest_path, 'w', encoding='utf-8') as f:
        f.write(filename)

    print(f'Saved model checkpoint: {path}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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

    forest_targets = [idx for idx, (_, target) in enumerate(dataset) if target == 4]
    dataset = Subset(dataset, forest_targets)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    nca_model = NCA(state_channels=STATE_CHANNELS, hidden_channels=HIDDEN_CHANNELS, alpha_channel=ALPHA_CHANNEL, update_prob=UPDATE_PROB).to(device)

    vgg_model = models.vgg16(pretrained=True).features.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(nca_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (targets, _) in enumerate(train_loader):
            images = generate_images(targets=targets, model=nca_model)
            loss = calculate_loss(images=images, targets=targets, model=vgg_model, criterion=criterion)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(nca_model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

        num_val_batches = len(val_loader)
        val_loss = 0.0

        for batch_idx, (targets, _) in enumerate(val_loader):
            with torch.no_grad():
                images = generate_images(targets=targets, model=nca_model)
                loss = calculate_loss(images=images, targets=targets, model=vgg_model, criterion=criterion)
            val_loss += loss.item()

        val_loss /= num_val_batches
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}')

        fig = plt.figure(figsize=(8, 6))
        plt.plot(range(1, epoch + 2), [val_loss] * (epoch + 1), label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss over Epochs')
        plt.grid(True)
        plt.tight_layout()

        filename = f'session_loss_plot_{session_timestamp}.png'
        plot_path = os.path.join('./data/plots', filename)
        fig.savefig(plot_path)
        plt.close(fig)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_checkpoint(nca_model, epoch + 1)

if __name__ == '__main__':
    main()