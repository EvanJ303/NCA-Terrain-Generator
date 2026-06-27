import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import NCA

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
STATE_CHANNELS = 16
HIDDEN_CHANNELS = 32
UPDATE_PROB = 0.5
MIN_STEPS = 64
MAX_STEPS = 96
NUM_EPOCHS = 10
CLIP_GRAD_NORM = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = ToTensor()

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

model = NCA(state_channels=STATE_CHANNELS, hidden_channels=HIDDEN_CHANNELS, update_prob=UPDATE_PROB).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

for epoch in range(NUM_EPOCHS):