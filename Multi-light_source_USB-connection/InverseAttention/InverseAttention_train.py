import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CUDA = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader import MultiViewWithNormalDataset
from InverseAttentionClassifier import InverseAttentionClassifier
from util import *
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.optim as optim
import time
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

epochs = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataSet, dataLoader
num_cores = os.cpu_count()
num_workers = max(1, num_cores // 2)  # 코어 수의 50%로 설정

train_dir = '/home/johs/Multi-light_source_USB-Connection/dataset/train'
valid_dir = '/home/johs/Multi-light_source_USB-Connection/dataset/valid'
train_dataset = MultiViewWithNormalDataset(root_dir=train_dir, augmentation=True)
valid_dataset = MultiViewWithNormalDataset(root_dir=valid_dir)
train_loader = DataLoader(train_dataset, batch_size=2 ** 7, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=2 ** 7, shuffle=False, num_workers=num_workers)

# Set up model and tools
model = InverseAttentionClassifier(output_size=5, temperature=0.7).to(device)

# weightsloss = torch.tensor([2, 3, 1, 1, 1], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)

weight_path = f'weights/Metric_{epochs}_seed{SEED}_weightloss_aug_newinverse07.pth'

print(f"\nTraining model {model.__class__.__name__}, GPU: {CUDA}, "
      f"Criterion: {criterion}, Optimizer: {optimizer.__class__.__name__}, Weight path: {os.path.basename(weight_path)}\n")

total_time = time.time()
best_accuracy = 0.0
for epoch in range(1, epochs + 1):

    # Train model
    model.train()

    epoch_loss = 0.0
    train_outputs, train_targets = [], []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        losses = criterion(logits, labels)

        # Backward pass, Update weights
        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        _, pred = torch.max(logits, 1)
        train_outputs.extend(pred.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

        epoch_loss += losses.item()

    train_matrix = confusion_matrix(train_targets, train_outputs)

    val_outputs, val_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            _, pred = torch.max(logits, 1)
            val_outputs.extend(pred.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_matrix = confusion_matrix(val_targets, val_outputs)
    calculate_accuracies(val_matrix, epoch)

    val_accuracy = accuracy_score(val_targets, val_outputs)

    # accuracy = accuracy_score(targets, preds)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), weight_path)
        print(f"Saved weight_path: {weight_path}\n")

    if epoch % 10 == 0:
        print(f"Loss: {epoch_loss: .4f}, Weight: {weight_path}. Train matrix: \n{train_matrix}")
        print(f"Valid matrix: \n{val_matrix}")

    if epoch % 100 == 0:
        print(f'\nEpoch: {epoch}/{epochs}. Middle training Time: {(time.time() - total_time) / 3600: .2f} hours\n')

print(f'\nTraining end. Total epoch: {epochs}, Total training Time: {((time.time() - total_time) / 3600): .2f} hours\n')
