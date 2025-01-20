import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CUDA = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader import MultiViewDataset
from AWTClassifier import AWTClassifier
from util import *
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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

epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataSet, dataLoader
num_cores = os.cpu_count()
num_workers = max(1, num_cores // 2)  # 코어 수의 50%로 설정

train_dir = '/home/johs/Multi-light_source_USB-Connection/dataset/train'
valid_dir = '/home/johs/Multi-light_source_USB-Connection/dataset/valid'
train_dataset = MultiViewDataset(root_dir=train_dir)
valid_dataset = MultiViewDataset(root_dir=valid_dir)
train_loader = DataLoader(train_dataset, batch_size=2 ** 6, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=2 ** 6, shuffle=False, num_workers=num_workers)

# Set up model and tools
model = AWTClassifier(output_size=5).to(device)

# weights = torch.tensor([2, 2, 1, 1, 1], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)

weight_path1 = f'weights/Proposed_{epochs}_seed{SEED}_Transformer.pth'
weight_path2 = f'weights/Proposed_{epochs}_seed{SEED}_Attention.pth'

print(f"\nTraining model {model.__class__.__name__}, GPU: {CUDA}, "
      f"Criterion: {criterion}, Optimizer: {optimizer.__class__.__name__}, Weight path: {os.path.basename(weight_path1)}\n")

total_time = time.time()
best_accuracy1, best_accuracy2 = 0.0, 0.0
for epoch in range(1, epochs + 1):

    # Train model
    model.train()

    epoch_loss = 0.0
    train_output1, train_output2, train_targets = [], [], []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        losses = torch.tensor(0.0, requires_grad=True).to(device)

        logit1, logit2 = model(images)
        loss1 = criterion(logit1, labels)  # logit1의 Loss
        loss2 = criterion(logit2, labels)  # logit2의 Loss
        total_loss = 0.5 * loss1 + 0.5 * loss2

        # Backward pass, Update weights
        optimizer.zero_grad()
        total_loss.backward()  # 두 Loss의 총합에 대해 역전파 수행
        optimizer.step()

        _, pred1 = torch.max(logit1, 1)
        _, pred2 = torch.max(logit2, 1)
        train_output1.extend(pred1.cpu().numpy())
        train_output2.extend(pred2.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

        epoch_loss += losses.item()

    train_matrix1 = confusion_matrix(train_targets, train_output1)
    calculate_accuracies(train_matrix1, epoch)
    train_matrix2 = confusion_matrix(train_targets, train_output2)
    calculate_accuracies(train_matrix2, epoch)

    if epoch % 10 == 0:
        print(f"\nEpoch {epoch}, Loss: {epoch_loss: .4f}, Weight: {weight_path1}, Confusion matrix: \n{train_matrix1}")
        print(f"\nEpoch {epoch}, Loss: {epoch_loss: .4f}, Weight: {weight_path2}, Confusion matrix: \n{train_matrix2}")

    if epoch % 100 == 0:
        print(f'\nEpoch: {epoch}/{epochs}. Middle training Time: {(time.time() - total_time) / 3600: .2f} hours\n')

    val_output1, val_output2, val_targets = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            logit1, logit2 = model(images)

            _, pred1 = torch.max(logit1, 1)
            _, pred2 = torch.max(logit2, 1)
            val_output1.extend(pred1.cpu().numpy())
            val_output2.extend(pred2.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_matrix1 = confusion_matrix(val_targets, val_output1)
    calculate_accuracies(val_matrix1, epoch)
    val_matrix2 = confusion_matrix(val_targets, val_output2)
    calculate_accuracies(val_matrix2, epoch)

    val_accuracy1 = accuracy_score(val_targets, val_output1)
    val_accuracy2 = accuracy_score(val_targets, val_output1)

    # accuracy = accuracy_score(targets, preds)
    if val_accuracy1 > best_accuracy1:
        best_accuracy1 = val_accuracy1
        torch.save(model.state_dict(), weight_path1)
        print(f"Saved weight_path1: {weight_path1}\n")

    if val_accuracy2 > best_accuracy2:
        best_accuracy2 = val_accuracy2
        torch.save(model.state_dict(), weight_path2)
        print(f"Saved weight_path2: {weight_path2}\n")

print(f'\nTraining end. Total epoch: {epochs}, Total training Time: {((time.time() - total_time) / 3600): .2f} hours\n')
