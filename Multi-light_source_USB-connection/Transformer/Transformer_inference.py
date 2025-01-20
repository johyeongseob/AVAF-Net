import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from torch.utils.data import DataLoader
from DataLoader import MultiViewDataset
from CLSClassifier import CLSClassifier
from util import *
from sklearn.metrics import confusion_matrix
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataSet, dataLoader
test_dir = '/home/johs/Multi-light_source_USB-Connection/dataset/test'
test_dataset = MultiViewDataset(root_dir=test_dir)
test_loader = DataLoader(test_dataset, batch_size=2 ** 6, shuffle=False)

weight_path = f'weights/Transformer_100_seed42_cls_dropout1.pth'

# Set up model and tools
model = CLSClassifier(num_classes=5).to(device)
model.load_state_dict(torch.load(weight_path))

print(f"\nTest model {model.__class__.__name__}, Weight path: {os.path.basename(weight_path)}\n")


model.eval()

with torch.no_grad():
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            _, pred = torch.max(logits, 1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    calculate_accuracies(confusion_matrix(targets, preds), 0)
