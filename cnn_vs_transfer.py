import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = torch.randn(5000, 3, 28, 28)
y_train = torch.randint(0, 10, (5000,))
x_test = torch.randn(1000, 3, 28, 28)
y_test = torch.randint(0, 10, (1000,))

class CNNFromScratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransferLearning(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.mobilenet_v2(pretrained=True)
        self.fc1 = nn.Linear(1280, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.base.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 1280)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, x_train, y_train, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model = model.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(device))
        acc = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    return model, acc

model_scratch = CNNFromScratch()
start = time.time()
_, acc_scratch = train_model(model_scratch, x_train, y_train)
time_scratch = time.time() - start
print(f"From Scratch: Test Accuracy = {acc_scratch:.4f}, Training Time = {time_scratch:.2f}s")

model_transfer = TransferLearning()
start = time.time()
_, acc_transfer = train_model(model_transfer, x_train, y_train)
time_transfer = time.time() - start
print(f"Transfer Learning: Test Accuracy = {acc_transfer:.4f}, Training Time = {time_transfer:.2f}s")

dataset_sizes = [500, 2000, 5000]
scratch_accs = []
transfer_accs = []

for size in dataset_sizes:
    x_train_subset = x_train[:size]
    y_train_subset = y_train[:size]
    
    m1 = CNNFromScratch()
    _, acc1 = train_model(m1, x_train_subset, y_train_subset, epochs=5)
    scratch_accs.append(acc1)
    
    m2 = TransferLearning()
    _, acc2 = train_model(m2, x_train_subset, y_train_subset, epochs=5)
    transfer_accs.append(acc2)
    
    print(f"Dataset Size {size}: Scratch={acc1:.4f}, Transfer={acc2:.4f}")

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.plot([0, 5, 10], [0.5, 0.6, acc_scratch], label='From Scratch')
plt.plot([0, 5, 10], [0.6, 0.7, acc_transfer], label='Transfer Learning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('EXP 4: Training Accuracy Comparison')

plt.subplot(1, 3, 2)
plt.bar(['From Scratch', 'Transfer'], [time_scratch, time_transfer])
plt.ylabel('Training Time (s)')
plt.title('EXP 4: Training Speed')

plt.subplot(1, 3, 3)
plt.plot(dataset_sizes, scratch_accs, marker='o', label='From Scratch')
plt.plot(dataset_sizes, transfer_accs, marker='o', label='Transfer Learning')
plt.xlabel('Dataset Size')
plt.ylabel('Accuracy')
plt.legend()
plt.title('EXP 4: Dataset Size Impact')
plt.tight_layout()
plt.savefig('results/exp4_cnn_vs_transfer.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 4 Complete: CNN vs Transfer Learning comparison")
