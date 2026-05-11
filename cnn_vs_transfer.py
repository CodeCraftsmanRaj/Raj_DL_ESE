import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import models

# Prefer GPU when available but avoid small-GPU OOM by falling back to CPU
if torch.cuda.is_available():
    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        total_mem = 0
    # If GPU < 8GB, use CPU to avoid OOM for MobileNet upsampling
    if total_mem and total_mem < 8 * 1024 ** 3:
        print("GPU detected but memory is small — using CPU to avoid OOM for transfer experiments")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.manual_seed(0)
# synthetic inputs and deterministic, correlated labels to avoid random-label noise
x_train = torch.randn(5000, 3, 28, 28)
flat = x_train.view(x_train.size(0), -1)
proj = torch.randn(flat.size(1), 10)
y_train = flat.matmul(proj).argmax(dim=1)
x_test = torch.randn(1000, 3, 28, 28)
flat_t = x_test.view(x_test.size(0), -1)
y_test = flat_t.matmul(proj).argmax(dim=1)

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
        # Upsample to the size MobileNet expects
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.base.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 1280)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, x_train, y_train, x_val=None, y_val=None, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    epoch_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(epoch_loss)

        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                pred = model(x_val.to(device))
                acc = (pred.argmax(1) == y_val.to(device)).float().mean().item()
            val_accs.append(acc)

    final_val = val_accs[-1] if len(val_accs) > 0 else 0.0
    return model, final_val, epoch_losses, val_accs

model_scratch = CNNFromScratch()
start = time.time()
model_scratch, acc_scratch, losses_scratch, accs_scratch = train_model(model_scratch, x_train, y_train, x_test, y_test, epochs=10, lr=1e-3)
time_scratch = time.time() - start
model_scratch.eval()
with torch.no_grad():
    pred = model_scratch(x_test.to(device))
    acc_scratch_final = (pred.argmax(1) == y_test.to(device)).float().mean().item()
print(f"From Scratch: Test Accuracy = {acc_scratch_final:.4f}, Training Time = {time_scratch:.2f}s")

model_transfer = TransferLearning()
# freeze base to simulate feature reuse
for p in model_transfer.base.parameters():
    p.requires_grad = False
start = time.time()
model_transfer, acc_transfer, losses_transfer, accs_transfer = train_model(model_transfer, x_train, y_train, x_test, y_test, epochs=10, lr=1e-3)
time_transfer = time.time() - start
model_transfer.eval()
with torch.no_grad():
    pred = model_transfer(x_test.to(device))
    acc_transfer_final = (pred.argmax(1) == y_test.to(device)).float().mean().item()
print(f"Transfer Learning: Test Accuracy = {acc_transfer_final:.4f}, Training Time = {time_transfer:.2f}s")

dataset_sizes = [500, 2000, 5000]
scratch_accs = []
transfer_accs = []

for size in dataset_sizes:
    x_train_subset = x_train[:size]
    y_train_subset = y_train[:size]
    
    m1 = CNNFromScratch()
    m1, acc1, _, _ = train_model(m1, x_train_subset, y_train_subset, x_test, y_test, epochs=5, lr=1e-3)
    m1.eval()
    with torch.no_grad():
        pred = m1(x_test.to(device))
        acc1 = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    scratch_accs.append(acc1)
    
    m2 = TransferLearning()
    for p in m2.base.parameters():
        p.requires_grad = False
    m2, acc2, _, _ = train_model(m2, x_train_subset, y_train_subset, x_test, y_test, epochs=5, lr=1e-3)
    m2.eval()
    with torch.no_grad():
        pred = m2(x_test.to(device))
        acc2 = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    transfer_accs.append(acc2)
    
    print(f"Dataset Size {size}: Scratch={acc1:.4f}, Transfer={acc2:.4f}")

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
if len(accs_scratch) > 0:
    plt.plot(accs_scratch, marker='o', label='From Scratch')
else:
    plt.plot([acc_scratch_final], marker='o', label='From Scratch')
if len(accs_transfer) > 0:
    plt.plot(accs_transfer, marker='o', label='Transfer Learning')
else:
    plt.plot([acc_transfer_final], marker='o', label='Transfer Learning')
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
