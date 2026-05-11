import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x_train = torch.stack([mnist_train[i][0] for i in range(min(5000, len(mnist_train)))])
y_train = torch.tensor([mnist_train[i][1] for i in range(min(5000, len(mnist_train)))])
x_test = torch.stack([mnist_test[i][0] for i in range(min(1000, len(mnist_test)))])
y_test = torch.tensor([mnist_test[i][1] for i in range(min(1000, len(mnist_test)))])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
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

def train_model(model, x_train, y_train, lr=0.001, epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    losses = []
    
    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    
    return model, losses

learning_rates = [0.001, 0.01, 0.1]
lr_histories = {}

for lr in learning_rates:
    model = CNN()
    model, losses = train_model(model, x_train, y_train, lr=lr, epochs=15)
    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(device))
        acc = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    lr_histories[lr] = losses
    print(f"LR {lr}: Test Accuracy = {acc:.4f}, Best Loss Epoch = {np.argmin(losses)}")

epoch_ranges = [5, 10, 15, 20]
epoch_histories = {}

for epochs in epoch_ranges:
    model = CNN()
    model, losses = train_model(model, x_train, y_train, epochs=epochs)
    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(device))
        acc = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    epoch_histories[epochs] = {'accuracy': acc, 'losses': losses}
    overfit_gap = 0.5 - acc if acc < 0.5 else 0
    print(f"Epochs {epochs}: Test Accuracy = {acc:.4f}, Overfit Gap = {overfit_gap:.4f}")

model_baseline = CNN()
model_baseline, losses_baseline = train_model(model_baseline, x_train, y_train, epochs=20)

hyperparam_configs = [
    {'lr': 0.001, 'batch_size': 16},
    {'lr': 0.01, 'batch_size': 32},
    {'lr': 0.1, 'batch_size': 64}
]

hyperparam_results = {}

for config in hyperparam_configs:
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    model = model.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config['batch_size'], shuffle=True)
    
    for epoch in range(15):
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
    hyperparam_results[str(config)] = acc
    print(f"Config {config}: Test Accuracy = {acc:.4f}")

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
for lr in learning_rates:
    plt.plot(lr_histories[lr], label=f'LR {lr}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 5: Learning Rate Impact on Loss')

plt.subplot(2, 3, 2)
for lr in learning_rates:
    plt.plot(lr_histories[lr], label=f'LR {lr}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 5: LR Impact on Convergence')

plt.subplot(2, 3, 3)
plt.plot(losses_baseline, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 5: Train vs Validation Loss')

plt.subplot(2, 3, 4)
for epochs in epoch_ranges:
    plt.plot(epoch_histories[epochs]['losses'], label=f'{epochs} Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 5: Epoch Count Effect')

plt.subplot(2, 3, 5)
plt.plot(losses_baseline, label='Train Acc')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 5: Generalization Analysis')

plt.subplot(2, 3, 6)
configs_str = list(hyperparam_results.keys())
accs = list(hyperparam_results.values())
plt.bar(range(len(configs_str)), accs)
plt.xticks(range(len(configs_str)), ['LR:0.001', 'LR:0.01', 'LR:0.1'], rotation=45)
plt.ylabel('Accuracy')
plt.title('EXP 5: Hyperparameter Comparison')
plt.tight_layout()
plt.savefig('results/exp5_performance_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 5 Complete: Performance and hyperparameter analysis")
