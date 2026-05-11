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

class FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, x_train, y_train, x_val=None, y_val=None, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model = model.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    losses = []
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
        losses.append(epoch_loss)

        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                pred = model(x_val.to(device))
                acc = (pred.argmax(1) == y_val.to(device)).float().mean().item()
            val_accs.append(acc)

    return model, losses, val_accs

model_cnn = CNN()
model_cnn, losses_cnn, accs_cnn = train_model(model_cnn, x_train, y_train, x_test, y_test)
final_acc_cnn = accs_cnn[-1] if len(accs_cnn) > 0 else 0.0
print(f"CNN: Test Accuracy = {final_acc_cnn:.4f}")

model_fc = FC()
model_fc, losses_fc, accs_fc = train_model(model_fc, x_train, y_train, x_test, y_test)
final_acc_fc = accs_fc[-1] if len(accs_fc) > 0 else 0.0
print(f"FC: Test Accuracy = {final_acc_fc:.4f}")

class CNNPool(nn.Module):
    def __init__(self, pool_type='max', pool_size=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(pool_size) if pool_type == 'max' else nn.AvgPool2d(pool_size)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

for name, pool_type in [('MaxPool 2x2', 'max'), ('AvgPool 2x2', 'avg')]:
    model = CNNPool(pool_type)
    model, losses, accs = train_model(model, x_train, y_train, x_test, y_test)
    final_acc = accs[-1] if len(accs) > 0 else 0.0
    print(f"{name}: Test Accuracy = {final_acc:.4f}")

class CNNKernel(nn.Module):
    def __init__(self, kernel, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel, padding=kernel//2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(filters, 64, kernel, padding=kernel//2)
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

for kernel, filters in [(3, 32), (5, 32), (3, 64)]:
    model = CNNKernel(kernel, filters)
    model, losses, accs = train_model(model, x_train, y_train, x_test, y_test)
    final_acc = accs[-1] if len(accs) > 0 else 0.0
    print(f"Kernel {kernel}, Filters {filters}: Test Accuracy = {final_acc:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses_cnn, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('EXP 2: CNN Training Curves')

plt.subplot(1, 2, 2)
if len(accs_cnn) > 0:
    plt.plot(accs_cnn, marker='o', label='Val Accuracy')
else:
    plt.plot([0], [final_acc_cnn], marker='o', label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('EXP 2: CNN Accuracy')
plt.tight_layout()
plt.savefig('results/exp2_cnn.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 2 Complete: CNN architecture analysis")
