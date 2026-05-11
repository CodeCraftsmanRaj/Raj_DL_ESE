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

x_train = x_train.view(-1, 784)
x_test = x_test.view(-1, 784)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, x_train, y_train, epochs=10, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model = model.to(device)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    losses = []
    
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    
    return model, losses

activations = ['relu', 'tanh', 'sigmoid']
histories = {}

for act in activations:
    class MLPActivation(nn.Module):
        def __init__(self, activation):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 10)
            
            if activation == 'relu':
                self.act = nn.ReLU()
            elif activation == 'tanh':
                self.act = nn.Tanh()
            else:
                self.act = nn.Sigmoid()
        
        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            x = self.fc4(x)
            return x
    
    model = MLPActivation(act)
    model, losses = train_model(model, x_train, y_train, epochs=10)
    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(device))
        acc = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    histories[act] = losses
    print(f"{act}: Test Accuracy = {acc:.4f}")

for act in activations:
    plt.plot(histories[act], label=f'{act} - Train Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('EXP 1: Activation Function Impact on Training Stability')
plt.savefig('exp1_mlp.png', dpi=100, bbox_inches='tight')
plt.close()

depths = [1, 2, 3, 4]
depth_results = {}

for d in depths:
    class MLPDepth(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(depth-1)])
            self.fc_out = nn.Linear(128, 10)
            self.depth = depth
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            for layer in self.layers:
                x = torch.relu(layer(x))
            x = self.fc_out(x)
            return x
    
    model = MLPDepth(d)
    model, losses = train_model(model, x_train, y_train, epochs=10)
    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(device))
        acc = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    depth_results[d] = {'accuracy': acc, 'val_loss': losses}
    print(f"Depth {d}: Test Accuracy = {acc:.4f}, Convergence Epochs = {np.argmin(losses)}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(depth_results.keys(), [v['accuracy'] for v in depth_results.values()])
plt.xlabel('Model Depth')
plt.ylabel('Accuracy')
plt.title('EXP 1: Depth vs Accuracy')

plt.subplot(1, 2, 2)
for d in depths:
    plt.plot(depth_results[d]['val_loss'], label=f'Depth {d}')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('EXP 1: Convergence by Depth')
plt.tight_layout()
plt.savefig('exp1_depth_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 1 Complete: MLP analysis with activation functions and depth variations")
