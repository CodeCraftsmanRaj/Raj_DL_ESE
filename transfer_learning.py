import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use a deterministic seed and correlated labels so results are stable and
# behave sensibly when dataset size changes.
torch.manual_seed(0)
x_train = torch.randn(100, 3, 224, 224)
# derive labels correlated with input to avoid random-label noise
y_train = (x_train.mean(dim=(1, 2, 3)) > 0.0).long()
x_val = torch.randn(20, 3, 224, 224)
y_val = (x_val.mean(dim=(1, 2, 3)) > 0.0).long()

base_model = models.mobilenet_v2(pretrained=True)

class TransferWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.base.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

for param in base_model.parameters():
    param.requires_grad = False

model_frozen = TransferWrapper(base_model)

def train_model(model, x_train, y_train, x_val=None, y_val=None, epochs=5, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.float().unsqueeze(1).to(device)
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
                val_out = model(x_val.to(device))
                val_acc = ((val_out.squeeze() > 0.5) == y_val.to(device)).float().mean().item()
            val_accs.append(val_acc)

    final_val = val_accs[-1] if len(val_accs) > 0 else 0.0
    return model, losses, val_accs

model_frozen, losses_frozen, val_accs_frozen = train_model(model_frozen, x_train, y_train, x_val, y_val)
print(f"Frozen: Val Accuracy = {val_accs_frozen[-1]:.4f}")

base_model2 = models.mobilenet_v2(pretrained=True)
for param in list(base_model2.parameters())[:-20]:
    param.requires_grad = False
for param in list(base_model2.parameters())[-20:]:
    param.requires_grad = True

model_finetune = TransferWrapper(base_model2)

model_finetune, losses_finetune, val_accs_finetune = train_model(model_finetune, x_train, y_train, x_val, y_val, lr=1e-5)
print(f"Fine-tuned: Val Accuracy = {val_accs_finetune[-1]:.4f}")

x_small = torch.randn(50, 3, 224, 224)
y_small = torch.randint(0, 2, (50,))
x_small_val = torch.randn(20, 3, 224, 224)
y_small_val = torch.randint(0, 2, (20,))

base_model3 = models.mobilenet_v2(pretrained=True)
for param in base_model3.parameters():
    param.requires_grad = False

model_small = TransferWrapper(base_model3)

model_small, losses_small, val_accs_small = train_model(model_small, x_small, y_small, x_small_val, y_small_val)
print(f"Small Dataset: Val Accuracy = {val_accs_small[-1]:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses_frozen, label='Frozen')
plt.plot(losses_finetune, label='Fine-tuned')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 3: Frozen vs Fine-tuned')

plt.subplot(1, 3, 2)
plt.plot(losses_frozen, label='Frozen Loss')
plt.plot(losses_finetune, label='Fine-tuned Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 3: Training Loss Comparison')

plt.subplot(1, 3, 3)
plt.bar(['Frozen', 'Fine-tuned', 'Small Dataset'], [val_accs_frozen[-1] if len(val_accs_frozen)>0 else 0.0, val_accs_finetune[-1] if len(val_accs_finetune)>0 else 0.0, val_accs_small[-1] if len(val_accs_small)>0 else 0.0])
plt.ylabel('Accuracy')
plt.title('EXP 3: Performance Summary')
plt.tight_layout()
plt.savefig('results/exp3_transfer_learning.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 3 Complete: Transfer Learning analysis")
