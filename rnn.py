import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

with open('data.txt', 'r') as f:
    text = f.read().strip()

if len(text) < 500:
    text = (
        "Machine learning systems are evaluated across data quality, model capacity, and optimization stability. "
        "In professional lab environments, results must be reproducible, measurable, and clearly interpretable. "
        "Sequence models are expected to learn context, preserve signal over longer spans, and avoid unstable training dynamics. "
        "A strong evaluation pipeline reports accuracy, convergence behavior, error modes, and computational efficiency. "
        "Teams should compare architectures under the same protocol, document assumptions, and justify hyperparameter choices. "
        "Reliable model development requires disciplined experiments, transparent reporting, and structured analysis of trade offs. "
    )

chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}

seq_length = 20
x_data, y_data = [], []

for i in range(len(text) - seq_length):
    x_data.append([char_to_idx[c] for c in text[i:i+seq_length]])
    y_data.append(char_to_idx[text[i+seq_length]])

x_data = torch.tensor(x_data, dtype=torch.float32).unsqueeze(-1) / len(chars)
y_data = torch.tensor(y_data)

split = int(0.8 * len(x_data))
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 64, batch_first=True)
        self.fc = nn.Linear(64, len(chars))
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, x_train, y_train, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
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

model_rnn = RNNModel()
model_rnn, losses_rnn = train_model(model_rnn, x_train, y_train, epochs=20)
model_rnn.eval()
with torch.no_grad():
    pred = model_rnn(x_test.to(device))
    acc = (pred.argmax(1) == y_test.to(device)).float().mean().item()
print(f"RNN: Test Accuracy = {acc:.4f}")

seq_lengths = [10, 20, 40]
seq_results = {}

for seq_len in seq_lengths:
    x_temp, y_temp = [], []
    for i in range(len(text) - seq_len):
        x_temp.append([char_to_idx[c] for c in text[i:i+seq_len]])
        y_temp.append(char_to_idx[text[i+seq_len]])
    
    x_temp = torch.tensor(x_temp, dtype=torch.float32).unsqueeze(-1) / len(chars)
    y_temp = torch.tensor(y_temp)
    
    split = int(0.8 * len(x_temp))
    x_tr, x_te = x_temp[:split], x_temp[split:]
    y_tr, y_te = y_temp[:split], y_temp[split:]
    
    class RNNLen(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(1, 64, batch_first=True)
            self.fc = nn.Linear(64, len(chars))
        
        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.fc(out[:, -1, :])
            return out
    
    m = RNNLen()
    m, losses = train_model(m, x_tr, y_tr, epochs=20)
    m.eval()
    with torch.no_grad():
        pred = m(x_te.to(device))
        acc = (pred.argmax(1) == y_te.to(device)).float().mean().item()
    seq_results[seq_len] = {'accuracy': acc, 'losses': losses}
    print(f"Seq Len {seq_len}: Test Accuracy = {acc:.4f}")

short_text = 'the quick brown fox '
long_text = 'the quick brown fox jumps over the lazy dog ' * 5

for text_type, txt in [('Short', short_text), ('Long', long_text)]:
    local_chars = sorted(set(txt))
    local_char_to_idx = {c: i for i, c in enumerate(local_chars)}
    
    x_local, y_local = [], []
    for i in range(max(1, len(txt) - 20)):
        x_local.append([local_char_to_idx[c] for c in txt[i:i+20]])
        y_local.append(local_char_to_idx[txt[i+20]] if i+20 < len(txt) else local_char_to_idx[txt[-1]])
    
    if len(x_local) > 0:
        x_local = torch.tensor(x_local, dtype=torch.float32).unsqueeze(-1) / len(local_chars)
        y_local = torch.tensor(y_local)
        
        class RNNText(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.RNN(1, 64, batch_first=True)
                self.fc = nn.Linear(64, len(local_chars))
            
            def forward(self, x):
                out, _ = self.rnn(x)
                out = self.fc(out[:, -1, :])
                return out
        
        m = RNNText()
        m, losses = train_model(m, x_local, y_local, epochs=15)
        m.eval()
        with torch.no_grad():
            pred = m(x_local.to(device))
            acc = (pred.argmax(1) == y_local.to(device)).float().mean().item()
        print(f"{text_type} Sequence Performance: Accuracy = {acc:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(losses_rnn, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 7: RNN Training Curves')

plt.subplot(1, 3, 2)
seq_accs = [seq_results[sl]['accuracy'] for sl in seq_lengths]
plt.plot(seq_lengths, seq_accs, marker='o')
plt.xlabel('Sequence Length')
plt.ylabel('Accuracy')
plt.title('EXP 7: Sequence Length Impact')

plt.subplot(1, 3, 3)
for seq_len in seq_lengths:
    plt.plot(seq_results[seq_len]['losses'], label=f'Seq Len {seq_len}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 7: Convergence Comparison')
plt.tight_layout()
plt.savefig('results/exp7_rnn.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 7 Complete: Simple RNN sequence prediction")
