import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

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

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 64, batch_first=True)
        self.fc = nn.Linear(64, len(chars))
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc = nn.Linear(64, len(chars))
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, x_train, y_train, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model = model.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    losses = []

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

    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(device))
        acc = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    return model, acc, losses

model_gru = GRU()
start_gru = time.time()
model_gru, acc_gru, losses_gru = train_model(model_gru, x_train, y_train, epochs=20)
time_gru = time.time() - start_gru
print(f"GRU: Test Accuracy = {acc_gru:.4f}, Training Time = {time_gru:.2f}s")

model_lstm = LSTM()
start_lstm = time.time()
model_lstm, acc_lstm, losses_lstm = train_model(model_lstm, x_train, y_train, epochs=20)
time_lstm = time.time() - start_lstm
print(f"LSTM: Test Accuracy = {acc_lstm:.4f}, Training Time = {time_lstm:.2f}s")

print(f"GRU vs LSTM - Accuracy Diff: {acc_gru - acc_lstm:.4f}, Speed Improvement: {time_lstm/time_gru:.2f}x")

gru_results = {}

for units in [32, 64, 128]:
    class GRUUnits(nn.Module):
        def __init__(self, u):
            super().__init__()
            self.gru = nn.GRU(1, u, batch_first=True)
            self.fc = nn.Linear(u, len(chars))
        
        def forward(self, x):
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])
            return out
    
    m = GRUUnits(units)
    _, acc, _ = train_model(m, x_train, y_train, epochs=20)
    gru_results[units] = acc
    print(f"GRU Units {units}: Test Accuracy = {acc:.4f}")

efficiency_configs = [
    {'units': 32, 'epochs': 10},
    {'units': 64, 'epochs': 20},
    {'units': 128, 'epochs': 30}
]

efficiency_results = {}

for config in efficiency_configs:
    class GRUEff(nn.Module):
        def __init__(self, u):
            super().__init__()
            self.gru = nn.GRU(1, u, batch_first=True)
            self.fc = nn.Linear(u, len(chars))
        
        def forward(self, x):
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])
            return out
    
    m = GRUEff(config['units'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(m.parameters())
    m = m.to(device)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    
    start = time.time()
    for epoch in range(config['epochs']):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = m(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    
    m.eval()
    with torch.no_grad():
        pred = m(x_test.to(device))
        acc = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    
    efficiency_results[str(config)] = {'accuracy': acc, 'time': train_time}
    print(f"Config {config}: Accuracy = {acc:.4f}, Time = {train_time:.2f}s")

inference_times = []
realtime_model = GRUUnits(32)
realtime_model = realtime_model.to(device)

for _ in range(100):
    start = time.time()
    with torch.no_grad():
        _ = realtime_model(x_test[:10].to(device))
    inference_times.append(time.time() - start)

avg_inference_time = np.mean(inference_times)
print(f"Avg Inference Time per batch: {avg_inference_time*1000:.2f}ms - Suitable for real-time: {avg_inference_time < 0.01}")

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
# Use epoch-averaged losses collected during training for smoother curves
plt.plot(losses_gru, label='GRU')
plt.plot(losses_lstm, label='LSTM')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 9: GRU vs LSTM Loss')

plt.subplot(2, 3, 2)
plt.bar(['GRU', 'LSTM'], [time_gru, time_lstm])
plt.ylabel('Training Time (s)')
plt.title('EXP 9: Training Speed')

plt.subplot(2, 3, 3)
plt.plot(losses_gru[:50], label='GRU')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 9: GRU Performance')

plt.subplot(2, 3, 4)
units = list(gru_results.keys())
accs = [gru_results[u] for u in units]
plt.plot(units, accs, marker='o')
plt.xlabel('GRU Units')
plt.ylabel('Accuracy')
plt.title('EXP 9: Architecture Scaling')

plt.subplot(2, 3, 5)
configs_str = list(efficiency_results.keys())
eff_accs = [efficiency_results[c]['accuracy'] for c in configs_str]
eff_times = [efficiency_results[c]['time'] for c in configs_str]
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.bar(range(len(configs_str)), eff_accs, alpha=0.7, label='Accuracy')
ax2.plot(range(len(configs_str)), eff_times, 'ro-', label='Time')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Training Time (s)')
plt.title('EXP 9: Efficiency Trade-off')

plt.subplot(2, 3, 6)
plt.hist(inference_times, bins=20, edgecolor='black')
plt.xlabel('Inference Time (s)')
plt.ylabel('Frequency')
plt.title('EXP 9: Real-time Capability')
plt.tight_layout()
plt.savefig('results/exp9_gru.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 9 Complete: GRU efficiency analysis")
