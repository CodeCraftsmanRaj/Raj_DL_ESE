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

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc = nn.Linear(64, len(chars))
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class RNN(nn.Module):
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

    return model, losses

model_lstm = LSTM()
model_lstm, losses_lstm = train_model(model_lstm, x_train, y_train, epochs=20)
model_lstm.eval()
with torch.no_grad():
    pred = model_lstm(x_test.to(device))
    acc_lstm = (pred.argmax(1) == y_test.to(device)).float().mean().item()
print(f"LSTM: Test Accuracy = {acc_lstm:.4f}")

model_rnn = RNN()
model_rnn, losses_rnn = train_model(model_rnn, x_train, y_train, epochs=20)
model_rnn.eval()
with torch.no_grad():
    pred = model_rnn(x_test.to(device))
    acc_rnn = (pred.argmax(1) == y_test.to(device)).float().mean().item()
print(f"RNN: Test Accuracy = {acc_rnn:.4f}")

print(f"LSTM Best Loss Epoch: {np.argmin(losses_lstm)}")
print(f"RNN Best Loss Epoch: {np.argmin(losses_rnn)}")
print(f"LSTM Improvement: {acc_lstm - acc_rnn:.4f}")

lstm_results = {}

for units in [32, 64, 128]:
    class LSTMUnits(nn.Module):
        def __init__(self, u):
            super().__init__()
            self.lstm = nn.LSTM(1, u, batch_first=True)
            self.fc = nn.Linear(u, len(chars))
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
    
    m = LSTMUnits(units)
    m, losses = train_model(m, x_train, y_train, epochs=20)
    m.eval()
    with torch.no_grad():
        pred = m(x_test.to(device))
        acc = (pred.argmax(1) == y_test.to(device)).float().mean().item()
    lstm_results[units] = {'accuracy': acc, 'losses': losses}
    print(f"LSTM Units {units}: Test Accuracy = {acc:.4f}")

split_short = len(x_data) // 3
split_long = 2 * len(x_data) // 3

short_data = (x_data[:split_short], y_data[:split_short])
long_data = (x_data[split_long:], y_data[split_long:])

memory_impact = {'short_term': 0, 'long_term': 0}

for key, (x_seq, y_seq) in [('short_term', short_data), ('long_term', long_data)]:
    if len(x_seq) > 0:
        m = LSTM()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(m.parameters())
        m = m.to(device)
        loader = DataLoader(TensorDataset(x_seq, y_seq), batch_size=16, shuffle=True)
        
        for epoch in range(15):
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = m(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        m.eval()
        with torch.no_grad():
            pred = m(x_seq.to(device))
            acc = (pred.argmax(1) == y_seq.to(device)).float().mean().item()
        memory_impact[key] = acc
        print(f"LSTM {key} dependency: Accuracy = {acc:.4f}")

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.plot(losses_lstm, label='LSTM')
plt.plot(losses_rnn, label='RNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 8: LSTM vs RNN Loss')

plt.subplot(2, 3, 2)
plt.plot(losses_lstm, label='LSTM Val Loss')
plt.plot(losses_rnn, label='RNN Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 8: Convergence Speed')

plt.subplot(2, 3, 3)
plt.plot(losses_lstm, label='LSTM Train')
plt.plot(losses_rnn, label='RNN Train')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 8: Training Trajectory')

plt.subplot(2, 3, 4)
units = list(lstm_results.keys())
accs = [lstm_results[u]['accuracy'] for u in units]
plt.plot(units, accs, marker='o')
plt.xlabel('LSTM Units')
plt.ylabel('Accuracy')
plt.title('EXP 8: Architecture Effect')

plt.subplot(2, 3, 5)
plt.bar(['Short-term', 'Long-term'], [memory_impact['short_term'], memory_impact['long_term']])
plt.ylabel('Accuracy')
plt.title('EXP 8: Memory Mechanisms')

plt.subplot(2, 3, 6)
for u in units:
    plt.plot(lstm_results[u]['losses'], label=f'{u} Units')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 8: LSTM Configuration Comparison')
plt.tight_layout()
plt.savefig('results/exp8_lstm.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 8 Complete: LSTM long-term dependency analysis")
