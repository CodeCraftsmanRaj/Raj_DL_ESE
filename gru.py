import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

with open('data.txt', 'w') as f:
    f.write('the quick brown fox jumps over the lazy dog ' * 50)

with open('data.txt', 'r') as f:
    text = f.read()

chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}

seq_length = 20
x_data, y_data = [], []

for i in range(len(text) - seq_length):
    x_data.append([char_to_idx[c] for c in text[i:i+seq_length]])
    y_data.append(char_to_idx[text[i+seq_length]])

x_data = np.array(x_data) / len(chars)
y_data = tf.keras.utils.to_categorical(y_data, len(chars))

split = int(0.8 * len(x_data))
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

model_gru = tf.keras.Sequential([
    tf.keras.layers.GRU(64, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model_gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
start_gru = time.time()
hist_gru = model_gru.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
time_gru = time.time() - start_gru
loss_gru, acc_gru = model_gru.evaluate(x_test, y_test, verbose=0)
print(f"GRU: Test Accuracy = {acc_gru:.4f}, Training Time = {time_gru:.2f}s")

model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
start_lstm = time.time()
hist_lstm = model_lstm.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
time_lstm = time.time() - start_lstm
loss_lstm, acc_lstm = model_lstm.evaluate(x_test, y_test, verbose=0)
print(f"LSTM: Test Accuracy = {acc_lstm:.4f}, Training Time = {time_lstm:.2f}s")

print(f"GRU vs LSTM - Accuracy Diff: {acc_gru - acc_lstm:.4f}, Speed Improvement: {time_lstm/time_gru:.2f}x")

gru_results = {}

for units in [32, 64, 128]:
    m = tf.keras.Sequential([
        tf.keras.layers.GRU(units, activation='relu', input_shape=(seq_length, 1)),
        tf.keras.layers.Dense(len(chars), activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    h = m.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    _, acc = m.evaluate(x_test, y_test, verbose=0)
    gru_results[units] = {'accuracy': acc, 'history': h}
    print(f"GRU Units {units}: Test Accuracy = {acc:.4f}")

efficiency_configs = [
    {'units': 32, 'epochs': 10},
    {'units': 64, 'epochs': 20},
    {'units': 128, 'epochs': 30}
]

efficiency_results = {}

for config in efficiency_configs:
    m = tf.keras.Sequential([
        tf.keras.layers.GRU(config['units'], activation='relu', input_shape=(seq_length, 1)),
        tf.keras.layers.Dense(len(chars), activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    h = m.fit(x_train, y_train, epochs=config['epochs'], batch_size=32, validation_split=0.2, verbose=0)
    train_time = time.time() - start
    _, acc = m.evaluate(x_test, y_test, verbose=0)
    efficiency_results[str(config)] = {'accuracy': acc, 'time': train_time}
    print(f"Config {config}: Accuracy = {acc:.4f}, Time = {train_time:.2f}s")

realtime_model = tf.keras.Sequential([
    tf.keras.layers.GRU(32, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

realtime_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
realtime_model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

inference_times = []
for _ in range(100):
    start = time.time()
    _ = realtime_model.predict(x_test[:10], verbose=0)
    inference_times.append(time.time() - start)

avg_inference_time = np.mean(inference_times)
print(f"Avg Inference Time per batch: {avg_inference_time*1000:.2f}ms - Suitable for real-time: {avg_inference_time < 0.01}")

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.plot(hist_gru.history['loss'], label='GRU')
plt.plot(hist_lstm.history['loss'], label='LSTM')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 9: GRU vs LSTM Loss')

plt.subplot(2, 3, 2)
plt.bar(['GRU', 'LSTM'], [time_gru, time_lstm])
plt.ylabel('Training Time (s)')
plt.title('EXP 9: Training Speed')

plt.subplot(2, 3, 3)
plt.plot(hist_gru.history['accuracy'], label='GRU Train')
plt.plot(hist_gru.history['val_accuracy'], label='GRU Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('EXP 9: GRU Performance')

plt.subplot(2, 3, 4)
units = list(gru_results.keys())
accs = [gru_results[u]['accuracy'] for u in units]
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
plt.savefig('exp9_gru.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 9 Complete: GRU efficiency analysis")
