import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model_rnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist_lstm = model_lstm.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
loss_lstm, acc_lstm = model_lstm.evaluate(x_test, y_test, verbose=0)
print(f"LSTM: Test Accuracy = {acc_lstm:.4f}")

model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist_rnn = model_rnn.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
loss_rnn, acc_rnn = model_rnn.evaluate(x_test, y_test, verbose=0)
print(f"RNN: Test Accuracy = {acc_rnn:.4f}")

print(f"LSTM Best Val Loss Epoch: {np.argmin(hist_lstm.history['val_loss'])}")
print(f"RNN Best Val Loss Epoch: {np.argmin(hist_rnn.history['val_loss'])}")

print(f"LSTM Improvement: {acc_lstm - acc_rnn:.4f}")

layer_configs = [
    {'units': 32},
    {'units': 64},
    {'units': 128}
]

lstm_results = {}

for config in layer_configs:
    m = tf.keras.Sequential([
        tf.keras.layers.LSTM(config['units'], activation='relu', input_shape=(seq_length, 1)),
        tf.keras.layers.Dense(len(chars), activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    h = m.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    _, acc = m.evaluate(x_test, y_test, verbose=0)
    lstm_results[config['units']] = {'accuracy': acc, 'history': h}
    print(f"LSTM Units {config['units']}: Test Accuracy = {acc:.4f}")

memory_impact = {
    'short_term': [],
    'long_term': []
}

short_seq_data = x_data[:len(x_data)//2]
long_seq_data = x_data[len(x_data)//2:]

for seq_data, key in [(short_seq_data, 'short_term'), (long_seq_data, 'long_term')]:
    if len(seq_data) > 0:
        split = int(0.8 * len(seq_data))
        x_tr = seq_data[:split]
        x_te = seq_data[split:]
        y_tr = y_data[:len(x_tr)]
        y_te = y_data[len(x_tr):len(x_tr)+len(x_te)]
        
        m = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, activation='relu', input_shape=(seq_length, 1)),
            tf.keras.layers.Dense(len(chars), activation='softmax')
        ])
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        m.fit(x_tr, y_tr, epochs=15, batch_size=32, verbose=0)
        _, acc = m.evaluate(x_te, y_te, verbose=0)
        memory_impact[key] = acc
        print(f"LSTM {key} dependency: Accuracy = {acc:.4f}")

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.plot(hist_lstm.history['loss'], label='LSTM')
plt.plot(hist_rnn.history['loss'], label='RNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 8: LSTM vs RNN Loss')

plt.subplot(2, 3, 2)
plt.plot(hist_lstm.history['val_loss'], label='LSTM Val Loss')
plt.plot(hist_rnn.history['val_loss'], label='RNN Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('EXP 8: Convergence Speed')

plt.subplot(2, 3, 3)
plt.plot(hist_lstm.history['accuracy'], label='LSTM Train')
plt.plot(hist_lstm.history['val_accuracy'], label='LSTM Val')
plt.plot(hist_rnn.history['accuracy'], label='RNN Train')
plt.plot(hist_rnn.history['val_accuracy'], label='RNN Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
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
for units in units:
    plt.plot(lstm_results[units]['history'].history['val_loss'], label=f'{units} Units')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('EXP 8: LSTM Configuration Comparison')
plt.tight_layout()
plt.savefig('exp8_lstm.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 8 Complete: LSTM long-term dependency analysis")
