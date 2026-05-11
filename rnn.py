import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with open('data.txt', 'w') as f:
    f.write('the quick brown fox jumps over the lazy dog ' * 50)

with open('data.txt', 'r') as f:
    text = f.read()

chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

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

model_rnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist_rnn = model_rnn.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
loss_rnn, acc_rnn = model_rnn.evaluate(x_test, y_test, verbose=0)
print(f"RNN: Test Accuracy = {acc_rnn:.4f}")

seq_lengths = [10, 20, 40]
seq_results = {}

for seq_len in seq_lengths:
    x_temp, y_temp = [], []
    for i in range(len(text) - seq_len):
        x_temp.append([char_to_idx[c] for c in text[i:i+seq_len]])
        y_temp.append(char_to_idx[text[i+seq_len]])
    
    x_temp = np.array(x_temp) / len(chars)
    y_temp = tf.keras.utils.to_categorical(y_temp, len(chars))
    
    split = int(0.8 * len(x_temp))
    x_tr, x_te = x_temp[:split], x_temp[split:]
    y_tr, y_te = y_temp[:split], y_temp[split:]
    
    m = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(seq_len, 1)),
        tf.keras.layers.Dense(len(chars), activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    h = m.fit(x_tr, y_tr, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    _, acc = m.evaluate(x_te, y_te, verbose=0)
    seq_results[seq_len] = {'accuracy': acc, 'history': h}
    print(f"Seq Len {seq_len}: Test Accuracy = {acc:.4f}")

short_text = 'the quick brown fox '
long_text = 'the quick brown fox jumps over the lazy dog ' * 5

for text_type, txt in [('Short', short_text), ('Long', long_text)]:
    local_chars = sorted(set(txt))
    local_char_to_idx = {c: i for i, c in enumerate(local_chars)}
    
    x_local, y_local = [], []
    for i in range(len(txt) - 20):
        x_local.append([local_char_to_idx[c] for c in txt[i:i+20]])
        y_local.append(local_char_to_idx[txt[i+20]])
    
    if len(x_local) > 0:
        x_local = np.array(x_local) / len(local_chars)
        y_local = tf.keras.utils.to_categorical(y_local, len(local_chars))
        
        m = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(20, 1)),
            tf.keras.layers.Dense(len(local_chars), activation='softmax')
        ])
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        h = m.fit(x_local, y_local, epochs=15, batch_size=16, verbose=0)
        _, acc = m.evaluate(x_local, y_local, verbose=0)
        print(f"{text_type} Sequence Performance: Accuracy = {acc:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(hist_rnn.history['loss'], label='Train Loss')
plt.plot(hist_rnn.history['val_loss'], label='Val Loss')
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
    plt.plot(seq_results[seq_len]['history'].history['loss'], label=f'Seq Len {seq_len}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 7: Convergence Comparison')
plt.tight_layout()
plt.savefig('exp7_rnn.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 7 Complete: Simple RNN sequence prediction")
