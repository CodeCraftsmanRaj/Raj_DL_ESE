import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[:5000]/255, x_test[:1000]/255
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

learning_rates = [0.001, 0.01, 0.1]
lr_histories = {}

for lr in learning_rates:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lr_histories[lr] = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"LR {lr}: Test Accuracy = {acc:.4f}, Best Val Loss Epoch = {np.argmin(lr_histories[lr].history['val_loss'])}")

epoch_ranges = [5, 10, 15, 20]
epoch_histories = {}

for epochs in epoch_ranges:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    epoch_histories[epochs] = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    overfit_gap = epoch_histories[epochs].history['accuracy'][-1] - epoch_histories[epochs].history['val_accuracy'][-1]
    print(f"Epochs {epochs}: Test Accuracy = {acc:.4f}, Overfit Gap = {overfit_gap:.4f}")

model_baseline = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_baseline.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist_baseline = model_baseline.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

hyperparam_configs = [
    {'lr': 0.001, 'batch_size': 16},
    {'lr': 0.01, 'batch_size': 32},
    {'lr': 0.1, 'batch_size': 64}
]

hyperparam_results = {}

for config in hyperparam_configs:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, epochs=15, batch_size=config['batch_size'], validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    hyperparam_results[str(config)] = {'accuracy': acc, 'history': hist}
    print(f"Config {config}: Test Accuracy = {acc:.4f}")

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
for lr in learning_rates:
    plt.plot(lr_histories[lr].history['loss'], label=f'LR {lr}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 5: Learning Rate Impact on Loss')

plt.subplot(2, 3, 2)
for lr in learning_rates:
    plt.plot(lr_histories[lr].history['val_loss'], label=f'LR {lr}')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('EXP 5: LR Impact on Convergence')

plt.subplot(2, 3, 3)
plt.plot(hist_baseline.history['loss'], label='Train Loss')
plt.plot(hist_baseline.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 5: Train vs Validation Loss')

plt.subplot(2, 3, 4)
for epochs in epoch_ranges:
    plt.plot(epoch_histories[epochs].history['val_loss'], label=f'{epochs} Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('EXP 5: Epoch Count Effect')

plt.subplot(2, 3, 5)
plt.plot(hist_baseline.history['accuracy'], label='Train Acc')
plt.plot(hist_baseline.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('EXP 5: Generalization Analysis')

plt.subplot(2, 3, 6)
configs_str = [str(c) for c in hyperparam_configs]
accs = [hyperparam_results[c]['accuracy'] for c in configs_str]
plt.bar(range(len(configs_str)), accs)
plt.xticks(range(len(configs_str)), ['LR:0.001', 'LR:0.01', 'LR:0.1'], rotation=45)
plt.ylabel('Accuracy')
plt.title('EXP 5: Hyperparameter Comparison')
plt.tight_layout()
plt.savefig('exp5_performance_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 5 Complete: Performance and hyperparameter analysis")
