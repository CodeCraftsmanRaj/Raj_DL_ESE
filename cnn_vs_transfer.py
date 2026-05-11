import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[:5000]/255, x_test[:1000]/255
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

base_model = tf.keras.applications.MobileNetV2(input_shape=(28, 28, 3), include_top=False, weights='imagenet')
x_train_rgb = np.concatenate([x_train]*3, axis=-1)
x_test_rgb = np.concatenate([x_test]*3, axis=-1)

model_scratch = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_transfer = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

for model, name in [(model_scratch, 'From Scratch'), (model_transfer, 'Transfer Learning')]:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    hist = model.fit(x_train_rgb if name == 'Transfer Learning' else x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    train_time = time.time() - start
    loss, acc = model.evaluate(x_test_rgb if name == 'Transfer Learning' else x_test, y_test, verbose=0)
    print(f"{name}: Test Accuracy = {acc:.4f}, Training Time = {train_time:.2f}s")

model_scratch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist_scratch = model_scratch.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

model_transfer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist_transfer = model_transfer.fit(x_train_rgb, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.plot(hist_scratch.history['loss'], label='From Scratch')
plt.plot(hist_transfer.history['loss'], label='Transfer Learning')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 4: Training Loss Comparison')

plt.subplot(1, 3, 2)
plt.plot(hist_scratch.history['accuracy'], label='From Scratch')
plt.plot(hist_transfer.history['accuracy'], label='Transfer Learning')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('EXP 4: Training Accuracy Comparison')

plt.subplot(1, 3, 3)
dataset_sizes = [500, 2000, 5000]
scratch_accs = []
transfer_accs = []
for size in dataset_sizes:
    m1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    m1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m1.fit(x_train[:size], y_train[:size], epochs=5, batch_size=32, verbose=0)
    _, acc1 = m1.evaluate(x_test, y_test, verbose=0)
    scratch_accs.append(acc1)
    
    m2 = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    m2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m2.fit(np.concatenate([x_train[:size]]*3, axis=-1), y_train[:size], epochs=5, batch_size=32, verbose=0)
    _, acc2 = m2.evaluate(x_test_rgb, y_test, verbose=0)
    transfer_accs.append(acc2)

plt.plot(dataset_sizes, scratch_accs, marker='o', label='From Scratch')
plt.plot(dataset_sizes, transfer_accs, marker='o', label='Transfer Learning')
plt.xlabel('Dataset Size')
plt.ylabel('Accuracy')
plt.legend()
plt.title('EXP 4: Dataset Size Impact')
plt.tight_layout()
plt.savefig('exp4_cnn_vs_transfer.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 4 Complete: CNN vs Transfer Learning comparison")
