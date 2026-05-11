import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[:5000]/255, x_test[:1000]/255
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_fc = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

for name, model in [('CNN', model_cnn), ('FC', model_fc)]:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: Test Accuracy = {acc:.4f}, Params = {model.count_params()}")

model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist_cnn = model_cnn.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

pooling_configs = [('MaxPool 2x2', 2), ('MaxPool 3x3', 3), ('AvgPool 2x2', 'avg')]

for name, pool_size in pooling_configs:
    if pool_size == 'avg':
        layers = [
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.AveragePooling2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.AveragePooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    else:
        layers = [
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(pool_size),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    
    model = tf.keras.Sequential(layers)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: Test Accuracy = {acc:.4f}")

conv_params = [(3, 32), (5, 32), (3, 64)]

for kernel, filters in conv_params:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Kernel {kernel}, Filters {filters}: Test Accuracy = {acc:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_cnn.history['loss'], label='Train Loss')
plt.plot(hist_cnn.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('EXP 2: CNN Training Curves')

plt.subplot(1, 2, 2)
plt.plot(hist_cnn.history['accuracy'], label='Train Acc')
plt.plot(hist_cnn.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('EXP 2: CNN Accuracy')
plt.tight_layout()
plt.savefig('exp2_cnn.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 2 Complete: CNN architecture analysis")
