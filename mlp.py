import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[:5000]/255, x_test[:1000]/255
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

activations = ['relu', 'tanh', 'sigmoid']
histories = {}

for act in activations:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=act, input_shape=(784,)),
        tf.keras.layers.Dense(64, activation=act),
        tf.keras.layers.Dense(32, activation=act),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    histories[act] = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{act}: Test Accuracy = {acc:.4f}")

for act in activations:
    plt.plot(histories[act].history['loss'], label=f'{act} - Train Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('EXP 1: Activation Function Impact on Training Stability')
plt.savefig('exp1_mlp.png', dpi=100, bbox_inches='tight')
plt.close()

depths = [1, 2, 3, 4]
depth_results = {}

for d in depths:
    layers = [tf.keras.layers.Dense(128, activation='relu', input_shape=(784,))]
    for _ in range(d-1):
        layers.append(tf.keras.layers.Dense(64, activation='relu'))
    layers.append(tf.keras.layers.Dense(10, activation='softmax'))
    
    model = tf.keras.Sequential(layers)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    depth_results[d] = {'accuracy': acc, 'val_loss': hist.history['val_loss']}
    print(f"Depth {d}: Test Accuracy = {acc:.4f}, Convergence Epochs = {np.argmin(hist.history['val_loss'])}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(depth_results.keys(), [v['accuracy'] for v in depth_results.values()])
plt.xlabel('Model Depth')
plt.ylabel('Accuracy')
plt.title('EXP 1: Depth vs Accuracy')

plt.subplot(1, 2, 2)
for d in depths:
    plt.plot(depth_results[d]['val_loss'], label=f'Depth {d}')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('EXP 1: Convergence by Depth')
plt.tight_layout()
plt.savefig('exp1_depth_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 1 Complete: MLP analysis with activation functions and depth variations")
