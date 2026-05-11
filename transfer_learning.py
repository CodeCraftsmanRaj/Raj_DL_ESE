import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    dataset = tf.keras.utils.get_file('cats_and_dogs', 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip', extract=True)
    path = os.path.join(os.path.dirname(dataset), 'PetImages')
except:
    print("Using placeholder data for transfer learning")
    path = None

if path:
    img_size = (150, 150)
    batch_size = 32
    train_ds = tf.keras.utils.image_dataset_from_directory(path, seed=42, image_size=img_size, batch_size=batch_size, subset='training', validation_split=0.2, label_mode='binary')
    val_ds = tf.keras.utils.image_dataset_from_directory(path, seed=42, image_size=img_size, batch_size=batch_size, subset='validation', validation_split=0.2, label_mode='binary')
    test_sample = next(iter(train_ds.take(1)))[0][:5]
else:
    img_size = (150, 150)
    train_ds = tf.data.Dataset.from_tensor_slices((np.random.randn(100, 150, 150, 3).astype('float32'), np.random.randint(0, 2, 100))).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((np.random.randn(20, 150, 150, 3).astype('float32'), np.random.randint(0, 2, 20))).batch(32)
    test_sample = next(iter(train_ds.take(1)))[0][:5]

base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

model_frozen = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_frozen.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist_frozen = model_frozen.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
val_loss_frozen, val_acc_frozen = model_frozen.evaluate(val_ds, verbose=0)
print(f"Frozen: Val Accuracy = {val_acc_frozen:.4f}")

for layer in base_model.layers[-20:]:
    layer.trainable = True

model_finetune = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_finetune.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
hist_finetune = model_finetune.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
val_loss_finetune, val_acc_finetune = model_finetune.evaluate(val_ds, verbose=0)
print(f"Fine-tuned: Val Accuracy = {val_acc_finetune:.4f}")

print(f"Frozen improvement over epochs: {hist_frozen.history['accuracy']}")
print(f"Fine-tuned improvement over epochs: {hist_finetune.history['accuracy']}")

small_train = tf.data.Dataset.from_tensor_slices((np.random.randn(50, 150, 150, 3).astype('float32'), np.random.randint(0, 2, 50))).batch(32)
small_val = tf.data.Dataset.from_tensor_slices((np.random.randn(20, 150, 150, 3).astype('float32'), np.random.randint(0, 2, 20))).batch(32)

model_small = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_small.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist_small = model_small.fit(small_train, validation_data=small_val, epochs=5, verbose=0)
val_loss_small, val_acc_small = model_small.evaluate(small_val, verbose=0)
print(f"Small Dataset: Val Accuracy = {val_acc_small:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(hist_frozen.history['accuracy'], label='Frozen')
plt.plot(hist_finetune.history['accuracy'], label='Fine-tuned')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('EXP 3: Frozen vs Fine-tuned')

plt.subplot(1, 3, 2)
plt.plot(hist_frozen.history['loss'], label='Frozen Loss')
plt.plot(hist_finetune.history['loss'], label='Fine-tuned Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('EXP 3: Training Loss Comparison')

plt.subplot(1, 3, 3)
plt.bar(['Frozen', 'Fine-tuned', 'Small Dataset'], [val_acc_frozen, val_acc_finetune, val_acc_small])
plt.ylabel('Accuracy')
plt.title('EXP 3: Performance Summary')
plt.tight_layout()
plt.savefig('exp3_transfer_learning.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 3 Complete: Transfer Learning analysis")
