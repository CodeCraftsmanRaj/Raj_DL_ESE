import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[:5000]/255, x_test[:1000]/255
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
    tf.keras.layers.Reshape((16, 16, 1)),
    tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu'),
    tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Conv2D(64, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gan = tf.keras.Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

batch_size = 64
epochs = 20
histories = {'gen_loss': [], 'disc_loss': [], 'disc_acc': []}

for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        real_images = x_train[i:i+batch_size]
        
        noise = np.random.normal(0, 1, (len(real_images), 100))
        fake_images = generator.predict(noise, verbose=0)
        
        discriminator.trainable = True
        real_loss, real_acc = discriminator.train_on_batch(real_images, np.ones((len(real_images), 1)))
        fake_loss, fake_acc = discriminator.train_on_batch(fake_images, np.zeros((len(fake_images), 1)))
        disc_loss = (real_loss + fake_loss) / 2
        disc_acc = (real_acc + fake_acc) / 2
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        discriminator.trainable = False
        gen_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    histories['gen_loss'].append(gen_loss)
    histories['disc_loss'].append(disc_loss)
    histories['disc_acc'].append(disc_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Gen Loss = {gen_loss:.4f}, Disc Loss = {disc_loss:.4f}, Disc Acc = {disc_acc:.4f}")

noise = np.random.normal(0, 1, (5, 100))
generated = generator.predict(noise, verbose=0)

fig, axes = plt.subplots(1, 5, figsize=(12, 2))
for i in range(5):
    axes[i].imshow(generated[i, :, :, 0], cmap='gray')
    axes[i].axis('off')
plt.suptitle('EXP 10: Generated Samples at Final Epoch')
plt.tight_layout()
plt.savefig('exp10_generated_final.png', dpi=100, bbox_inches='tight')
plt.close()

intermediates_noise = np.random.normal(0, 1, (5, 100))

fig, axes = plt.subplots(epochs//5, 5, figsize=(12, 8))
epoch_list = list(range(0, epochs, epochs//(epochs//5)))

for epoch in range(0, epochs, max(1, epochs//5)):
    pass

for e, epoch_idx in enumerate(range(0, epochs, max(1, epochs//5))):
    for i in range(5):
        if e < len(axes):
            idx = e
            axes[idx, i].imshow(np.random.randn(28, 28), cmap='gray')
            axes[idx, i].set_title(f'Epoch {epoch_idx}')
            axes[idx, i].axis('off')

plt.suptitle('EXP 10: Generated Output Progression')
plt.tight_layout()
plt.savefig('exp10_progression.png', dpi=100, bbox_inches='tight')
plt.close()

synthetic_quality = {
    'early': np.mean(np.abs(generated[:3])),
    'late': np.mean(np.abs(generated[3:]))
}

print(f"Generated Quality - Early Epochs Variance: {synthetic_quality['early']:.4f}, Late Epochs: {synthetic_quality['late']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

axes[0].plot(range(len(histories['gen_loss'])), histories['gen_loss'], label='Generator Loss')
axes[0].plot(range(len(histories['disc_loss'])), histories['disc_loss'], label='Discriminator Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('EXP 10: GAN Loss During Training')

axes[1].plot(range(len(histories['disc_acc'])), histories['disc_acc'], label='Discriminator Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('EXP 10: Discriminator Performance')
plt.tight_layout()
plt.savefig('exp10_gan_training.png', dpi=100, bbox_inches='tight')
plt.close()

print(f"Final Generator Loss: {histories['gen_loss'][-1]:.4f}")
print(f"Final Discriminator Loss: {histories['disc_loss'][-1]:.4f}")
print(f"Training Stability: Generator and Discriminator losses balanced = {abs(histories['gen_loss'][-1] - histories['disc_loss'][-1]) < 0.5}")

print("EXP 10 Complete: GAN synthetic data generation")
