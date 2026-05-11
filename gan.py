import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

x_train = torch.stack([mnist_train[i][0] for i in range(min(5000, len(mnist_train)))])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

generator = Generator()
discriminator = Discriminator()

gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

generator = generator.to(device)
discriminator = discriminator.to(device)

batch_size = 64
epochs = 50
histories = {'gen_loss': [], 'disc_loss': [], 'disc_acc': []}
loader = DataLoader(x_train, batch_size=batch_size, shuffle=True)
saved_samples = {}

for epoch in range(epochs):
    epoch_gen_losses = []
    epoch_disc_losses = []
    epoch_correct = 0
    epoch_total = 0

    for real_images in loader:
        real_images = real_images.to(device)
        # label smoothing
        real_labels = torch.full((real_images.size(0), 1), 0.9, device=device)
        fake_labels = torch.full((real_images.size(0), 1), 0.0, device=device)
        
        # Train discriminator
        noise = torch.randn(real_images.size(0), 100).to(device)
        fake_images = generator(noise)

        discriminator.zero_grad()
        real_output = discriminator(real_images)
        real_loss = criterion(real_output, real_labels)

        fake_output = discriminator(fake_images.detach())
        fake_loss = criterion(fake_output, fake_labels)

        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        disc_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        noise = torch.randn(real_images.size(0), 100).to(device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        gen_loss = criterion(fake_output, real_labels)
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        gen_optimizer.step()

        epoch_gen_losses.append(gen_loss.item())
        epoch_disc_losses.append(disc_loss.item())

        epoch_correct += (real_output > 0.5).sum().item() + (fake_output < 0.5).sum().item()
        epoch_total += 2 * real_images.size(0)

    # epoch averages
    avg_gen = float(np.mean(epoch_gen_losses)) if epoch_gen_losses else 0.0
    avg_disc = float(np.mean(epoch_disc_losses)) if epoch_disc_losses else 0.0
    disc_acc = float(epoch_correct) / epoch_total if epoch_total > 0 else 0.0

    histories['gen_loss'].append(avg_gen)
    histories['disc_loss'].append(avg_disc)
    histories['disc_acc'].append(disc_acc)

    # save samples at checkpoints for progression visualization
    if epoch in [0, 5, 10, 15, epochs-1]:
        with torch.no_grad():
            sample_noise = torch.randn(5, 100).to(device)
            saved_samples[epoch] = generator(sample_noise).detach().cpu()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Gen Loss = {avg_gen:.4f}, Disc Loss = {avg_disc:.4f}, Disc Acc = {disc_acc:.4f}")

# final generated samples
final_samples = saved_samples.get(epochs-1, None)
if final_samples is None:
    with torch.no_grad():
        final_samples = generator(torch.randn(5, 100).to(device)).detach().cpu()

fig, axes = plt.subplots(1, 5, figsize=(12, 2))
for i in range(5):
    axes[i].imshow(final_samples[i, 0, :, :], cmap='gray')
    axes[i].axis('off')
plt.suptitle('EXP 10: Generated Samples at Final Epoch')
plt.tight_layout()
plt.savefig('results/exp10_generated_final.png', dpi=100, bbox_inches='tight')
plt.close()

# progression grid using saved_samples
epochs_saved = sorted(saved_samples.keys())
n_rows = len(epochs_saved)
fig, axes = plt.subplots(n_rows, 5, figsize=(12, 2 * n_rows))
for r, ep in enumerate(epochs_saved):
    samples = saved_samples[ep]
    for i in range(5):
        axes[r, i].imshow(samples[i, 0, :, :], cmap='gray')
        axes[r, i].set_title(f'E{ep}')
        axes[r, i].axis('off')

plt.suptitle('EXP 10: Generated Output Progression')
plt.tight_layout()
plt.savefig('results/exp10_progression.png', dpi=100, bbox_inches='tight')
plt.close()

print(f"Final Generator Loss: {histories['gen_loss'][-1]:.4f}")
print(f"Final Discriminator Loss: {histories['disc_loss'][-1]:.4f}")
print(f"Training Stability: Generator and Discriminator losses balanced = {abs(histories['gen_loss'][-1] - histories['disc_loss'][-1]) < 0.5}")

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
plt.savefig('results/exp10_gan_training.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 10 Complete: GAN synthetic data generation")
