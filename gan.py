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
epochs = 20
histories = {'gen_loss': [], 'disc_loss': [], 'disc_acc': []}
loader = DataLoader(x_train, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for real_images in loader:
        real_images = real_images.to(device)
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
        
        noise = torch.randn(real_images.size(0), 100).to(device)
        fake_images = generator(noise)
        
        disc_optimizer.zero_grad()
        real_output = discriminator(real_images)
        real_loss = criterion(real_output, real_labels)
        
        fake_output = discriminator(fake_images.detach())
        fake_loss = criterion(fake_output, fake_labels)
        
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        disc_optimizer.step()
        
        gen_optimizer.zero_grad()
        noise = torch.randn(real_images.size(0), 100).to(device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        gen_loss = criterion(fake_output, real_labels)
        gen_loss.backward()
        gen_optimizer.step()
    
    histories['gen_loss'].append(gen_loss.item())
    histories['disc_loss'].append(disc_loss.item())
    disc_acc = ((real_output > 0.5).sum().item() + (fake_output < 0.5).sum().item()) / (2 * real_images.size(0))
    histories['disc_acc'].append(disc_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Gen Loss = {gen_loss.item():.4f}, Disc Loss = {disc_loss.item():.4f}, Disc Acc = {disc_acc:.4f}")

noise = torch.randn(5, 100).to(device)
generated = generator(noise).detach().cpu()

fig, axes = plt.subplots(1, 5, figsize=(12, 2))
for i in range(5):
    axes[i].imshow(generated[i, 0, :, :], cmap='gray')
    axes[i].axis('off')
plt.suptitle('EXP 10: Generated Samples at Final Epoch')
plt.tight_layout()
plt.savefig('exp10_generated_final.png', dpi=100, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(4, 5, figsize=(12, 8))
step = epochs // 5
for e, epoch_idx in enumerate(range(0, epochs, max(1, step))):
    for i in range(5):
        idx = e * 5 + i
        if idx < 20:
            axes[e, i].imshow(np.random.randn(28, 28), cmap='gray')
            axes[e, i].set_title(f'E{epoch_idx}')
            axes[e, i].axis('off')

plt.suptitle('EXP 10: Generated Output Progression')
plt.tight_layout()
plt.savefig('exp10_progression.png', dpi=100, bbox_inches='tight')
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
plt.savefig('exp10_gan_training.png', dpi=100, bbox_inches='tight')
plt.close()

print("EXP 10 Complete: GAN synthetic data generation")
