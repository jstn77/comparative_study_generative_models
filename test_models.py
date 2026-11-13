import torch

from configurations.configuration import DEVICE

from models.vanilla_ae import Autoencoder
from models.sparse_ae import SparseAutoencoder
from models.vae import VAE
from models.gan import VanillaGAN
from models.dcgan import DCGAN
from models.ddpm import DDPM
from models.mlp_classifier import MLPClassifier


# DUMMY INPUT
x = torch.randn(4, 1, 28, 28).to(DEVICE)

print("Testing all models...\n")


# 1. AUTOENCODER
ae = Autoencoder().to(DEVICE)
print("AE:", ae(x).shape)


# 2. SPARSE AUTOENCODER
sparse_ae = SparseAutoencoder().to(DEVICE)
print("Sparse AE:", sparse_ae(x).shape)


# 3. VAE
vae = VAE().to(DEVICE)
reconstruction, mu, logvar = vae(x)

print("VAE:", reconstruction.shape)
print("VAE mu:", mu.shape)
print("VAE logvar:", logvar.shape)


# 4. VANILLA GAN
gan = VanillaGAN().to(DEVICE)
print("Vanilla GAN:", gan.generate(4, DEVICE).shape)


# 5. DCGAN
dcgan = DCGAN().to(DEVICE)
print("DCGAN:", dcgan.generate(4, DEVICE).shape)


# 6. DDPM
ddpm = DDPM().to(DEVICE)
t = torch.randint(0, ddpm.timesteps, (4,), device=DEVICE)
noisy_images, noise = ddpm.q_sample(x, t)

print("DDPM noisy image:", noisy_images.shape)
print("DDPM predicted noise:", ddpm.predict_noise(noisy_images, t).shape)


# 7. MLP CLASSIFIER
mlp = MLPClassifier().to(DEVICE)
print("MLP:", mlp(x).shape)
print("MLP latent:", mlp.encode(x).shape)


print("\nAll model tests completed!")