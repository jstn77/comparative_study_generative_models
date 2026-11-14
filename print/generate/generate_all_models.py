import os
import sys
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configurations.configuration import (
    DEVICE, NUM_GENERATED_SAMPLES, LATENT_DIM,
    GAN_NOISE_DIM, DCGAN_NOISE_DIM, CHECKPOINT_DIR, RESULT_DIR
)
from data.dataset import get_data_loaders
from models.vanilla_ae import Autoencoder
from models.sparse_ae import SparseAutoencoder
from models.vae import VAE
from models.gan import VanillaGenerator
from models.dcgan import DCGANGenerator


OUTPUT_DIR = Path(RESULT_DIR) / "visualizations"
GENERATED_DIR = OUTPUT_DIR / "generated_img"
RECON_DIR = OUTPUT_DIR / "reconstructions"
NUM_SAMPLES = NUM_GENERATED_SAMPLES  


def load_checkpoint_state(path, key="model_state_dict"):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:  
        checkpoint = torch.load(path, map_location=DEVICE)
    if key not in checkpoint:
        raise KeyError(
            f"Key '{key}' tidak ditemukan di {path.name}. "
            f"Key checkpoint tersedia: {list(checkpoint.keys())}"
        )
    return checkpoint[key]


def save_grid(images, path, nrow=4):
    images = images.detach().cpu().clamp(-1, 1)
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)
    save_image(grid, path)
    print(f"[SAVED] {path}")


def load_autoencoder(model_class, checkpoint_name, label):
    model = model_class().to(DEVICE)
    checkpoint_path = Path(CHECKPOINT_DIR) / checkpoint_name
    model.load_state_dict(load_checkpoint_state(checkpoint_path))
    model.eval()
    print(f"[LOADED] {label}: {checkpoint_path.name}")
    return model


@torch.no_grad()
def create_reconstructions(test_loader):
    print("\n=== RECONSTRUCTION: AE / SPARSE AE / VAE ===")
    batch_images, _ = next(iter(test_loader))
    originals = batch_images[:NUM_SAMPLES].to(DEVICE)

    ae = load_autoencoder(Autoencoder, "vanilla_ae_best.pth", "Vanilla AE")
    sparse_ae = load_autoencoder(SparseAutoencoder, "sparse_ae_best.pth", "Sparse AE")
    vae = load_autoencoder(VAE, "vae_best.pth", "VAE")

    ae_recon = ae(originals)
    sparse_recon = sparse_ae(originals)
    vae_recon, _, _ = vae(originals)

    save_grid(originals, RECON_DIR / "original_test_images_16.png")
    save_grid(ae_recon, RECON_DIR / "vanilla_ae_reconstruction_16.png")
    save_grid(sparse_recon, RECON_DIR / "sparse_ae_reconstruction_16.png")
    save_grid(vae_recon, RECON_DIR / "vae_reconstruction_16.png")


@torch.no_grad()
def create_random_generations():
    print("\n=== RANDOM GENERATION: VAE / VANILLA GAN / DCGAN ===")

    #checkpoint stores the full VAE
    vae = load_autoencoder(VAE, "vae_best.pth", "VAE")
    vae_samples = vae.sample(NUM_SAMPLES, DEVICE)
    save_grid(vae_samples, GENERATED_DIR / "vae_generated_16.png")

    #GAN checkpoint stores generator_state_dict
    gan = VanillaGenerator().to(DEVICE)
    gan_path = Path(CHECKPOINT_DIR) / "vanilla_gan_best.pth"
    gan.load_state_dict(load_checkpoint_state(gan_path, "generator_state_dict"))
    gan.eval()
    gan_noise = torch.randn(NUM_SAMPLES, GAN_NOISE_DIM, device=DEVICE)
    gan_samples = gan(gan_noise)
    save_grid(gan_samples, GENERATED_DIR / "vanilla_gan_generated_16.png")
    print(f"[LOADED] Vanilla GAN: {gan_path.name}")

    #DCGAN checkpoint stores generator_state_dict
    dcgan = DCGANGenerator().to(DEVICE)
    dcgan_path = Path(CHECKPOINT_DIR) / "dcgan_best.pth"
    dcgan.load_state_dict(load_checkpoint_state(dcgan_path, "generator_state_dict"))
    dcgan.eval()
    dcgan_noise = torch.randn(NUM_SAMPLES, DCGAN_NOISE_DIM, 1, 1, device=DEVICE)
    dcgan_samples = dcgan(dcgan_noise)
    save_grid(dcgan_samples, GENERATED_DIR / "dcgan_generated_16.png")
    print(f"[LOADED] DCGAN: {dcgan_path.name}")


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    RECON_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("VISUALIZATION GENERATION - WITHOUT DDPM")
    print("Device:", DEVICE)
    print("Samples per grid:", NUM_SAMPLES)
    print("=" * 68)

    _, _, test_loader = get_data_loaders()
    create_reconstructions(test_loader)
    create_random_generations()

    print("\nSelesai. Semua gambar tersimpan di:")
    print(" -", RECON_DIR)
    print(" -", GENERATED_DIR)


if __name__ == "__main__":
    main()
