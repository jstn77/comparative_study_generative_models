import argparse
import csv
import sys
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
import os
os.environ["TORCH_HOME"] = r"D:\TorchCache"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configurations.configuration import (
    DEVICE, CHECKPOINT_DIR, METRIC_DIR, GAN_NOISE_DIM, DCGAN_NOISE_DIM,
    DDPM_TIMESTEPS
)
from data.dataset import get_data_loaders
from models.vae import VAE
from models.gan import VanillaGenerator
from models.dcgan import DCGANGenerator

FID_SAMPLES = 1000
GEN_BATCH_SIZE = 64

def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)

def to_inception_uint8(images):
    images = ((images.clamp(-1, 1) + 1.0) / 2.0)
    images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    return (images * 255.0).round().to(torch.uint8)

def load_fid_metric():
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as exc:
        raise RuntimeError(
            "Dependensi FID belum tersedia. Jalankan:\n"
            "pip install torchmetrics torch-fidelity"
        ) from exc
    return FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)

def update_real_images(fid, test_loader, count):
    seen = 0
    for images, _ in test_loader:
        remaining = count - seen
        if remaining <= 0:
            break
        images = images[:remaining].to(DEVICE)
        fid.update(to_inception_uint8(images), real=True)
        seen += images.size(0)
    if seen != count:
        raise RuntimeError(f"Real image count {seen}, expected {count}")

def corrected_ddpm_p_sample(self, x, t):
    beta_t = self.betas[t].view(-1, 1, 1, 1)
    alpha_t = self.alphas[t].view(-1, 1, 1, 1)
    alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
    predicted_noise = self.predict_noise(x, t)
    mean = (
        x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
    ) / torch.sqrt(alpha_t)
    if t[0].item() == 0:
        return mean
    return mean + torch.sqrt(beta_t) * torch.randn_like(x)

@torch.no_grad()
def generate_for_model(name, count, batch_size):
    if name == "VAE":
        model = VAE().to(DEVICE).eval()
        ckpt = load_checkpoint(Path(CHECKPOINT_DIR) / "vae_best.pth")
        model.load_state_dict(ckpt["model_state_dict"])
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            yield model.sample(n, DEVICE)

    elif name == "Vanilla GAN":
        model = VanillaGenerator().to(DEVICE).eval()
        ckpt = load_checkpoint(Path(CHECKPOINT_DIR) / "vanilla_gan_best.pth")
        model.load_state_dict(ckpt["generator_state_dict"])
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            z = torch.randn(n, GAN_NOISE_DIM, device=DEVICE)
            yield model(z)

    elif name == "DCGAN":
        model = DCGANGenerator().to(DEVICE).eval()
        ckpt = load_checkpoint(Path(CHECKPOINT_DIR) / "dcgan_best.pth")
        model.load_state_dict(ckpt["generator_state_dict"])
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            z = torch.randn(n, DCGAN_NOISE_DIM, 1, 1, device=DEVICE)
            yield model(z)

    elif name == "DDPM":
        from models.ddpm import DDPM
        model = DDPM(timesteps=DDPM_TIMESTEPS).to(DEVICE).eval()
        ckpt = load_checkpoint(Path(CHECKPOINT_DIR) / "ddpm_best.pth")
        model.load_state_dict(ckpt["model_state_dict"])
        # Fixes per-batch scalar broadcasting in the existing reverse sampler.
        model.p_sample = MethodType(corrected_ddpm_p_sample, model)
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            yield model.sample(n, DEVICE)
    else:
        raise ValueError(name)

def calculate_fid(name, test_loader, count):
    fid = load_fid_metric()
    update_real_images(fid, test_loader, count)
    generated = 0
    for batch in generate_for_model(name, count, GEN_BATCH_SIZE):
        fid.update(to_inception_uint8(batch), real=False)
        generated += batch.size(0)
        print(f"{name}: generated {generated}/{count}")
    if generated != count:
        raise RuntimeError(f"Generated {generated}, expected {count}")
    return float(fid.compute().item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-ddpm", action="store_true")
    args = parser.parse_args()

    _, _, test_loader = get_data_loaders()
    if FID_SAMPLES > len(test_loader.dataset):
        raise ValueError("FID_SAMPLES cannot exceed test set size.")

    names = ["VAE", "Vanilla GAN", "DCGAN"]
    if args.include_ddpm:
        names.append("DDPM")

    results = []
    for name in names:
        print(f"\n=== FID: {name} | N={FID_SAMPLES} | device={DEVICE} ===")
        score = calculate_fid(name, test_loader, FID_SAMPLES)
        results.append({"model": name, "fid_score": score, "sample_count": FID_SAMPLES})
        print(f"[RESULT] {name} FID = {score:.6f}")

    out = Path(METRIC_DIR) / "fid_test_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        with out.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["model"]] = row
    for row in results:
        existing[row["model"]] = row
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "fid_score", "sample_count"])
        writer.writeheader()
        writer.writerows(existing.values())
    print(f"\n[SAVED] {out}")

if __name__ == "__main__":
    main()
