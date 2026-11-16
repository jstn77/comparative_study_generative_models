import csv
import sys
import time
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from configurations.configuration import (
    DEVICE, CHECKPOINT_DIR, METRIC_DIR, LATENCY_BATCH_SIZE,
    GAN_NOISE_DIM, DCGAN_NOISE_DIM
)
from models.vanilla_ae import Autoencoder
from models.sparse_ae import SparseAutoencoder
from models.vae import VAE
from models.gan import VanillaGenerator
from models.dcgan import DCGANGenerator

WARMUP = 5
REPEATS = 30

def read_epoch_time(csv_name):
    path = Path(METRIC_DIR) / csv_name
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    values = [float(row["epoch_time_seconds"]) for row in rows if row.get("epoch_time_seconds")]
    return {
        "epochs_completed": len(rows),
        "avg_epoch_time_seconds": sum(values) / len(values) if values else None,
        "total_recorded_training_seconds": sum(values) if values else None,
    }

def load_model(model, filename, state_key="model_state_dict"):
    path = Path(CHECKPOINT_DIR) / filename
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt[state_key])
    return model.to(DEVICE).eval()

def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def time_forward(label, callable_fn):
    for _ in range(WARMUP):
        callable_fn()
    sync()
    start = time.perf_counter()
    for _ in range(REPEATS):
        callable_fn()
    sync()
    elapsed = (time.perf_counter() - start) / REPEATS
    return elapsed * 1000.0

def main():
    n = LATENCY_BATCH_SIZE
    x = torch.randn(n, 1, 28, 28, device=DEVICE).clamp(-1, 1)
    results = []

    model_specs = [
        ("Vanilla AE", "vanilla_ae_history.csv", "vanilla_ae_best.pth", Autoencoder()),
        ("Sparse AE", "sparse_ae_history.csv", "sparse_ae_best.pth", SparseAutoencoder()),
        ("VAE", "vae_history.csv", "vae_best.pth", VAE()),
    ]

    for name, history_file, ckpt_file, model in model_specs:
        model = load_model(model, ckpt_file)
        latency = time_forward(name, lambda: model(x))
        timing = read_epoch_time(history_file)
        results.append({
            "model": name,
            **timing,
            "inference_batch_size": n,
            "inference_latency_ms_batch": latency,
            "device": str(DEVICE),
        })
        print(f"{name}: {latency:.3f} ms/batch {n}")

    gan = load_model(VanillaGenerator(), "vanilla_gan_best.pth", "generator_state_dict")
    z_gan = torch.randn(n, GAN_NOISE_DIM, device=DEVICE)
    gan_latency = time_forward("Vanilla GAN", lambda: gan(z_gan))
    gan_time = read_epoch_time("vanilla_gan_history.csv")
    results.append({
        "model": "Vanilla GAN",
        **gan_time,
        "inference_batch_size": n,
        "inference_latency_ms_batch": gan_latency,
        "device": str(DEVICE),
    })
    print(f"Vanilla GAN: {gan_latency:.3f} ms/batch {n}")

    dcgan = load_model(DCGANGenerator(), "dcgan_best.pth", "generator_state_dict")
    z_dcgan = torch.randn(n, DCGAN_NOISE_DIM, 1, 1, device=DEVICE)
    dcgan_latency = time_forward("DCGAN", lambda: dcgan(z_dcgan))
    dcgan_time = read_epoch_time("dcgan_history.csv")
    results.append({
        "model": "DCGAN",
        **dcgan_time,
        "inference_batch_size": n,
        "inference_latency_ms_batch": dcgan_latency,
        "device": str(DEVICE),
    })
    print(f"DCGAN: {dcgan_latency:.3f} ms/batch {n}")

    out = Path(METRIC_DIR) / "training_time_and_latency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "epochs_completed", "avg_epoch_time_seconds",
        "total_recorded_training_seconds", "inference_batch_size",
        "inference_latency_ms_batch", "device"
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[SAVED] {out}")

if __name__ == "__main__":
    main()