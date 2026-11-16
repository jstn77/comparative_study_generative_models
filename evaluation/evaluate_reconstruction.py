import csv
import sys
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import structural_similarity
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from configurations.configuration import DEVICE, CHECKPOINT_DIR, METRIC_DIR
from data.dataset import get_data_loaders
from models.vanilla_ae import Autoencoder
from models.sparse_ae import SparseAutoencoder
from models.vae import VAE

BATCH_SIZE = 128
DATA_RANGE = 2.0

def load_state(model, filename):
    path = Path(CHECKPOINT_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=DEVICE)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"model_state_dict tidak ditemukan pada {filename}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"[LOAD] {filename} | best epoch: {ckpt.get('epoch', 'unknown')}")
    return model

def calculate_metrics(model, test_loader, model_name):
    total_squared_error = 0.0
    total_elements = 0
    psnr_values = []
    ssim_values = []
    criterion = torch.nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            output = model(images)
            if isinstance(output, tuple):  #VAE (reconstruction, mu, logvar)
                output = output[0]
            output = output.clamp(-1, 1)

            total_squared_error += criterion(output, images).item()
            total_elements += images.numel()

            real_np = images.detach().cpu().numpy()
            pred_np = output.detach().cpu().numpy()

            for i in range(real_np.shape[0]):
                mse_i = float(np.mean((real_np[i] - pred_np[i]) ** 2))
                psnr_values.append(
                    float("inf") if mse_i == 0
                    else 10.0 * np.log10((DATA_RANGE ** 2) / mse_i)
                )
                ssim_values.append(
                    structural_similarity(real_np[i, 0],pred_np[i, 0],data_range=DATA_RANGE)
                )

    mse = total_squared_error / total_elements
    return {
        "model": model_name,
        "test_mse": mse,
        "test_psnr_db": float(np.mean(psnr_values)),
        "test_ssim": float(np.mean(ssim_values)),
        "test_images": len(test_loader.dataset),
    }

def main():
    _, _, test_loader = get_data_loaders()
    print(f"Test images: {len(test_loader.dataset)} | Device: {DEVICE}")

    models = [
        ("Vanilla AE", Autoencoder(), "vanilla_ae_best.pth"),
        ("Sparse AE", SparseAutoencoder(), "sparse_ae_best.pth"),
        ("VAE", VAE(), "vae_best.pth"),
    ]

    results = []
    for name, model, checkpoint in models:
        model = load_state(model, checkpoint)
        result = calculate_metrics(model, test_loader, name)
        results.append(result)
        print(
            f"{name}: MSE={result['test_mse']:.6f}, "
            f"PSNR={result['test_psnr_db']:.4f} dB, "
            f"SSIM={result['test_ssim']:.6f}"
        )

    output = Path(METRIC_DIR) / "reconstruction_test_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SAVED] {output}")

if __name__ == "__main__":
    main()