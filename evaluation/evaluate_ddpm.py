import csv
import time
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from configurations.configuration import (
    CHECKPOINT_DIR,
    METRIC_DIR,
    DEVICE,
    DDPM_TIMESTEPS,
)
from data.dataset import get_data_loaders
from models.ddpm import DDPM

FID_SAMPLES = 1000
FID_BATCH_SIZE = 32

LATENCY_BATCH_SIZE = 64
LATENCY_REPEATS = 2

CHECKPOINT_PATH = Path(CHECKPOINT_DIR) / "ddpm_best.pth"
HISTORY_PATH = Path(METRIC_DIR) / "ddpm_history.csv"
OUTPUT_PATH = Path(METRIC_DIR) / "ddpm_evaluation.csv"

def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)

def fix_p_sample(model):
    def corrected_p_sample(self, x, t):
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        predicted_noise = self.predict_noise(x, t)

        mean = (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t))* predicted_noise) / torch.sqrt(alpha_t)

        if t[0].item() == 0:
            return mean

        return mean + torch.sqrt(beta_t) * torch.randn_like(x)

    model.p_sample = MethodType(corrected_p_sample, model)

def prepare_fid_images(images):
    images = (images.clamp(-1, 1) + 1) / 2
    images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(
        images,
        size=(299, 299),
        mode="bilinear",
        align_corners=False
    )
    return (images * 255).round().clamp(0, 255).to(torch.uint8)

def get_images(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch

def evaluate_fid(model, test_loader):
    print("\nCalculating DDPM FID")

    fid = FrechetInceptionDistance(feature=2048,normalize=False).to(DEVICE)
    real_count = 0
    for batch in test_loader:
        images = get_images(batch).to(DEVICE)
        remaining = FID_SAMPLES - real_count
        if remaining <= 0:
            break
        images = images[:remaining]
        fid.update(prepare_fid_images(images),real=True)
        real_count += images.size(0)

    if real_count < FID_SAMPLES:
        raise RuntimeError(
            f"Only {real_count} real images available."
        )

    fake_count = 0
    model.eval()

    with torch.inference_mode():
        while fake_count < FID_SAMPLES:
            current_batch = min(FID_BATCH_SIZE,FID_SAMPLES - fake_count)
            generated = model.sample(current_batch,DEVICE)
            fid.update(prepare_fid_images(generated),real=False)

            fake_count += generated.size(0)
            print(f"Generated: {fake_count}/{FID_SAMPLES}")

    return fid.compute().item()

def evaluate_latency(model):
    print("\nMeasuring DDPM inference latency")

    model.eval()
    with torch.inference_mode():
        model.sample(LATENCY_BATCH_SIZE, DEVICE)

    synchronize()
    times = []

    for _ in range(LATENCY_REPEATS):
        synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            model.sample(LATENCY_BATCH_SIZE, DEVICE)
        synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return sum(times) / len(times)

def evaluate_training_history():
    with open(
        HISTORY_PATH,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("DDPM history CSV is empty.")

    epoch_times = [
        float(row["epoch_time_seconds"])
        for row in rows
        if row.get("epoch_time_seconds")
    ]

    avg_epoch_time = (
        sum(epoch_times) / len(epoch_times)
        if epoch_times else None
    )

    return len(rows), avg_epoch_time

def main():

    print("=" * 60)
    print("DDPM EVALUATION")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(HISTORY_PATH)

    _, _, test_loader = get_data_loaders()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = DDPM(timesteps=DDPM_TIMESTEPS).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fix_p_sample(model)

    #metrics
    fid_score = evaluate_fid(model,test_loader)
    latency = evaluate_latency(model)
    epochs, avg_epoch_time = evaluate_training_history()

    #save
    results = {
        "model": "DDPM",
        "fid_score": fid_score,
        "fid_sample_count": FID_SAMPLES,
        "inference_batch_size": LATENCY_BATCH_SIZE,
        "inference_latency_ms_batch": latency,
        "reverse_diffusion_steps": DDPM_TIMESTEPS,
        "epochs_completed": epochs,
        "avg_epoch_time_seconds": avg_epoch_time,
        "test_mse": "N/A",
        "test_psnr": "N/A",
        "test_ssim": "N/A"
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results.keys()
        )
        writer.writeheader()
        writer.writerow(results)

    print("\nDDPM EVALUATION RESULTS")

    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()