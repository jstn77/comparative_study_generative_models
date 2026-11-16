import os
import sys
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import csv
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from configurations.configuration import (
    DEVICE, SEED, IMAGE_DIM, IMAGE_SIZE, NUM_CLASSES, HIDDEN_DIM,
    LATENT_DIM, LEARNING_RATE, BATCH_SIZE, CHECKPOINT_DIR, METRIC_DIR,
    RESULT_DIR
)
from data.dataset import get_data_loaders
from models.vanilla_ae import Autoencoder

MLP_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "mlp_best.pth")
AE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "vanilla_ae_best.pth")
DECODER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "mlp_frozen_decoder_best.pth")
MLP_OUTPUT_DIR = os.path.join(RESULT_DIR, "mlp_experiments")
FIGURE_DIR = os.path.join(MLP_OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(MLP_OUTPUT_DIR, "metrics")
DECODER_EPOCHS = 30
DECODER_BATCH_SIZE = 256
DECODER_LR = LEARNING_RATE
N_DISPLAY = 8
N_PRIOR_SAMPLES = 16

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IMAGE_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM, NUM_CLASSES)
        self.relu = nn.ReLU()

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h1 = self.relu(self.fc1(x))
        z = self.relu(self.fc2(h1))
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.fc3(z)

class FrozenLatentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, IMAGE_DIM),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.network(z).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)

def load_checkpoint_state(path, model):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint

def to_unit_range(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

def collect_latents(model, loader):
    latent_parts, image_parts, label_parts = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            z = model.encode(images)
            latent_parts.append(z.cpu())
            image_parts.append(to_unit_range(images).cpu())
            label_parts.append(labels.cpu())
    return (
        torch.cat(latent_parts),
        torch.cat(image_parts),
        torch.cat(label_parts)
    )

def evaluate_classifier(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss_sum += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, loss_sum / total


def ssim_per_image(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    try:
        from skimage.metrics import structural_similarity
        values = [
            structural_similarity(target_np[i, 0], pred_np[i, 0], data_range=1.0)
            for i in range(pred_np.shape[0])
        ]
        return np.asarray(values, dtype=np.float64)
    except ImportError:
        import torch.nn.functional as F
        x, y = pred, target
        mu_x = F.avg_pool2d(x, 7, stride=1, padding=3)
        mu_y = F.avg_pool2d(y, 7, stride=1, padding=3)
        sigma_x = F.avg_pool2d(x * x, 7, stride=1, padding=3) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, 7, stride=1, padding=3) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, 7, stride=1, padding=3) - mu_x * mu_y
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        return score.mean(dim=(1, 2, 3)).detach().cpu().numpy()

def reconstruction_metrics(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse_each = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    psnr_each = 10.0 * torch.log10(1.0 / mse_each.clamp_min(1e-12))
    ssim_each = ssim_per_image(pred, target)
    return float(mse_each.mean()), float(psnr_each.mean()), float(np.mean(ssim_each))

def save_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def save_grid(images, path, titles=None, ncols=8, cmap="gray", vmin=0, vmax=1):
    images = images.detach().cpu().clamp(0, 1)
    n = len(images)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.8))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, 0].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

#no1
def stage1_load_and_evaluate_mlp(test_loader):
    print("\n[STAGE 1] Load checkpoint and evaluate MLP Classifier")
    model = MLPClassifier().to(DEVICE)
    checkpoint = load_checkpoint_state(MLP_CHECKPOINT, model)
    test_accuracy, test_loss = evaluate_classifier(model, test_loader)
    saved_epoch = checkpoint.get("epoch", "unknown")
    print(f"Checkpoint epoch: {saved_epoch}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.6f}")
    if test_accuracy < 0.88:
        print("WARNING: Test accuracy belum mencapai target minimal 88%. Jangan klaim target tercapai.")
    else:
        print("Target test accuracy >= 88% tercapai.")
    save_csv(
        os.path.join(METRICS_DIR, "mlp_classifier_test_metrics.csv"),
        ["model", "checkpoint_epoch", "test_accuracy", "test_loss", "target_accuracy"],
        [["MLP Classifier", saved_epoch, test_accuracy, test_loss, 0.88]]
    )
    return model


#no2&3
def transpose_class_weights(model):
    one_hot = torch.eye(NUM_CLASSES, device=DEVICE)
    with torch.no_grad():
        z_hat = torch.relu(one_hot @ model.fc3.weight)
        h1_hat = torch.relu(z_hat @ model.fc2.weight)
        x_hat = torch.sigmoid(h1_hat @ model.fc1.weight)
    return x_hat.view(NUM_CLASSES, 1, IMAGE_SIZE, IMAGE_SIZE).cpu()

def transpose_latent_features(model, z):
    with torch.no_grad():
        h1_hat = torch.relu(z.to(DEVICE) @ model.fc2.weight)
        x_hat = torch.sigmoid(h1_hat @ model.fc1.weight)
    return x_hat.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE).cpu()

def stage2_and_stage3(model, test_loader):
    print("\n[STAGE 2] Top-down weight transposition")
    class_images = transpose_class_weights(model)
    save_grid(
        class_images,
        os.path.join(FIGURE_DIR, "stage2_class_weight_transpose.png"),
        titles=[f"Class {i}" for i in range(NUM_CLASSES)], ncols=5
    )

    print("[STAGE 3] Latent transpose projection")
    z_test, x_test, y_test = collect_latents(model, test_loader)
    n = min(N_DISPLAY, len(z_test))
    projected = transpose_latent_features(model, z_test[:n])
    originals = x_test[:n]

    fig, axes = plt.subplots(2, n, figsize=(n * 1.7, 4))
    for i in range(n):
        axes[0, i].imshow(originals[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Original\n{int(y_test[i])}", fontsize=8)
        axes[1, i].imshow(projected[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("W transpose", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "stage3_latent_transpose_projection.png"), dpi=300, bbox_inches="tight")
    plt.close()

    direct_mse = float(((projected - originals[:n]) ** 2).mean().item())
    save_csv(
        os.path.join(METRICS_DIR, "mlp_transpose_projection.csv"),
        ["experiment", "n_images", "mse_01_range"],
        [["Latent transpose projection", n, direct_mse]]
    )
    return z_test, x_test, y_test

#no4
def make_latent_dataset(model, loader):
    z_all, x_all, _ = collect_latents(model, loader)
    return TensorDataset(z_all, x_all)

def train_frozen_decoder(model, train_loader, val_loader):
    print("\n[STAGE 4] Train separate decoder while keeping MLP frozen")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    train_ds = make_latent_dataset(model, train_loader)
    val_ds = make_latent_dataset(model, val_loader)
    train_dl = DataLoader(train_ds, batch_size=DECODER_BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=DECODER_BATCH_SIZE, shuffle=False)

    decoder = FrozenLatentDecoder().to(DEVICE)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR)
    criterion = nn.MSELoss()
    best_val = float("inf")
    history = []

    for epoch in range(1, DECODER_EPOCHS + 1):
        start = time.time()
        decoder.train()
        train_sum, train_count = 0.0, 0
        for z, target in train_dl:
            z, target = z.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            prediction = decoder(z)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * z.size(0)
            train_count += z.size(0)

        decoder.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for z, target in val_dl:
                z, target = z.to(DEVICE), target.to(DEVICE)
                loss = criterion(decoder(z), target)
                val_sum += loss.item() * z.size(0)
                val_count += z.size(0)
        train_loss = train_sum / train_count
        val_loss = val_sum / val_count
        elapsed = time.time() - start
        history.append([epoch, train_loss, val_loss, elapsed])
        print(f"Epoch {epoch:02d}/{DECODER_EPOCHS} | train={train_loss:.6f} | val={val_loss:.6f} | {elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, DECODER_CHECKPOINT)

    save_csv(
        os.path.join(METRICS_DIR, "mlp_frozen_decoder_history.csv"),
        ["epoch", "train_loss", "val_loss", "epoch_time_seconds"], history
    )
    best_ckpt = torch.load(DECODER_CHECKPOINT, map_location=DEVICE, weights_only=False)
    decoder.load_state_dict(best_ckpt["model_state_dict"])
    return decoder

def evaluate_decoder_vs_ae(decoder, mlp, test_loader):
    print("Evaluating frozen MLP decoder against Vanilla AE on test set")
    ae = Autoencoder().to(DEVICE)
    load_checkpoint_state(AE_CHECKPOINT, ae)
    ae.eval()
    decoder.eval()
    mlp.eval()

    all_targets, all_mlp, all_ae = [], [], []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            z = mlp.encode(images)
            mlp_recon = decoder(z)
            ae_recon = to_unit_range(ae(images))
            target = to_unit_range(images)
            all_targets.append(target.cpu())
            all_mlp.append(mlp_recon.cpu())
            all_ae.append(ae_recon.cpu())

    targets = torch.cat(all_targets)
    mlp_recons = torch.cat(all_mlp)
    ae_recons = torch.cat(all_ae)
    mlp_metrics = reconstruction_metrics(mlp_recons, targets)
    ae_metrics = reconstruction_metrics(ae_recons, targets)

    save_csv(
        os.path.join(METRICS_DIR, "mlp_vs_vanilla_ae_reconstruction.csv"),
        ["model", "test_mse", "test_psnr_db", "test_ssim", "test_images"],
        [
            ["Frozen MLP Latent Decoder", *mlp_metrics, len(targets)],
            ["Vanilla AE", *ae_metrics, len(targets)],
        ]
    )

    n = min(N_DISPLAY, len(targets))
    fig, axes = plt.subplots(3, n, figsize=(n * 1.7, 5.2))
    rows = [(targets, "Original"), (mlp_recons, "Frozen MLP Decoder"), (ae_recons, "Vanilla AE")]
    for r, (images, row_title) in enumerate(rows):
        for c in range(n):
            axes[r, c].imshow(images[c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(row_title, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "stage4_frozen_decoder_vs_ae.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return targets, mlp_recons, ae_recons

#no5
def stage5_prior_sampling(decoder, z_test):
    print("\n[STAGE 5] Random prior sampling in MLP latent space")
    mu = z_test.mean(dim=0)
    std = z_test.std(dim=0, unbiased=False).clamp_min(1e-6)
    z_random = mu.unsqueeze(0) + torch.randn(N_PRIOR_SAMPLES, LATENT_DIM) * std.unsqueeze(0)
    decoder.eval()
    with torch.no_grad():
        generated = decoder(z_random.to(DEVICE)).cpu()
    save_grid(
        generated,
        os.path.join(FIGURE_DIR, "stage5_mlp_latent_prior_sampling.png"),
        titles=[f"Sample {i+1}" for i in range(N_PRIOR_SAMPLES)], ncols=4
    )
    save_csv(
        os.path.join(METRICS_DIR, "mlp_latent_distribution.csv"),
        ["latent_dimension", "mean", "std"],
        [[i, float(mu[i]), float(std[i])] for i in range(LATENT_DIM)]
    )
    return generated

#fig3
def create_figure3_summary(class_images, test_loader, mlp, decoder, z_test, x_test, y_test):
    n = min(4, len(z_test))
    latent_projection = transpose_latent_features(mlp, z_test[:n])
    decoder.eval()
    with torch.no_grad():
        frozen_decoded = decoder(z_test[:n].to(DEVICE)).cpu()

    ae = Autoencoder().to(DEVICE)
    load_checkpoint_state(AE_CHECKPOINT, ae)
    ae.eval()
    with torch.no_grad():
        ae_reconstruction = to_unit_range(ae((x_test[:n].to(DEVICE) * 2.0) - 1.0)).cpu()

    mu = z_test.mean(dim=0)
    std = z_test.std(dim=0, unbiased=False).clamp_min(1e-6)
    z_rand = mu.unsqueeze(0) + torch.randn(4, LATENT_DIM) * std.unsqueeze(0)
    with torch.no_grad():
        prior = decoder(z_rand.to(DEVICE)).cpu()

    panels = [
        (x_test[:n], "(A) Ground truth"),
        (class_images[:n], "(B1) Class-vector W transpose"),
        (latent_projection, "(B2) Latent W transpose"),
        (frozen_decoded, "(C1) Frozen MLP latent decoder"),
        (ae_reconstruction, "(C2) Vanilla AE decoder"),
        (prior, "(D) Random latent prior samples"),
    ]
    fig, axes = plt.subplots(len(panels), n, figsize=(n * 1.8, len(panels) * 1.65))
    for r, (imgs, label) in enumerate(panels):
        for c in range(n):
            axes[r, c].imshow(imgs[c, 0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=8)
    fig.suptitle("Empirical Limitations of MLP Classifier for Image Generation", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "figure3_mlp_classifier_limitations.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    set_seed()
    print("Device:", DEVICE)
    print("Same deterministic Fashion-MNIST split")
    train_loader, val_loader, test_loader = get_data_loaders()

    mlp = stage1_load_and_evaluate_mlp(test_loader)
    class_images = transpose_class_weights(mlp)
    stage2_and_stage3(mlp, test_loader)
    z_test, x_test, y_test = collect_latents(mlp, test_loader)

    decoder = train_frozen_decoder(mlp, train_loader, val_loader)
    evaluate_decoder_vs_ae(decoder, mlp, test_loader)
    stage5_prior_sampling(decoder, z_test)
    create_figure3_summary(class_images, test_loader, mlp, decoder, z_test, x_test, y_test)

    print("\nMLP experiment completed")
    print("Figures:", FIGURE_DIR)
    print("Metrics:", METRICS_DIR)
    print("Decoder checkpoint:", DECODER_CHECKPOINT)

if __name__ == "__main__":
    main()